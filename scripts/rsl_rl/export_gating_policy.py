"""ゲーティング方策 (凍結 expert 2 本 + ゲート MLP) を 1 つの ONNX に融合して書き出す (2026-09-29)。

デプロイ側 (booster_k1_locomotion の rl_policy_isaaclab_node) は歩行ポリシーと同じ
フレーム形式の履歴 1 本を持ち、1 回の推論で混合後のアクションを得る。C++ 側の契約:

入力
  command     (1, 3)      現モードの意味のコマンド。walk: (vx, vy, ωz) / turn: (0, 0, Δψ/π)
  obs_history (1, H, 49)  歩行ポリシーと同一レイアウトのフレーム履歴 (古い → 新しい)。
                          コマンド枠 [6:9] は現モードの意味の値。**モード切替の瞬間に
                          全フレームのコマンド枠を 0 にする** (学習環境 TransitionCommand と同じ)。
  mode        (1, 1)      0.0 = walk, 1.0 = turn
出力
  actions     (1, 12)     混合後の生アクション (スケール未適用、歩行ポリシーと同じ)
  turn_weight (1, 1)      ゲート重み w (回転 expert 側)。ログ用

非アクティブ expert へのコマンド枠マスク (GatingActorCritic.mask_inactive_command) と
ゲート特徴の組み立て・ハード切替の事前 logit はモデル内部に焼き込む。

使い方 (Isaac Sim アプリが必要: isaaclab_tasks の import 経路のため)::

    uv run python export_gating_policy.py --headless \\
        --checkpoint logs/rsl_rl/k1_gating/<run>/model_2500.pt \\
        --out ../../frozen_policies/best_gating/gating_2500.onnx

書き出し後、同じ乱数入力で torch 方策 (GatingActorCritic.act_inference_blended) と
onnxruntime の出力を比較して等価性を検証する (walk / turn 両モード)。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="k1_gating の model_*.pt")
parser.add_argument("--out", required=True, help="出力 ONNX パス")
parser.add_argument("--num_checks", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import os  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_k1_locomotion.tasks.manager_based.locomotion.agents.gating_policy import (  # noqa: E402
    EXPERT_MODES,
    GatingActorCritic,
    RslRlGatingActorCriticCfg,
)
from isaaclab_k1_locomotion.tasks.manager_based.locomotion.history_layout import (  # noqa: E402
    CRITIC_TERM_SPECS,
    HISTORY_LENGTH,
    POLICY_TERM_SPECS,
    term_specs_dim,
)

OBS_GROUPS = {"policy": ["command", "policy"], "critic": ["command", "critic"]}
STEP_DIM = term_specs_dim(POLICY_TERM_SPECS)


def fake_obs(n: int) -> TensorDict:
    return TensorDict(
        {
            "command": torch.zeros(n, 3),
            "policy": torch.zeros(n, HISTORY_LENGTH * STEP_DIM),
            "critic": torch.zeros(n, HISTORY_LENGTH * term_specs_dim(CRITIC_TERM_SPECS)),
            "expert_mode": torch.zeros(n, 1),
        },
        batch_size=[n],
    )


class GatingExporter(nn.Module):
    """(command, obs_history, mode) → (actions, turn_weight)。GatingActorCritic と同じ計算を tensor だけで行う。"""

    def __init__(self, policy: GatingActorCritic):
        super().__init__()
        self.experts = nn.ModuleList([policy.experts[str(m)] for m in EXPERT_MODES])
        self.gate = policy.actor
        self.gate_norm = policy.actor_obs_normalizer
        self.prior = float(policy.hard_switch_prior_logit)
        self.mask_inactive = bool(policy.mask_inactive_command)
        self.uses_expert_actions = bool(policy.gate_uses_expert_actions)
        self.register_buffer("recent_idx", policy._policy_recent_idx.clone())
        self.register_buffer("cmd_idx", policy._policy_cmd_idx.clone())
        self.term_dims = tuple(dim for _, dim, _ in POLICY_TERM_SPECS)

    def flatten_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        """(N, H, C) フレーム形式 → history_layout の項ごと (H × dim) ブロック flatten。"""
        parts = []
        channel = 0
        for dim in self.term_dims:
            parts.append(obs_history[:, :, channel : channel + dim].reshape(obs_history.shape[0], -1))
            channel += dim
        return torch.cat(parts, dim=-1)

    def forward(self, command: torch.Tensor, obs_history: torch.Tensor, mode: torch.Tensor):
        hist = self.flatten_history(obs_history)
        turn = mode.round()  # (N, 1)
        keeps = [1.0 - turn, turn]  # expert 順 = EXPERT_MODES (walk, turn)
        expert_actions = []
        for expert, keep in zip(self.experts, keeps):
            if self.mask_inactive:
                cmd = command * keep
                h = hist.clone()
                h[:, self.cmd_idx] = h[:, self.cmd_idx] * keep
            else:
                cmd, h = command, hist
            x = torch.cat([cmd, h], dim=-1)
            expert_actions.append(expert.actor(expert.actor_obs_normalizer(x)))
        parts = [torch.cat([1.0 - turn, turn], dim=-1), command, hist.index_select(-1, self.recent_idx)]
        if self.uses_expert_actions:
            parts.extend(expert_actions)
        z = self.gate(self.gate_norm(torch.cat(parts, dim=-1))) + self.prior * (2.0 * turn - 1.0)
        w = torch.sigmoid(z)
        actions = (1.0 - w) * expert_actions[0] + w * expert_actions[1]
        return actions, w


def main() -> None:
    cfg = RslRlGatingActorCriticCfg().to_dict()
    cfg.pop("class_name")
    policy = GatingActorCritic(fake_obs(1), OBS_GROUPS, 12, **cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"], strict=True)
    policy.eval()

    exporter = GatingExporter(policy).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    example = (torch.zeros(1, 3), torch.zeros(1, HISTORY_LENGTH, STEP_DIM), torch.zeros(1, 1))
    torch.onnx.export(
        exporter,
        example,
        args.out,
        export_params=True,
        opset_version=18,
        input_names=["command", "obs_history", "mode"],
        output_names=["actions", "turn_weight"],
        dynamic_axes={},
        dynamo=False,
    )

    # --- 等価性: torch 方策 (TensorDict 経路) vs ONNX (フレーム形式) ---
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    max_err = 0.0
    for i in range(args.num_checks):
        mode = float(i % 2)
        command = torch.randn(1, 3, generator=gen)
        frames = torch.randn(1, HISTORY_LENGTH, STEP_DIM, generator=gen)
        obs = fake_obs(1)
        obs["command"] = command
        obs["policy"] = exporter.flatten_history(frames)
        obs["expert_mode"] = torch.full((1, 1), mode)
        with torch.no_grad():
            ref = policy.act_inference_blended(obs).numpy()
        out = sess.run(
            None,
            {
                "command": command.numpy(),
                "obs_history": frames.numpy(),
                "mode": np.full((1, 1), mode, dtype=np.float32),
            },
        )
        max_err = max(max_err, float(np.abs(out[0] - ref).max()))
    print(f"[export_gating_policy] wrote {args.out}")
    print(f"[export_gating_policy] parity vs torch over {args.num_checks} random inputs: max |Δaction| = {max_err:.3e}")
    assert max_err < 1e-4, "ONNX と torch 方策の出力が一致しません"


if __name__ == "__main__":
    main()
    app.close()
