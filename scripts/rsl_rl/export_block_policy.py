"""ボール阻止方策 (BlockActorCritic) の actor を実機ノード向け ONNX に書き出す (2026-11-02)。

書き出すのは上位 actor だけ (観測正規化 + ボール履歴の causal transformer + MLP + コマンドのクランプ /
デッドゾーン + 残差スケール)。凍結歩行 expert は含めない。実機では既存の歩行 ONNX (best_walk /
gating 融合モデルの walk モード) に ``command`` を渡し、その出力に ``residual`` を足す
(booster_k1_locomotion rl_policy_isaaclab_node.cpp の block 機能)。

C++ 側の契約 (block_env_cfg.py K1BlockPolicyCfg / K1BallNoisyCfg, agents/block_policy.py):

入力  block_obs (1, 65) float32 =
        [歩行 K1PolicyCfg と同じ 49 次元フレーム: ang_vel 3, projected_gravity 3, 前 tick のコマンド 3,
         joint_pos_rel 12, joint_vel_rel 12, last_action 12 (= expert + residual の合成), gait_phase 4]
      + [前 tick の residual (12)] + [ゴールライン: 原点の相対位置 xy (2), 向き誤差 cos/sin (2)] … base yaw 系
      ball_history (1, 20, 4) float32 = 50 Hz × 20 step (古い → 新しい) の
        [ボール相対位置 x, y (base yaw 系), ボール速度 vx, vy (world 速度を base yaw 系で表したもの)]
        学習時はカメラ 30 fps・遅延 160〜240 ms・欠け 10% の値を保持したもの (新フレームが無い step は前回値)。
出力  command (1, 3) float32 = (vx, vy, ωz)。クランプ (2.0, 0.9, 2.0) と ||cmd|| < 0.1 → 0 を適用済み。
      residual (1, 12) float32 = 歩行 ONNX の出力 (action 単位) に足す関節残差 (residual_scale 0.5 を掛け済み)。
      q_target = default + 0.5 × (walk_action + residual)。次 tick の last_action は walk_action + residual。

使い方 (Isaac Sim 不要)::

    uv run python export_block_policy.py \\
        --checkpoint logs/rsl_rl/k1_block/<run>/model_999.pt \\
        --out ../../frozen_policies/best_block/block_999.onnx

ネットワーク構成は <run>/params/agent.yaml から読む。書き出し後に torch と onnxruntime の等価性を確認する。
"""

import argparse
import importlib.util
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from rsl_rl.networks import EmpiricalNormalization

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="k1_block の model_*.pt")
parser.add_argument("--out", required=True, help="出力 ONNX パス")
parser.add_argument("--num_checks", type=int, default=64)
args = parser.parse_args()

BALL_DIM = 4
NETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/agents/block_nets.py",
)


def load_block_nets():
    # パッケージ __init__ が Isaac Sim を要求するので、ファイル単体で読む
    spec = importlib.util.spec_from_file_location("block_nets", NETS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BlockExporter(nn.Module):
    """(block_obs, ball_history) → (command, residual)。BlockActorCritic.act_inference + env_actions と同じ計算。"""

    def __init__(self, pcfg: dict, obs_dim: int, ball_history: int):
        super().__init__()
        nets = load_block_nets()
        self.actor_obs_normalizer = EmpiricalNormalization(shape=[obs_dim + ball_history * BALL_DIM], until=1.0e8)
        encoder = nets.CausalTransformerEncoder(
            BALL_DIM, ball_history, pcfg["d_model"], pcfg["nhead"], pcfg["num_layers"], pcfg["latent_dim"]
        )
        self.actor = nets.BlockHead(obs_dim, ball_history, encoder, 3 + 12, list(pcfg["actor_hidden_dims"]), pcfg["activation"])
        self.register_buffer("lim", torch.tensor(pcfg["command_limits"], dtype=torch.float), persistent=False)
        self.deadzone = float(pcfg["command_deadzone"])
        self.residual_scale = float(pcfg["residual_scale"])

    def forward(self, block_obs: torch.Tensor, ball_history: torch.Tensor):
        x = torch.cat([block_obs, ball_history.reshape(block_obs.shape[0], -1)], dim=-1)
        raw = self.actor(self.actor_obs_normalizer(x))
        cmd = torch.maximum(torch.minimum(raw[:, :3], self.lim), -self.lim)
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.deadzone
        cmd = torch.where(stop, torch.zeros_like(cmd), cmd)
        return cmd, self.residual_scale * raw[:, 3:]


def main() -> None:
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        pcfg = yaml.unsafe_load(f)["policy"]
    assert pcfg["actor_obs_normalization"], "actor_obs_normalization=False は未対応"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith(("actor.", "actor_obs_normalizer."))}
    ball_history = int(pcfg["ball_history"])
    obs_dim = int(sd["actor.mlp.0.weight"].shape[1]) - int(pcfg["latent_dim"])

    exporter = BlockExporter(pcfg, obs_dim, ball_history)
    exporter.load_state_dict(sd, strict=True)
    exporter.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    example = (torch.zeros(1, obs_dim), torch.zeros(1, ball_history, BALL_DIM))
    torch.onnx.export(
        exporter,
        example,
        args.out,
        export_params=True,
        opset_version=18,
        input_names=["block_obs", "ball_history"],
        output_names=["command", "residual"],
        dynamic_axes={},
        dynamo=False,
    )

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    max_err = 0.0
    for _ in range(args.num_checks):
        obs = torch.randn(1, obs_dim, generator=gen)
        ball = torch.randn(1, ball_history, BALL_DIM, generator=gen) * 2.0
        with torch.no_grad():
            ref = [t.numpy() for t in exporter(obs, ball)]
        out = sess.run(None, {"block_obs": obs.numpy(), "ball_history": ball.numpy()})
        max_err = max(max_err, max(float(np.abs(o - r).max()) for o, r in zip(out, ref)))
    print(
        f"[export_block_policy] wrote {args.out} (block_obs {obs_dim}, ball_history {ball_history}x{BALL_DIM}, "
        f"limits {pcfg['command_limits']}, deadzone {pcfg['command_deadzone']}, residual_scale {pcfg['residual_scale']})"
    )
    print(f"[export_block_policy] parity torch vs onnxruntime over {args.num_checks} inputs: max |Δ| = {max_err:.3e}")
    assert max_err < 1e-4, "ONNX と torch の出力が一致しません"


if __name__ == "__main__":
    main()
