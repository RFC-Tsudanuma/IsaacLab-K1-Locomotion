"""ボール奪取方策 (StealActorCritic) の actor を実機ノード向け ONNX に書き出す (2026-11-03)。

書き出すのは上位 actor だけ (観測正規化 + MLP + コマンドのクランプ / デッドゾーン + 残差スケール)。
凍結歩行 expert は含めない。実機では既存の歩行 ONNX に ``command`` を渡し、その出力に ``residual`` を足す
(booster_k1_locomotion rl_policy_isaaclab_node.cpp の steal 機能。block と同じ流儀)。

C++ 側の契約 (steal_env_cfg.py K1StealPolicyCfg, agents/steal_policy.py):

入力  steal_obs (1, 67) float32 =
        [歩行 K1PolicyCfg と同じ 49 次元フレーム: ang_vel 3, projected_gravity 3, 前 tick のコマンド 3,
         joint_pos_rel 12, joint_vel_rel 12, last_action 12 (= expert + residual の合成), gait_phase 4]
      + [前 tick の residual (12)]
      + [ボール相対位置 x, y (2)] + [ボール速度 vx, vy (2)] + [障害物 (敵) 中心の相対位置 x, y (2)] … base yaw 系
出力  command (1, 3) float32 = (vx, vy, ωz)。クランプ (2.0, 0.9, 2.0) と ||cmd|| < 0.1 → 0 を適用済み。
      residual (1, 12) float32 = 歩行 ONNX の出力 (action 単位) に足す関節残差 (residual_scale 0.5 を掛け済み)。
      q_target = default + 0.5 × (walk_action + residual)。次 tick の last_action は walk_action + residual。

使い方 (Isaac Sim 不要)::

    uv run python export_steal_policy.py \\
        --checkpoint ../../frozen_policies/best_steal/model_2999.pt \\
        --out ../../frozen_policies/best_steal/steal_2999.onnx

ネットワーク構成は <run>/params/agent.yaml から読む (checkpoint をコピーした場合は --run_dir で指定)。
書き出し後に torch と onnxruntime の等価性を確認する。
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from rsl_rl.networks import MLP, EmpiricalNormalization

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="k1_steal の model_*.pt")
parser.add_argument("--out", required=True, help="出力 ONNX パス")
parser.add_argument("--run_dir", default=None, help="params/agent.yaml を持つ run dir (既定: checkpoint の親)")
parser.add_argument("--num_checks", type=int, default=64)
args = parser.parse_args()

NUM_ACTIONS = 3 + 12


class StealExporter(nn.Module):
    """steal_obs → (command, residual)。StealActorCritic.act_inference + env_actions と同じ計算。"""

    def __init__(self, pcfg: dict, obs_dim: int):
        super().__init__()
        self.actor_obs_normalizer = EmpiricalNormalization(shape=[obs_dim], until=1.0e8)
        self.actor = MLP(obs_dim, NUM_ACTIONS, list(pcfg["actor_hidden_dims"]), pcfg["activation"])
        self.register_buffer("lim", torch.tensor(pcfg["command_limits"], dtype=torch.float), persistent=False)
        self.deadzone = float(pcfg["command_deadzone"])
        self.residual_scale = float(pcfg["residual_scale"])

    def forward(self, steal_obs: torch.Tensor):
        raw = self.actor(self.actor_obs_normalizer(steal_obs))
        cmd = torch.maximum(torch.minimum(raw[:, :3], self.lim), -self.lim)
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.deadzone
        cmd = torch.where(stop, torch.zeros_like(cmd), cmd)
        return cmd, self.residual_scale * raw[:, 3:]


def main() -> None:
    run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        pcfg = yaml.unsafe_load(f)["policy"]
    assert pcfg["actor_obs_normalization"], "actor_obs_normalization=False は未対応"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith(("actor.", "actor_obs_normalizer."))}
    obs_dim = int(sd["actor.0.weight"].shape[1])

    exporter = StealExporter(pcfg, obs_dim)
    exporter.load_state_dict(sd, strict=True)
    exporter.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        exporter,
        (torch.zeros(1, obs_dim),),
        args.out,
        export_params=True,
        opset_version=18,
        input_names=["steal_obs"],
        output_names=["command", "residual"],
        dynamic_axes={},
        dynamo=False,
    )

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    max_err = 0.0
    for _ in range(args.num_checks):
        obs = torch.randn(1, obs_dim, generator=gen)
        with torch.no_grad():
            ref = [t.numpy() for t in exporter(obs)]
        out = sess.run(None, {"steal_obs": obs.numpy()})
        max_err = max(max_err, max(float(np.abs(o - r).max()) for o, r in zip(out, ref)))
    print(
        f"[export_steal_policy] wrote {args.out} (steal_obs {obs_dim}, limits {pcfg['command_limits']}, "
        f"deadzone {pcfg['command_deadzone']}, residual_scale {pcfg['residual_scale']})"
    )
    print(f"[export_steal_policy] parity torch vs onnxruntime over {args.num_checks} inputs: max |Δ| = {max_err:.3e}")
    assert max_err < 1e-4, "ONNX と torch の出力が一致しません"


if __name__ == "__main__":
    main()
