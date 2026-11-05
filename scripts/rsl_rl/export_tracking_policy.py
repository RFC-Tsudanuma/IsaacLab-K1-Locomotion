"""経路追従 + 姿勢到達 / 見ながら到達の上位方策 (TrackingActorCritic) を実機ノード向け ONNX に書き出す (2026-11-05)。

書き出すのは上位 actor だけ (観測正規化 + MLP + 後処理)。凍結ゲート / expert は含めない
(実機では gating 融合 ONNX にこの出力を mode / コマンドとして渡す)。

C++ 側の契約 (mdp/tracking.py, tracking_env_cfg.py K1TrackingNavPolicyCfg):

入力  nav_obs (1, 36) float32 =
        [次の 4 点 xy (8), ゴール xy (2), cos/sin(ゴール yaw 誤差) (2, look では 0),
         見つめる対象 xy (2, pose では 0), task_mode one-hot (pose, look) (2),
         pivot 既定目標 cos/sin (2)]                                                … base yaw 座標系
      pivot 既定目標 (現在 yaw からの相対角) = look: 見つめる対象の方位 / pose: 全経由点通過後はゴール yaw 誤差、
      それ以前は次の目標点の方位。turn 突入時の pivot 目標 = 既定目標 + 出力の Δψ 残差 (±1.0 rad)。
      + [前 tick の上位出力 (vx, vy, ωz, Δψ, mode) (5)]
      + [gate mode one-hot (walk, turn) (2), 現モード滞在時間 clip 1 s (1), turn 中の残り角 (rad, walk では 0) (1)]
      + [base 線速度 body 系 (3)] + [base 角速度 (3)] + [projected gravity (3)]
出力  command (1, 5) float32 = (vx, vy, ωz, Δψ 残差, mode)。クランプ (2.0, 0.9, 1.0) / ||cmd|| < 0.1 → 0 /
      Δψ 残差 ∈ ±1.0 / mode ∈ {0, 1} を適用済み。上位出力は次 tick のコマンドとして使う (学習と同じ 1 step 遅れ)。
      mode の切替は最小滞在 0.3 s を満たすときだけ受け付け、turn 突入時に Δψ をラッチする (C++ 側で実装)。

使い方 (Isaac Sim 不要)::

    uv run python export_tracking_policy.py \\
        --checkpoint logs/rsl_rl/k1_tracking/<run>/model_2999.pt \\
        --out ../../frozen_policies/best_tracking/tracking_2999.onnx
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="k1_tracking の model_*.pt")
parser.add_argument("--out", required=True, help="出力 ONNX パス")
parser.add_argument("--num_checks", type=int, default=64)
args = parser.parse_args()


class NavExporter(nn.Module):
    """nav_obs → command。TrackingActorCritic.act_inference + command() と同じ計算。"""

    def __init__(self, policy: ActorCritic, limits, deadzone: float, pivot_limit: float):
        super().__init__()
        self.norm = policy.actor_obs_normalizer
        self.actor = policy.actor
        self.register_buffer("lim", torch.tensor(limits, dtype=torch.float))
        self.deadzone = float(deadzone)
        self.pivot_limit = float(pivot_limit)

    def forward(self, nav_obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor(self.norm(nav_obs))
        cmd = torch.maximum(torch.minimum(raw[:, :3], self.lim), -self.lim)
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.deadzone
        cmd = torch.where(stop, torch.zeros_like(cmd), cmd)
        dpsi = raw[:, 3:4].clamp(-self.pivot_limit, self.pivot_limit)
        mode = (raw[:, 4:5] > 0.0).float()
        return torch.cat([cmd, dpsi, mode], dim=-1)


def main() -> None:
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        pcfg = yaml.unsafe_load(f)["policy"]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("gate.")}
    obs = TensorDict(
        {"nav_policy": torch.zeros(1, sd["actor.0.weight"].shape[1]), "nav_critic": torch.zeros(1, sd["critic.0.weight"].shape[1])},
        batch_size=[1],
    )
    policy = ActorCritic(
        obs,
        {"policy": ["nav_policy"], "critic": ["nav_critic"]},
        5,
        actor_obs_normalization=pcfg["actor_obs_normalization"],
        critic_obs_normalization=pcfg["critic_obs_normalization"],
        actor_hidden_dims=list(pcfg["actor_hidden_dims"]),
        critic_hidden_dims=list(pcfg["critic_hidden_dims"]),
        activation=pcfg["activation"],
        init_noise_std=pcfg["init_noise_std"],
    )
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    exporter = NavExporter(policy, pcfg["command_limits"], pcfg["command_deadzone"], pcfg["pivot_limit"]).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    obs_dim = obs["nav_policy"].shape[1]
    torch.onnx.export(
        exporter, (torch.zeros(1, obs_dim),), args.out, export_params=True, opset_version=18,
        input_names=["nav_obs"], output_names=["command"], dynamic_axes={}, dynamo=False,
    )

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    max_err = 0.0
    for _ in range(args.num_checks):
        x = torch.randn(1, obs_dim, generator=gen) * 2.0
        with torch.no_grad():
            ref = exporter(x).numpy()
        out = sess.run(None, {"nav_obs": x.numpy()})[0]
        max_err = max(max_err, float(np.abs(out - ref).max()))
    print(f"[export_tracking_policy] wrote {args.out} (nav_obs dim {obs_dim}, limits {pcfg['command_limits']}, "
          f"deadzone {pcfg['command_deadzone']}, pivot_limit {pcfg['pivot_limit']})")
    print(f"[export_tracking_policy] parity torch vs onnxruntime over {args.num_checks} inputs: max |Δ| = {max_err:.3e}")
    assert max_err < 1e-5, "ONNX と torch の出力が一致しません"


if __name__ == "__main__":
    main()
