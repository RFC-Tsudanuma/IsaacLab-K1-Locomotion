"""経路追従の上位方策 (WaypointActorCritic) を実機ノード向け ONNX に書き出す (2026-10-29)。

書き出すのは上位 actor だけ (観測正規化 + MLP + コマンドのクランプ / デッドゾーン)。
凍結歩行 expert は含めない。実機では既存の歩行 ONNX (best_walk / gating 融合モデルの walk モード)
にこの出力をコマンドとして渡す (booster_k1_locomotion rl_policy_isaaclab_node.cpp の waypoint 機能)。

C++ 側の契約 (mdp/waypoint.py WaypointCommand.command / waypoint_env_cfg.py K1WaypointNavPolicyCfg):

入力  nav_obs (1, 22) float32 =
        [次の 4 点の xy (8), ゴール xy (2), cos/sin(ゴール yaw 誤差) (2), 通過半径 (1)]  … base yaw 座標系
      + [前 tick のコマンド (vx, vy, ωz) (3)]
      + [base 線速度 (vx, vy, vz) body 系 (3)] + [base 角速度 (gyro) (3)]
出力  command (1, 3) float32 = (vx, vy, ωz)。クランプ (1.8, 0.9, 1.0) と ||cmd|| < 0.1 → 0 を適用済み。

使い方 (Isaac Sim 不要)::

    uv run python export_waypoint_policy.py \\
        --checkpoint logs/rsl_rl/k1_waypoint/<run>/model_1499.pt \\
        --out ../../frozen_policies/best_waypoint/waypoint_1499.onnx

ネットワーク構成は <run>/params/agent.yaml から読む。書き出し後に torch と onnxruntime の等価性を確認する。
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
parser.add_argument("--checkpoint", required=True, help="k1_waypoint の model_*.pt")
parser.add_argument("--out", required=True, help="出力 ONNX パス")
parser.add_argument("--num_checks", type=int, default=64)
args = parser.parse_args()


class NavExporter(nn.Module):
    """nav_obs → command。WaypointActorCritic.act_inference + command() と同じ計算。"""

    def __init__(self, policy: ActorCritic, limits, deadzone: float):
        super().__init__()
        self.norm = policy.actor_obs_normalizer
        self.actor = policy.actor
        self.register_buffer("lim", torch.tensor(limits, dtype=torch.float))
        self.deadzone = float(deadzone)

    def forward(self, nav_obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor(self.norm(nav_obs))
        cmd = torch.maximum(torch.minimum(raw, self.lim), -self.lim)
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.deadzone
        return torch.where(stop, torch.zeros_like(cmd), cmd)


def main() -> None:
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        pcfg = yaml.unsafe_load(f)["policy"]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("expert.")}
    obs = TensorDict(
        {"nav_policy": torch.zeros(1, sd["actor.0.weight"].shape[1]), "nav_critic": torch.zeros(1, sd["critic.0.weight"].shape[1])},
        batch_size=[1],
    )
    policy = ActorCritic(
        obs,
        {"policy": ["nav_policy"], "critic": ["nav_critic"]},
        3,
        actor_obs_normalization=pcfg["actor_obs_normalization"],
        critic_obs_normalization=pcfg["critic_obs_normalization"],
        actor_hidden_dims=list(pcfg["actor_hidden_dims"]),
        critic_hidden_dims=list(pcfg["critic_hidden_dims"]),
        activation=pcfg["activation"],
        init_noise_std=pcfg["init_noise_std"],
    )
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    exporter = NavExporter(policy, pcfg["command_limits"], pcfg["command_deadzone"]).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    obs_dim = obs["nav_policy"].shape[1]
    torch.onnx.export(
        exporter,
        (torch.zeros(1, obs_dim),),
        args.out,
        export_params=True,
        opset_version=18,
        input_names=["nav_obs"],
        output_names=["command"],
        dynamic_axes={},
        dynamo=False,
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
    print(f"[export_waypoint_policy] wrote {args.out} (nav_obs dim {obs_dim}, limits {pcfg['command_limits']}, deadzone {pcfg['command_deadzone']})")
    print(f"[export_waypoint_policy] parity torch vs onnxruntime over {args.num_checks} inputs: max |Δcmd| = {max_err:.3e}")
    assert max_err < 1e-5, "ONNX と torch の出力が一致しません"


if __name__ == "__main__":
    main()
