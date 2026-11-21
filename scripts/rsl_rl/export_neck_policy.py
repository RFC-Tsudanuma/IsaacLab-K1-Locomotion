"""首安定化の残差方策 (NeckActorCritic) を実機ノード向け ONNX に書き出す (2026-11-21)。

書き出すのは actor だけ (観測正規化 + MLP)。凍結歩行 expert は含めない。
実機側は booster_k1_locomotion の src/neck_stabilizer.hpp (平滑化・FK/IK・観測の組み立て・残差の加算)。

入力  neck_obs (1, 130) float32 = 項ごとに履歴 5 step (古い → 新しい) を並べたもの:
        [安定化器の観測 17 × 5, 前回の出力 2 × 5, 歩行位相 4 × 5, 歩行コマンド 3 × 5]  (neck_env_cfg.py K1NeckPolicyCfg)
出力  action (1, 2) float32 = 生の出力 [yaw, pitch]。残差 [rad] = clamp(0.1 × action, ±0.3) (mdp/neck.py NeckActionCfg)。

使い方 (Isaac Sim 不要)::

    _labpython2 export_neck_policy.py --checkpoint ../../frozen_policies/best_neck/model_e2_delay35_1499.pt \\
        --out ../../frozen_policies/best_neck/neck_e2_delay35_1499.onnx
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="k1_neck の model_*.pt")
parser.add_argument("--out", required=True, help="出力 ONNX パス")
args = parser.parse_args()

sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
mean, std = sd["actor_obs_normalizer._mean"], sd["actor_obs_normalizer._std"]
weights = sorted((int(k.split(".")[1]), k) for k in sd if k.startswith("actor.") and k.endswith(".weight"))
layers: list[nn.Module] = []
for i, k in weights:
    lin = nn.Linear(sd[k].shape[1], sd[k].shape[0])
    lin.load_state_dict({"weight": sd[k], "bias": sd[f"actor.{i}.bias"]})
    layers += [lin, nn.ELU()]  # RslRlNeckActorCriticCfg.activation = "elu"
actor = nn.Sequential(*layers[:-1]).eval()


class Exporter(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, neck_obs: torch.Tensor) -> torch.Tensor:
        return self.actor((neck_obs - self.mean) / (self.std + 1e-2))  # rsl_rl EmpiricalNormalization (eps = 1e-2)


obs_dim = mean.shape[-1]
torch.onnx.export(
    Exporter().eval(), (torch.zeros(1, obs_dim),), args.out, export_params=True, opset_version=18,
    input_names=["neck_obs"], output_names=["action"], dynamic_axes={}, dynamo=False,
)

# rsl_rl 本体の ActorCritic と一致することを確認する (正規化の式・層の並びを取り違えていないか)
from rsl_rl.modules import ActorCritic  # noqa: E402
from tensordict import TensorDict  # noqa: E402

critic_dim = sd["critic.0.weight"].shape[1]
ref = ActorCritic(
    TensorDict({"p": torch.zeros(1, obs_dim), "c": torch.zeros(1, critic_dim)}, batch_size=[1]),
    {"policy": ["p"], "critic": ["c"]}, sd[weights[-1][1]].shape[0],
    actor_obs_normalization=True, critic_obs_normalization=True,
    actor_hidden_dims=[sd[k].shape[0] for _, k in weights[:-1]],
    critic_hidden_dims=[v.shape[0] for k, v in sd.items() if k.startswith("critic.") and k.endswith(".weight")][:-1],
    activation="elu",
)
ref.load_state_dict({k: v for k, v in sd.items() if not k.startswith("expert.")}, strict=True)
ref.eval()
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
gen = torch.Generator().manual_seed(0)
max_err = 0.0
for _ in range(64):
    x = mean + torch.randn(1, obs_dim, generator=gen) * std * 2.0
    with torch.no_grad():
        want = ref.act_inference(TensorDict({"p": x, "c": torch.zeros(1, critic_dim)}, batch_size=[1])).numpy()
    max_err = max(max_err, float(np.abs(sess.run(None, {"neck_obs": x.numpy()})[0] - want).max()))
print(f"[export_neck_policy] wrote {args.out} (obs dim {obs_dim}); rsl_rl vs onnxruntime: max |Δ| = {max_err:.3e}")
assert max_err < 1e-4, "ONNX と rsl_rl の出力が一致しません"
