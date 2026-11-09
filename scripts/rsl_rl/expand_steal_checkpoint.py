"""奪取方策の checkpoint に観測入力を末尾追加して、方向指定版 (Isaac-StealDir-K1) の初期値にする (2026-11-09)。

追加入力 (target_dir 2 次元) の初段重みは 0、正規化は平均 0・分散 1 で埋めるので、変換直後の出力は元の方策と同一。
optimizer は形が合わないので捨てる (resume 時は --reset_noise_std を付けて optimizer 読み込みを省くこと)。

    uv run python expand_steal_checkpoint.py --src <k1_steal/.../model_N.pt> --dst <k1_steal_dir/<run>/model_N.pt> --extra 2
"""

import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--extra", type=int, default=2)
args = parser.parse_args()

ck = torch.load(args.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
for net in ("actor", "critic"):
    w = sd[f"{net}.0.weight"]
    sd[f"{net}.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], args.extra)], dim=1)
    for stat, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        k = f"{net}_obs_normalizer.{stat}"
        sd[k] = torch.cat([sd[k], sd[k].new_full((1, args.extra), fill)], dim=1)
ck.pop("optimizer_state_dict", None)
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
torch.save(ck, args.dst)

# 自己チェック: 追加入力が何であっても出力は元と同じ
from rsl_rl.networks import MLP, EmpiricalNormalization  # noqa: E402

src = torch.load(args.src, map_location="cpu", weights_only=False)["model_state_dict"]
n_in, n_out = src["actor.0.weight"].shape[1], src["actor.4.weight"].shape[0]
def actor(sd, n):
    norm, mlp = EmpiricalNormalization([n]), MLP(n, n_out, [256, 128], "elu")
    norm.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("actor_obs_normalizer.")})
    mlp.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("actor.")})
    return lambda x: mlp(norm.eval()(x))
x = torch.randn(64, n_in)
with torch.no_grad():
    err = (actor(src, n_in)(x) - actor(sd, n_in + args.extra)(torch.cat([x, torch.randn(64, args.extra)], 1))).abs().max()
assert err < 1e-5, err
print(f"[expand] {args.src} -> {args.dst}: obs {n_in} -> {n_in + args.extra}, max |Δ| {err:.1e}")
