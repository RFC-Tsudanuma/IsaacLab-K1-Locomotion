"""StealDir (MLP) の checkpoint を StealHist (Block 型: 観測履歴 encoder ×2 + MLP) の初期値に変換する (2026-11-22)。

MLP の重みはそのまま引き継ぎ、latent 入力 (末尾 latent_dim 列) の初段重みは 0、encoder は乱数初期化、
履歴入力の正規化は平均 0・分散 1。よって変換直後の actor 出力は履歴の値によらず元の方策と同じ。
optimizer は捨てる (resume 時は --reset_noise_std を付けること)。

    uv run python convert_steal_to_hist.py --src <k1_steal_dir/.../model_N.pt> --dst <k1_steal_hist/<run>/model_N.pt>
"""

import argparse
import importlib.util
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--history", type=int, default=30)
parser.add_argument("--actor_frame", type=int, default=8)
parser.add_argument("--critic_frame", type=int, default=6)
parser.add_argument("--latent_dim", type=int, default=32)
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "block_nets", os.path.join(here, "../../source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/agents/block_nets.py")
)
nets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nets)

ck = torch.load(args.src, map_location="cpu", weights_only=False)
src = ck["model_state_dict"]
out = {"std": src["std"]}
T, L = args.history, args.latent_dim
heads = {}
for net, frame, n_out in (("actor", args.actor_frame, 15), ("critic", args.critic_frame, 1)):
    direct = src[f"{net}.0.weight"].shape[1]
    enc = nets.CausalTransformerEncoder(frame, T, 64, 4, 2, L)
    head = nets.BlockHead(direct, T, enc, n_out, [256, 128], "elu")
    sd = head.state_dict()
    for k in [k for k in src if k.startswith(f"{net}.") and not k.startswith(f"{net}_")]:
        sd["mlp." + k.split(".", 1)[1]] = src[k].clone()
    w = src[f"{net}.0.weight"]
    sd["mlp.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], L)], dim=1)
    head.load_state_dict(sd)
    heads[net] = head
    out.update({f"{net}.{k}": v for k, v in head.state_dict().items()})
    for stat, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        k = f"{net}_obs_normalizer.{stat}"
        out[k] = torch.cat([src[k], src[k].new_full((1, T * frame), fill)], dim=1)
    out[f"{net}_obs_normalizer.count"] = src[f"{net}_obs_normalizer.count"]
ck["model_state_dict"] = out
ck.pop("optimizer_state_dict", None)
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
torch.save(ck, args.dst)

# 自己チェック: 履歴が何であっても actor の出力は元の MLP と同じ
from rsl_rl.networks import MLP  # noqa: E402

n_in = src["actor.0.weight"].shape[1]
old = MLP(n_in, 15, [256, 128], "elu")
old.load_state_dict({k.split(".", 1)[1]: v for k, v in src.items() if k.startswith("actor.")})
x = torch.randn(64, n_in)
with torch.no_grad():
    err = (old(x) - heads["actor"](torch.cat([x, torch.randn(64, T * args.actor_frame)], 1))).abs().max()
assert err < 1e-5, err
print(f"[convert] {args.src} -> {args.dst}: actor {n_in}+{T}x{args.actor_frame}, critic "
      f"{src['critic.0.weight'].shape[1]}+{T}x{args.critic_frame}, actor max |Δ| {err:.1e}")
