"""履歴 encoder 版の奪取方策 (StealHistActorCritic、教師-生徒の生徒) を実機ノード向け ONNX に書き出す (2026-11-23)。

書き出すのは上位 actor だけ: 観測正規化 + 観測履歴の causal transformer + MLP + コマンドのクランプ /
デッドゾーン / 膝 + 残差スケール。凍結歩行 expert は含めない (export_steal_policy.py と同じ流儀)。

C++ 側の契約 (steal_env_cfg.py K1StealDirPolicyCfg / K1StealHistObsCfg、mdp/steal.py StealCommand):

入力  steal_obs (1, 69) float32 = 従来の MLP 版 (export_steal_policy.py) と同じ並び
        [歩行 49 次元フレーム] + [前 tick の residual (12)] + [ボール相対位置 x, y (2)] + [ボール速度 vx, vy (2)]
        + [敵の相対位置 x, y (2)] + [目標方向 cos, sin (2)]   … base yaw 系。ボール・敵はノードの代替則を通した値
      steal_hist (1, 30, 8) float32 = 50 Hz × 30 step (古い → 新しい、最後が今の tick) の
        [ボール位置 x, y, ボール速度 vx, vy, 敵位置 x, y, ボールが見えている (0/1), 敵が仮想 (0/1)]
        各 tick の値を world (odom) 系で保持し、**毎 tick 現在の base 位置・yaw 基準に直して** 渡す
        (位置は現在の base からの相対を base yaw 系で、速度は base yaw 系へ回転のみ)。
        ボール・敵の値は steal_obs と同じ (見失い中は最後に見た位置・速度 0、敵はボールから 1 m 超 / 不検出なら仮想の敵)。
        「見えている」= その tick にボールを検出した、「仮想」= その tick の敵が仮想の敵。
出力  command (1, 3) float32 / residual (1, 12) float32 … MLP 版と同じ (クランプ・デッドゾーン・膝、residual_scale 済み)。

使い方 (Isaac Sim 不要)::

    uv run python export_steal_hist_policy.py --checkpoint <k1_steal_hist/.../model_N.pt> \\
        --run_dir <params/agent.yaml を持つ run (蒸留 run には無いので教師の run)> --out <out.onnx>
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
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--run_dir", default=None, help="params/agent.yaml を持つ run dir (既定: checkpoint の親)")
parser.add_argument("--frame_dim", type=int, default=8)
parser.add_argument("--num_checks", type=int, default=64)
args = parser.parse_args()

NETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/locomotion/agents/block_nets.py",
)


def load_nets():
    spec = importlib.util.spec_from_file_location("block_nets", NETS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StealHistExporter(nn.Module):
    """(steal_obs, steal_hist) → (command, residual)。StealHistActorCritic.act_inference + env_actions (w=1) と同じ計算。"""

    def __init__(self, pcfg: dict, obs_dim: int, hist: int, frame: int):
        super().__init__()
        nets = load_nets()
        self.actor_obs_normalizer = EmpiricalNormalization(shape=[obs_dim + hist * frame], until=1.0e8)
        enc = nets.CausalTransformerEncoder(frame, hist, pcfg["d_model"], pcfg["nhead"], pcfg["num_layers"], pcfg["latent_dim"])
        self.actor = nets.BlockHead(obs_dim, hist, enc, 3 + 12, list(pcfg["actor_hidden_dims"]), pcfg["activation"])
        self.register_buffer("lim", torch.tensor(pcfg["command_limits"], dtype=torch.float), persistent=False)
        self.deadzone = float(pcfg["command_deadzone"])
        self.residual_scale = float(pcfg["residual_scale"])
        self.knee = float(pcfg.get("lateral_x_knee", 1.0))
        self.x_min = float(pcfg.get("lateral_x_min", 0.3))
        self.knee_active = "lateral_x_knee" in pcfg and float(pcfg["command_limits"][1]) > self.knee

    def forward(self, steal_obs: torch.Tensor, steal_hist: torch.Tensor):
        x = torch.cat([steal_obs, steal_hist.reshape(steal_obs.shape[0], -1)], dim=-1)
        raw = self.actor(self.actor_obs_normalizer(x))
        cmd = torch.maximum(torch.minimum(raw[:, :3], self.lim), -self.lim)
        stop = torch.norm(cmd, dim=-1, keepdim=True) < self.deadzone
        cmd = torch.where(stop, torch.zeros_like(cmd), cmd)
        if self.knee_active:
            t = ((cmd[:, 1:2].abs() - self.knee) / (self.lim[1] - self.knee)).clamp(0.0, 1.0)
            x_lim = self.lim[0] - t * (self.lim[0] - self.x_min)
            cmd = torch.cat([torch.maximum(torch.minimum(cmd[:, :1], x_lim), -x_lim), cmd[:, 1:]], dim=-1)
        return cmd, self.residual_scale * raw[:, 3:]


def main() -> None:
    run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        pcfg = yaml.unsafe_load(f)["policy"]
    assert pcfg["actor_obs_normalization"], "actor_obs_normalization=False は未対応"
    hist = int(pcfg["ball_history"])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith(("actor.", "actor_obs_normalizer."))}
    obs_dim = int(sd["actor_obs_normalizer._mean"].shape[1]) - hist * args.frame_dim
    assert obs_dim == 69, f"steal_obs は 69 次元のはず ({obs_dim})"

    exporter = StealHistExporter(pcfg, obs_dim, hist, args.frame_dim)
    exporter.load_state_dict(sd, strict=True)
    exporter.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        exporter,
        (torch.zeros(1, obs_dim), torch.zeros(1, hist, args.frame_dim)),
        args.out,
        export_params=True,
        opset_version=18,
        input_names=["steal_obs", "steal_hist"],
        output_names=["command", "residual"],
        dynamic_axes={},
        dynamo=False,
    )

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    max_err = 0.0
    for _ in range(args.num_checks):
        obs = torch.randn(1, obs_dim, generator=gen)
        h = torch.randn(1, hist, args.frame_dim, generator=gen) * 2.0
        with torch.no_grad():
            ref = [t.numpy() for t in exporter(obs, h)]
        out = sess.run(None, {"steal_obs": obs.numpy(), "steal_hist": h.numpy()})
        max_err = max(max_err, max(float(np.abs(o - r).max()) for o, r in zip(out, ref)))
    print(f"[export_steal_hist_policy] wrote {args.out} (steal_obs {obs_dim}, steal_hist {hist}x{args.frame_dim}, "
          f"limits {pcfg['command_limits']}, knee {'on' if exporter.knee_active else 'off'})")
    print(f"[export_steal_hist_policy] parity torch vs onnxruntime: max |Δ| = {max_err:.3e}")
    assert max_err < 1e-4, "ONNX と torch の出力が一致しません"


if __name__ == "__main__":
    main()
