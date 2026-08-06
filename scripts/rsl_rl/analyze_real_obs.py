"""実機/シミュで記録した観測ログ (obs_log) の分布解析ツール。

booster_k1_locomotion の rl_policy_isaaclab_node (obs_log_path パラメータ) が
記録した CSV を読み、学習チェックポイントに焼き込まれた EmpiricalNormalization
の per-チャネル統計 (mean/std) で z スコア化して「どのチャネルが学習分布から
外れているか」を特定する。実機とシミュ (mujoco) の両方で記録して比較すると、
sim2real ギャップの犯人チャネルを絞り込める。

CSV フォーマット (ヘッダ付き):
    step,event,cmd_vx,cmd_vy,cmd_wz,f0..f45,a0..a11
    event: 0=通常 tick, 1=fall guard 発動 (行は 0 埋め), 2=on_enter マーカー

使い方 (学習 PC の kit python で実行):
    _labpython2 analyze_real_obs.py --csv real_backward.csv \
        --checkpoint logs/rsl_rl/k1_flat/dual_first/model_19999.pt \
        [--ref_csv mujoco_backward.csv] \
        [--onnx logs/.../exported/policy.onnx]

--onnx を与えると、記録フレームから C++ ノードと同じ規則 (enter/fall guard で
最新フレームをバックフィル → 毎 tick スライド) で履歴を再構築して ONNX を
リプレイし、記録された action と一致するか (= C++ ノードの数値検証) も行う。

このスクリプトは isaaclab に依存しない (numpy + torch、リプレイのみ
onnxruntime)。レイアウト定数は history_layout.py と一致させてある。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

# history_layout.py (POLICY_TERM_SPECS / HISTORY_LENGTH / COMMAND_DIM) と一致させること
HISTORY_LENGTH = 100
COMMAND_DIM = 3
# 現行レイアウト (2026-08-05〜): フレームに velocity_commands を含む 49 次元
POLICY_TERM_SPECS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("last_action", 12),
    ("gait_phase", 4),
)
# 旧レイアウト (dual_first 等): コマンド無しの 46 次元フレーム
LEGACY_TERM_SPECS = tuple(
    (name, dim) for name, dim in POLICY_TERM_SPECS if name != "velocity_commands"
)
NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization.eps と同値


def frame_dim(specs) -> int:
    return sum(d for _, d in specs)


def channel_names(specs) -> list[str]:
    return [f"{name}[{i}]" for name, dim in specs for i in range(dim)]


def load_csv(path: str, specs):
    data = np.genfromtxt(path, delimiter=",", names=True)
    n_cols = len(data.dtype.names)
    fdim = frame_dim(specs)
    expected = 2 + 3 + fdim + 12
    if n_cols != expected:
        raise ValueError(
            f"{path}: 列数 {n_cols} が想定 {expected} と違います "
            f"(checkpoint 由来のフレーム次元 {fdim} と CSV のレイアウトが不一致?)"
        )
    rows = np.stack([data[name] for name in data.dtype.names], axis=1)
    return {
        "step": rows[:, 0].astype(int),
        "event": rows[:, 1].astype(int),
        "cmd": rows[:, 2:5],
        "frame": rows[:, 5 : 5 + fdim],
        "action": rows[:, 5 + fdim :],
    }


def load_normalizer_stats(checkpoint_path: str):
    """checkpoint から actor 正規化器の「現在フレーム相当スロット」の mean/std を返す。

    actor 正規化器は [command(3), 項ごとの (H×dim) 履歴ブロック] の
    4903 次元 (現行 49 次元フレーム) または 4603 次元 (旧 46 次元フレーム)。
    次元からレイアウトを自動判別し、各項の最新スロット (t=H-1) の統計を
    フレームチャネル順に並べ替え、command 3 次元の統計と使用レイアウトを返す。
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    mean = sd["actor_obs_normalizer._mean"].numpy().reshape(-1)
    std = sd["actor_obs_normalizer._std"].numpy().reshape(-1)
    by_dim = {
        COMMAND_DIM + frame_dim(specs) * HISTORY_LENGTH: specs
        for specs in (POLICY_TERM_SPECS, LEGACY_TERM_SPECS)
    }
    specs = by_dim.get(mean.shape[0])
    if specs is None:
        raise ValueError(
            f"正規化器の次元 {mean.shape[0]} が想定 {sorted(by_dim)} のどれとも"
            " 違います (モデル構成が異なる?)"
        )
    fdim = frame_dim(specs)
    cmd_mean, cmd_std = mean[:COMMAND_DIM], std[:COMMAND_DIM]
    frame_mean = np.zeros(fdim)
    frame_std = np.zeros(fdim)
    offset = COMMAND_DIM
    ch = 0
    for _name, dim in specs:
        latest = offset + (HISTORY_LENGTH - 1) * dim
        frame_mean[ch : ch + dim] = mean[latest : latest + dim]
        frame_std[ch : ch + dim] = std[latest : latest + dim]
        offset += HISTORY_LENGTH * dim
        ch += dim
    return (cmd_mean, cmd_std), (frame_mean, frame_std), specs


def zscore_stats(frames: np.ndarray, frame_mean, frame_std):
    z = (frames - frame_mean) / (frame_std + NORMALIZER_EPS)
    return {
        "mean_z": z.mean(axis=0),
        "mean_abs_z": np.abs(z).mean(axis=0),
        "p99_abs_z": np.percentile(np.abs(z), 99, axis=0),
    }


def print_report(tag: str, log, stats, specs, ref_stats=None):
    names = channel_names(specs)
    n = len(log["event"])
    normal = log["event"] == 0
    print(f"\n===== {tag} =====")
    print(f"rows={n} ({n / 50.0:.1f}s @50Hz)  normal={normal.sum()}"
          f"  fall_guard={int((log['event'] == 1).sum())}  enter={int((log['event'] == 2).sum())}")
    guard_idx = np.nonzero(log["event"] == 1)[0]
    if len(guard_idx) > 0:
        print(f"fall_guard 行位置 (先頭10): {guard_idx[:10].tolist()}")
    cmd = log["cmd"][normal]
    print(f"cmd range: vx [{cmd[:, 0].min():+.2f}, {cmd[:, 0].max():+.2f}]"
          f"  vy [{cmd[:, 1].min():+.2f}, {cmd[:, 1].max():+.2f}]"
          f"  wz [{cmd[:, 2].min():+.2f}, {cmd[:, 2].max():+.2f}]")

    # 項ごとのまとめ
    print(f"\n--- 項ごとの |z| (学習分布からの乖離; 1.0 ≒ 学習時の 1σ) ---")
    header = f"{'term':22s} {'mean|z|':>8s} {'p99|z|':>8s} {'worst channel':>24s}"
    if ref_stats is not None:
        header += f" {'ref mean|z|':>11s}"
    print(header)
    ch = 0
    for name, dim in specs:
        sl = slice(ch, ch + dim)
        mz = stats["mean_abs_z"][sl]
        worst = int(np.argmax(mz))
        line = (f"{name:22s} {mz.mean():8.3f} {stats['p99_abs_z'][sl].max():8.3f}"
                f" {names[ch + worst]:>20s}={mz[worst]:.3f}")
        if ref_stats is not None:
            line += f" {ref_stats['mean_abs_z'][sl].mean():11.3f}"
        print(line)
        ch += dim

    # ワーストチャネル
    order = np.argsort(-stats["mean_abs_z"])
    print(f"\n--- ワーストチャネル (mean|z| 上位 12) ---")
    for i in order[:12]:
        line = (f"{names[i]:24s} mean_z={stats['mean_z'][i]:+7.3f}"
                f"  mean|z|={stats['mean_abs_z'][i]:6.3f}  p99|z|={stats['p99_abs_z'][i]:6.3f}")
        if ref_stats is not None:
            line += f"  (ref mean|z|={ref_stats['mean_abs_z'][i]:.3f})"
        print(line)


def replay_onnx(onnx_path: str, log):
    """記録フレームから履歴を再構築して ONNX をリプレイし、記録 action と比較する。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    hist = np.zeros((HISTORY_LENGTH, log["frame"].shape[1]), dtype=np.float32)
    primed = False
    max_diff = 0.0
    diffs = []
    for i in range(len(log["event"])):
        ev = log["event"][i]
        if ev != 0:
            # enter / fall guard: C++ ノードは次の通常 tick でバックフィルし直す
            primed = False
            continue
        frame = log["frame"][i].astype(np.float32)
        if not primed:
            hist[:] = frame
            primed = True
        else:
            hist[:-1] = hist[1:]
            hist[-1] = frame
        cmd = log["cmd"][i].astype(np.float32).reshape(1, 3)
        out = sess.run(["actions"], {"command": cmd, "obs_history": hist[None]})[0][0]
        # 記録側は ±10 クリップ後の値
        out = np.clip(out, -10.0, 10.0)
        d = float(np.max(np.abs(out - log["action"][i])))
        diffs.append(d)
        max_diff = max(max_diff, d)
    diffs = np.array(diffs)
    print(f"\n--- ONNX リプレイ検証 (C++ ノードの数値一致) ---")
    print(f"compared {len(diffs)} ticks: max|Δaction|={max_diff:.2e}  mean={diffs.mean():.2e}")
    if max_diff > 1e-3:
        print("  ※ 一致しません。C++ ノードの履歴構築か前処理に差異がある可能性。")
    else:
        print("  OK: C++ ノードの推論経路は記録データ上で ONNX と一致。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="実機の obs_log CSV")
    ap.add_argument("--checkpoint", required=True, help="学習チェックポイント (model_*.pt)")
    ap.add_argument("--ref_csv", default=None, help="比較用 (mujoco 等) の obs_log CSV")
    ap.add_argument("--onnx", default=None, help="エクスポート済み policy.onnx (リプレイ検証)")
    args = ap.parse_args()

    (cmd_mean, cmd_std), (frame_mean, frame_std), specs = load_normalizer_stats(
        args.checkpoint
    )
    print(f"frame layout: {frame_dim(specs)} dims "
          f"({'velocity_commands 込み' if specs is POLICY_TERM_SPECS else '旧コマンド無し'})")

    log = load_csv(args.csv, specs)
    stats = zscore_stats(log["frame"][log["event"] == 0], frame_mean, frame_std)

    ref_stats = None
    if args.ref_csv:
        ref_log = load_csv(args.ref_csv, specs)
        ref_stats = zscore_stats(ref_log["frame"][ref_log["event"] == 0], frame_mean, frame_std)
        print_report(f"REF: {args.ref_csv}", ref_log, ref_stats, specs)

    print_report(f"TARGET: {args.csv}", log, stats, specs, ref_stats)

    if args.onnx:
        replay_onnx(args.onnx, log)


if __name__ == "__main__":
    main()
