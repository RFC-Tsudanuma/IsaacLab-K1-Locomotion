#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""直接版 Stage2 の ckpt を **ボール履歴版の観測次元へ拡張** する。

ボール履歴版は観測の末尾にボール履歴を足すだけなので、第1層の重みは

    * 既存の 59 列 → そのままコピー
    * 新規の履歴列 → **ゼロで初期化**

とすれば、拡張直後の方策は元の方策と **数学的に完全に同一** になる
(ゼロ列は出力に寄与しないため)。そこから追加学習すれば、方策は履歴を
使うことを少しずつ学べる。**Stage1 からの作り直しが不要**になるのが要点。

観測正規化 (EmpiricalNormalization) の統計も同様に拡張する:

    * mean → 既存はコピー、新規は 0
    * var/std → 既存はコピー、新規は **1**  (0 にすると割り算で発散する)
    * count → そのまま (既存列の統計を捨てないため)

★ 使い方 (ホストの python3 で動く。torch のみ必要):

    python3 scripts/rsl_rl/expand_ckpt_for_ballhist.py \\
        --src logs/rsl_rl/k1_gk_direct_stage2/2026-08-17_10-37-49/model_35200.pt \\
        --dst logs/rsl_rl/k1_gk_direct_stage2/2026-08-17_10-37-49/ballhist_seed.pt

    # そのあと ボール履歴版タスクを --resume で開始する
    STAGE1_CKPT=<dst> ./scripts/rsl_rl/train_gk_ballhist.sh --max_iterations 20000

★ 追加次元は ボール履歴版の cfg (BALLHIST_FRAMES x FRAME_DIM) と一致させること。
  既定は 10 フレーム x 3 = 30。cfg を変えたら --extra も変えること。
"""

import argparse
import torch


def _expand_linear(w: torch.Tensor, extra: int) -> torch.Tensor:
    """(out, in) の重みを (out, in + extra) へ拡張し、新規列を 0 で埋める。"""
    out, n_in = w.shape
    new = torch.zeros(out, n_in + extra, dtype=w.dtype)
    new[:, :n_in] = w
    return new


def _expand_stat(v: torch.Tensor, extra: int, fill: float) -> torch.Tensor:
    """正規化統計 (1, D) または (D,) を拡張する。"""
    flat = v.reshape(-1)
    new = torch.full((flat.numel() + extra,), float(fill), dtype=v.dtype)
    new[: flat.numel()] = flat
    return new.reshape(*v.shape[:-1], -1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand a direct-GK ckpt to the ボール履歴版 observation size.")
    ap.add_argument("--src", required=True, help="直接版 Stage2 の model_*.pt")
    ap.add_argument("--dst", required=True, help="出力先")
    ap.add_argument("--extra", type=int, default=30,
                    help="actor に足す観測次元 (BALLHIST_FRAMES x 3)。既定 30")
    ap.add_argument("--extra-critic", type=int, default=None,
                    help="critic に足す次元。省略時は --extra と同じ")
    args = ap.parse_args()

    extra_c = args.extra if args.extra_critic is None else args.extra_critic
    ckpt = torch.load(args.src, weights_only=False, map_location="cpu")
    msd = ckpt["model_state_dict"]

    n_w = n_s = 0
    for key in list(msd.keys()):
        t = msd[key]
        # --- 第1層の重み (actor.0.weight / critic.0.weight など) ---
        if key.endswith(".weight") and t.dim() == 2:
            is_actor = "actor" in key and "normalizer" not in key
            is_critic = "critic" in key and "normalizer" not in key
            if not (is_actor or is_critic):
                continue
            # 第1層だけが対象。層番号 0 を含むキーに限定する。
            if ".0.weight" not in key:
                continue
            msd[key] = _expand_linear(t, args.extra if is_actor else extra_c)
            n_w += 1
            print(f"[weight] {key}: {tuple(t.shape)} -> {tuple(msd[key].shape)}")

        # --- 観測正規化の統計 ---
        elif "normalizer" in key and key.rsplit(".", 1)[-1] in ("_mean", "_var", "_std"):
            extra = args.extra if "actor" in key else extra_c
            fill = 0.0 if key.endswith("_mean") else 1.0
            msd[key] = _expand_stat(t, extra, fill)
            n_s += 1
            print(f"[stat]   {key}: {tuple(t.shape)} -> {tuple(msd[key].shape)}  (fill={fill})")

    # --- Adam のモーメント (optimizer_state_dict) も同じ形に拡張する ---
    #
    # ★ これを忘れると resume 時に
    #     RuntimeError: The size of tensor a (59) must match the size of tensor b (89)
    #   で落ちる。rsl_rl は optimizer state をそのまま読み込むため、パラメータだけ
    #   拡張しても噛み合わない。
    # ★ 新規列のモーメントは **0**。まだ勾配を受けていない重みなので、蓄積が無いのが正しい。
    #   形だけ合わせて中身を捨てる (load_optimizer=False) より、既存列の学習状態を
    #   保てるぶん収束が滑らかになる。
    n_o = 0
    opt = ckpt.get("optimizer_state_dict")
    if opt is not None:
        for pid, state in opt.get("state", {}).items():
            for mkey in ("exp_avg", "exp_avg_sq"):
                m = state.get(mkey)
                if m is None or m.dim() != 2:
                    continue
                # 拡張対象は第1層の重みと同じ形 (out, 59) / (out, 64) のものだけ
                if m.shape[1] == args.extra + 0 or m.shape[1] not in (59, 64):
                    continue
                extra = args.extra if m.shape[1] == 59 else extra_c
                state[mkey] = _expand_linear(m, extra)
                n_o += 1
                print(f"[optim]  param {pid}.{mkey}: {tuple(m.shape)} -> {tuple(state[mkey].shape)}")

    if n_w == 0:
        raise SystemExit(
            "第1層の重みが見つかりませんでした。キー名の規約が違う可能性があります。\n"
            "python3 -c \"import torch;print(list(torch.load('<src>',weights_only=False,"
            "map_location='cpu')['model_state_dict'].keys()))\" で確認してください。"
        )

    torch.save(ckpt, args.dst)
    print(f"\n拡張した重み {n_w} 個 / 統計 {n_s} 個 / optimizer {n_o} 個 を書き出しました: {args.dst}")
    print("★ 拡張直後の方策は元の方策と数学的に同一です (新規列はゼロなので出力に寄与しない)。")


if __name__ == "__main__":
    main()
