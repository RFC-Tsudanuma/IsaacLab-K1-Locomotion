# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""long_pass の checkpoint を walk_long_pass_flag の次元へゼロパディングする。

なぜ必要か
----------
``train.py --load_pretrained`` は **形の合わないテンソルを捨てる** 実装なので、
次元が変わった checkpoint をそのまま渡すと::

    actor.0.weight   (512, 55) -> (512, 56)   … 捨てられる = 入力層がランダム初期化
    actor.6.weight   (12, 128) -> (13, 128)   … 捨てられる = 出力層がランダム初期化
    std              (12,)     -> (13,)       … 捨てられる
    *_obs_normalizer.*                        … 捨てられる = 統計リセット

隠れ層だけ残っても、入力層と出力層が両方ランダムではポリシーとして死んでいる。
これまでの stage 遷移が ``--load_pretrained`` で成立していたのは、全 stage が
観測 55 次元・行動 12 次元で「たまたま全テンソルの形が一致していた」ため。

このスクリプトは捨てる代わりに **0 で埋めて拡張** する。新しい入力列が 0 なので
追加した観測は無視され、新しい出力行が 0 なのでフラグは常に bias=0 を出す。つまり
**拡張後のポリシーは元の checkpoint と挙動が完全に一致する**。安全に fine-tune の
出発点にできる。

拡張する内容 (既定値)
---------------------
============================  =============  =============  ================
tensor                        before         after          新しい部分
============================  =============  =============  ================
``actor.<first>.weight``      (H, 55)        (H, 56)        0
``actor.<last>.weight``       (12, H)        (13, H)        0
``actor.<last>.bias``         (12,)          (13,)          0
``std`` / ``log_std``         (12,)          (13,)          ``--flag-std``
``actor_obs_normalizer._mean``  (1, 55)      (1, 56)        0
``actor_obs_normalizer._var``   (1, 55)      (1, 56)        1
``actor_obs_normalizer._std``   (1, 55)      (1, 56)        1
``critic.<first>.weight``     (H, 61)        (H, 69)        0
``critic_obs_normalizer.*``   (1, 61)        (1, 69)        0 / 1 / 1
============================  =============  =============  ================

``count`` と隠れ層と ``critic.<last>.*`` はそのまま。

なぜ新しい ``std`` が既定 0.5 か
--------------------------------
収束済み checkpoint の std は 0.07-0.20 まで落ちている。``init_noise_std`` の 0.72 を
入れると 1 次元だけ突出して entropy と adaptive KL に効きすぎる。逆に小さすぎると
探索できずフラグが学習されない。0.5 は sigmoid の非飽和域 (±3) に対して十分な探索幅。

使い方::

    python scripts/rsl_rl/expand_checkpoint_kick_flag.py \\
        logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \\
        -o /tmp/long_pass_flag_init.pt

    # 検算: --load_pretrained のログが "Loaded N tensors" だけで
    # "Skipped ... tensors" の行が出なければ、全テンソルが載っている。

IsaacLab の python でなくても torch さえあれば動く (isaaclab を import しない)。
"""

from __future__ import annotations

import argparse
import re
import sys

import torch

# 既定の拡張量。walk_long_pass_flag_env_cfg の構成に対応する。
#   policy 観測 55 -> 56 (prev_kick_flag)
#   行動      12 -> 13 (kick_flag)
#   critic 観測 61 -> 69 (prev_kick_flag 1 + kick_latch 2 + kick_frozen 5)
DEFAULT_ACTOR_OBS_PAD = 1
DEFAULT_ACTION_PAD = 1
DEFAULT_CRITIC_OBS_PAD = 8
DEFAULT_FLAG_STD = 0.5


def _pad(t: torch.Tensor, dim: int, n: int, fill: float) -> torch.Tensor:
    """``t`` の ``dim`` 方向の末尾に ``n`` 要素を ``fill`` で足す。"""
    if n <= 0:
        return t
    shape = list(t.shape)
    shape[dim] = n
    block = torch.full(shape, fill, dtype=t.dtype, device=t.device)
    return torch.cat([t, block], dim=dim)


def _layer_indices(msd: dict[str, torch.Tensor], prefix: str) -> tuple[int, int]:
    """``<prefix>.<i>.weight`` の最小/最大インデックス (= 入力層/出力層)。"""
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    idx = sorted(int(m.group(1)) for k in msd if (m := pattern.match(k)))
    if not idx:
        raise KeyError(f"'{prefix}.<i>.weight' が checkpoint に見つかりません。")
    return idx[0], idx[-1]


def expand(
    msd: dict[str, torch.Tensor],
    actor_obs_pad: int,
    action_pad: int,
    critic_obs_pad: int,
    flag_std: float,
) -> list[str]:
    """state_dict をその場で拡張し、変更内容の説明行を返す。"""
    log: list[str] = []

    def note(key: str, before: tuple[int, ...]) -> None:
        log.append(f"  {key:34s} {tuple(before)} -> {tuple(msd[key].shape)}")

    # -- actor: 入力層に観測を、出力層に行動を足す
    a_first, a_last = _layer_indices(msd, "actor")
    if a_first == a_last:
        raise RuntimeError("actor が単層です。想定していない構造なので中断します。")

    key = f"actor.{a_first}.weight"
    before = msd[key].shape
    msd[key] = _pad(msd[key], dim=1, n=actor_obs_pad, fill=0.0)
    note(key, before)

    key = f"actor.{a_last}.weight"
    before = msd[key].shape
    msd[key] = _pad(msd[key], dim=0, n=action_pad, fill=0.0)
    note(key, before)

    key = f"actor.{a_last}.bias"
    before = msd[key].shape
    msd[key] = _pad(msd[key], dim=0, n=action_pad, fill=0.0)
    note(key, before)

    # -- 行動ノイズ std (scalar 型なら std、log 型なら log_std)
    if "std" in msd:
        before = msd["std"].shape
        msd["std"] = _pad(msd["std"], dim=0, n=action_pad, fill=flag_std)
        note("std", before)
    elif "log_std" in msd:
        import math

        before = msd["log_std"].shape
        msd["log_std"] = _pad(msd["log_std"], dim=0, n=action_pad, fill=math.log(flag_std))
        note("log_std", before)
    else:
        raise KeyError("'std' も 'log_std' も見つかりません。ノイズ表現が想定外です。")

    # -- critic: 入力層のみ (出力は value 1 次元のまま)
    c_first, _ = _layer_indices(msd, "critic")
    key = f"critic.{c_first}.weight"
    before = msd[key].shape
    msd[key] = _pad(msd[key], dim=1, n=critic_obs_pad, fill=0.0)
    note(key, before)

    # -- 観測正規化の統計。mean は 0、var/std は 1 で埋める。
    #
    # count は **触らない**。EmpiricalNormalization は rate = batch/count で更新するので、
    # 収束済み checkpoint の count (1e9 オーダー) では足した次元の統計は実質動かない。
    # そのぶん観測側で O(1) にスケールしてある (mdp.observations.kick_frozen_values 参照)。
    for prefix, pad in (("actor_obs_normalizer", actor_obs_pad), ("critic_obs_normalizer", critic_obs_pad)):
        for suffix, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            key = f"{prefix}.{suffix}"
            if key not in msd:
                continue
            before = msd[key].shape
            msd[key] = _pad(msd[key], dim=-1, n=pad, fill=fill)
            note(key, before)

    return log


def main() -> int:
    parser = argparse.ArgumentParser(
        description="long_pass の checkpoint を walk_long_pass_flag の次元へゼロパディングする。"
    )
    parser.add_argument("checkpoint", help="入力 checkpoint (.pt)")
    parser.add_argument("-o", "--output", required=True, help="出力先 (.pt)")
    parser.add_argument("--actor-obs-pad", type=int, default=DEFAULT_ACTOR_OBS_PAD,
                        help=f"policy 観測に足す次元数 (既定 {DEFAULT_ACTOR_OBS_PAD})")
    parser.add_argument("--action-pad", type=int, default=DEFAULT_ACTION_PAD,
                        help=f"行動に足す次元数 (既定 {DEFAULT_ACTION_PAD})")
    parser.add_argument("--critic-obs-pad", type=int, default=DEFAULT_CRITIC_OBS_PAD,
                        help=f"critic 観測に足す次元数 (既定 {DEFAULT_CRITIC_OBS_PAD})")
    parser.add_argument("--flag-std", type=float, default=DEFAULT_FLAG_STD,
                        help=f"フラグ次元の初期ノイズ std (既定 {DEFAULT_FLAG_STD})")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        print("[ERROR] 'model_state_dict' がありません。rsl_rl の checkpoint ではないようです。",
              file=sys.stderr)
        return 1

    msd = ckpt["model_state_dict"]
    print(f"[INFO] loaded: {args.checkpoint} (iter={ckpt.get('iter', '?')})")

    log = expand(msd, args.actor_obs_pad, args.action_pad, args.critic_obs_pad, args.flag_std)
    print("[INFO] expanded tensors:")
    print("\n".join(log))

    ckpt["model_state_dict"] = msd
    # optimizer_state_dict は触らない。--load_pretrained は runner を作り直してから
    # policy の重みだけ上書きするので、optimizer は元から使われない。
    torch.save(ckpt, args.output)
    print(f"[INFO] saved: {args.output}")
    print(f"[INFO] 次: train.py --load_pretrained でこのファイルを渡し、ログが "
          f"'Loaded {len(msd)} tensors' だけで 'Skipped ... tensors' の行が"
          " 出ないことを確認すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
