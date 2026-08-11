# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""long_pass の checkpoint を walk_long_pass_history の次元へ並べ替え + ゼロ埋めする。

なぜ専用スクリプトが要るか
--------------------------
``expand_checkpoint_kick_flag.py`` は **末尾にゼロを足す** だけなので使えない。
履歴化は「観測を末尾に足す」のではなく **各項をその場で H 倍に展開する** ので、
55 次元の並びが 223 次元の中に散らばる::

    項                       before      after
    projected_gravity        0-2         0-14      (5 スロット)
    base_ang_vel             3-5         15-29     (5 スロット)
    sole_pos                 6-8         30-32
    gait_phase               9-10        33-34
    joint_pos                11-22       35-94     (5 スロット)
    joint_vel                23-34       95-154    (5 スロット)
    prev_joint_request       35-46       155-214   (5 スロット)
    gait_phase_factor_offset 47          215
    kick_direction           48-49       216-217
    target_kick_velocity     50          218
    ball_vel                 51-52       219-220
    prev_ball_pos            53-54       221-222

``train.py --load_pretrained`` は形の合わないテンソルを **捨てる** 実装なので、生の
checkpoint を渡すと ``actor.0.weight`` (512, 55) と ``actor_obs_normalizer.*`` が
落ちて入力層がランダム初期化される = ポリシーとして死ぬ。

このスクリプトがやること
------------------------
1. ``actor.<first>.weight`` (H_hidden, 55) → (H_hidden, 223)。
   元の列を各履歴ブロックの **最新スロット** (ブロック末尾) に置き、残り H-1 スロットは
   **0** で埋める。CircularBuffer は「古い順・最新が末尾」で flatten するので、
   最新スロット = 従来の 1 フレーム観測そのもの。
   → **拡張直後のポリシーは元の checkpoint と挙動が完全に一致する** (過去フレームは
   重み 0 で無視される)。ここから fine-tune するのが安全な出発点。

2. ``actor_obs_normalizer.{_mean,_var,_std}`` (1, 55) → (1, 223)。
   こちらは 0/1 埋めではなく **元の統計を全スロットに複製する**。履歴スロットに流れる
   のは最新スロットと同じ分布 (同じ項の 1-4 ステップ前) なので、複製が正しい統計。

   flag 版が新しい次元を 0/1 で埋めていたのは「そこに未知の量が来る」からで、事情が違う。
   ``count`` を触らない点は同じ (EmpiricalNormalization は rate = batch/count で
   更新するので、収束済み checkpoint の count では足した次元の統計が実質動かない。
   正しい値を最初から入れておく必要がある)。

3. それ以外 (隠れ層・出力層・``std``・``critic.*``・``critic_obs_normalizer.*``) は
   **そのまま**。walk_long_pass_history は行動 12 次元も critic 観測 61 次元も
   変えていない。

使い方::

    python scripts/rsl_rl/expand_checkpoint_history.py \\
        logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \\
        -o /tmp/long_pass_history_init.pt

    # 検算: --load_pretrained のログが "Loaded N tensors" だけで
    # "Skipped ... tensors" の行が出なければ、全テンソルが載っている。

IsaacLab の python でなくても torch さえあれば動く (isaaclab を import しない)。

保守について
------------
下の ``_POLICY_TERMS`` は
:class:`~isaaclab_k1_locomotion.tasks.manager_based.walk_kick.walk_kick_env_cfg.K1WalkKickPolicyCfg`
の **宣言順と次元** の写しで、``history`` 列は
``walk_long_pass_history_env_cfg._HISTORY_TERMS`` の写し。cfg 側を変えたらここも直すこと。
片方だけ変えると checkpoint が黙って壊れた並びで読まれる (エラーにはならない)。
合計次元だけは checkpoint の実寸と突き合わせて検算している。
"""

from __future__ import annotations

import argparse
import re
import sys

import torch

# --------------------------------------------------------------------------- #
# policy 観測の項テーブル: (名前, 次元, 履歴を付けるか)
#
# 宣言順 = ObservationManager の連結順。K1WalkKickPolicyCfg の並びそのまま。
# --------------------------------------------------------------------------- #
_POLICY_TERMS: tuple[tuple[str, int, bool], ...] = (
    ("projected_gravity", 3, True),
    ("base_ang_vel", 3, True),
    ("sole_pos", 3, False),
    ("gait_phase", 2, False),
    ("joint_pos", 12, True),
    ("joint_vel", 12, True),
    ("prev_joint_request", 12, True),  # actions
    ("gait_phase_factor_offset", 1, False),
    ("kick_direction", 2, False),
    ("target_kick_velocity", 1, False),
    ("ball_vel", 2, False),
    ("prev_ball_pos", 2, False),
)

# walk_long_pass_history_env_cfg._HISTORY_LEN と同じ値。
DEFAULT_HISTORY_LEN = 5
# critic 観測は変えないので、この幅で来ているかだけ確認する (policy 55 + 特権 6)。
EXPECTED_CRITIC_DIM = 61


def build_column_maps(
    terms: tuple[tuple[str, int, bool], ...], history_len: int
) -> tuple[list[int], list[int]]:
    """新しい列 → 元の列 の写像を 2 本作る。

    Args:
        terms: ``(名前, 次元, 履歴を付けるか)`` を宣言順に並べたもの。
        history_len: 履歴スロット数。

    Returns:
        ``(weight_src, stat_src)``。どちらも長さが新次元のリストで、要素は元の列番号。

        * ``weight_src[k]`` … 重みの写像。**最新スロット以外は -1** (= 0 で埋める)。
        * ``stat_src[k]``   … 正規化統計の写像。全スロットに元の列を複製するので -1 は無い。
    """
    weight_src: list[int] = []
    stat_src: list[int] = []
    old_offset = 0
    for _name, dim, has_history in terms:
        n_slots = history_len if has_history else 1
        for slot in range(n_slots):
            # CircularBuffer.buffer は「古い順・最新が末尾」。最新スロットが従来の観測。
            is_newest = slot == n_slots - 1
            for j in range(dim):
                stat_src.append(old_offset + j)
                weight_src.append(old_offset + j if is_newest else -1)
        old_offset += dim
    return weight_src, stat_src


def _layer_indices(msd: dict[str, torch.Tensor], prefix: str) -> tuple[int, int]:
    """``<prefix>.<i>.weight`` の最小/最大インデックス (= 入力層/出力層)。"""
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    idx = sorted(int(m.group(1)) for k in msd if (m := pattern.match(k)))
    if not idx:
        raise KeyError(f"'{prefix}.<i>.weight' が checkpoint に見つかりません。")
    return idx[0], idx[-1]


def _scatter_weight(w: torch.Tensor, weight_src: list[int]) -> torch.Tensor:
    """``w`` (out, old_dim) の列を新しい並びへ散らし、未対応の列を 0 にする。"""
    out = w.new_zeros((w.shape[0], len(weight_src)))
    dst = [k for k, s in enumerate(weight_src) if s >= 0]
    src = [weight_src[k] for k in dst]
    out[:, dst] = w[:, src]
    return out


def _scatter_stat(t: torch.Tensor, stat_src: list[int]) -> torch.Tensor:
    """正規化統計 (..., old_dim) の最終次元を新しい並びへ複製展開する。"""
    index = torch.as_tensor(stat_src, dtype=torch.long, device=t.device)
    return t.index_select(dim=-1, index=index).contiguous()


def expand(msd: dict[str, torch.Tensor], history_len: int) -> list[str]:
    """state_dict をその場で拡張し、変更内容の説明行を返す。"""
    log: list[str] = []

    weight_src, stat_src = build_column_maps(_POLICY_TERMS, history_len)
    old_dim = sum(dim for _n, dim, _h in _POLICY_TERMS)
    new_dim = len(weight_src)

    # -- actor 入力層
    a_first, a_last = _layer_indices(msd, "actor")
    if a_first == a_last:
        raise RuntimeError("actor が単層です。想定していない構造なので中断します。")

    key = f"actor.{a_first}.weight"
    w = msd[key]
    if w.shape[1] != old_dim:
        raise RuntimeError(
            f"{key} の入力幅が {w.shape[1]} です ( _POLICY_TERMS の合計 {old_dim} と不一致)。\n"
            "  この checkpoint は walk_long_pass 系ではないか、観測の構成が変わっています。\n"
            "  _POLICY_TERMS を実際の cfg に合わせて直してください。"
        )
    before = tuple(w.shape)
    msd[key] = _scatter_weight(w, weight_src)
    log.append(f"  {key:34s} {before} -> {tuple(msd[key].shape)}  (最新スロットへ写像、他は 0)")

    # -- 観測正規化の統計。履歴スロットには同じ分布が流れるので元の統計を複製する。
    #
    # count は触らない (収束済み checkpoint の count は 1e9 オーダーで、
    # rate = batch/count のため足した次元の統計は実質もう動かない)。
    #
    # キー名を ``actor_obs_normalizer`` 決め打ちにせず、``*_obs_normalizer._{mean,var,std}``
    # を総なめして **幅が old_dim のものだけ** 拡張する。rsl_rl のバージョンで
    # ``obs_normalizer`` / ``actor_obs_normalizer`` と名前が揺れるため。critic 側は
    # 幅が違う (61) ので自然に外れるが、念のため名前でも弾いておく。
    normalizer_pattern = re.compile(r"^(?P<prefix>\w*obs_normalizer)\.(_mean|_var|_std)$")
    found_normalizer = False
    for key in list(msd):
        m = normalizer_pattern.match(key)
        if m is None or m.group("prefix").startswith("critic"):
            continue
        t = msd[key]
        if t.shape[-1] != old_dim:
            print(
                f"[WARN] {key} の幅が {t.shape[-1]} です ({old_dim} を期待)。拡張せずそのまま残します。",
                file=sys.stderr,
            )
            continue
        found_normalizer = True
        before = tuple(t.shape)
        msd[key] = _scatter_stat(t, stat_src)
        log.append(f"  {key:34s} {before} -> {tuple(msd[key].shape)}  (全スロットへ複製)")
    if not found_normalizer:
        print(
            "[WARN] policy 側の観測正規化テンソル (*_obs_normalizer._mean など) が\n"
            "       見つかりませんでした。正規化を使っていない checkpoint なら問題ありませんが、\n"
            "       使っているのに見つからない場合は --load_pretrained に捨てられ、統計が\n"
            "       リセットされます。checkpoint のキー名を確認してください。",
            file=sys.stderr,
        )

    # -- critic は据え置き。幅だけ検算して、想定外なら気づけるようにする。
    c_first, _ = _layer_indices(msd, "critic")
    critic_shape = tuple(msd[f"critic.{c_first}.weight"].shape)
    if critic_shape[1] != EXPECTED_CRITIC_DIM:
        print(
            f"[WARN] critic 入力幅が {critic_shape[1]} です ({EXPECTED_CRITIC_DIM} を期待)。\n"
            "       walk_long_pass_history は critic を変更しないので、このまま渡すと\n"
            "       critic 側だけ形が合わず --load_pretrained に捨てられます。",
            file=sys.stderr,
        )
    log.append(f"  {f'critic.{c_first}.weight':34s} {critic_shape} -> 変更なし")
    log.append(f"  {f'actor.{a_last}.* / std / critic_obs_normalizer.*':34s} -> 変更なし")
    log.append(f"  policy 観測: {old_dim} -> {new_dim} (履歴長 {history_len})")
    return log


def main() -> int:
    parser = argparse.ArgumentParser(
        description="long_pass の checkpoint を walk_long_pass_history の次元へ並べ替え + ゼロ埋めする。"
    )
    parser.add_argument("checkpoint", help="入力 checkpoint (.pt)")
    parser.add_argument("-o", "--output", required=True, help="出力先 (.pt)")
    parser.add_argument(
        "--history-len",
        type=int,
        default=DEFAULT_HISTORY_LEN,
        help=f"履歴スロット数 (既定 {DEFAULT_HISTORY_LEN} = 0.1 s / 0.02 s)。"
        " walk_long_pass_history_env_cfg._HISTORY_LEN と揃えること。",
    )
    args = parser.parse_args()

    if args.history_len < 1:
        print("[ERROR] --history-len は 1 以上にしてください。", file=sys.stderr)
        return 1

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        print("[ERROR] 'model_state_dict' がありません。rsl_rl の checkpoint ではないようです。",
              file=sys.stderr)
        return 1

    msd = ckpt["model_state_dict"]
    print(f"[INFO] loaded: {args.checkpoint} (iter={ckpt.get('iter', '?')})")

    log = expand(msd, args.history_len)
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
