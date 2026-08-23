# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""履歴つき横移動ポリシーの観測レイアウト定義。

``feat/inoue_walk_double_encoder`` の ``locomotion/history_layout.py`` を横移動タスク
(``Isaac-GKLateralDH-K1-v0``) 向けに移植したもの。方式もそちらに合わせて
**IsaacLab 標準の ``ObservationGroupCfg.history_length``** を使う (自前のリングバッファは
持たない)。ノイズは ObservationManager が履歴へ push する **前** に 1 度だけ掛かるので、
各ステップのノイズがそのまま履歴に固定される = 実機と同じ性質になる。

観測は 3 グループに分ける:

* ``direct``: 履歴を持たない直接入力。最新の速度指令 (3) + GK タスクスロット (10)。
* ``policy``: actor 用の歩行観測 49 次元を ``HISTORY_LENGTH`` ステップ分バッファしたもの。
* ``critic``: critic 用 (特権情報込み) 54 次元の同様の履歴。

各グループの flatten 後のレイアウトは「項ごとに (履歴長 × 項次元) のブロック」が
**項の定義順** に並ぶ。ブロック内は 古い → 新しい (IsaacLab の CircularBuffer 準拠)。

☠ この項リストは ``rough_env_cfg.py`` の ``K1PolicyCfg`` / ``K1CriticCfg`` の定義順と
  **正確に一致させること**。:mod:`.networks` と :mod:`.symmetry` と :mod:`.exporter` の
  全てがこのレイアウトを参照する。ずれても例外にはならず、静かに間違った列を読む。
"""

# 履歴バッファの長さ [ステップ]。50Hz なので 100 = 2.0s (論文 arXiv:2401.16889 と同じ長さ)。
# ☠ GPU メモリ律速。4096 env で足りない場合はここを 50 (1.0s) に下げるか num_envs を減らす。
HISTORY_LENGTH = 50

# 履歴のうち、CNN を通さず **生のまま** MLP に入れる直近ステップ数。
# 論文の短期 I/O 履歴 (0.1s ≒ 4〜5 frame) に対応。接触イベント・高周波振動への即応担当。
MLP_HISTORY_STEPS = 4

# ``direct`` グループの次元 = velocity_commands(3) + GK タスクスロット(10)
DIRECT_DIM = 13

# 各項: (項名, 次元, 左右反転の種別)。種別は :mod:`.symmetry` の _KIND_TRANSFORMS のキー。
POLICY_TERM_SPECS = (
    ("base_ang_vel", 3, "ang_vel"),
    ("projected_gravity", 3, "gravity"),
    ("velocity_commands", 3, "vel_cmd"),
    ("joint_pos", 12, "joint"),
    ("joint_vel", 12, "joint"),
    ("actions", 12, "joint"),
    ("gait_phase", 4, "phase"),
)

CRITIC_TERM_SPECS = (
    ("base_lin_vel", 3, "lin_vel"),
) + POLICY_TERM_SPECS + (
    ("zmp_position", 2, "zmp"),
)

# ``direct`` グループ内の項 (履歴なし)。critic も同じものを共有する。
DIRECT_TERM_SPECS = (
    ("velocity_commands", 3, "vel_cmd"),
    ("ball_pos_rel", 2, "xy"),
    ("ball_vel", 2, "xy"),
    ("ball_active", 1, "scalar"),
    ("target_y", 1, "flip"),
    ("self_state", 4, "self_state"),
)


def term_specs_dim(term_specs) -> int:
    """項リストの 1 ステップ分の合計次元を返す。"""
    return sum(dim for _, dim, _ in term_specs)


# mirror loss 用に symmetry 関数が返す「直近 1 ステップの生の policy 観測」グループ名。
# :class:`~.networks.HistoryActorCritic` がこれを履歴長ぶんタイル展開して方策入力を作る
# (エピソード開始直後の CircularBuffer バックフィルと同じ、実在しうる観測)。
LATEST_FRAME_GROUP = "policy_latest"
