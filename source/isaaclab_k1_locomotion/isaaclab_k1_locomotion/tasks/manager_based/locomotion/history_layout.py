# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat 環境の「観測履歴バッファ + 最新コマンド」構成の観測レイアウト定義。

K1FlatEnvCfg では観測を 3 グループに分ける:

* ``command``: 最新の歩行コマンドのみ (履歴なし, 3 次元)。
* ``policy``:  Actor 用観測 (コマンド込み, 2026-08-05〜) を ``HISTORY_LENGTH`` ステップ分
  バッファしたもの (ノイズあり)。ノイズは各ステップで 1 度だけ掛かり、そのまま
  履歴に固定される (IsaacLab の ObservationManager がノイズ適用後に CircularBuffer
  へ push するため)。
* ``critic``:  Critic 用観測 (特権情報込み・ノイズなし・コマンド込み) の同様の履歴。

NOTE (2026-08-05): 以前は履歴グループからコマンドを除いていたが、
「4 ステップ MLP 履歴にも 100 ステップ CNN 履歴にもコマンド系列を含める」
仕様に変更した。command グループ (最新値の直接入力) はそのまま残している。

各グループの flatten 後のレイアウトは「項ごとに (履歴長 × 項次元) のブロック」が
項の定義順に並ぶ。各ブロック内は 古い → 新しい の順 (CircularBuffer.buffer 準拠)。

このモジュールの項リストは flat_env_cfg の K1PolicyCfg / K1CriticCfg の
項定義順と正確に一致させること。HistoryActorCritic と mdp/symmetry.py と
history_policy_exporter の全てがこのレイアウトを参照する。
"""

# 履歴バッファの長さ (ステップ数)
HISTORY_LENGTH = 100

# actor-critic の MLP に入力する直近ステップ数
MLP_HISTORY_STEPS = 4

# command グループの次元 (vx, vy, ωz)
COMMAND_DIM = 3

# 各項: (項名, 次元, 左右反転の種別)
# 左右反転の種別は mdp/symmetry.py の _KIND_TRANSFORMS のキーに対応する。
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


def term_specs_dim(term_specs) -> int:
    """項リストの 1 ステップ分の合計次元を返す。"""
    return sum(dim for _, dim, _ in term_specs)


# mirror loss 用に symmetry 関数が返す「直近 1 ステップの生の policy 観測」グループ名。
# HistoryActorCritic がこのグループを履歴長ぶんタイル展開して方策入力を構築する。
LATEST_FRAME_GROUP = "policy_latest"


# ---------------------------------------------------------------------------
# B-Human 版キック (bhuman_walk_kick) のレイアウト
# ---------------------------------------------------------------------------
# 歩行の項 (POLICY_TERM_SPECS / CRITIC_TERM_SPECS) は**そのまま先頭に置き**、キック用の項を
# 後ろに足す。こうすると歩行の checkpoint の重みを「足した入力の重みを 0」で移せる
# (scripts/rsl_rl/convert_walk_to_bhuman_kick.py)。並びを変えたら変換スクリプトも直すこと。
KICK_EXTRA_TERM_SPECS = (
    ("ball_position", 2, "ball_xy"),           # 両足裏の中点から見たボール位置 (yaw フレーム) * 0.1
    ("kick_direction", 1, "neg"),              # キック方向 [rad] / pi (胴体の向き基準)
    ("kick_direction_sincos", 2, "dir_sincos"),  # (sin, cos)
    ("kick_flags", 3, "none"),                 # 強キック / 精度不問 / 逸脱可
    ("ball_velocity", 2, "ball_xy"),           # ボール速度 (yaw フレーム) * 0.1
    ("expected_ball_velocity", 1, "none"),     # 目標距離に必要なボール初速 * 0.1
    ("gait_frequency_action", 1, "none"),      # 歩行周波数の action (前ステップ)
)

KICK_POLICY_TERM_SPECS = POLICY_TERM_SPECS + KICK_EXTRA_TERM_SPECS

KICK_CRITIC_TERM_SPECS = CRITIC_TERM_SPECS + KICK_EXTRA_TERM_SPECS + (
    ("kick_pose_state", 2, "none"),            # B-Human の inside_kick_pose_counter, ball_close_to_kick_pose
)

_ALL_POLICY_SPECS = (POLICY_TERM_SPECS, KICK_POLICY_TERM_SPECS)
_ALL_CRITIC_SPECS = (CRITIC_TERM_SPECS, KICK_CRITIC_TERM_SPECS)


def policy_specs_for_dim(dim: int, history_length: int = HISTORY_LENGTH):
    """履歴グループの次元から policy の項レイアウトを選ぶ (歩行 / キック)。"""
    for specs in _ALL_POLICY_SPECS:
        if term_specs_dim(specs) * history_length == dim:
            return specs
    raise ValueError(f"policy 履歴の次元 {dim} に合う項レイアウトがありません (history_layout.py)")


def critic_specs_for_dim(dim: int, history_length: int = HISTORY_LENGTH):
    """履歴グループの次元から critic の項レイアウトを選ぶ (歩行 / キック)。"""
    for specs in _ALL_CRITIC_SPECS:
        if term_specs_dim(specs) * history_length == dim:
            return specs
    raise ValueError(f"critic 履歴の次元 {dim} に合う項レイアウトがありません (history_layout.py)")
