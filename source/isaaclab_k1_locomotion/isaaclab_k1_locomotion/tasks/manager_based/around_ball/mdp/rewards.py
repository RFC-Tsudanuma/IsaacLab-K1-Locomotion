# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用の報酬関数。

タスク定義:
    1. キック方向 (``kick_direction`` コマンド, ワールド座標の単位ベクトル) が与えられる。
    2. ロボットはボールを動かさないように回り込み、「ボールの後方 standoff [m]
       (キック方向の反対側)」の目標点に到達する。
    3. 体の向きがキック方向と揃ったら、止まらずにボールへ突進して歩いたまま
       ボールに突っ込み、キック方向へボールを動かす。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .observations import ball_offset_and_bearing

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# ハンドオフ (突進→通常歩行の引き渡し) 中の報酬ゲーティング
#
# ``HierarchicalVecEnvWrapper`` (scripts/rsl_rl/dribble_helpers.py) が
# ``env._handoff_active`` を立てている間、上位ポリシーの action は捨てられ、
# 代わりに通常歩行コマンドが frozen に流れている。つまりその区間の挙動は
# **上位ポリシーの制御下にない** ので、上位 action に紐づく報酬・ペナルティを
# 課金してはいけない (ポリシーが起こしていない事象で加点/減点されてしまう)。
#
# 一方で ``termination_penalty`` は **残す**。ハンドオフ後に転倒したら -500 が
# エピソードに乗り、GAE で遡って「どんな速度・姿勢で接触したか」に信用割当される。
# これが「ハンドオフを生き延びられる状態で突っ込む」を学習させる本体。
# ``ball_moved_along_kick`` も残す (蹴りの成果は評価したいので)。
# ============================================================================


def handoff_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ハンドオフ中は 0、それ以外は 1 の per-env マスク (N,)。

    状態機が無効 (handoff なしの学習・他タスク) なら ``_handoff_active`` が存在せず
    常に 1 を返すので、ゲート版の報酬をそのまま使っても従来と同じ挙動になる。
    """
    flag = getattr(env, "_handoff_active", None)
    if flag is None:
        return torch.ones(env.num_envs, device=env.device)
    return (~flag).float()


def gated_high_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """:func:`locomotion.mdp.rewards.high_action_rate_l2` のハンドオフゲート版。"""
    from ...locomotion.mdp.rewards import high_action_rate_l2

    return high_action_rate_l2(env) * handoff_mask(env)


def gated_high_action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """:func:`locomotion.mdp.rewards.high_action_smoothness_l2` のハンドオフゲート版。"""
    from ...locomotion.mdp.rewards import high_action_smoothness_l2

    return high_action_smoothness_l2(env) * handoff_mask(env)


def gated_high_action_xy_coactivation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """:func:`locomotion.mdp.rewards.high_action_xy_coactivation` のハンドオフゲート版。

    ``normal_cmd`` はランダム化されていて vx・vy が同時に立つことがあるが、それは
    上位ポリシーの出力ではないので罰さない。
    """
    from ...locomotion.mdp.rewards import high_action_xy_coactivation

    return high_action_xy_coactivation(env) * handoff_mask(env)


def _kick_geometry(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """回り込みの幾何量を返すヘルパ。

    Returns:
        dist: ロボット↔ボールの xy 距離 (N,)
        theta_pos: 「ロボット→ボール方向」とキック方向のなす角 (N,) [rad]。
            0 のときロボットはボールの真後ろ (= 蹴る位置) にいる。
        heading_err: ロボットの向きとキック方向のなす角の絶対値 (N,) [rad]。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    to_ball = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.norm(to_ball, dim=1)
    kick_dir = env.command_manager.get_term(command_name).command  # (N, 2) 単位ベクトル

    cos_pos = (to_ball * kick_dir).sum(dim=1) / dist.clamp(min=1e-6)
    theta_pos = torch.acos(cos_pos.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    kick_angle = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
    heading_err = wrap_to_pi(kick_angle - robot.data.heading_w).abs()
    return dist, theta_pos, heading_err


def standoff_point_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    standoff: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """「ボール後方 standoff [m] の目標点」への 1 ステップ接近量 (m) を返すポテンシャル報酬。

    目標点 p* = ball_xy - standoff * kick_dir (ワールド座標)。
    ``progress = dist_{t-1} - dist_t`` で、近づくと正、離れると負。ポテンシャル差なので
    エピソード総和は telescope し、行ったり来たりで報酬を稼ぐことはできない。
    ボールのどちら側から回り込むかは指定しない (近い方を自然に選ぶ)。
    locomotion の :func:`approach_ball_progress` と同じパターン。
    リセット直後 (``episode_length_buf < 2``) は距離が不連続に飛ぶので 0 を返す。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    kick_dir = env.command_manager.get_term(command_name).command  # (N, 2)
    target = ball.data.root_pos_w[:, :2] - standoff * kick_dir
    dist = torch.norm(target - robot.data.root_pos_w[:, :2], dim=1)

    prev = getattr(env, "_prev_standoff_dist", None)
    if prev is None or prev.shape != dist.shape:
        env._prev_standoff_dist = dist.clone()
        return torch.zeros_like(dist)
    progress = prev - dist
    env._prev_standoff_dist = dist.clone()
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(progress), progress)


def aligned_pose_hold(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    target_distance: float = 0.3,
    dist_std: float = 0.15,
    angle_std: float = 0.35,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """最終姿勢 (ボールの真後ろ・体はキック方向・蹴れる距離) に居るほど大きい [0, 1] の状態報酬。

    3 つのガウスカーネルの積:
        * 距離: |robot↔ball| が ``target_distance`` に近い
        * 配置: ロボットがボールのキック方向の真後ろにいる (theta_pos ≈ 0)
        * 向き: 体の向きがキック方向と一致 (heading_err ≈ 0)

    ポテンシャル (差分) ではなく毎ステップの状態報酬なので、「良い姿勢に到達して
    留まり続ける」ことが最適になる。standoff_point_progress が目標点から先へ進む分を
    わずかに罰する (~0.2m × weight) が、この報酬を数ステップ稼げば上回る設計。
    """
    dist, theta_pos, heading_err = _kick_geometry(env, command_name, robot_cfg, ball_cfg)
    r_dist = torch.exp(-torch.square(dist - target_distance) / dist_std**2)
    r_pos = torch.exp(-torch.square(theta_pos) / angle_std**2)
    r_heading = torch.exp(-torch.square(heading_err) / angle_std**2)
    return r_dist * r_pos * r_heading


def misaligned_ball_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    min_clearance: float = 0.45,
    angle_tol: float = 0.6,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """姿勢が合っていないのにボールへ近づきすぎたら 1 を返すペナルティ項 (weight < 0 で使う)。

    「ボールの真後ろ (theta_pos <= angle_tol) 以外の方向から距離 min_clearance 未満に
    入る」= 回り込み中にボールを突っ切るショートカットを直接罰する。これにより
    ボールの周囲 min_clearance の円が「正面ゲート付きの障害物」になる。
    """
    dist, theta_pos, _ = _kick_geometry(env, command_name, robot_cfg, ball_cfg)
    return ((dist < min_clearance) & (theta_pos > angle_tol)).float()


def ball_out_of_fov(
    env: ManagerBasedRLEnv,
    fov_half_angle_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ボールが視野 (±fov_half_angle_deg) の外にいる間 1 を返すペナルティ項 (weight < 0 で使う)。

    観測の hold-last-seen (``ball_pos_rel_fov``) と対になる項で、「ボールを見失わない
    ように動く」ことを直接教える。回り込み中は常にボールへ正対し続ければ 0 になる。
    """
    _, bearing = ball_offset_and_bearing(env, asset_cfg)
    return (bearing > math.radians(fov_half_angle_deg)).float()


def charge_to_ball_when_aligned(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    gate_pos_std: float = 0.5,
    gate_heading_std: float = 0.9,
    max_speed: float = 1.0,
    min_distance: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """「キック方向に回り込めた状態でのみ」ボールへの接近速度を報酬にする [0, 1] の項。

    タスクの 2 段目「回り込めたらボールへ突っ込む」を直接駆動する。

    構成:
        * ``gate``  : ロボットがボールの真後ろ (theta_pos≈0) かつ体がキック方向を
          向いている (heading_err≈0) ほど 1 に近づく滑らかなゲート [0, 1]。
          2 つのガウスの積。視野内で揃うほど突進報酬が「開く」。
        * ``v_along``: ロボットのワールド xy 速度のボール方向成分 (前進のみ、負は 0)。
          ``max_speed`` で正規化して [0, 1]。**速度そのもの**を見るので「速く突っ込む」
          ほど高い (ポテンシャル型の接近報酬と違い、速さが直接効く)。

    報酬 = gate * v_along。揃っていなければ gate≈0 で突進しても報酬が出ない
    (= 先に回り込まないと突進報酬が開かない) ので、回り込み→突進の順序が強制される。
    横移動 (orbit) 中は v_along≈0 なので回り込み自体には報酬を与えない。

    ``min_distance`` > 0 を渡すとその距離より近くで報酬を 0 にできる (ボール前で
    止めたい場合用)。0 (既定) なら接触するまで報酬が続く = 歩いたまま突っ込む。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    dist, theta_pos, heading_err = _kick_geometry(env, command_name, robot_cfg, ball_cfg)

    gate = torch.exp(-torch.square(theta_pos) / gate_pos_std**2) * torch.exp(
        -torch.square(heading_err) / gate_heading_std**2
    )

    to_ball = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    direction = to_ball / (dist.unsqueeze(1) + 1e-6)
    v_along = (robot.data.root_lin_vel_w[:, :2] * direction).sum(dim=1)
    v_along = torch.clamp(v_along, min=0.0)
    reward = gate * torch.clamp(v_along / max_speed, 0.0, 1.0)

    # min_distance > 0 のときのみ、その距離より近くで報酬を切る (既定 0 = 切らない)。
    reward = torch.where(dist < min_distance, torch.zeros_like(reward), reward)

    # ★ハンドオフ後は 0。これが無いと「蹴った後もボールを追い続ける」ことに
    # 満額 (weight 18.0 × gate 1.0 × v_along 1.0) が払われ続ける。蹴った後は
    # ボールが正面に飛ぶので gate≈1 のまま、全速で追えば v_along≈1 になり、
    # 転倒 (-500) を差し引いても黒字になってしまう (実測された転倒の主因)。
    # ハンドオフ後は上位 action が捨てられているので、そもそもポリシーの手柄でもない。
    return reward * handoff_mask(env)


def ball_disturbance_when_misaligned(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    angle_tol: float = 0.6,
    max_speed: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """回り込みが完了していない状態でボールを動かしたら罰する項 (weight < 0 で使う)。

    ボールの xy 速度ノルムを ``max_speed`` で正規化 [0, 1] し、ロボットがボールの
    真後ろに揃っていない (theta_pos > angle_tol) ときだけ返す。

    無条件の ball_speed ペナルティと違い、**揃った後の突進でボールを弾くのは無罪**。
    「回り込み中に触るな、揃ってから突っ込め」だけを教える。
    angle_tol は misaligned_ball_proximity と同じ値に揃えること (境界の一貫性)。
    """
    ball = env.scene[ball_cfg.name]
    _, theta_pos, _ = _kick_geometry(env, command_name, robot_cfg, ball_cfg)
    speed = torch.norm(ball.data.root_com_vel_w[:, :2], dim=1)
    speed_norm = torch.clamp(speed / max_speed, 0.0, 1.0)
    return speed_norm * (theta_pos > angle_tol).float()
