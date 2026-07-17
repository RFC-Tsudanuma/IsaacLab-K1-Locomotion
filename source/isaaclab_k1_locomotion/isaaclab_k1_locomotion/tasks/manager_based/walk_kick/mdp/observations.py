# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick 用の観測関数。

B-Human "A Modular Ball Kicking Behavior with Reinforcement Learning"
(Reichenberg & Frese) の observation 空間を K1 向けに移植したもの。
Flags (3次元) は使わない。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from ...locomotion.mdp.events import get_phase_freq

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos_rel(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットベースフレーム（yaw aligned）でのボール相対位置。shape: (N, 3)

    policy には遅延なしの現在位置は渡さない (:func:`prev_ball_pos_b` を使う)。
    critic の特権情報としてのみ使用する。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    rel_pos_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)
    return rel_pos_b


def ball_vel_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットベースフレーム（yaw aligned）でのボール水平速度。shape: (N, 2)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    vel_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), ball.data.root_lin_vel_w[:, :3])
    return vel_b[:, :2]


def sole_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_foot_link"),
) -> torch.Tensor:
    """足裏 (sole) のベース相対 3D 位置。shape: (N, 3)

    論文の評価表と同じく左足裏を既定とする。ベースの姿勢を完全に打ち消した
    ボディフレーム (yaw だけでなく roll/pitch も除去) で表現する。
    """
    robot = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]

    offset_w = robot.data.body_pos_w[:, body_idx, :] - robot.data.root_pos_w[:, :3]
    return quat_rotate_inverse(robot.data.root_quat_w, offset_w)


def gait_phase_sincos(
    env: ManagerBasedRLEnv,
    phase_freq: float = 1.6,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
) -> torch.Tensor:
    """歩行位相を (sin, cos) で返す。shape: (N, 2)

    locomotion 側の ``phase_obs`` は左右両脚の位相 (4次元) を返すが、
    右脚位相は左脚位相 + π で一意に決まるため、ここでは左脚位相のみを渡す。
    コマンド速度が ``cmd_threshold`` 未満のときは ``phase_obs`` と同様にゼロで
    埋め、停止すべき状況であることを明示する (feet_phase 報酬のゲートと揃える)。
    """
    t = env.episode_length_buf * env.step_dt
    pf = get_phase_freq(env, phase_freq)
    phase = 2.0 * math.pi * pf * t

    phase_sincos = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed < cmd_threshold
    return torch.where(is_stopped, torch.zeros_like(phase_sincos), phase_sincos)


def gait_phase_factor_offset(
    env: ManagerBasedRLEnv,
    base_phase_freq: float = 1.6,
) -> torch.Tensor:
    """この env の歩行位相周波数が基準値からどれだけずれているか。shape: (N, 1)

    ``randomize_phase_freq`` イベントが startup で env ごとに ±offset を振るので、
    その offset (= pf - base_phase_freq) をポリシーに開示する。イベントが無い
    構成ではゼロを返す。
    """
    pf = get_phase_freq(env, base_phase_freq)
    if not isinstance(pf, torch.Tensor):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (pf - base_phase_freq).unsqueeze(-1)


def kick_dir_b(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """目標蹴り方向のベース相対 2D 単位ベクトル。shape: (N, 2)

    ``KickDirectionCommand`` は world frame の角度 θ を [sin θ, cos θ, v] で持つので、
    world の方向ベクトル (cos θ, sin θ) をロボットの yaw frame に変換して返す。
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)  # (N, 3) [sin θ, cos θ, v]

    dir_w = torch.zeros(env.num_envs, 3, device=env.device)
    dir_w[:, 0] = command[:, 1]  # cos θ
    dir_w[:, 1] = command[:, 0]  # sin θ

    dir_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), dir_w)
    return dir_b[:, :2]


def target_kick_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """目標ボール速度 [m/s]。shape: (N, 1)

    ``KickDirectionCommand`` が command[:, 2] に格納する。
    """
    command = env.command_manager.get_command(command_name)
    return command[:, 2].unsqueeze(-1)


# --------------------------------------------------------------------------- #
# walk phase 用: ボール/キック由来のスロットを歩行コマンドで置き換える
#
# 観測の次元と並びを kick phase と 1 bit も違わないように保ったまま、中身だけ差し替える。
# こうすると walk phase の checkpoint をそのまま kick phase に引き継げる。
#
# スロットの対応は kick phase の BallFollowVelocityCommand の式から決めている:
#   vx, vy = clamp(<G のロボット相対位置>, ±max_vel)  ← prev_ball_pos スロットと同じ土俵
#   wz     = clamp(<kick_dir_b の偏角>,   ±max_ang_vel) ← kick_direction スロットそのもの
# なので walk phase では「仮想的な目標点」を prev_ball_pos に、「仮想的な目標向き」を
# kick_direction に載せる。policy から見た「スロットが指す方へ歩き、指す向きへ回る」という
# 入力→挙動の対応が両 phase で共通になるので、歩容がそのまま転移する。
#
# NOTE: kick phase の vx,vy は G (ボールの後方 reach の点) を指すので、対応は厳密な恒等では
#       なく reach 分のオフセットが乗る。歩容そのものの転移が目的なので許容し、その差は
#       kick phase 側のフェードイン期間で吸収させる。
# --------------------------------------------------------------------------- #


def walk_command_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """walk phase: 速度コマンドの (vx, vy) を prev_ball_pos スロットに載せる。shape: (N, 2)"""
    return env.command_manager.get_command(command_name)[:, :2]


def walk_command_yaw_dir(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """walk phase: 角速度コマンド wz を kick_direction スロットに載せる。shape: (N, 2)

    kick phase の kick_dir_b は単位ベクトルで、その偏角が wz 指令に対応する。
    合わせて (cos wz, sin wz) の単位ベクトルとして渡す。
    """
    wz = env.command_manager.get_command(command_name)[:, 2]
    return torch.stack([torch.cos(wz), torch.sin(wz)], dim=-1)


def zero_obs(env: ManagerBasedRLEnv, dim: int = 1) -> torch.Tensor:
    """walk phase: ボールが存在しないスロットをゼロで埋める。shape: (N, dim)"""
    return torch.zeros(env.num_envs, dim, device=env.device)


def prev_ball_pos_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """前ステップのボール水平位置（ベース相対）。shape: (N, 2)

    知覚遅延を模すため、現在位置ではなく 1 ステップ前の値を渡す。
    エピソード開始直後は前ステップが存在しないので現在位置で初期化する
    (ゼロ埋めするとボールがベース原点にあるという誤った観測になるため)。

    同一ステップ内で policy / critic の両グループから呼ばれても値が二重に
    進まないよう、``common_step_counter`` でステップ境界を検出する。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    cur = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)[:, :2]

    step = int(env.common_step_counter)
    state = getattr(env, "_prev_ball_pos_state", None)
    if state is None:
        state = {"prev": cur.clone(), "cur": cur.clone(), "step": step}
        env._prev_ball_pos_state = state
    elif state["step"] != step:
        # ステップが進んだ: 前ステップに観測した「現在位置」が今ステップの「前回位置」になる
        state["prev"] = state["cur"]
        state["cur"] = cur.clone()
        state["step"] = step

    # リセット直後の env は前エピソードの値を引き継がせない。
    # episode_length_buf は step() 内で加算された後に _reset_idx で 0 に戻されるので、
    # 「今このステップでリセットされた env」だけが 0 になる。
    just_reset = env.episode_length_buf == 0
    state["prev"][just_reset] = cur[just_reset]
    state["cur"][just_reset] = cur[just_reset]

    return state["prev"]
