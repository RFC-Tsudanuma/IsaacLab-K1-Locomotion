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

import isaaclab.envs.mdp as base_mdp
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


# --------------------------------------------------------------------------- #
# センサ遅延の domain randomization
#
# 実機では IMU も関節エンコーダも「測ってから policy に届くまで」に遅れがある
# (バス転送・フィルタ・制御ループの位相)。sim で遅延ゼロのまま学習すると、実機の
# 遅れた観測に対して過剰に反応する (特に base_ang_vel は歩行の安定化に直結する)。
#
# 実装は「過去フレームの線形補間」。制御周期 dt = 0.02 s に対して遅延 0.02 s は
# ちょうど 1 ステップなので、整数ステップの遅延だと 0 か 1 の 2 値にしかならない。
# hist[i0] と hist[i0+1] を補間することで [0, max_delay_s] の連続値を表現する。
#
#   lag [steps] = delay_s / dt,  i0 = floor(lag),  w = lag - i0
#   out = (1 - w) * hist[i0] + w * hist[i0 + 1]
#
# 遅延量は **env ごと・エピソードごと** に一様サンプリングする (エピソード内では
# 一定)。実機のレイテンシは機体・起動ごとにほぼ一定で、ステップ単位で揺れるもの
# ではないため。ジッタまでは模擬していない。
#
# ``group`` が同じ項は **同じ遅延量を共有する**。projected_gravity と base_ang_vel は
# どちらも同じ IMU から来るので独立に遅れることはなく、joint_pos と joint_vel も
# 同じエンコーダ読み出しから来る。独立に引くと物理的にあり得ない組み合わせ
# (重力は最新・角速度だけ 1 ステップ古い) を学習させることになる。
#
# NOTE: policy 観測にだけ掛けること。critic は特権情報なので遅延させない
#       (遅延した観測から価値を推定させる理由が無く、学習が難しくなるだけ)。
# NOTE: ObservationManager のノイズはこの関数の **後** に乗る。実機の
#       「遅れて届いた値にセンサノイズが乗る」順序と一致する。
# --------------------------------------------------------------------------- #
_OBS_DELAY_STATE_ATTR = "_obs_delay_state"


def _delayed_signal(
    env: ManagerBasedRLEnv,
    key: str,
    group: str,
    value: torch.Tensor,
    max_delay_s: float,
    base_delay_s: float = 0.0,
) -> torch.Tensor:
    """``value`` を ``base_delay_s + [0, max_delay_s]`` だけ遅延させて返す。

    Args:
        key: 項ごとの履歴バッファを引くキー (項ごとに一意にすること)。
        group: 遅延量を共有するセンサ名 ("imu" / "encoder" / "vision" など)。
            同じ group の項は同じ乱数を引く。
        value: 今ステップの生の観測 (num_envs, dim)。
        max_delay_s: ランダム成分の上限 [s]。
        base_delay_s: 全 env 共通の固定遅延 [s]。同じセンサから来るのに片方の項だけ
            設計上すでに遅れている場合 (:func:`prev_ball_pos_b` の 1 ステップ)、
            遅れていない方にこれを与えて実効遅延を揃える。
    """
    if max_delay_s <= 0.0 and base_delay_s <= 0.0:
        return value

    dt = env.step_dt
    max_lag = max_delay_s / dt
    base_lag = base_delay_s / dt
    # 補間には hist[i0] と hist[i0+1] が要るので、最大遅延ぶん + 1 フレーム持つ。
    n_frames = int(math.ceil(base_lag + max_lag)) + 1
    step = int(env.common_step_counter)
    num_envs = value.shape[0]

    root = getattr(env, _OBS_DELAY_STATE_ATTR, None)
    if root is None:
        root = {"groups": {}, "terms": {}}
        setattr(env, _OBS_DELAY_STATE_ATTR, root)

    # prev_ball_pos_b と同じ判定: episode_length_buf は step() 内で加算された後に
    # _reset_idx で 0 に戻るので、「今このステップでリセットされた env」だけが 0 になる。
    just_reset = env.episode_length_buf == 0

    # -- 1. グループ単位の遅延量。エピソード開始時に引き直す。
    gate = root["groups"].get(group)
    if gate is None or gate["lag"].shape[0] != num_envs:
        gate = {"lag": torch.rand(num_envs, device=value.device) * max_lag, "step": step}
        root["groups"][group] = gate
    elif gate["step"] != step:
        # 同じグループの 2 項目以降が同じステップで引き直さないよう step で守る。
        gate["step"] = step
        n_reset = int(just_reset.sum())
        if n_reset > 0:
            gate["lag"][just_reset] = torch.rand(n_reset, device=value.device) * max_lag
    lag = gate["lag"]

    # -- 2. 項ごとの履歴。hist[0] が現在フレーム、hist[k] が k ステップ前。
    hist_state = root["terms"].get(key)
    if hist_state is None or hist_state["hist"].shape[1:] != value.shape:
        hist_state = {"hist": value.unsqueeze(0).repeat(n_frames, 1, 1), "step": step}
        root["terms"][key] = hist_state
    elif hist_state["step"] != step:
        hist_state["hist"] = torch.roll(hist_state["hist"], shifts=1, dims=0)
        hist_state["hist"][0] = value
        hist_state["step"] = step
    else:
        # 同一ステップ内で 2 回呼ばれても履歴をずらさない (先頭を上書きするだけ)。
        hist_state["hist"][0] = value
    hist = hist_state["hist"]

    # リセット直後は前エピソードの値を引きずらせない (全フレームを現在値で埋める)。
    if bool(just_reset.any()):
        hist[:, just_reset] = value[just_reset].unsqueeze(0)

    # -- 3. 線形補間 (固定遅延ぶんを足してから)
    total_lag = lag + base_lag
    i0 = torch.floor(total_lag).long().clamp_(min=0, max=n_frames - 2)
    weight = (total_lag - i0.to(total_lag.dtype)).unsqueeze(-1)
    env_idx = torch.arange(num_envs, device=value.device)
    return (1.0 - weight) * hist[i0, env_idx] + weight * hist[i0 + 1, env_idx]


def delayed_projected_gravity(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "imu",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IMU 由来の重力方向を遅延させたもの。"""
    value = base_mdp.projected_gravity(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "projected_gravity", group, value, max_delay_s)


def delayed_base_ang_vel(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "imu",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IMU 由来の base 角速度を遅延させたもの。"""
    value = base_mdp.base_ang_vel(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "base_ang_vel", group, value, max_delay_s)


def delayed_joint_pos_rel(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "encoder",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """エンコーダ由来の関節角 (デフォルト姿勢からの相対) を遅延させたもの。"""
    value = base_mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "joint_pos_rel", group, value, max_delay_s)


def delayed_joint_vel_rel(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "encoder",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """エンコーダ由来の関節速度 (デフォルトからの相対) を遅延させたもの。"""
    value = base_mdp.joint_vel_rel(env, asset_cfg=asset_cfg)
    return _delayed_signal(env, "joint_vel_rel", group, value, max_delay_s)


def delayed_ball_vel_b(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "vision",
    base_delay_s: float = 0.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """視覚由来のボール速度を遅延させたもの。

    ``base_delay_s`` は :func:`prev_ball_pos_b` が設計上持っている固定 1 ステップと
    実効遅延を揃えるためのもの。同じカメラフレームから出る 2 つの量なので、
    レイテンシが 1 ステップずれているのは実機ではあり得ない。
    """
    value = ball_vel_b(env, ball_cfg=ball_cfg)
    return _delayed_signal(env, "ball_vel_b", group, value, max_delay_s, base_delay_s)


def delayed_prev_ball_pos_b(
    env: ManagerBasedRLEnv,
    max_delay_s: float,
    group: str = "vision",
    base_delay_s: float = 0.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """視覚由来のボール位置を遅延させたもの。

    NOTE: :func:`prev_ball_pos_b` は **元から 1 ステップ (0.02 s) 遅らせた値** を返す
          (知覚遅延を模す設計。関数名の "prev" がそれ)。ここでの遅延はその上に載るので、
          この項の実効遅延は ``0.02 + [0, max_delay_s]`` 秒。``base_delay_s`` は
          ここでは 0 のまま使うこと (二重に足すことになる)。
    """
    value = prev_ball_pos_b(env, ball_cfg=ball_cfg)
    return _delayed_signal(env, "prev_ball_pos_b", group, value, max_delay_s, base_delay_s)
