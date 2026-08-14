# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用のイベント関数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# 「蹴ったらエピソード終了」の蹴り検出用 per-env バッファ名
_BALL_SPAWN_POS_ATTR = "_ball_spawn_pos_w"          # 直近スポーン位置 (N, 2)
_BALL_KICK_CD_ATTR = "_ball_kick_countdown"          # 蹴り後の終了カウントダウン (N,), -1=非発火


def _place_ball_in_front_cone(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    dist_range: tuple[float, float],
    half_angle: float,
    ball_radius: float,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """ロボットの現在の向き基準の正面扇形にボールを静止配置し、新しい xy 位置を返す。"""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    n = len(env_ids)

    dist = torch.empty(n, device=env.device).uniform_(float(dist_range[0]), float(dist_range[1]))
    ang = torch.empty(n, device=env.device).uniform_(-float(half_angle), float(half_angle))
    theta = robot.data.heading_w[env_ids] + ang

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = robot.data.root_pos_w[env_ids, 0] + dist * torch.cos(theta)
    pose[:, 1] = robot.data.root_pos_w[env_ids, 1] + dist * torch.sin(theta)
    pose[:, 2] = ball_radius  # 平面地形 (z=0) 前提
    pose[:, 3:7] = ball.data.default_root_state[env_ids, 3:7]  # 単位クォータニオン

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    return pose[:, :2]


def _ball_tracking_buffers(env: "ManagerBasedEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """スポーン位置・蹴り後カウントダウンのバッファを (遅延生成して) 返す。"""
    spawn = getattr(env, _BALL_SPAWN_POS_ATTR, None)
    if spawn is None or spawn.shape != (env.num_envs, 2):
        spawn = torch.zeros(env.num_envs, 2, device=env.device)
        setattr(env, _BALL_SPAWN_POS_ATTR, spawn)
    cd = getattr(env, _BALL_KICK_CD_ATTR, None)
    if cd is None or cd.shape != (env.num_envs,):
        cd = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        setattr(env, _BALL_KICK_CD_ATTR, cd)
    return spawn, cd


def reset_base_forward_velocity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    speed_range: tuple[float, float] = (0.0, 0.6),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """リセット時にロボットへ「体の正面方向（≒ボール方向）への前進初速」を与える。

    実運用では AroundBall は Approach (ボールへ前進中) の後に起動するので、回り込みは
    「動きながら開始」される。学習が毎回ほぼ静止から始まると、この引き渡し遷移が
    分布外になり、切り替え直後にコマンドが跳ねて不安定になる。本イベントで前進初速を
    与えて「動きながら回り込み開始」を学習分布に含める。

    ボールは reset_ball_in_front_cone がロボット正面±57°に置くので、体の正面方向への
    初速はおおよそボール方向への前進になる。速度はワールド座標の xy に heading 方向で
    設定し、z・角速度は据え置く。

    Note:
        脚は静止ポーズのままなので「歩行中」ではなく「静止ポーズで前進」する近似。
        速すぎると開始時につまずくため、frozen が押し外乱で慣れている範囲に収める。
        ★ reset_base (ロボット姿勢リセット) より後に実行されること。
    """
    robot = env.scene[robot_cfg.name]
    n = len(env_ids)
    speed = torch.empty(n, device=env.device).uniform_(float(speed_range[0]), float(speed_range[1]))
    heading = robot.data.heading_w[env_ids]  # world yaw
    lin = robot.data.root_lin_vel_w[env_ids].clone()  # (n, 3)
    ang = robot.data.root_ang_vel_w[env_ids].clone()  # (n, 3)
    lin[:, 0] = speed * torch.cos(heading)
    lin[:, 1] = speed * torch.sin(heading)
    robot.write_root_velocity_to_sim(torch.cat([lin, ang], dim=1), env_ids=env_ids)


def reset_ball_in_front_cone(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    dist_range: tuple[float, float] = (0.6, 2.0),
    half_angle: float = 1.0,
    ball_radius: float = 0.11,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """リセット後のロボットの正面扇形内にボールを静止状態で配置する。

    walk_kick の ``reset_ball_in_front_of_robot`` と同方式: ロボットの現在位置・向き
    (heading) を基準に、距離 d ∈ ``dist_range``・方位 φ ∈ ±``half_angle`` [rad] を
    一様サンプリングして配置する。ロボット基準なので、ロボットの初期 yaw を全周
    ランダム化しても「ボールは必ず視野内 (half_angle ≤ FOV)」が保証される。

    Note: reset_base (ロボットの姿勢リセット) より後に実行されること
    (EventManager は cfg の定義順に reset イベントを適用する)。
    """
    new_pos = _place_ball_in_front_cone(
        env, env_ids, dist_range, half_angle, ball_radius, robot_cfg, ball_cfg
    )
    # 蹴り検出用のスポーン位置を記録し、蹴り後カウントダウンを解除。
    # 通常リセット (タイムアウト/転倒) でも走るので、中断された蹴り検出も
    # ここで確実に初期化される (別要因のリセットで幽霊カウントダウンが残らない)。
    spawn, cd = _ball_tracking_buffers(env)
    spawn[env_ids] = new_pos
    cd[env_ids] = -1


def reset_ball_last_seen(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
):
    """リセットされた env の ``_ball_last_seen_pos_b`` バッファ (hold-last-seen 観測) を 0 にする。

    バッファ実体は観測関数 :func:`mdp.observations.ball_pos_rel_fov` が遅延生成するので、
    まだ無ければ no-op。リセットイベントは観測計算より先に走るため、ここで 0 化しても
    ボールが視野内にスポーンする限り新エピソード最初の観測で即座に真値へ上書きされる。
    """
    buf = getattr(env, "_ball_last_seen_pos_b", None)
    if buf is None:
        return
    if env_ids is None:
        buf.zero_()
        return
    buf[env_ids] = 0.0


def reset_ball_perception(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    max_latency: int = 8,
    # 既定値はビジョン約30Hz上限 @ 制御50Hz (step_dt=0.02s) 前提。
    # 30Hz = 1.67 tick なので period=1 (=50Hz認識) は実機に存在しない。
    # レイテンシは実測なし (ビジョン担当も不明) → 広め 40〜160ms をDR。
    latency_range: tuple[int, int] = (2, 8),
    update_period_range: tuple[int, int] = (2, 4),
    bias_sigma: float = 0.03,
    yaw_err_sigma: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """知覚DR (:func:`mdp.observations.ball_pos_rel_perceived` / ``kick_direction_b_perceived``)
    の per-episode パラメータを採番し、状態バッファを初期化する。

    - レイテンシ・更新周期: エピソード中は固定。範囲から一様サンプリング。
    - 系統バイアス (ボール位置): ``bias_sigma`` の等方ガウス、エピソード固定。
    - ヨー推定誤差 (キック方向): ``yaw_err_sigma`` [rad] のガウス、エピソード固定。
    - 履歴・認識出力バッファ: 現在の真のボール相対位置で初期化 (起動時トランジェント回避)。
      ※ ``reset_ball`` (ボール配置) より **後** に登録すること。

    観測関数のバッファ実体は遅延生成なので、まだ無ければここで生成する。
    """
    from .observations import _ensure_perc_buffers, ball_offset_and_bearing

    _ensure_perc_buffers(env, max_latency)
    n_ids = len(env_ids)

    lo, hi = int(latency_range[0]), int(latency_range[1])
    env._ball_perc_latency[env_ids] = torch.randint(lo, hi + 1, (n_ids,), device=env.device)
    # 更新間隔レンジ [lo, hi] を per-episode で確定。実際の間隔は観測側 (ball_pos_rel_perceived)
    # が更新のたびにこのレンジ内で引き直すので、エピソード内でフレーム間隔がジッタする
    # (= ビジョンが常に 30Hz とは限らない、負荷で遅くなる状況を再現)。
    lo2, hi2 = int(update_period_range[0]), int(update_period_range[1])
    env._ball_perc_update_period_lo[env_ids] = lo2
    env._ball_perc_update_period_hi[env_ids] = hi2
    # 最初の更新までの間隔もレンジからサンプル (以降は観測側で毎回引き直す)。
    env._ball_perc_update_period[env_ids] = torch.randint(lo2, hi2 + 1, (n_ids,), device=env.device)
    env._ball_perc_update_ctr[env_ids] = 0  # 次tickを更新tickにする
    env._ball_perc_bias[env_ids] = torch.randn(n_ids, 2, device=env.device) * bias_sigma

    yaw_err = getattr(env, "_kick_yaw_err", None)
    if yaw_err is None or yaw_err.shape != (env.num_envs,):
        yaw_err = torch.zeros(env.num_envs, device=env.device)
        env._kick_yaw_err = yaw_err
    env._kick_yaw_err[env_ids] = torch.randn(n_ids, device=env.device) * yaw_err_sigma

    # 履歴・認識出力を現在の真値で埋める (レイテンシ分の起動トランジェントを消す)
    offset_b, _ = ball_offset_and_bearing(env, robot_cfg)
    env._ball_perc_hist[env_ids] = offset_b[env_ids].unsqueeze(1)  # (n_ids, H+1, 2) にブロードキャスト
    env._ball_perceived[env_ids] = offset_b[env_ids]
    # ボールは視野内にスポーンするので「今まさに検出済み」とみなし ball_in_fov を
    # エピソード開始時に 1 (フレッシュ) にする。
    env._ball_perc_last_update_step[env_ids] = int(env.common_step_counter)
