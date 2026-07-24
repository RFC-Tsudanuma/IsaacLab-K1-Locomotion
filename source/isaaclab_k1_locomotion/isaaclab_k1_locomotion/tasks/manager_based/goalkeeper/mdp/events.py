# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用のイベント関数。

ボールのスポーン・初速・狙い先などの可変パラメータは、EventTerm の params ではなく
env cfg の ``goalkeeper`` フィールド (GoalkeeperParamsCfg) から実行時に読む。
これにより ``--override_json`` の ``{"env": {"goalkeeper.ball_speed_max": 1.5}}``
のようなドットパス上書きで、ステージ遷移条件・初速レンジを設定ファイルから制御できる。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from ...around_ball.mdp.observations import _high_action_cmd
from .observations import gk_buffers, robot_pos_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _gk_params(env: "ManagerBasedEnv"):
    """env cfg に取り付けられた GoalkeeperParamsCfg を返す。"""
    return env.cfg.goalkeeper


def reset_gk_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor):
    """リセットされた env の goalkeeper 状態バッファを初期化する。"""
    bufs = gk_buffers(env)
    bufs["target_y"][env_ids] = 0.0
    bufs["ball_active"][env_ids] = False
    bufs["touched"][env_ids] = False
    bufs["touch_rewarded"][env_ids] = False
    bufs["save_cd"][env_ids] = -1
    bufs["hold_ctr"][env_ids] = 0


def _sample_stage1_targets(env: "ManagerBasedEnv", robot_y: torch.Tensor) -> torch.Tensor:
    """ステージ1 の目標 y をサンプルする (robot_y と同形状)。

    速度学習の圧を保つための分布制御 (パラメータは GoalkeeperParamsCfg):

    * 確率 ``stage1_far_prob``: **反対側のポスト際ゾーン** (|y| ∈ stage1_far_zone) から
      採る。ロボットがポスト際にいるときは逆ポストまでの最大距離 (~2.5m) の
      スプリントになる — 「ゴール幅全域をなるべく速く横移動」のワーストケースを
      確実に学習分布へ入れるための枝。
    * それ以外: ゴール幅内の一様ランダム。ただし現在位置 ±``stage1_min_move`` の
      近距離帯を **除外した区間から直接サンプル** する (棄却ループなし)。
      近すぎる目標は移動なしで達成できてしまい、速度学習に寄与しないため。
    """
    p = _gk_params(env)
    n = robot_y.shape[0]
    device = env.device
    r = float(p.stage1_target_range)
    m = float(p.stage1_min_move)
    y = robot_y.clamp(-r, r)

    # --- 一様ランダム枝: [-r, r] から近距離帯 (y ± m) を除いた区間を直接サンプル ---
    # 除外窓とゴール幅の共通部分 [a, b] を除いた左区間 [-r, a] / 右区間 [b, r] の
    # 長さ比で振り分ける。m < r なので合計長は必ず正 (最低 2r - 2m)。
    a = (y - m).clamp(min=-r)
    b = (y + m).clamp(max=r)
    left_len = (a + r).clamp(min=0.0)
    right_len = (r - b).clamp(min=0.0)
    total = (left_len + right_len).clamp(min=1e-6)
    u = torch.rand(n, device=device) * total
    uniform_far = torch.where(u < left_len, -r + u, b + (u - left_len))

    # --- ポスト際ゾーン枝: 現在位置の反対側の |y| ∈ far_zone ---
    z_lo, z_hi = float(p.stage1_far_zone[0]), float(p.stage1_far_zone[1])
    side = -torch.sign(y)
    rand_side = torch.sign(torch.rand(n, device=device) - 0.5)
    side = torch.where(y.abs() < 0.1, rand_side, side)          # 中央付近なら左右ランダム
    side = torch.where(side == 0, torch.ones_like(side), side)  # sign(0)=0 の保険
    post_zone = side * (z_lo + torch.rand(n, device=device) * (z_hi - z_lo))

    use_far = torch.rand(n, device=device) < float(p.stage1_far_prob)
    return torch.where(use_far, post_zone, uniform_far)


def reset_stage1_target_and_park(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """ステージ1: ボールを遠方にパーク (非アクティブ) し、ランダム目標 y を採番する。

    ボール観測は ``_gk_ball_active`` = False によりダミー 0 になるが、物理体としては
    フィールド奥 (park_pos) に静止させておき、観測次元・シーン構成をステージ2 以降と
    完全に一致させる。★ reset_gk_buffers より後に登録すること (目標を上書きするため)。
    目標のサンプリング分布は :func:`_sample_stage1_targets` 参照。
    """
    p = _gk_params(env)
    ball = env.scene[ball_cfg.name]
    bufs = gk_buffers(env)
    n = len(env_ids)

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = env.scene.env_origins[env_ids, 0] + float(p.park_pos[0])
    pose[:, 1] = env.scene.env_origins[env_ids, 1] + float(p.park_pos[1])
    pose[:, 2] = float(p.ball_radius)
    pose[:, 3] = 1.0  # 単位クォータニオン (w, x, y, z)
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)

    bufs["ball_active"][env_ids] = False
    bufs["target_y"][env_ids] = _sample_stage1_targets(env, robot_pos_goal(env)[env_ids, 1])
    bufs["hold_ctr"][env_ids] = 0


def stage1_target_tick(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None = None,
):
    """ステージ1: 目標に到達して静止を保てた env の目標 y を再サンプルする毎ステップイベント。

    ``EventTerm(mode="interval", interval_range_s=(dt, dt), is_global_time=True)`` で
    毎制御ステップ呼ぶ。到達判定は

        * |robot_y − target_y| < reach_tol
        * 上位コマンドノルム < stage1_cmd_tol (足踏み対策: around_ball_ready の教訓。
          frozen が「初期姿勢で立つ」のはコマンドノルム < 0.05 のときだけなので、
          実ベース速度だけで判定するとコマンドを残した足踏みで達成できてしまう)
        * ベース並進速度 < stage1_speed_tol

    を ``stage1_hold_steps`` 連続で満たすこと。達成した env は新しい目標を採番し、
    カウンタを 0 に戻す (エピソードは切らない)。
    """
    p = _gk_params(env)
    bufs = gk_buffers(env)
    robot = env.scene["robot"]

    robot_y = robot_pos_goal(env)[:, 1]
    y_err = (robot_y - bufs["target_y"]).abs()
    cmd_norm = torch.norm(_high_action_cmd(env)[:, :3], dim=1)
    lin_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    reached = (
        (y_err < float(p.stage1_reach_tol))
        & (cmd_norm < float(p.stage1_cmd_tol))
        & (lin_speed < float(p.stage1_speed_tol))
    )
    # リセット直後の過渡では判定しない
    reached = reached & (env.episode_length_buf >= 2)

    ctr = bufs["hold_ctr"]
    ctr[reached] += 1
    ctr[~reached] = 0

    done = ctr >= int(p.stage1_hold_steps)
    if int(done.sum()) > 0:
        bufs["target_y"][done] = _sample_stage1_targets(env, robot_y[done])
        ctr[done] = 0


def reset_ball_shot(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """ステージ2/3: ボールをフィールド側にスポーンし、ゴール内の狙い先へ初速を与える。

    ランダム化 (すべて GoalkeeperParamsCfg から):
        * スポーン位置: ゴール中央からの距離 d ∈ spawn_dist_range、
          方位 θ ∈ ±spawn_half_angle (+x 正面基準)。ロボットに近すぎる位置
          (< 0.6m) はロボットから見て radial に押し出して重なりを防ぐ。
        * 狙い先: ゴールライン上の y_aim ∈ ±aim_y_range (ポスト内側)
        * 初速: v ∈ [ball_speed_min, hi]。hi は通常 ball_speed_max だが、
          ステージ3 の適応カリキュラム (adaptive_ball_speed) が ``_gk_speed_hi``
          バッファを持っている場合はそちらを使う (成功率に応じて連続的に引き上げ)。
        * 実現可能性クランプ: ゴールライン到達までの時間が min_time_to_line [s] を
          下回らないよう初速に上限を掛ける (近距離×高速の「原理的にセーブ不可能な
          球」を作らない。距離と速度に自然な相関がつく: 近い球は遅く、遠い球は速い)。

    ボールには転がり整合の角速度 ω = (-vy, vx, 0)/r も与える (滑り減速の過渡を消す)。
    ★ reset_gk_buffers より後、reset_base (ロボット配置) より後に登録すること。
    """
    p = _gk_params(env)
    ball = env.scene[ball_cfg.name]
    bufs = gk_buffers(env)
    n = len(env_ids)
    r = float(p.ball_radius)

    dist = torch.empty(n, device=env.device).uniform_(*[float(v) for v in p.spawn_dist_range])
    ang = torch.empty(n, device=env.device).uniform_(-float(p.spawn_half_angle), float(p.spawn_half_angle))
    spawn_x = dist * torch.cos(ang)
    spawn_y = dist * torch.sin(ang)

    # ロボットとの重なり防止: 近すぎるスポーンはロボットから radial に 0.6m まで押し出す
    robot_xy = robot_pos_goal(env)[env_ids, :2]
    d_rel_x = spawn_x - robot_xy[:, 0]
    d_rel_y = spawn_y - robot_xy[:, 1]
    d_rel = torch.sqrt(d_rel_x**2 + d_rel_y**2).clamp(min=1e-6)
    too_close = d_rel < 0.6
    scale = torch.where(too_close, 0.6 / d_rel, torch.ones_like(d_rel))
    spawn_x = robot_xy[:, 0] + d_rel_x * scale
    spawn_y = robot_xy[:, 1] + d_rel_y * scale

    aim_y = torch.empty(n, device=env.device).uniform_(-float(p.aim_y_range), float(p.aim_y_range))

    hi_buf = getattr(env, "_gk_speed_hi", None)
    hi = float(hi_buf.item()) if hi_buf is not None else float(p.ball_speed_max)
    speed = torch.empty(n, device=env.device).uniform_(float(p.ball_speed_min), hi)

    # スポーン点 → 狙い先 (x=0, y=aim_y) の方向単位ベクトル
    dir_x = -spawn_x
    dir_y = aim_y - spawn_y
    norm = torch.sqrt(dir_x**2 + dir_y**2).clamp(min=1e-6)
    # 実現可能性クランプ: 到達時間 = 距離/速度 ≥ min_time_to_line
    v_feasible = norm / max(float(p.min_time_to_line), 1e-3)
    speed = torch.minimum(speed, v_feasible)
    vx = speed * dir_x / norm
    vy = speed * dir_y / norm

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = env.scene.env_origins[env_ids, 0] + spawn_x
    pose[:, 1] = env.scene.env_origins[env_ids, 1] + spawn_y
    pose[:, 2] = r
    pose[:, 3] = 1.0
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)

    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 0] = vx
    vel[:, 1] = vy
    # 転がり整合角速度: v_contact = v + ω × (-r ẑ) = 0 ⇔ ω = (-vy, vx, 0)/r
    vel[:, 3] = -vy / r
    vel[:, 4] = vx / r
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)

    bufs["ball_active"][env_ids] = True


def reset_ball_perception(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
):
    """知覚DR (観測側の遅延・更新レート・ノイズ) の per-episode パラメータを採番し、
    履歴・認識出力バッファを現在の真値で初期化する (起動時トランジェント回避)。

    ★ reset_ball (ボール配置イベント) より後に登録すること (配置後の真値で初期化)。
    パラメータは GoalkeeperParamsCfg の perc_*。PLAY 環境ではこれらを 0 / (1,1) に
    上書きすればクリーン観測になる。
    """
    from .observations import _gk_perc_buffers, _gk_true_rel_state

    p = _gk_params(env)
    _gk_perc_buffers(env)
    n = len(env_ids)

    lo, hi = int(p.perc_latency_range[0]), int(p.perc_latency_range[1])
    env._gkp_latency[env_ids] = torch.randint(lo, hi + 1, (n,), device=env.device)
    # ビジョンの更新レート [Hz] → 制御 tick 単位の周期 (小数)。
    # 例: 制御 50Hz・ビジョン 30Hz なら 50/30 = 1.67 tick。
    control_hz = 1.0 / float(env.step_dt)
    lo2, hi2 = float(p.perc_update_rate_hz[0]), float(p.perc_update_rate_hz[1])
    rate = torch.empty(n, device=env.device).uniform_(lo2, hi2).clamp(min=1e-3)
    env._gkp_period[env_ids] = (control_hz / rate).clamp(min=1.0)
    env._gkp_ctr[env_ids] = 0.0  # 次 tick を更新 tick にする
    env._gkp_bias[env_ids] = torch.randn(n, 2, device=env.device) * float(p.perc_bias_sigma)
    # 速度バイアス: x, y 各軸独立。大きさを perc_vel_bias_range から一様サンプルし、
    # 符号はランダム。エピソード中は固定 (遅延由来の系統誤差)。
    vlo, vhi = float(p.perc_vel_bias_range[0]), float(p.perc_vel_bias_range[1])
    vmag = torch.empty(n, 2, device=env.device).uniform_(vlo, vhi)
    vsign = torch.where(
        torch.rand(n, 2, device=env.device) < 0.5,
        torch.ones(n, 2, device=env.device),
        -torch.ones(n, 2, device=env.device),
    )
    env._gkp_vel_bias[env_ids] = vmag * vsign

    pos_b, vel_b = _gk_true_rel_state(env)
    env._gkp_hist_pos[env_ids] = pos_b[env_ids].unsqueeze(1)  # (n, H, 2) にブロードキャスト
    env._gkp_hist_vel[env_ids] = vel_b[env_ids].unsqueeze(1)
    env._gkp_out_pos[env_ids] = pos_b[env_ids]
    env._gkp_out_vel[env_ids] = vel_b[env_ids]
