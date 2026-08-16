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
from isaaclab.utils.math import wrap_to_pi

from ...around_ball.mdp.observations import _high_action_cmd
from .observations import ball_pos_goal, gk_buffers, robot_pos_goal

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
    bufs["respawn_cd"][env_ids] = -1
    bufs["save_count"][env_ids] = 0
    bufs["save_quality"][env_ids] = 0.0
    bufs["hard_ball"][env_ids] = False
    bufs["unreachable"][env_ids] = False

    # ★ 2026-08-15: 歩行位相のアキュムレータと指令ローパスの状態をクリアする。
    #   どちらも前エピソードの値を持ち越すと、開始直後に「途中の位相」や
    #   「前の球への指令」が出てしまう。
    # ★ 2026-08-16: 歩行ゲートの sim2real 用の状態も同様にクリアする。
    #   ノイズは前エピソードの相関を持ち越さないよう 0 から、ゲートは停止状態から、
    #   遅延バッファは 0 から始める (位相が 0 から始まる規約に合わせる)。
    #   env ごとの個体差 (_BASE_VEL_NOISE_SCALE_ATTR = ノイズ倍率、
    #   _BASE_VEL_DELAY_ATTR = 遅延段数) だけは **リセットしない** (startup DR と同じ扱い)。
    from .observations import (
        _BASE_VEL_HIST_ATTR,
        _BASE_VEL_NOISE_ATTR,
        _DRIVE_FILT_ATTR,
        _TASK_PHASE_ATTR,
        _WALK_GATE_ATTR,
    )

    phase = getattr(env, _TASK_PHASE_ATTR, None)
    if phase is not None:
        phase[env_ids] = 0.0
    filt = getattr(env, _DRIVE_FILT_ATTR, None)
    if filt is not None:
        filt[env_ids] = 0.0
    vnoise = getattr(env, _BASE_VEL_NOISE_ATTR, None)
    if vnoise is not None:
        vnoise[env_ids] = 0.0
    vhist = getattr(env, _BASE_VEL_HIST_ATTR, None)
    if vhist is not None:
        vhist[:, env_ids] = 0.0   # (depth, N, 2) なので env 軸は 1 番目
    gate = getattr(env, _WALK_GATE_ATTR, None)
    if gate is not None:
        gate[env_ids] = False


def _sample_stage1_targets(env: "ManagerBasedEnv", robot_y: torch.Tensor) -> torch.Tensor:
    """ステージ1 の目標 y をサンプルする (robot_y と同形状)。

    速度学習の圧を保つための分布制御 (パラメータは GoalkeeperParamsCfg):

    * 確率 ``stage1_far_prob``: **反対側のポスト際ゾーン** (|y| ∈ stage1_far_zone) から
      採る。ロボットがポスト際にいるときは逆ポストまでの最大距離 (~2.6m) の
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
          適応カリキュラム (adaptive_difficulty) が ``_gk_speed_hi``
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

    # --- 初速を先に引き、距離をその初速から決める (2026-08-14 に順序を反転) ---
    #
    # 旧方式は「距離を一様サンプル → 初速を min_time_to_line でクランプ」だった。
    # これは「近い = 必ず遅い」を強制するため、
    #   * 速い球が近距離から来ることが原理的に無い (実機では普通に起きる)
    #   * 逆に遅い球が最遠 (6.5m) から来る。ボール位置ノイズは σ(d)=0.124d+0.149 で
    #     6.5m では 0.96m = ゴール幅の 3/4 に達し、到達点予測が無意味になる。
    #     知覚クリーン実験で、この遠距離ノイズが学習速度を約 10 倍遅くしていることを確認
    #     (199 iter 時点でセーブ率 55.2% → クリーン 70.9%)。
    # の 2 つの問題があった。
    #
    # 新方式は「初速 v を引き、距離を [v·t_near, max(d_floor, v·t_far)] から引く」:
    #   * t_far  (spawn_time_far)  = 距離の上限。v=3 → 4.2m (ユーザー指定)、v=6 → 8.4m
    #   * t_near (spawn_time_near) = 距離の下限 = 「反応が成立する最短距離」。
    #     v=6 なら 3.3m で、守備面まで 0.40s・知覚レイテンシを引いて反応 0.24s。
    #     その場から 6cm 移動 + タッチ判定 0.5m = 0.56m 幅をカバーできる水準。
    #     これ以上詰める (t_near=0.4) と反応時間がほぼゼロになり、立っていた場所に
    #     偶然来たときだけ止まる運任せの球になって学習信号がノイズ化する。
    #   * d_floor は遅い球が至近距離に湧くのを防ぐ下限 (v=1 で 1.4m は近すぎるため)。
    # 「それより厳しい球」は従来どおり hard_ball 側で少量だけ混ぜる (5.3 の二層構造)。
    hi_buf = getattr(env, "_gk_speed_hi", None)
    hi = float(hi_buf.item()) if hi_buf is not None else float(p.ball_speed_max)
    speed = torch.empty(n, device=env.device).uniform_(float(p.ball_speed_min), hi)

    hard_prob = float(getattr(p, "hard_ball_prob", 0.0))
    hard = torch.rand(n, device=env.device) < hard_prob
    if hard_prob > 0.0:
        mult = float(getattr(p, "hard_ball_speed_mult", 1.6))
        hard_speed = torch.empty(n, device=env.device).uniform_(hi, max(hi * mult, hi + 1e-3))
        speed = torch.where(hard, hard_speed, speed)

    t_near = float(getattr(p, "spawn_time_near", 0.55))
    t_far = float(getattr(p, "spawn_time_far", 1.4))
    d_floor = float(getattr(p, "spawn_dist_floor", 2.0))
    # 到達不能球は下限をさらに詰める (= 反応が間に合わない距離から撃つ)
    if hard_prob > 0.0:
        t_near_hard = float(getattr(p, "hard_ball_time_near", 0.35))
        t_near_t = torch.where(
            hard,
            torch.full((n,), t_near_hard, device=env.device),
            torch.full((n,), t_near, device=env.device),
        )
    else:
        t_near_t = torch.full((n,), t_near, device=env.device)

    # ★ 下限にも床を入れる。v×t_near だけだと遅い球 (0.5 m/s) が 0.28m = ロボットの
    #   足元に湧く。reset_ball_shot は後段で「ロボットから 0.6m 未満なら radial に
    #   押し出す」補正をするが、その補正はボールの狙い方向を変えてしまうので、
    #   最初から湧かせない方がよい。
    ang = torch.empty(n, device=env.device).uniform_(-float(p.spawn_half_angle), float(p.spawn_half_angle))

    d_near_floor = float(getattr(p, "spawn_dist_near_floor", 1.0))
    d_lo = torch.clamp(speed * t_near_t, min=d_near_floor)

    # ★ スポーン点が守備面より前になることを保証する (2026-08-14)。
    #   距離だけで下限を決めると、広角の球が **キーパーの背後** に湧く。
    #   sx = d·cos(ang) なので、ang=±1.1rad(63°) では sx = 0.45d。d=1.5m でも
    #   sx = 0.68m となり guard_x=0.8 より内側で、猶予ゼロの球になる。
    #   実測: この制約が無いと最易段でも 40% が到達不能判定、入れると 10.4% に落ちる。
    #   spawn_ahead_min は「守備面のどれだけ前に湧かせるか」の最小値 [m]。
    ahead = float(getattr(p, "spawn_ahead_min", 0.7))
    guard_x = float(p.guard_x)
    d_geom = (guard_x + ahead) / torch.cos(ang).clamp(min=0.1)
    d_lo = torch.maximum(d_lo, d_geom)

    d_hi = torch.clamp(speed * t_far, min=d_floor)
    d_hi = torch.maximum(d_hi, d_lo + 1e-3)          # 下限が上限を超える組み合わせの保険
    dist = d_lo + torch.rand(n, device=env.device) * (d_hi - d_lo)
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

    # 狙い先 y の範囲。適応カリキュラム (adaptive_difficulty) が段階的に広げるので、
    aim_buf = getattr(env, "_gk_aim_y", None)
    aim_range = float(aim_buf.item()) if aim_buf is not None else float(p.aim_y_range)
    aim_y = torch.empty(n, device=env.device).uniform_(-aim_range, aim_range)

    # スポーン点 → 狙い先 (x=0, y=aim_y) の方向単位ベクトル
    dir_x = -spawn_x
    dir_y = aim_y - spawn_y
    norm = torch.sqrt(dir_x**2 + dir_y**2).clamp(min=1e-6)
    # ★ 速度クランプ (旧 min_time_to_line) は廃止。距離を速度から決めるようになったので
    #   「近い×高速」は距離サンプリングの下限 (spawn_time_near) で直接制御している。
    #   ここで再度クランプすると、狙い先が横に振れて norm が伸びた分だけ二重に効く。
    bufs["hard_ball"][env_ids] = hard
    vx = speed * dir_x / norm
    vy = speed * dir_y / norm

    # --- 「この球はこのキーパー位置から物理的に取れるか」を発射時に判定する ---
    #
    # ★ 2026-08-14 (ユーザー指示): 取れない球を成功率の集計から外すため。
    #   閾値 (adaptive_success_threshold=0.85) を下げるのではなく **分母から外す**。
    #   「キーパーは可能な限り全部止めるべき」という要求を保ったまま、物理的に不可能な
    #   球でカリキュラムが止まるのを防ぐ。失点ペナルティ (-500) は除外しないので、
    #   ポリシーは取れない球でも諦めずに向かう (集計だけの話)。
    #
    #   これにより「右端にいるときに左端へ速い球」は自動的に除外され、
    #   「右端にいるときに左端へ**遅い**球」は除外されない (時間があるので取れるべき)。
    #   難易度ではなく物理で線を引くので恣意性が無い。
    #
    #   判定は真値で行う (観測ではなく集計用のブックキーピングなので)。
    _mark_unreachable(env, env_ids, spawn_x, spawn_y, vx, vy, speed)

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


def _mark_unreachable(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    speed: torch.Tensor,
) -> None:
    """発射した球が「そのキーパー位置から物理的に取れるか」を判定して記録する。

    :func:`~.curriculums._update_success_ema` が、ここで True になった球での失点を
    成功率の集計から除外する (失点ペナルティは除外しない = ポリシーは諦めない)。

    判定 (すべて真値。集計用のブックキーピングなので観測は使わない):

        守備面 x = guard_x を通過する y   … 等速直線で外挿し ±goal_half_width にクランプ
        猶予時間 t_avail                  … そこまでの飛行時間 − 知覚レイテンシ
        必要移動 Δy                       … |通過 y − キーパーの現在 y| − タッチ判定半径
        必要時間 t_need                   … 下位の実測エンベロープ (定常速度と立ち上がり)
                                            による加速込みの横移動時間

    ``t_need > t_avail`` なら到達不能。パラメータはすべて GoalkeeperParamsCfg にあり、
    下位ポリシーを差し替えたら ``reach_v_max`` / ``reach_t_acc`` を実測値に更新すること。
    """
    p = _gk_params(env)
    dev = env.device

    v_max = float(getattr(p, "reach_v_max", 1.278))     # 下位の定常横速度 [m/s]
    t_acc = float(getattr(p, "reach_t_acc", 0.6))       # 静止→定常の立ち上がり [s]
    lat = float(getattr(p, "reach_latency_s", 0.156))   # 知覚レイテンシ + 更新間隔 [s]
    touch = float(getattr(p, "touch_proximity", 0.5))
    guard_x = float(p.guard_x)
    max_y = float(p.goal_half_width)
    d_acc = 0.5 * v_max * t_acc                          # 加速中に進む距離 [m]

    # 守備面 x=guard_x を通過するまでの時間と、その時点の y
    vx_in = (-vx).clamp(min=1e-3)                        # ゴール方向 (=-x) の速度成分
    t_guard = ((spawn_x - guard_x) / vx_in).clamp(min=0.0)
    y_guard = (spawn_y + vy * t_guard).clamp(-max_y, max_y)
    t_avail = t_guard - lat

    ky = robot_pos_goal(env)[env_ids, 1]
    dy = (y_guard - ky).abs() - touch                    # タッチ判定ぶんは動かなくてよい
    dy = dy.clamp(min=0.0)

    # 加速込みの横移動時間: 短距離は等加速、長距離は加速 + 定常
    t_short = torch.sqrt((2.0 * dy * t_acc / v_max).clamp(min=0.0))
    t_long = t_acc + (dy - d_acc) / v_max
    t_need = torch.where(dy <= d_acc, t_short, t_long)

    unreachable = t_need > t_avail
    bufs = gk_buffers(env)
    bufs["unreachable"][env_ids] = unreachable


def reset_ball_perception(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
):
    """VirtualPerception のリングバッファ/per-env パラメータをリセットし、速度バイアスを
    採番する。

    位置・遅延・ノイズ・検出率・occlusion は VirtualPerception (実機カメラ準拠) が担当。
    速度バイアス (0.5〜1.0 m/s のエピソード固定系統誤差) だけは goalkeeper 側で持つので
    ここで採番する。自己位置推定 (実機は MCL) の誤差もここで採番する。
    ★ reset_ball (ボール配置イベント) より後に登録すること。
    """
    import math

    from .observations import _gk_loc_buffers, _gk_perception

    p = _gk_params(env)
    vp = _gk_perception(env)
    vp.reset(env_ids)
    n = len(env_ids)

    # 速度バイアス: x, y 各軸独立。大きさを perc_vel_bias_range から一様サンプルし、
    if bool(getattr(p, "perception_clean", False)):
        env._gkp_vel_bias[env_ids] = 0.0
    else:
        vlo, vhi = float(p.perc_vel_bias_range[0]), float(p.perc_vel_bias_range[1])
        vmag = torch.empty(n, 2, device=env.device).uniform_(vlo, vhi)
        vsign = torch.where(
            torch.rand(n, 2, device=env.device) < 0.5,
            torch.ones(n, 2, device=env.device),
            -torch.ones(n, 2, device=env.device),
        )
        env._gkp_vel_bias[env_ids] = vmag * vsign
    env._gkp_out_vel[env_ids] = 0.0

    # 自己位置推定の誤差: エピソード固定バイアス + ドリフト速度 + 跳びの頻度。
    _gk_loc_buffers(env)
    env._gk_loc_err[env_ids] = 0.0
    # 跳び由来のボール速度の漏れも持ち越さない (前エピソードの山が残ると開始直後に
    # 存在しない「接近中」信号が出る)。
    env._gk_loc_vel_leak[env_ids] = 0.0
    if bool(getattr(p, "perception_clean", False)):
        env._gk_loc_bias[env_ids] = 0.0
        env._gk_loc_drift[env_ids] = 0.0
        env._gk_loc_jump_p[env_ids] = 0.0
        return

    deg = math.pi / 180.0
    bias = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
    bias[:, :2] *= float(p.loc_bias_xy_m)
    bias[:, 2] *= float(p.loc_bias_yaw_deg) * deg
    env._gk_loc_bias[env_ids] = bias

    drift = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
    drift[:, :2] *= float(p.loc_drift_xy_mps)
    drift[:, 2] *= float(p.loc_drift_yaw_dps) * deg
    env._gk_loc_drift[env_ids] = drift

    jlo, jhi = float(p.loc_jump_hz_range[0]), float(p.loc_jump_hz_range[1])
    env._gk_loc_jump_p[env_ids] = (
        torch.empty(n, device=env.device).uniform_(jlo, jhi) * float(env.step_dt)
    )


def _save_clearance_quality(
    env: "ManagerBasedEnv",
    safe_dist: float = 1.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """セーブ品質 [0, 1]: ボールを「ゴール枠」からどれだけ遠ざけたか (N,)。

    「止めた」と「危険を除去した」を区別するための指標。ゴール正面 0.3m に
    ボールを止めただけでも従来はセーブ成功扱いだったが、実戦ではそのまま
    押し込まれる。ゴール枠 (x=0, |y| <= goal_half_width) の矩形からの
    ユークリッド距離で測り、``safe_dist`` [m] 離れれば満点とする。

        * 前方へ押し出した  → dx が伸びる
        * ポストの外へ弾いた → dy が伸びる (枠外に出た球もここで加点される)

    明示的なキック方向は与えない。「なるべく外へ弾け」という圧力だけを与え、
    体の当て方はポリシーに任せる (キック技術は ball_kick タスクの担当)。
    """
    p = _gk_params(env)
    pos = ball_pos_goal(env, ball_cfg)
    dx = pos[:, 0].clamp(min=0.0)                                       # ライン前方への距離
    dy = (pos[:, 1].abs() - float(p.goal_half_width)).clamp(min=0.0)    # ポスト外へのはみ出し
    dist = torch.sqrt(dx * dx + dy * dy)
    return (dist / max(float(safe_dist), 1e-3)).clamp(0.0, 1.0)


def sync_task_command(
    env: "ManagerBasedEnv",
    # ★ env_ids にデフォルト値を付けないこと (理由は relaunch_ball_after_save 参照)。
    env_ids: torch.Tensor | None,
    command_name: str = "base_velocity",
):
    """``base_velocity`` コマンドをタスク由来の移動要求で上書きする毎ステップイベント。

    ステージ2/3 では観測スロットだけを :func:`~.observations.task_drive_vector` に
    差し替えていたため、コマンドを参照する報酬 (``feet_phase`` / ``foot_clearance``)
    がランダムなコマンドを見て常に「歩け」と要求し、足踏みが止まらなかった。
    コマンド自体を差し替えて観測・報酬・位相を同じ信号で駆動する。

    ★ 併せて cfg 側で ``heading_command`` / ``rel_standing_envs`` /
      ``resampling_time_range`` を無効化すること。
    """
    from .observations import task_drive_vector

    cmd_term = env.command_manager.get_term(command_name)
    buf = getattr(cmd_term, "vel_command_b", None)
    if buf is None:
        return
    # ★ 推定値を使うこと。ここは feet_phase / foot_clearance の停止判定に使われる。
    #   真値にすると「ポリシーが観測できない情報で足上げ報酬が ON/OFF する」状態になり、
    #   ポリシーは報酬が付くかを予測できなくなる。実測で 28% 食い違い、足上げの期待値が
    #   下がって foot_clearance が 0.70 → 0.25 に落ちた。自己位置のバイアスは設計上
    #   エピソード内で一定なので、真値のままだとロボットが正しく定位置へ行くほど
    #   食い違いが恒久化する (平均回帰では解消できない)。このスイッチは実機に存在しない
    #   学習時だけのものなので、推定値にしても sim-to-real のリアルさは損なわれない。
    drive = task_drive_vector(env, use_perceived=True)
    buf[:, :2] = drive[:, :2]
    # ★ 向き成分は 0 にする。feet_phase / foot_clearance の停止判定は
    buf[:, 2] = 0.0


def relaunch_ball_after_save(
    env: "ManagerBasedEnv",
    # ★ env_ids にデフォルト値を付けないこと (理由は下記)。EventManager の引数検証
    env_ids: torch.Tensor | None,
    respawn_delay_steps: int = 50,
    clearance_safe_dist: float = 1.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """**エピソード継続モード**: セーブが確定した env に、次のボールを撃ち直す。

    ``EventTerm(mode="interval", interval_range_s=(dt, dt), is_global_time=True)`` で
    毎制御ステップ呼ぶ。ステージ1 の :func:`resample_stage1_target` と同じ思想で、
    「成功したらエピソードを切らずに次の課題を出す」。

    なぜ継続にするか:
        従来は ``save_success`` が ``DoneTerm`` で、**セーブ成功＝エピソード終了**
        だった。そのため
          * 1 エピソード (10s) に報酬イベントが 1 回しか無く、ただでさえスパースな
            セーブ報酬がさらに希薄になっていた。
          * ``return_to_center_after_save`` が発火する間もなく終了するため、
            「弾いた後に定位置へ戻る」がほぼ学習されていなかった。
        継続にすると 1 エピソードに複数回のセーブが入り、単位時間あたりの学習信号が
        増える。さらに「セーブ後すぐ中央へ戻らないと次が取れない」という圧力が
        自然に生まれる (実機の連続シュート対応に近い)。

    失敗 (失点・転倒・場外) は従来通り ``DoneTerm`` でエピソードを切る。
    ロボットの位置はリセットせず、**セーブした場所からそのまま次の球に備える**。

    Args:
        respawn_delay_steps: セーブ確定から次の発射までの待ちステップ数。
            弾いたボールが転がりきる前に撃つと接触判定が混線するので間を置く。
            50 step = 1.0s @ 50Hz。
    """
    bufs = gk_buffers(env)
    cd = bufs["respawn_cd"]

    # --- 1. セーブ確定を検知して待ちカウンタを開始する ---
    # update_save_state は save_cd を進め、確定時に 0 になる (冪等なので多重呼び出し可)。
    from .terminations import update_save_state

    update_save_state(env)
    newly_saved = (bufs["save_cd"] == 0) & (cd < 0)
    if bool(newly_saved.any()):
        cd[newly_saved] = int(respawn_delay_steps)
        # 適応カリキュラムが「1 球あたりの成功率」を出すための実績カウント
        bufs["save_count"][newly_saved] += 1
        # セーブ品質 (どれだけ危険圏から遠ざけたか) を記録する。
        q = _save_clearance_quality(env, safe_dist=float(clearance_safe_dist))
        bufs["save_quality"][newly_saved] = q[newly_saved]
        # 確定済みとして save_cd を無効化 (save_success と同じ後始末)
        bufs["save_cd"][newly_saved] = -1
        # ボールを非アクティブにして観測をダミーに戻す (次の発射までは「脅威なし」)
        bufs["ball_active"][newly_saved] = False

    # --- 2. 待ちカウンタを進め、0 になった env に次の球を撃つ ---
    waiting = cd > 0
    cd[waiting] -= 1
    fire = cd == 0
    if not bool(fire.any()):
        return

    fire_ids = torch.nonzero(fire).flatten()
    # ボール 1 球ぶんの状態をリセットしてから再発射する (touched/touch_rewarded を
    # 残すと 2 球目以降の save_touch_bonus が払われない)。
    bufs["touched"][fire_ids] = False
    bufs["touch_rewarded"][fire_ids] = False
    bufs["save_cd"][fire_ids] = -1
    cd[fire_ids] = -1

    reset_ball_shot(env, fire_ids, ball_cfg=ball_cfg)
    reset_ball_perception(env, fire_ids)


# ---------------------------------------------------------------------------
# 横移動特化タスク (goalkeeper_lateral_env_cfg.py) 用の状態バッファ
# ---------------------------------------------------------------------------

_LATERAL_ATTR = "_gk_lateral_buffers"


def lateral_buffers(env: "ManagerBasedEnv") -> dict:
    """横移動特化タスクの状態バッファ (遅延生成)。

    * ``cmd_prev``     : 前ステップの速度コマンド。変化検出に使う。
    * ``since_change`` : 速度コマンドが「大きく変わって」からの経過ステップ数。
                         立ち上がり (加速) 報酬のゲートに使う。負値は「次回の更新で
                         基準を取り直す」ためのセンチネル (リセット直後)。
    * ``ref_yaw``      : 指令角速度 wz を積分した **基準ヘディング**。
                         実ヨーとのズレが「積分 yaw 誤差」= ドリフトそのもの。
    """
    bufs = getattr(env, _LATERAL_ATTR, None)
    if bufs is None:
        n, dev = env.num_envs, env.device
        bufs = {
            "cmd_prev": torch.zeros(n, 3, device=dev),
            "since_change": torch.full((n,), -1, dtype=torch.long, device=dev),
            "ref_yaw": torch.zeros(n, device=dev),
            # 今のコマンドで「指令速度の所定割合に到達した」ボーナスを払い済みか
            # (:func:`~.rewards.onset_reach_bonus` が 1 コマンドにつき 1 回だけ払うため)
            "reach_paid": torch.zeros(n, dtype=torch.bool, device=dev),
            "last_step": -1,
        }
        setattr(env, _LATERAL_ATTR, bufs)
    return bufs


def update_lateral_buffers(
    env: "ManagerBasedEnv",
    command_name: str = "base_velocity",
    change_tol: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict:
    """1 ステップにつき 1 回だけバッファを更新する (何度呼んでも安全)。

    報酬関数の先頭から呼ぶ想定。``common_step_counter`` でガードしてあるので、
    EventTerm として登録しなくても、複数の報酬から呼んでも二重更新しない。
    EventTerm の実行順に依存しないぶん、こちらの方が壊れにくい。

    基準ヘディングの取り直し (resync) は **線速度コマンドが大きく変わったとき**と
    **リセット直後**に行う。取り直さないと誤差が上限まで飽和して勾配が消え、
    取り直しすぎると積分ドリフトを検出できない。コマンド再サンプル周期
    (1.5〜4.0s) がそのまま「ドリフトを溜めて測る窓」になる。
    """
    bufs = lateral_buffers(env)
    step = int(env.common_step_counter)
    if bufs["last_step"] == step:
        return bufs
    bufs["last_step"] = step

    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dt = env.step_dt

    # 線速度指令の変化だけを見る。wz は heading 制御を使うと毎ステップ動くので、
    # 変化検出に混ぜると常時 onset 扱いになってしまう。
    changed = torch.norm(cmd[:, :2] - bufs["cmd_prev"][:, :2], dim=1) > change_tol
    resync = changed | (bufs["since_change"] < 0)

    bufs["since_change"] += 1
    bufs["since_change"][resync] = 0
    bufs["reach_paid"][resync] = False

    # 基準ヘディングは指令角速度を積分する (wz=0 なら「向きを保て」と同義)。
    bufs["ref_yaw"] = wrap_to_pi(bufs["ref_yaw"] + cmd[:, 2] * dt)
    bufs["ref_yaw"][resync] = asset.data.heading_w[resync]

    bufs["cmd_prev"] = cmd.clone()
    return bufs


def reset_lateral_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor):
    """リセットされた env の横移動バッファを無効化する (次の更新で取り直す)。

    ここでヨーを読まないのは、EventTerm の実行順 (reset_base より前か後か) に
    依存しないようにするため。センチネル -1 を入れておけば、次のステップで
    :func:`update_lateral_buffers` が新しい姿勢を基準に取り直す。
    """
    bufs = lateral_buffers(env)
    bufs["since_change"][env_ids] = -1
    bufs["ref_yaw"][env_ids] = 0.0
    bufs["cmd_prev"][env_ids] = 0.0
    bufs["reach_paid"][env_ids] = False
