# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用の観測関数。

座標系の規約 (goalkeeper_env_cfg.py と共有):
    * 「ゴール座標系」= env ローカル座標 (env origin がゴール中央・ゴールライン上)。
      +x がフィールド側 (ボールが来る方向)、y がゴールライン方向。
    * ゴールライン: x = 0。失点判定はボール中心 x < -(ボール半径)。
    * ロボットはゴール中央 (原点付近) に +x 向きでスポーンする。

ステージ間で観測次元を固定するため、ボール観測 (相対位置・速度) はステージ1
(ボールなし) でもスロットを確保し、非アクティブ時は 0 (ダミー値) を返す。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# per-env 状態バッファ (遅延生成)
# ---------------------------------------------------------------------------

def gk_buffers(env: "ManagerBasedRLEnv") -> dict[str, torch.Tensor]:
    """goalkeeper タスクの per-env 状態バッファを (無ければ生成して) 返す。

    - ``_gk_target_y``      : ステージ1 のランダム目標 y [m] (ゴール座標系)
    - ``_gk_ball_active``   : ボールがアクティブ (発射済み) か。False = ステージ1 の
                              パーク状態 (観測はダミー 0)
    - ``_gk_touched``       : このエピソードでロボットがボールに触れたか
    - ``_gk_touch_rewarded``: save_touch_bonus を既に払ったか (一回限りの報酬用)
    - ``_gk_save_cd``       : セーブ成功終了までのカウントダウン [-1=非発火]
    - ``_gk_hold_ctr``      : ステージ1 の目標保持カウンタ
    """
    n = env.num_envs
    if getattr(env, "_gk_target_y", None) is None or env._gk_target_y.shape != (n,):
        env._gk_target_y = torch.zeros(n, device=env.device)
        env._gk_ball_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_touched = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_touch_rewarded = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_save_cd = torch.full((n,), -1, dtype=torch.long, device=env.device)
        env._gk_hold_ctr = torch.zeros(n, dtype=torch.long, device=env.device)
    return {
        "target_y": env._gk_target_y,
        "ball_active": env._gk_ball_active,
        "touched": env._gk_touched,
        "touch_rewarded": env._gk_touch_rewarded,
        "save_cd": env._gk_save_cd,
        "hold_ctr": env._gk_hold_ctr,
    }


# ---------------------------------------------------------------------------
# 座標ヘルパ
# ---------------------------------------------------------------------------

def ball_pos_goal(env: "ManagerBasedRLEnv", ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    """ボール位置をゴール座標系 (env origin 基準) で返す (N, 3)。"""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w[:, :3] - env.scene.env_origins


def robot_pos_goal(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ロボット base 位置をゴール座標系で返す (N, 3)。"""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins


def compute_target_y(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.25,
    approach_vx_threshold: float = -0.05,
    use_perceived: bool = False,
) -> torch.Tensor:
    """ロボットが向かうべき目標 y 座標 (ゴール座標系) を返す (N,)。

    * ボール非アクティブ (ステージ1): ``_gk_target_y`` バッファのランダム目標。
    * ボールがゴールへ接近中 (vx < approach_vx_threshold): 現在のボール位置・速度から
      **守備面 (x = guard_x、ロボットの定位置)** への到達 y を外挿して返す
      (±max_y にクランプ)。ロボットはゴールラインの 0.3m 前で守るので、予測面も
      ゴールライン (x=0) ではなく守備面に合わせる (斜めの球ほどズレるため必須)。
      転がりの減速は進行方向に沿って一様に効くため軌道は直線のままであり、
      「到達するなら到達点 y」は等速外挿と一致する。
    * ボールが接近していない (弾いた後・停止後): ゴール中央 0 (復帰)。

    ``use_perceived=True`` なら知覚DR後のボール状態 (遅延・ノイズ入り) から計算する
    (policy 観測用。実機では認識出力から同じ計算をする)。報酬と critic は真値を使う。

    観測と報酬の両方からこの関数で同じ目標を参照する (整合を一箇所に集約)。
    """
    bufs = gk_buffers(env)
    guard_x = float(env.cfg.goalkeeper.guard_x)
    if use_perceived:
        pos, vel = _gk_perceived_goal_state(env)
    else:
        pos = ball_pos_goal(env)
        ball: RigidObject = env.scene["soccer_ball"]
        vel = ball.data.root_com_vel_w[:, :3]

    approaching = vel[:, 0] < approach_vx_threshold
    # 守備面 (x=guard_x) 到達までの時間 (接近中のみ意味を持つ。ゼロ割り防止で clamp)
    t = ((pos[:, 0] - guard_x) / (-vel[:, 0]).clamp(min=1e-3)).clamp(min=0.0)
    y_pred = (pos[:, 1] + vel[:, 1] * t).clamp(-max_y, max_y)

    target = torch.where(approaching, y_pred, torch.zeros_like(y_pred))
    return torch.where(bufs["ball_active"], target, bufs["target_y"])


# ---------------------------------------------------------------------------
# 知覚DR (実機の認識出力を模擬: レイテンシ / 更新レート / ドロップ / 距離依存ノイズ)
#
# around_ball の ball_pos_rel_perceived と同系の設計だが、頭部が常にボールを追う
# 前提 (戦略側でボール追従) なので ±60° の FOV マスクと hold-last-seen は入れない。
# 品質劣化 (遅延・欠損・ノイズ) だけを模擬する。critic は真値を見る (非対称)。
# ---------------------------------------------------------------------------

_PERC_HIST_LEN = 6  # 履歴長 (最大レイテンシ 5 tick + 現在)


def _gk_perc_buffers(env: "ManagerBasedRLEnv") -> None:
    """知覚DRの per-env バッファを (無ければ) 生成する。"""
    n = env.num_envs
    if getattr(env, "_gkp_hist_pos", None) is None or env._gkp_hist_pos.shape[0] != n:
        dev = env.device
        env._gkp_hist_pos = torch.zeros(n, _PERC_HIST_LEN, 2, device=dev)   # 真の相対位置履歴 (base frame)
        env._gkp_hist_vel = torch.zeros(n, _PERC_HIST_LEN, 2, device=dev)   # 真のボール速度履歴 (base frame)
        env._gkp_out_pos = torch.zeros(n, 2, device=dev)                    # 認識出力 (保持)
        env._gkp_out_vel = torch.zeros(n, 2, device=dev)
        env._gkp_latency = torch.ones(n, dtype=torch.long, device=dev)      # per-episode レイテンシ [tick]
        # 更新周期は **小数 tick** (制御50Hz / ビジョン30Hz = 1.67 tick)。整数だと
        # 25Hz か 50Hz にしか置けず実測レートを表現できないため float で保持する。
        env._gkp_period = torch.ones(n, device=dev)                         # per-episode 更新周期 [tick, float]
        env._gkp_ctr = torch.zeros(n, device=dev)                           # 次回更新までのカウンタ [float]
        env._gkp_bias = torch.zeros(n, 2, device=dev)                       # per-episode 位置バイアス [m]
        env._gkp_vel_bias = torch.zeros(n, 2, device=dev)                   # per-episode 速度バイアス [m/s] (x,y 各軸独立)
        env._gkp_step = -1                                                  # 1 step 1 回ガード


def _gk_true_rel_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """真のボール相対位置とボール速度 (base yaw frame, 各 (N,2)) を返す。"""
    ball: RigidObject = env.scene["soccer_ball"]
    robot: Articulation = env.scene["robot"]
    q = yaw_quat(robot.data.root_quat_w)
    offset_b = quat_apply_inverse(q, ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3])[:, :2]
    vel_b = quat_apply_inverse(q, ball.data.root_com_vel_w[:, :3])[:, :2]
    return offset_b, vel_b


def _gk_perception_tick(env: "ManagerBasedRLEnv") -> None:
    """知覚DRの状態を 1 制御ステップ 1 回だけ更新する (冪等)。

    毎ステップ: 真値を履歴に積む → 更新周期が来た env だけ「レイテンシ分過去の真値
    + 距離依存ノイズ + 系統バイアス」を認識出力に反映。ドロップ発生時は出力を
    据え置く (実機の検出取りこぼし)。パラメータは GoalkeeperParamsCfg の perc_*。
    """
    _gk_perc_buffers(env)
    step = int(env.common_step_counter)
    if env._gkp_step == step:
        return
    env._gkp_step = step

    p = env.cfg.goalkeeper
    pos_b, vel_b = _gk_true_rel_state(env)

    # 履歴を 1 tick 進める (index 0 が現在)
    env._gkp_hist_pos = torch.roll(env._gkp_hist_pos, shifts=1, dims=1)
    env._gkp_hist_vel = torch.roll(env._gkp_hist_vel, shifts=1, dims=1)
    env._gkp_hist_pos[:, 0] = pos_b
    env._gkp_hist_vel[:, 0] = vel_b

    # 更新周期の到来した env を選ぶ。カウンタは float で、到来時に周期を **加算** する
    # (代入ではない)。こうすると小数周期の位相が保たれ、1.67 tick 周期なら
    # 2,2,1,2,2,1... と実測 30Hz の更新間隔を平均的に再現できる。
    env._gkp_ctr -= 1.0
    due = env._gkp_ctr <= 0.0
    env._gkp_ctr[due] += env._gkp_period[due]
    # ドロップ: due のうち一部は更新をスキップ (出力据え置き)
    keep = torch.rand(env.num_envs, device=env.device) < float(p.perc_dropout_prob)
    update = due & ~keep
    if not bool(update.any()):
        return

    idx = torch.arange(env.num_envs, device=env.device)
    lat = env._gkp_latency.clamp(max=_PERC_HIST_LEN - 1)
    delayed_pos = env._gkp_hist_pos[idx, lat]
    delayed_vel = env._gkp_hist_vel[idx, lat]

    dist = torch.norm(delayed_pos, dim=1, keepdim=True)
    sigma = float(p.perc_noise_sigma) + float(p.perc_noise_per_m) * dist
    noisy_pos = delayed_pos + torch.randn_like(delayed_pos) * sigma + env._gkp_bias
    # 速度: 小さな毎フレームジッタ (perc_vel_noise_sigma) に加えて、エピソード固定の
    # 系統バイアス (_gkp_vel_bias, x/y 各軸独立で大きさ 0.5〜1.0 m/s) を乗せる。
    # 実機のボール速度推定は遅延で一定方向にずれるため、毎フレーム暴れるジッタではなく
    # 「そのエピソードでは +0.7, 別のエピソードでは -0.6」といった系統誤差で模擬する。
    noisy_vel = (
        delayed_vel
        + torch.randn_like(delayed_vel) * float(p.perc_vel_noise_sigma)
        + env._gkp_vel_bias
    )

    env._gkp_out_pos[update] = noisy_pos[update]
    env._gkp_out_vel[update] = noisy_vel[update]


def _gk_perceived_goal_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """知覚DR後のボール位置・速度をゴール座標系 (N,3) で返す (z は 0 埋め)。

    認識出力は base frame の相対値なので、ロボットの真の姿勢 (自己位置推定は
    別系統で十分正確とみなす) でゴール座標系へ戻す。到達予測 (policy 用) が使う。
    """
    _gk_perception_tick(env)
    robot: Articulation = env.scene["robot"]
    heading = robot.data.heading_w
    c, s = torch.cos(heading), torch.sin(heading)
    rel = env._gkp_out_pos
    off_x = c * rel[:, 0] - s * rel[:, 1]
    off_y = s * rel[:, 0] + c * rel[:, 1]
    rpos = robot_pos_goal(env)
    pos = torch.stack([rpos[:, 0] + off_x, rpos[:, 1] + off_y, torch.zeros_like(off_x)], dim=1)
    v = env._gkp_out_vel
    vel = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], torch.zeros_like(off_x)], dim=1)
    return pos, vel


# ---------------------------------------------------------------------------
# 観測項
# ---------------------------------------------------------------------------

def gk_ball_pos_rel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ボール相対位置の真値 (base yaw frame, 2D)。critic 用。非アクティブ時はダミー 0。"""
    bufs = gk_buffers(env)
    offset_b, _ = _gk_true_rel_state(env)
    return torch.where(bufs["ball_active"].unsqueeze(1), offset_b, torch.zeros_like(offset_b))


def gk_ball_vel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ボール速度の真値 (base yaw frame, 2D)。critic 用。非アクティブ時はダミー 0。"""
    bufs = gk_buffers(env)
    _, vel_b = _gk_true_rel_state(env)
    return torch.where(bufs["ball_active"].unsqueeze(1), vel_b, torch.zeros_like(vel_b))


def gk_ball_pos_rel_perceived(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """知覚DR後のボール相対位置 (base yaw frame, 2D)。policy 用。非アクティブ時はダミー 0。"""
    bufs = gk_buffers(env)
    _gk_perception_tick(env)
    out = env._gkp_out_pos
    return torch.where(bufs["ball_active"].unsqueeze(1), out, torch.zeros_like(out))


def gk_ball_vel_perceived(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """知覚DR後のボール速度 (base yaw frame, 2D)。policy 用。非アクティブ時はダミー 0。"""
    bufs = gk_buffers(env)
    _gk_perception_tick(env)
    out = env._gkp_out_vel
    return torch.where(bufs["ball_active"].unsqueeze(1), out, torch.zeros_like(out))


def gk_ball_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """ボールがアクティブ (発射済み) かのフラグ (N, 1)。ステージ1 では常に 0。"""
    return gk_buffers(env)["ball_active"].float().unsqueeze(1)


def gk_target_y(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.25,
    use_perceived: bool = False,
) -> torch.Tensor:
    """目標 y 座標 (ゴール座標系) の観測 (N, 1)。:func:`compute_target_y` 参照。

    policy には ``use_perceived=True`` (知覚DR後のボール状態から予測)、
    critic には既定の真値版を使う。
    """
    return compute_target_y(env, max_y=max_y, use_perceived=use_perceived).unsqueeze(1)


def zeros_obs(env: "ManagerBasedRLEnv", dim: int = 1) -> torch.Tensor:
    """常にゼロを返すダミー観測 (N, dim)。

    直接制御版のステージ1 (ボール不在) でボール系スロットの次元を確保するために使う。
    ゼロ入力の列には勾配が流れないので、該当する重みは初期値のままステージ2 へ渡る
    (ball_kick と同じ「次元一致方式」)。
    """
    return torch.zeros(env.num_envs, int(dim), device=env.device)


def task_drive_vector(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.25,
    vx_scale: float = 1.0,
    vy_scale: float = 1.5,
    use_perceived: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ステージ2/3 で ``velocity_commands`` スロットに入れる「移動要求」ベクトル (N, 3)。

    ステージ1 では同じスロットに locomotion の速度コマンド (vx, vy, wz) が入る。
    ステージ2/3 には外部から与えられるコマンドが無いので、タスク状態から等価な
    「どちらへどれだけ動きたいか」を作って同じスロットに入れる:

        [0] = 守備面 (guard_x) までの前後ずれ   → ステージ1 の vx と同じ意味
        [1] = 目標 y (ボール到達予測点) までの横ずれ → ステージ1 の vy と同じ意味
        [2] = 向きの誤差 (フィールド正面からのずれ)  → ステージ1 の wz と同じ意味

    位置誤差をステージ1 のコマンド範囲へクリップして渡すので、ステージ1 で獲得した
    「このスロットが大きい方向へ速く動く」という対応がそのまま流用でき、
    ステージ遷移が滑らかになる。ボールの到達予測を先読みして動く/構えるといった
    タスク固有の判断は、別スロットのボール観測から学習される。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    guard_x = float(env.cfg.goalkeeper.guard_x)
    pos = robot_pos_goal(env, asset_cfg)
    dx = (guard_x - pos[:, 0]).clamp(-vx_scale, vx_scale)
    dy = (compute_target_y(env, max_y=max_y, use_perceived=use_perceived) - pos[:, 1]).clamp(
        -vy_scale, vy_scale
    )
    # heading_w は +x を 0 とする world yaw。フィールド正面へ戻す向きを渡す。
    dyaw = (-robot.data.heading_w).clamp(-1.0, 1.0)
    return torch.stack([dx, dy, dyaw], dim=1)


def gk_self_state(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """自機のゴール座標系状態 (N, 4): (x オフセット, y オフセット, sin(yaw), cos(yaw))。

    yaw=0 はフィールド側 (+x, ボールが来る方向) を向いた姿勢。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    pos = robot_pos_goal(env, asset_cfg)
    heading = robot.data.heading_w
    return torch.stack([pos[:, 0], pos[:, 1], torch.sin(heading), torch.cos(heading)], dim=1)
