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

import math
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
    - ``_gk_respawn_cd``    : 次のボール発射までのカウントダウン [-1=非発火]。
                              エピソード継続モード (relaunch_ball_after_save) 用。
    - ``_gk_save_count``    : このエピソードでセーブした球数。適応カリキュラムが
                              **1 球あたり** の成功率を出すのに使う。
    - ``_gk_save_quality``  : 未払いのセーブ品質報酬 [0,1]。セーブ確定時にイベントが
                              書き、報酬 (save_clearance_bonus) が読んでゼロに戻す。
    """
    n = env.num_envs
    if getattr(env, "_gk_target_y", None) is None or env._gk_target_y.shape != (n,):
        env._gk_target_y = torch.zeros(n, device=env.device)
        env._gk_ball_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_touched = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_touch_rewarded = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_save_cd = torch.full((n,), -1, dtype=torch.long, device=env.device)
        env._gk_hold_ctr = torch.zeros(n, dtype=torch.long, device=env.device)
        env._gk_respawn_cd = torch.full((n,), -1, dtype=torch.long, device=env.device)
        env._gk_save_count = torch.zeros(n, dtype=torch.long, device=env.device)
        env._gk_save_quality = torch.zeros(n, device=env.device)
    return {
        "target_y": env._gk_target_y,
        "ball_active": env._gk_ball_active,
        "touched": env._gk_touched,
        "touch_rewarded": env._gk_touch_rewarded,
        "save_cd": env._gk_save_cd,
        "hold_ctr": env._gk_hold_ctr,
        "respawn_cd": env._gk_respawn_cd,
        "save_count": env._gk_save_count,
        "save_quality": env._gk_save_quality,
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
# 知覚DR: 実機カメラ準拠の VirtualPerception (booster_amp_lab の soccer_perception を
# goalkeeper に複製したもの) でボール位置観測を作る。カメラ仕様 (FOV 150°/80°、
# レイテンシ 116ms、25Hz、距離依存ノイズ σ(d)=0.124d+0.149、検出率90%/7m、occlusion、
# dead-zone) は soccer_vision_train_cfg がそのまま持つ。頭は戦略層が常にボールを
# 追う前提なので head_tracks_ball=True (FOV は実質常にパス、品質劣化のみ効く)。
#
# 速度は VirtualPerception が出さない (位置・mask・last_seen_dt のみ) ので、ボール速度
# だけは従来どおり「真値 + エピソード固定バイアス 0.5〜1.0 m/s」で別途作る (実機は
# 速度を直接測れず推定が一定方向にずれるため)。mask=0 (見えていない) のときは位置も
# 速度もゼロにして整合を取る。critic は真値を見る (非対称)。
# ---------------------------------------------------------------------------


def _gk_perception(env: "ManagerBasedRLEnv"):
    """VirtualPerception インスタンスを (無ければ生成して) 返す。"""
    from .perception import VirtualPerception, soccer_vision_train_cfg

    vp = getattr(env, "_gk_vp", None)
    if vp is None or vp.num_envs != env.num_envs:
        cfg = soccer_vision_train_cfg()
        # PLAY 用クリーン化: 検出100%・遅延0・ノイズ0・見失い(occlusion/dead-zone/blind)なし・
        # 50Hz。キーパーの動きそのものを純粋に評価するため (学習では False)。
        if bool(getattr(env.cfg.goalkeeper, "perception_clean", False)):
            cfg.detection_prob_in_fov_range = (1.0, 1.0)
            cfg.blind_prob = 0.0
            cfg.occlusion_prob = 0.0
            cfg.deadzone_prob = 0.0
            cfg.noise_a = 0.0
            cfg.noise_b = 0.0
            cfg.latency_mean_range = (0.0, 0.0)
            cfg.latency_std_s = 0.0
            cfg.update_hz_mean_range = (1.0 / float(env.step_dt), 1.0 / float(env.step_dt))
            cfg.update_hz_std = 0.0
        cfg.head_tracks_ball = True   # 戦略層がボールを追う前提。FOV は実質常にパス
        # K1_locomotion.urdf では頭リンク (Head_2→Head_1→Trunk) が全て Trunk にマージされ、
        # 独立した頭 body が存在しない (import 警告あり)。カメラは Trunk にマウントする。
        # head_tracks_ball=True でカメラ向きはボール方向に上書きするので、マウント先が頭でも
        # 胴体でも FOV 判定への影響はない (カメラ位置がわずかに下がるだけ)。
        # 実機の頭の高さに寄せるため、Trunk 原点からの上方オフセットを頭部相当に上げる。
        cfg.camera_body_name = "Trunk"
        cfg.camera_offset_pos = (0.06, 0.0, 0.45)  # 胴体原点からカメラ (頭部高さ相当) まで
        vp = VirtualPerception(
            cfg=cfg,
            robot=env.scene["robot"],
            num_envs=env.num_envs,
            dt=float(env.step_dt),
            device=env.device,
        )
        env._gk_vp = vp
        env._gkp_vel_bias = torch.zeros(env.num_envs, 2, device=env.device)  # 速度バイアス
        env._gkp_out_vel = torch.zeros(env.num_envs, 2, device=env.device)   # 誤差付き速度 (保持)
        env._gkp_step = -1
    return vp


def _gk_true_rel_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """真のボール相対位置とボール速度 (base yaw frame, 各 (N,2)) を返す。critic 用。"""
    ball: RigidObject = env.scene["soccer_ball"]
    robot: Articulation = env.scene["robot"]
    q = yaw_quat(robot.data.root_quat_w)
    offset_b = quat_apply_inverse(q, ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3])[:, :2]
    vel_b = quat_apply_inverse(q, ball.data.root_com_vel_w[:, :3])[:, :2]
    return offset_b, vel_b


def _gk_perception_tick(env: "ManagerBasedRLEnv") -> None:
    """VirtualPerception を 1 制御ステップ 1 回だけ進める (冪等)。

    位置・mask は VirtualPerception が担当 (レイテンシ・ノイズ・検出率・occlusion)。
    速度は真値 (base yaw frame) にエピソード固定バイアスを乗せ、mask=0 でゼロにする。
    """
    vp = _gk_perception(env)
    step = int(env.common_step_counter)
    if env._gkp_step == step:
        return
    env._gkp_step = step

    ball: RigidObject = env.scene["soccer_ball"]
    vp.update(env.scene["robot"], ball.data.root_pos_w[:, :3])

    # 速度: 真値 (base yaw frame) + エピソード固定バイアス。見えていない (mask=0) 間は 0。
    _, vel_b = _gk_true_rel_state(env)
    mask = vp.ball_mask.unsqueeze(1)
    env._gkp_out_vel = (vel_b + env._gkp_vel_bias) * mask


def _gk_perceived_goal_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """知覚DR後のボール位置・速度をゴール座標系 (N,3) で返す (z は 0 埋め)。

    VirtualPerception の相対位置 (base yaw frame) を、ロボットの真の姿勢 (自己位置推定は
    別系統で十分正確とみなす) でゴール座標系へ戻す。到達予測 (policy 用) が使う。
    """
    vp = _gk_perception(env)
    _gk_perception_tick(env)
    robot: Articulation = env.scene["robot"]
    heading = robot.data.heading_w
    c, s = torch.cos(heading), torch.sin(heading)
    rel = vp.ball_pos_b
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
    """VirtualPerception のボール相対位置 (base yaw frame, 2D)。policy 用。

    見えていない (mask=0) / 非アクティブ時は VirtualPerception が 0 を返す
    (hold_last_on_miss=False なので miss 時ゼロ)。ボールが park (非アクティブ) の
    ときは検出範囲外なので自然に 0 になる。"""
    _gk_perception_tick(env)
    return _gk_perception(env).ball_pos_b


def gk_ball_vel_perceived(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """誤差付きボール速度 (base yaw frame, 2D)。policy 用。mask=0 でゼロ (tick で連動済み)。"""
    _gk_perception_tick(env)
    return env._gkp_out_vel


def gk_ball_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """ボールが **今カメラで見えているか** のフラグ (N, 1) = VirtualPerception の mask。

    従来は「発射済みか (真値)」だったが、実機に寄せて「検出できているか」に変更。
    occlusion・検出漏れ・range 外では 0 になり、ポリシーが「今ボールを見失っている」
    ことを認識できる。ステージ1 (ボールなし) では観測スロット自体が zeros_obs で 0。"""
    _gk_perception_tick(env)
    return _gk_perception(env).ball_mask.unsqueeze(1)


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
    # ★ vy_scale はステージ1 のコマンド範囲 (commands.base_velocity.ranges.lin_vel_y)
    #   と必ず一致させること。ステージ1 は「スロットの値 = 出すべき速度」を学ぶので、
    #   ステージ2/3 でスロットの取りうる範囲が違うと対応がズレる。
    vy_scale: float = 1.3,
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


def task_drive_phase_obs(
    env: "ManagerBasedRLEnv",
    phase_freq: float = 1.6,
    # ★ しきい値の単位に注意。locomotion の phase_obs は **速度 [m/s]** のノルムを
    #   0.05 と比べるが、task_drive_vector は **位置ずれ [m]** なので同じ 0.05 を
    #   流用する根拠が無い (5cm 以内に入らないと止まれず、実質ほぼ止まらない)。
    #   本タスクが「到達した」とみなす尺度に合わせる:
    #   GoalkeeperParamsCfg.stage1_reach_tol = 0.15 / target_reach_velocity の
    #   deadband = 0.12。その下限側の 0.12 を既定にする。
    cmd_threshold: float = 0.12,
    max_y: float = 1.25,
    vx_scale: float = 1.0,
    vy_scale: float = 1.3,
    use_perceived: bool = True,
) -> torch.Tensor:
    """ステージ2/3 の ``gait_phase`` スロット (4 次元)。**タスク駆動**の歩行位相。

    locomotion の :func:`phase_obs` と同一フォーマット・同一規約 (左右 sin/cos、
    停止時はゼロ埋め) だが、**停止判定の駆動元が違う**。

    なぜ必要か:
        ``phase_obs`` は停止判定に ``base_velocity`` コマンドのノルムを使う。
        ステージ1 ではそれが実際の指令なので正しいが、ステージ2/3 では
        ``velocity_commands`` **スロットの中身**だけを :func:`task_drive_vector` に
        差し替えており、``base_velocity`` コマンド項自体はステージ1 の設定
        (10 秒ごとにランダム再サンプル) のまま残っている。その結果、位相が
        **タスクと無関係なランダムコマンド**で駆動され、ボールを止めた後も
        「歩き続けろ」という位相が入り続けて足踏みが止まらなかった。
        (階層版は同じ問題を ``high_action_phase_obs`` で解決済み。直接制御版に
        移植されていなかった。)

    本関数は観測スロットに入るのと同じ :func:`task_drive_vector` のノルムでゲートする。
    「動く必要がない = スロットが小さい = 位相ゼロ = その場で立つ」となり、
    ステージ1 で学んだ停止の仕方がそのまま流用できる。

    位相周波数はステージ1 と同じ ``get_phase_freq`` 経由なので、
    ``randomize_phase_freq`` による env ごとの ±0.05Hz ランダム化に自動追従する。
    """
    from ...locomotion.mdp.events import get_phase_freq

    t = env.episode_length_buf * env.step_dt
    pf = get_phase_freq(env, phase_freq)
    phase_left = 2.0 * math.pi * pf * t
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    drive = task_drive_vector(
        env, max_y=max_y, vx_scale=vx_scale, vy_scale=vy_scale, use_perceived=use_perceived
    )
    drive_norm = torch.norm(drive, dim=1, keepdim=True)
    return torch.where(drive_norm < cmd_threshold, torch.zeros_like(phase), phase)


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
