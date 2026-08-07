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

def ball_pos_goal(env: "ManagerBasedRLEnv", ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    """ボール位置をゴール座標系 (env origin 基準) で返す (N, 3)。"""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w[:, :3] - env.scene.env_origins


def robot_pos_goal(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ロボット base 位置をゴール座標系で返す (N, 3)。"""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins


# --------------------------------------------------------------- 自己位置推定誤差

def _gk_loc_buffers(env: "ManagerBasedRLEnv") -> None:
    """自己位置推定 (実機は MCL) の誤差状態を確保する。

    実機の MCL は白色ノイズではなく、
      * エピソード内でほぼ一定のバイアス
      * ランドマークが見えない間の odometry ドリフト
      * パーティクル群の再収束による**不連続な跳び** (平滑化が意図的に OFF で、
        1 フレームあたり最大 0.5 m / 0.5 rad まで動く)
      * そして跳んだりドリフトしたりした誤差が、ランドマークを見た時点で**戻る**
    という誤差を出す。跳びは学習中に一度も経験しないと実機で未知入力になるので、
    ここでモデル化する。
    """
    n = env.num_envs
    if getattr(env, "_gk_loc_err", None) is None or env._gk_loc_err.shape != (n, 3):
        env._gk_loc_err = torch.zeros(n, 3, device=env.device)     # 累積誤差 [dx, dy, dyaw]
        env._gk_loc_bias = torch.zeros(n, 3, device=env.device)    # エピソード固定バイアス
        env._gk_loc_drift = torch.zeros(n, 3, device=env.device)   # ドリフト速度 [m/s, rad/s]
        env._gk_loc_jump_p = torch.zeros(n, device=env.device)     # 跳びの毎ステップ確率
        env._gk_loc_step = -1


def _gk_loc_tick(env: "ManagerBasedRLEnv") -> None:
    """自己位置誤差を 1 制御ステップ進める (冪等)。"""
    _gk_loc_buffers(env)
    step = int(env.common_step_counter)
    if env._gk_loc_step == step:
        return
    env._gk_loc_step = step

    p = env.cfg.goalkeeper
    dt = float(env.step_dt)
    err = env._gk_loc_err
    err += env._gk_loc_drift * dt

    jump = torch.rand(env.num_envs, device=env.device) < env._gk_loc_jump_p
    if bool(jump.any()):
        jmag = float(getattr(p, "loc_jump_m", 0.5))
        jyaw = float(getattr(p, "loc_jump_rad", 0.2))
        delta = torch.empty(env.num_envs, 3, device=env.device).uniform_(-1.0, 1.0)
        delta[:, :2] *= jmag
        delta[:, 2] *= jyaw
        err = torch.where(jump.unsqueeze(-1), err + delta, err)

    # ★ 再収束 (2026-08-03)。MCL はランドマークが視野に入った時点で誤差が**戻る**。
    #   これが無いとドリフトと跳びで誤差が上限に張り付いたままになり、ロボットが
    #   「自分は 0.8m ずれている」と信じ込んで歩き続ける。実測で out_of_bounds が
    #   10 倍 (0.004 → 0.040) になったのはこれが原因。誤差のピーク値は変えず、
    #   「張り付いたまま」を「一時的なズレ」にするだけなので DR は弱まらない。
    tau = float(getattr(p, "loc_recover_tau_s", 5.0))
    if tau > 0.0:
        err *= max(0.0, 1.0 - dt / tau)

    # 再収束が追いつかない場合の保険。跳びの直後だけこの上限に触れる。
    max_xy = float(getattr(p, "loc_max_err_m", 0.6))
    max_yaw = float(getattr(p, "loc_max_err_rad", 0.3))
    err[:, :2] = err[:, :2].clamp(-max_xy, max_xy)
    err[:, 2] = err[:, 2].clamp(-max_yaw, max_yaw)
    env._gk_loc_err = err


def gk_self_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """自己位置推定の誤差 (N, 3) = [dx, dy, dyaw]。真値 + これ = ロボットが信じている自己位置。"""
    _gk_loc_tick(env)
    return env._gk_loc_bias + env._gk_loc_err


def robot_pose_est(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """ロボットが**自分で推定している**位置 (N,2) と heading (N,) を返す。

    実機のポリシーが使えるのはこれであって真値ではない。目標計算・指令生成・観測は
    すべてこちらを通す (報酬と critic だけが真値を使う)。
    """
    robot: Articulation = env.scene["robot"]
    e = gk_self_error(env)
    return robot_pos_goal(env)[:, :2] + e[:, :2], robot.data.heading_w + e[:, 2]


def compute_target_y(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.25,
    approach_vx_threshold: float = -0.05,
    use_perceived: bool = False,
) -> torch.Tensor:
    """ロボットが向かうべき目標 y 座標 (ゴール座標系) を返す (N,)。

    ★ ここで使う速度はビジョンから貰った値ではない。実機のビジョンは位置しか出さず、
      後段の PF も状態が [x, y] だけなので、``use_perceived=True`` の速度は
      :func:`_gk_perception_tick` の α-β フィルタが位置履歴から推定したものになる。
      実機でも同じ計算を自分の ROS2 ノードで再現できる。

    到達点予測 vs 位置追従 (2026-08-03 モンテカルロ、実測の知覚ノイズ込み・
    セーブ判定 0.5m):

        到達点予測 (この実装)                96.5%
        位置追従 (y_pred を pos[:,1] に置換)  92.7%

      以前「位置追従でも同等」と記録していたが、それは遅い球 (0.5〜1.2 m/s) かつ
      ノイズなしでの測定だった。現在のボール速度 (最大 2.35 m/s) では、ボールの横速度が
      キーパーの 1.3 m/s を超える球が 13% あり、現在位置を追う方式は原理的に間に合わない。
      位置追従へ落とす場合は下の ``y_pred`` を ``pos[:, 1].clamp(-max_y, max_y)`` に
      置き換え、:func:`task_drive_vector` の除数を固定時定数 (0.2s) に変えること。

    * ボール非アクティブ (ステージ1): ``_gk_target_y`` バッファのランダム目標。
    * ボールがゴールへ接近中 (vx < approach_vx_threshold): 現在のボール位置・速度から
      **守備面 (x = guard_x、ロボットの定位置)** への到達 y を外挿して返す
      (±max_y にクランプ)。ロボットはゴールラインの 0.3m 前で守るので、予測面も
      ゴールライン (x=0) ではなく守備面に合わせる (斜めの球ほどズレるため必須)。
      転がりの減速は進行方向に沿って一様に効くため軌道は直線のままであり、
      「到達するなら到達点 y」は等速外挿と一致する。
    * ボールが接近していない (弾いた後・停止後): **その場に留まる** (現在の自分の y)。

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

    # 脅威が無いときは自分の現在 y をそのまま目標にする。
    # ゼロ (ゴール中央) にすると中央復帰の指令が出てしまうため。
    # policy 側は推定した自己位置しか使えない (実機と同じ)。報酬・critic は真値。
    robot_y = (robot_pose_est(env)[0][:, 1] if use_perceived else robot_pos_goal(env)[:, 1])
    target = torch.where(approaching, y_pred, robot_y)
    return torch.where(bufs["ball_active"], target, bufs["target_y"])


_T_IDLE: float = 1.0
"""脅威が無いときの名目時間 [s]。位置ずれ [m] を速度 [m/s] に換算する除数。

1.0 にしてあるのは意図的: この状況では「ずれ ÷ 1.0」が恒等変換になるので、
定位置復帰の挙動も :func:`task_drive_phase_obs` の停止しきい値も、
必要速度化の前後で数値が変わらない (変更の影響をボール接近時だけに限定できる)。
"""


def guard_arrival_horizon(
    env: "ManagerBasedRLEnv",
    approach_vx_threshold: float = -0.05,
    use_perceived: bool = False,
    t_min: float = 0.25,
    t_idle: float = _T_IDLE,
) -> torch.Tensor:
    """ボールが守備面 (x = guard_x) に到達するまでの猶予時間 [s] (N,)。

    :func:`compute_target_y` が内部で計算しているのと同じ ``t`` を、
    「あとどれだけ時間があるか」として外に出したもの。
    :func:`task_drive_vector` が **位置ずれを必要速度に変換する** のに使う。

    Args:
        t_min: 下限クランプ [s]。到達直前に t→0 となって必要速度が発散するのを防ぐ。
        t_idle: ボールが脅威でないとき (弾いた後・非アクティブ) に使う名目時間 [s]。
            この値を 1.0 にしておくと、その状況では「位置ずれ [m] ÷ 1.0」= 従来と
            同じ数値になるので、定位置復帰の挙動と停止判定のしきい値が変わらない。
    """
    bufs = gk_buffers(env)
    guard_x = float(env.cfg.goalkeeper.guard_x)
    if use_perceived:
        pos, vel = _gk_perceived_goal_state(env)
    else:
        pos = ball_pos_goal(env)
        ball: RigidObject = env.scene["soccer_ball"]
        vel = ball.data.root_com_vel_w[:, :3]

    threat = (vel[:, 0] < approach_vx_threshold) & bufs["ball_active"]
    t = ((pos[:, 0] - guard_x) / (-vel[:, 0]).clamp(min=1e-3)).clamp(min=0.0)
    return torch.where(threat, t.clamp(min=float(t_min)), torch.full_like(t, float(t_idle)))


# ---------------------------------------------------------------------------


def _gk_perception(env: "ManagerBasedRLEnv"):
    """VirtualPerception インスタンスを (無ければ生成して) 返す。"""
    from .perception import VirtualPerception, soccer_vision_train_cfg

    vp = getattr(env, "_gk_vp", None)
    if vp is None or vp.num_envs != env.num_envs:
        cfg = soccer_vision_train_cfg()
        # PLAY 用クリーン化: 検出100%・遅延0・ノイズ0・見失い(occlusion/dead-zone/blind)なし・
        # 50Hz。キーパーの動きそのものを純粋に評価するため (学習では False)。
        p = env.cfg.goalkeeper
        # 実機のボール位置は姿勢誤差→地面投影で決まる (等方ガウスではない)。
        cfg.attitude_noise = bool(getattr(p, "perc_attitude_noise", True))
        cfg.attitude_bias_deg_range = tuple(getattr(p, "perc_attitude_bias_deg", (0.0, 1.2)))
        cfg.attitude_osc_deg_range = tuple(getattr(p, "perc_attitude_osc_deg", (0.0, 1.5)))
        cfg.attitude_osc_hz_range = tuple(getattr(p, "perc_attitude_osc_hz", (1.2, 2.0)))
        # 実機カメラ: fx=208.26、384px 入力で 5m のボールが約 9px。ここが検出限界。
        cfg.focal_px = 208.3
        cfg.max_detection_range = 5.0
        cfg.fov_h_deg = 105.0
        cfg.fov_v_deg = 94.0
        if bool(getattr(p, "perception_clean", False)):
            cfg.detection_prob_in_fov_range = (1.0, 1.0)
            cfg.blind_prob = 0.0
            cfg.occlusion_prob = 0.0
            cfg.deadzone_prob = 0.0
            cfg.noise_a = 0.0
            cfg.noise_b = 0.0
            cfg.attitude_noise = False
            cfg.pixel_noise_px = 0.0
            cfg.latency_mean_range = (0.0, 0.0)
            cfg.latency_std_s = 0.0
            cfg.update_hz_mean_range = (1.0 / float(env.step_dt), 1.0 / float(env.step_dt))
            cfg.update_hz_std = 0.0
        cfg.head_tracks_ball = True   # 戦略層がボールを追う前提。FOV は実質常にパス
        # K1_locomotion.urdf では頭リンク (Head_2→Head_1→Trunk) が全て Trunk にマージされ、
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
        # α-β フィルタの状態 (ゴール座標系)。実機の ROS2 ノードに置くものと同一。
        env._gkp_fpos = torch.zeros(env.num_envs, 2, device=env.device)
        env._gkp_fvel = torch.zeros(env.num_envs, 2, device=env.device)
        env._gkp_finit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._gkp_dt_meas = torch.zeros(env.num_envs, device=env.device)
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
    """VirtualPerception を 1 制御ステップ 1 回だけ進め、α-β フィルタを回す (冪等)。

    ★ ボール速度は **観測できない**。実機のビジョンは位置しか出さず、後段の PF も
      状態が [x, y] だけで速度を持たない (2026-08-03 実機コード調査)。そこで位置の
      時系列から α-β フィルタで速度を作る。ビジョン側の改修は不要で、実機では
      同じ 4 行を ROS2 ノードに置くだけで再現できる。

      フィルタは**ゴール座標系**で回す。ベース座標系のまま差分を取ると、キーパー
      自身の横移動が混ざって「ボールの速度」にならないため。ゴール座標系への変換は
      自己位置推定を使うので、その誤差も一緒に入る (実機と同じ経路)。

      カルマンフィルタも試したが結果は同じ (セーブ成立 96.5% で一致) だった。
      距離依存の観測ノイズにゲインを合わせる利点は、ボールが接近してノイズが自然に
      縮むぶんで相殺される。60 行 vs 6 行なので α-β を採る。
    """
    vp = _gk_perception(env)
    step = int(env.common_step_counter)
    if env._gkp_step == step:
        return
    env._gkp_step = step

    ball: RigidObject = env.scene["soccer_ball"]
    vp.update(env.scene["robot"], ball.data.root_pos_w[:, :3])

    p = env.cfg.goalkeeper
    dt = float(env.step_dt)
    alpha = float(getattr(p, "filter_alpha", 0.35))
    beta = float(getattr(p, "filter_beta", 0.05))

    # 観測: 知覚後のボール相対位置を、推定した自己位置でゴール座標系へ変換する。
    rpos, heading = robot_pose_est(env)
    c, s = torch.cos(heading), torch.sin(heading)
    rel = vp.ball_pos_b
    z = torch.stack(
        [rpos[:, 0] + c * rel[:, 0] - s * rel[:, 1], rpos[:, 1] + s * rel[:, 0] + c * rel[:, 1]],
        dim=1,
    )

    # 予測は毎制御ステップ、更新は新しい検出が来たときだけ。検出器は約 25 Hz で
    # 制御は 50 Hz なので、保持された値で更新すると同じ観測を二重に重み付けする。
    env._gkp_fpos = env._gkp_fpos + env._gkp_fvel * dt
    env._gkp_dt_meas = env._gkp_dt_meas + dt

    fresh = vp.fresh > 0.5
    active = gk_buffers(env)["ball_active"]
    upd = fresh & active
    dt_meas = env._gkp_dt_meas.clamp(dt, 0.4).unsqueeze(1)
    r = z - env._gkp_fpos
    fpos = env._gkp_fpos + alpha * r
    fvel = env._gkp_fvel + beta * r / dt_meas

    # 初回の検出はフィルタの初期化 (速度ゼロ)。ボール非アクティブ時も状態を落とす。
    first = upd & (~env._gkp_finit)
    env._gkp_fpos = torch.where(upd.unsqueeze(1), torch.where(first.unsqueeze(1), z, fpos), env._gkp_fpos)
    env._gkp_fvel = torch.where(
        upd.unsqueeze(1), torch.where(first.unsqueeze(1), torch.zeros_like(fvel), fvel), env._gkp_fvel
    )
    env._gkp_finit = (env._gkp_finit | upd) & active
    env._gkp_dt_meas = torch.where(upd, torch.zeros_like(env._gkp_dt_meas), env._gkp_dt_meas)
    env._gkp_fpos = torch.where(active.unsqueeze(1), env._gkp_fpos, torch.zeros_like(env._gkp_fpos))
    env._gkp_fvel = torch.where(active.unsqueeze(1), env._gkp_fvel, torch.zeros_like(env._gkp_fvel))


def _gk_filtered_rel(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """α-β フィルタの出力を base yaw frame へ戻して返す (位置 (N,2), 速度 (N,2))。

    実機のポリシーノードが ONNX に渡すのと同じ量。自己位置の推定値で逆変換するので、
    ゴール座標系へ行って戻る往復で自己位置誤差はほぼ相殺される (ヨー誤差だけ残る)。
    """
    _gk_perception_tick(env)
    rpos, heading = robot_pose_est(env)
    c, s = torch.cos(heading), torch.sin(heading)
    dx = env._gkp_fpos[:, 0] - rpos[:, 0]
    dy = env._gkp_fpos[:, 1] - rpos[:, 1]
    pos_b = torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=1)
    v = env._gkp_fvel
    vel_b = torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=1)
    mask = _gk_perception(env).ball_mask.unsqueeze(1)
    return pos_b * mask, vel_b * mask


def _gk_perceived_goal_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """α-β フィルタ後のボール位置・速度をゴール座標系 (N,3) で返す (z は 0 埋め)。

    フィルタ自体がゴール座標系で回っているので、そのまま取り出すだけ。自己位置推定の
    誤差は変換の時点で既に入っている。到達予測 (policy 用) が使う。
    """
    _gk_perception_tick(env)
    zero = torch.zeros(env.num_envs, device=env.device)
    pos = torch.stack([env._gkp_fpos[:, 0], env._gkp_fpos[:, 1], zero], dim=1)
    vel = torch.stack([env._gkp_fvel[:, 0], env._gkp_fvel[:, 1], zero], dim=1)
    return pos, vel


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
    """α-β フィルタ後のボール相対位置 (base yaw frame, 2D)。policy 用。

    生の検出値ではなくフィルタ出力を渡す。実機の PF は lpf_alpha=1.0 で平滑化が
    無効になっており生観測がほぼ素通しなので、平滑化はこちら側の責任になる。
    見えていない (mask=0) / 非アクティブ時は 0。"""
    return _gk_filtered_rel(env)[0]


def gk_ball_vel_perceived(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """α-β フィルタが位置履歴から推定したボール速度 (base yaw frame, 2D)。policy 用。

    ビジョンから速度は貰えない (実機の PF は状態が [x, y] だけ)。ここは位置の
    時系列だけから作った推定値で、実機でも同じ計算で再現できる。"""
    return _gk_filtered_rel(env)[1]


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
    vy_scale: float = 1.3,
    use_perceived: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ステージ2/3 で ``velocity_commands`` スロットに入れる「移動要求」ベクトル (N, 3)。

    ステージ1 では同じスロットに locomotion の速度コマンド (vx, vy, wz) が入る。
    ステージ2/3 には外部から与えられるコマンドが無いので、タスク状態から等価な
    「どちらへどれだけ速く動きたいか」を作って同じスロットに入れる:

        [0] = 守備面 (guard_x) へ戻る速度        → ステージ1 の vx と同じ [m/s]
        [1] = 目標 y へ間に合わせるのに要る横速度 → ステージ1 の vy と同じ [m/s]
        [2] = 正面へ戻す角速度                   → ステージ1 の wz と同じ [rad/s]

    ★ 2026-07-31: [1] を **位置ずれ [m] から必要速度 [m/s] に変更** した。
      旧実装は ``dy = 目標y − 自分のy`` をそのまま入れていたが、ステージ1 が学んだのは
      「スロットの値 = 出すべき速度 [m/s]」であり、**単位が違うものを渡していた**。
      その結果:
        * 目標までのズレは通常 0.3〜0.8m (ゴール幅 ±1.25m の内側) なので、
          指令される速度も 0.3〜0.8 m/s にしかならず、ステージ1 で獲得した
          1.18 m/s の能力を一度も使っていなかった (実測: 横移動が遅い)。
        * 目標に近づくほどズレが縮んで **さらに減速** し、到達直前が最も遅かった。
        * 結果としてボールに間に合わず、通過してから追いかける挙動になっていた
          (通過後は到達時間 t が 0 にクランプされ、予測点＝ボールの現在位置になるため)。
      距離を到達猶予時間で割れば「間に合わせるのに必要な速度」になり、
      速い球ほど大きな値 = 全力、遅い球は小さな値 = 落ち着いて、と自然に切り替わる。

    ★ [0] と [2] は従来通り「位置/向きのずれ」を ``t_idle`` (=1.0s) で割る。
      これは定位置維持の項であり、ボールの到達時間で割ると球が近いときに前後へ
      突っ込む挙動になる。1.0 で割るので数値は従来と同じまま、単位だけ揃う。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    guard_x = float(env.cfg.goalkeeper.guard_x)
    # policy 側は推定した自己位置しか使えない (実機と同じ)。報酬・critic は真値。
    if use_perceived:
        xy, heading = robot_pose_est(env)
    else:
        xy, heading = robot_pos_goal(env, asset_cfg)[:, :2], robot.data.heading_w

    # ボール到達までの猶予時間 [s]。脅威が無いときは t_idle=1.0 が返るので、
    # その状況では下の除算が恒等変換になり従来の挙動と一致する。
    horizon = guard_arrival_horizon(env, use_perceived=use_perceived)

    dx = ((guard_x - xy[:, 0]) / _T_IDLE).clamp(-vx_scale, vx_scale)
    dy = ((compute_target_y(env, max_y=max_y, use_perceived=use_perceived) - xy[:, 1]) / horizon).clamp(
        -vy_scale, vy_scale
    )
    # heading_w は +x を 0 とする world yaw。フィールド正面へ戻す向きを渡す。
    dyaw = ((-heading) / _T_IDLE).clamp(-1.0, 1.0)
    return torch.stack([dx, dy, dyaw], dim=1)


def task_drive_phase_obs(
    env: "ManagerBasedRLEnv",
    phase_freq: float = 1.6,
    cmd_threshold: float = 0.12,
    max_y: float = 1.25,
    vx_scale: float = 1.0,
    vy_scale: float = 1.3,
    use_perceived: bool = True,
) -> torch.Tensor:
    """ステージ2/3 の ``gait_phase`` スロット (4 次元)。タスク駆動の歩行位相。

    locomotion の :func:`phase_obs` と同一フォーマット (左右 sin/cos、停止時はゼロ埋め)
    だが、停止判定を ``base_velocity`` コマンドではなく :func:`task_drive_vector` の
    **並進成分 (dx, dy)** で行う。

    ★ 向き成分 (dyaw) は判定に含めない。足踏みでは向きは直らない (その場旋回が要る) のに、
      yaw drift は実測で恒常的に 7〜12° あり、それだけでしきい値を超えて
      「定位置にいるのに歩き続ける」状態になっていたため。
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
    drive_norm = torch.norm(drive[:, :2], dim=1, keepdim=True)
    return torch.where(drive_norm < cmd_threshold, torch.zeros_like(phase), phase)


def gk_self_state(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_perceived: bool = False,
) -> torch.Tensor:
    """自機のゴール座標系状態 (N, 4): (x オフセット, y オフセット, sin(yaw), cos(yaw))。

    yaw=0 はフィールド側 (+x, ボールが来る方向) を向いた姿勢。

    ``use_perceived=True`` なら自己位置推定 (実機は MCL) の誤差込み。policy 用。
    critic は既定の真値版を使う。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    if use_perceived:
        xy, heading = robot_pose_est(env)
    else:
        xy, heading = robot_pos_goal(env, asset_cfg)[:, :2], robot.data.heading_w
    return torch.stack([xy[:, 0], xy[:, 1], torch.sin(heading), torch.cos(heading)], dim=1)
