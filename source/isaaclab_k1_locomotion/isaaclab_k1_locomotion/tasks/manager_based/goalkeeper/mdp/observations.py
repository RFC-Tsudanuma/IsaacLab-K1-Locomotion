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
        # 今飛んでいる球が「到達不能球」(hard_ball_prob で混ぜたもの) か。
        # 適応カリキュラムはこの球での失点を成功率の集計から除外する。
        env._gk_hard_ball = torch.zeros(n, dtype=torch.bool, device=env.device)
        # 今飛んでいる球が「そのキーパー位置から物理的に取れない」か
        # (:func:`~.events._mark_unreachable` が発射時に幾何から判定する)。
        # hard_ball が確率で決め打ちするのに対し、こちらは実際の位置関係で決まる。
        env._gk_unreachable = torch.zeros(n, dtype=torch.bool, device=env.device)
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
        "unreachable": env._gk_unreachable,
        "hard_ball": env._gk_hard_ball,
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
        # 跳びがボール速度推定に漏れ込む分 (ゴール座標系 [m/s])。:func:`_gk_loc_tick` 参照。
        env._gk_loc_vel_leak = torch.zeros(n, 2, device=env.device)


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

    # ★ 跳びがボール速度に漏れ込む分を減衰させる (発火より先に進めること。発火した
    #   ステップの寄与がその場で 1 ステップ分減衰しないようにするため)。
    tau_v = float(getattr(p, "loc_vel_leak_tau_s", 0.79))
    if tau_v > 0.0:
        env._gk_loc_vel_leak *= math.exp(-dt / tau_v)

    jump = torch.rand(env.num_envs, device=env.device) < env._gk_loc_jump_p
    if bool(jump.any()):
        jmag = float(getattr(p, "loc_jump_m", 0.5))
        jyaw = float(getattr(p, "loc_jump_rad", 0.2))
        delta = torch.empty(env.num_envs, 3, device=env.device).uniform_(-1.0, 1.0)
        delta[:, :2] *= jmag
        delta[:, 2] *= jyaw
        err = torch.where(jump.unsqueeze(-1), err + delta, err)

        # ★ 2026-08-14: 自己位置の跳びを **ボール速度推定の過渡** に伝える。
        #   実機のボール速度は観測値ではなく、フィールド座標系のボール位置 (= 自己位置 +
        #   相対位置) を CVKF で微分した推定値なので、自己位置が跳ぶとボールが動いたのと
        #   区別が付かない。CVKF に自機移動の補償項は無く、原理的に分離できない。
        #
        #   応答の形は CVKF の確定パラメータ (measurement_noise_std=0.25m,
        #   process_acceleration_std=0.8m/s^2, NIS 閾値 9.21) から数値的に導出済み:
        #     * 位置カルマンゲイン 0.096 = 実効10フレーム平均の重い平滑化器
        #     * 0.5m のステップ入力 → ピーク 0.408 m/s、時定数 0.79s、約2秒の「山」
        #     * 係数 = 0.408 / 0.5 = 0.82 (25Hz/30Hz でほぼ不変)
        #     * 1.0m 超の跳びは NIS ゲート (実効 0.80m) で棄却されるので漏れない
        #
        #   ★ インパルスではなく「山」にすることが重要。1 フレームのスパイクを入れると
        #     「1 フレームだけ無視する」という実機に転移しない対処を学習してしまう。
        #     2 秒続く偽の「接近中」信号は、キーパーをポストまで走らせるのに十分な長さ。
        #
        #   なおドリフト由来の 1:1 漏れは実装しない。現行の loc_drift_xy_mps=0.03 では
        #   漏れも 0.03 m/s にしかならず、既存のエピソード固定バイアス
        #   (perc_vel_bias_range 0.5〜1.0 m/s) に埋もれる。「持続的に速度がずれている」
        #   という形は既にそちらでモデル化済み。
        gate = float(getattr(p, "loc_vel_leak_nis_gate_m", 0.80))
        coef = float(getattr(p, "loc_vel_leak_coef", 0.82))
        jump_xy = delta[:, :2]
        # NIS ゲートを通る跳びだけが速度に化ける (大きすぎる跳びは外れ値として棄却される)
        passed = jump & (torch.norm(jump_xy, dim=1) <= gate)
        env._gk_loc_vel_leak += torch.where(
            passed.unsqueeze(-1), coef * jump_xy, torch.zeros_like(jump_xy)
        )

    # ★ 再収束。MCL はランドマークが視野に入った時点で誤差が**戻る**。これが無いと
    #   ドリフトと跳びで誤差が上限に張り付いたままになり、ロボットが「自分は 0.8m
    #   ずれている」と信じ込んで歩き続ける (実測で out_of_bounds が 10 倍)。
    #   誤差のピーク値は変えず「張り付いたまま」を「一時的なズレ」にするだけなので
    #   DR は弱まらない。
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


def post_save_hold(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """「セーブ後の静止保持中」を表す bool マスク (N,)。

    ★ 2026-08-11 追加 (ユーザー指示): セーブ後に中央へ戻る動作をやめ、**止めた地点で
      初期姿勢のまま数秒間立たせる**ためのゲート。転倒しないかを目視・数値で確認する
      のが目的。

    区間は ``touched`` (ボールに触れた) が立ってから次の球が発射されるまで:

        タッチ → (save_delay_steps = 100 step = 2.0s) → セーブ確定
              → (respawn_delay_steps = 50 step = 1.0s) → 次の球

    合計 約3.0秒。``touched`` は :func:`~.events.relaunch_ball_after_save` が
    次の球の発射時に False へ戻すので、この 1 つのフラグで区間全体を覆える
    (``ball_active`` はセーブ確定時に False になってしまうので区間の途中で切れる)。
    """
    return gk_buffers(env)["touched"]


def compute_target_y(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
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
    out = torch.where(bufs["ball_active"], target, bufs["target_y"])

    # ★ 2026-08-11: セーブ後は中央 (y=0) ではなく **今いる場所** を目標にする。
    #   従来は接近中でない/非アクティブのとき目標 0 = ゴール中央へ戻る動作になっていた。
    #   自機の y をそのまま返せば task_drive_vector の dy = (目標 - 現在)/horizon が 0 に
    #   なり、横指令が出ない = その場に留まる。:func:`post_save_hold` 参照。
    if use_perceived:
        self_y = robot_pose_est(env)[0][:, 1]
    else:
        self_y = robot_pos_goal(env)[:, 1]
    return torch.where(post_save_hold(env), self_y, out)


_T_IDLE: float = 1.0
"""脅威が無いときの名目時間 [s]。位置ずれ [m] を速度 [m/s] に換算する除数。

1.0 にしてあるのは意図的: この状況では「ずれ ÷ 1.0」が恒等変換になるので、
定位置復帰の挙動も :func:`task_drive_phase_obs` の停止しきい値も、
必要速度化の前後で数値が変わらない (変更の影響をボール接近時だけに限定できる)。
"""


_T_FAST: float = 0.15
"""ボールが接近しているときに位置ずれ [m] を速度 [m/s] に換算する除数 [s]。

★ 2026-08-08: 「到達猶予時間で割る」方式をやめ、この固定値に変更した (ユーザー判断)。

旧方式は ``ずれ ÷ 到達猶予時間`` で「間に合う最小限の速度」を出していたが、
指令が即座に実速度になる (加速時間ゼロ) 前提だった。実際は静止から横 1.3 m/s に
乗るまで 2 歩 (約 0.6s) かかるため、全力指令が出るのが残り 0.5s では間に合わない。

固定値にすると ``ずれ > _T_FAST × 1.3 = 0.195m`` で常に全力になる。ボール接近中の
ずれは通常 0.3〜0.8m なので、**実質的に常時全力**。0 ではなく 0.15 にしてあるのは、
目標のごく近傍 (±0.195m) だけ速度を落として**停止できる余地を残す**ため。
完全な bang-bang にすると目標を通過し続けて振動する。

小さくするほど全力域が広がる (0.10 → 0.13m 以上で全力)。
"""


def guard_arrival_horizon(
    env: "ManagerBasedRLEnv",
    approach_vx_threshold: float = -0.05,
    use_perceived: bool = False,
    t_min: float = 0.25,
    t_idle: float = _T_IDLE,
) -> torch.Tensor:
    """位置ずれ [m] を速度 [m/s] に換算するための除数 [s] (N,)。

    ★ 2026-08-08 に意味が変わった。以前は「守備面への到達猶予時間」そのものを返して
      いたが、現在は **ボール接近中は固定値 ``_T_FAST`` を返す** (最速で向かう)。
      到達猶予時間そのものが要る場合は :func:`compute_target_y` 内の計算を参照。

    * ボールが接近中 (脅威あり): ``_T_FAST`` (= 0.15s)。ずれ 0.195m 以上で全力。
    * 脅威なし (弾いた後・非アクティブ): ``t_idle`` (= 1.0s)。

    脅威なし側を 1.0 のままにしてあるのは意図的で、この状況では「ずれ ÷ 1.0」が
    恒等変換になり、定位置復帰の挙動と :func:`task_drive_phase_obs` の停止しきい値
    (0.12) が従来と同じ数値のままになる。変更の影響をボール接近時だけに限定できる。

    Args:
        t_min: 未使用 (旧「到達直前の発散防止クランプ」。互換のため引数だけ残す)。
        t_idle: ボールが脅威でないときに使う名目時間 [s]。
    """
    bufs = gk_buffers(env)
    if use_perceived:
        _, vel = _gk_perceived_goal_state(env)
    else:
        ball: RigidObject = env.scene["soccer_ball"]
        vel = ball.data.root_com_vel_w[:, :3]

    threat = (vel[:, 0] < approach_vx_threshold) & bufs["ball_active"]
    t_fast = float(getattr(env.cfg.goalkeeper, "drive_t_fast", _T_FAST))
    return torch.where(
        threat,
        torch.full_like(vel[:, 0], t_fast),
        torch.full_like(vel[:, 0], float(t_idle)),
    )


# ---------------------------------------------------------------------------


def _gk_perception(env: "ManagerBasedRLEnv"):
    """VirtualPerception インスタンスを (無ければ生成して) 返す。"""
    from .perception import VirtualPerception, soccer_vision_train_cfg

    vp = getattr(env, "_gk_vp", None)
    if vp is None or vp.num_envs != env.num_envs:
        cfg = soccer_vision_train_cfg()
        p = env.cfg.goalkeeper
        # ビジョンの更新レート [Hz]。preset の既定は 25.36Hz 固定だが、cfg から
        # 上書きできるようにして DR の幅を持たせる (2026-08-08)。カメラは 30fps 出ても
        # 検出処理の取りこぼしで実効レートは落ちるので、上限は 25Hz に置く。
        cfg.update_hz_mean_range = tuple(
            float(v) for v in getattr(p, "perc_update_rate_hz", (20.0, 25.0))
        )
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

    # 速度: 真値 (base yaw frame) + エピソード固定バイアス + 自己位置の跳び由来の過渡。
    # 見えていない (mask=0) 間は 0。
    _, vel_b = _gk_true_rel_state(env)
    mask = vp.ball_mask.unsqueeze(1)

    # ★ 2026-08-14: 自己位置の跳びがボール速度推定に化ける分を加える。
    #   実機のボール速度はフィールド座標系で推定されるので、漏れもフィールド座標系で
    #   発生する (:func:`_gk_loc_tick`)。観測は base yaw frame なので、実機が
    #   local_velocity = R(-theta) * global_velocity で回し戻すのと同じ変換をする。
    #   使う角度は **推定 heading** (実機のポリシーが使えるのはそれだけ)。
    _gk_loc_tick(env)
    leak_g = env._gk_loc_vel_leak
    _, heading_est = robot_pose_est(env)
    c, s = torch.cos(heading_est), torch.sin(heading_est)
    leak_b = torch.stack(
        [c * leak_g[:, 0] + s * leak_g[:, 1], -s * leak_g[:, 0] + c * leak_g[:, 1]], dim=1
    )

    env._gkp_out_vel = (vel_b + env._gkp_vel_bias + leak_b) * mask


def _gk_perceived_goal_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """知覚DR後のボール位置・速度をゴール座標系 (N,3) で返す (z は 0 埋め)。

    VirtualPerception の相対位置 (base yaw frame) を、ロボットが**自分で推定している**
    姿勢 (:func:`robot_pose_est`) でゴール座標系へ戻す。到達予測 (policy 用) が使う。
    実機と同じく、ここでボール観測に自己位置推定の誤差が合流する。
    """
    vp = _gk_perception(env)
    _gk_perception_tick(env)
    rpos, heading = robot_pose_est(env)
    c, s = torch.cos(heading), torch.sin(heading)
    rel = vp.ball_pos_b
    off_x = c * rel[:, 0] - s * rel[:, 1]
    off_y = s * rel[:, 0] + c * rel[:, 1]
    pos = torch.stack([rpos[:, 0] + off_x, rpos[:, 1] + off_y, torch.zeros_like(off_x)], dim=1)
    v = env._gkp_out_vel
    vel = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], torch.zeros_like(off_x)], dim=1)
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
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
    use_perceived: bool = False,
) -> torch.Tensor:
    """目標 y 座標 (ゴール座標系) の観測 (N, 1)。:func:`compute_target_y` 参照。

    policy には ``use_perceived=True`` (知覚DR後のボール状態から予測)、
    critic には既定の真値版を使う。
    """
    return compute_target_y(env, max_y=max_y, use_perceived=use_perceived).unsqueeze(1)


def zmp_xy_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ZMP の水平位置を **自機の base yaw frame** で返す (N, 2)。critic 観測用。

    ★ 2026-08-09: locomotion の :func:`compute_zmp_xy` をそのまま critic 観測に
      入れていたが、あれは **world 絶対座標** を返す。IsaacLab の ``*_w`` は env 原点
      オフセットを含むため、env_spacing 6.0 × 4096 env では ±190m のレンジになり、
      本来見たい ±0.3m の ZMP 変位は全体の 0.1% 未満に埋もれる。critic の観測
      正規化 (critic_obs_normalization) は全 env 共通の統計で割るので、正規化後の
      有効信号は ~0.003 まで潰れ、実質「env ID を表すだけの定数」になっていた。

      ここでは自機位置を引いて base yaw frame へ回すことで、
        * env 原点オフセットが消えて信号が本来のスケールに戻る
        * 左右反転が y の符号反転だけで表せる (mirror / data augmentation が可能になる)
      の両方を満たす。
    """
    from ...locomotion.mdp.rewards import compute_zmp_xy

    robot: Articulation = env.scene[asset_cfg.name]
    rel = compute_zmp_xy(env, asset_cfg) - robot.data.root_pos_w[:, :2]
    rel3 = torch.cat([rel, torch.zeros_like(rel[:, :1])], dim=1)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel3)[:, :2]


def zeros_obs(env: "ManagerBasedRLEnv", dim: int = 1) -> torch.Tensor:
    """常にゼロを返すダミー観測 (N, dim)。

    直接制御版のステージ1 (ボール不在) でボール系スロットの次元を確保するために使う。
    ゼロ入力の列には勾配が流れないので、該当する重みは初期値のままステージ2 へ渡る
    (ball_kick と同じ「次元一致方式」)。
    """
    return torch.zeros(env.num_envs, int(dim), device=env.device)


def task_drive_vector(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
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

    ★ 2026-08-08: [1] の換算を **「到達猶予時間で割る」から「固定値 (_T_FAST=0.15s)
      で割る」に変更** した (ユーザー判断: 最速でボールを止める)。
      旧方式は「間に合う最小限の速度」を出す設計だったが、指令が即座に実速度になる
      前提 (加速時間ゼロ) だった。実際は静止から横 1.3 m/s に乗るまで約 0.6s かかり、
      全力指令が出るのが残り 0.5s では立ち上がり切らない。固定値にすると
      ずれ 0.195m 以上で常に全力になる (接近中のずれは通常 0.3〜0.8m)。
      到達猶予時間そのものは :func:`compute_target_y` の中で引き続き使う
      (予測点 y_pred の計算には必須)。

    ★ 2026-07-31 (旧): [1] を **位置ずれ [m] から必要速度 [m/s] に変更** した。
      旧実装は ``dy = 目標y − 自分のy`` をそのまま入れていたが、ステージ1 が学んだのは
      「スロットの値 = 出すべき速度 [m/s]」であり、**単位が違うものを渡していた**。
      その結果:
        * 目標までのズレは通常 0.3〜0.8m (ゴール幅 ±1.3m の内側) なので、
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
        pos_xy, heading = robot_pose_est(env)
    else:
        pos_xy, heading = robot_pos_goal(env, asset_cfg)[:, :2], robot.data.heading_w
    pos = pos_xy

    # ボール到達までの猶予時間 [s]。脅威が無いときは t_idle=1.0 が返るので、
    # その状況では下の除算が恒等変換になり従来の挙動と一致する。
    horizon = guard_arrival_horizon(env, use_perceived=use_perceived)

    dx = ((guard_x - pos[:, 0]) / _T_IDLE).clamp(-vx_scale, vx_scale)
    dy = ((compute_target_y(env, max_y=max_y, use_perceived=use_perceived) - pos[:, 1]) / horizon).clamp(
        -vy_scale, vy_scale
    )
    # heading は +x を 0 とする world yaw。フィールド正面へ戻す向きを渡す。
    dyaw = ((-heading) / _T_IDLE).clamp(-1.0, 1.0)
    drive = torch.stack([dx, dy, dyaw], dim=1)

    # ★ 2026-08-11: セーブ後の保持区間は指令を完全にゼロにする (ユーザー指示)。
    #   dy は compute_target_y 側で既に 0 になるが、dx (守備面へ戻る) と dyaw
    #   (正面へ向き直す) が残ると「止めた地点で立つ」にならないので3成分とも落とす。
    #   ノルム 0 < cmd_threshold なので task_drive_phase_obs も位相をゼロ埋めし、
    #   ステージ1 で学習済みの「停止」挙動がそのまま出る。
    return torch.where(post_save_hold(env).unsqueeze(1), torch.zeros_like(drive), drive)


def task_drive_phase_obs(
    env: "ManagerBasedRLEnv",
    phase_freq: float = 1.6,
    cmd_threshold: float = 0.12,
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
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
