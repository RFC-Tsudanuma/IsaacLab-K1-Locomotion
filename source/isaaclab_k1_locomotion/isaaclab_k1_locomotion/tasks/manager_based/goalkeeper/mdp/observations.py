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

    ★ 2026-08-15: ``GoalkeeperParamsCfg.post_save_hold_until_relaunch = False`` にすると
      **セーブ確定までで保持を終える** (``save_cd >= 0`` の区間だけ True)。残りの
      respawn 待ち時間は保持が解けるので、``compute_target_y`` が目標 0 (中央) を返し、
      次の球に備えて中央へ復帰する動きが出る。継続モードでは 2 球目以降が必ず「前の球を
      止めた場所」から始まり、中央から 0.6m ずれるだけで到達不能球が 33% → 42% に増える
      ため、復帰の有無はセーブ率に大きく効く (cfg 側のコメント参照)。
      既定は True = 従来どおり。
    """
    bufs = gk_buffers(env)
    touched = bufs["touched"]
    if bool(getattr(env.cfg.goalkeeper, "post_save_hold_until_relaunch", True)):
        return touched
    # セーブ確定までのカウントダウン中だけ保持する。確定後 (save_cd < 0) は解放され、
    # 次の球までの待ち時間が中央への復帰に使える。
    return touched & (bufs["save_cd"] >= 0)


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
        # ★ 2026-08-16: 生の 1 フレーム観測ではなく、直近ウィンドウの最小二乗フィットを使う
        #   (ball_fit_window_s > 0 のとき)。詳細は :func:`_gk_fitted_goal_state`。
        pos, vel = _gk_fitted_goal_state(env)
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


# --------------------------------------------------- ボール位置の時系列フィット
#
# ★ 2026-08-16 追加。**到達点予測が生の 1 フレーム観測を使っていた**のを直すもの。
#
#   VirtualPerception の位置ノイズは
#       sigma = noise_a * distance + noise_b     (実機準拠プリセットで 0.124d + 0.149)
#   で、``torch.randn_like`` によって **ビジョン更新のたびに独立にサンプル**される
#   (perception.py の該当行を参照)。エピソード固定なのはスケール係数の ±20% だけで、
#   ノイズ本体は白色。つまり **平均化すれば √N 分の 1 に減らせる**。
#
#   ところが ``compute_target_y`` の外挿
#       y_pred = pos_y + vel_y * t
#   は pos に生の 1 フレーム値を入れていた。飛翔時間 0.7s × ビジョン 20〜25Hz =
#   14〜17 サンプルぶんの情報があるのに 1 個しか使っていない、という状態:
#
#       速度上限 3.0 → sigma(p90) 0.51m   (タッチ判定半径 touch_proximity 0.50m と同等)
#       速度上限 6.0 → sigma(p90) 0.86m
#
#   これが適応カリキュラムが 2.07 m/s で止まった実体。予測誤差がタッチ半径に達すると
#   「予測した場所へ行っても届かない」が頻発し、成功率が昇格閾値に届かなくなる。
#
#   ボールは等速直線運動なので、直近ウィンドウの位置履歴に **直線を最小二乗フィット**
#   すれば位置と速度を同時に sigma/sqrt(N) の精度で得られる。学習不要・決定論的で、
#   **実機の C++ 側にも同じものをそのまま実装できる**。
#
#   ★ 既定は無効 (``ball_fit_window_s = 0.0``)。有効にすると到達点予測の入力が変わるので、
#     既存タスク・既存 ckpt の挙動を変えないため opt-in にしてある。
#   ★ 有効にしたら **実機側にも同じフィットを実装すること**。シムだけ賢くすると
#     そのぶんまるごと sim-to-real ギャップになる。


def _gk_fit_buffers(env: "ManagerBasedRLEnv", capacity: int) -> None:
    """フィット用のリングバッファを (無ければ) 確保する。"""
    n = env.num_envs
    buf = getattr(env, "_gk_fit_buf", None)
    if buf is None or buf.shape[0] != n or buf.shape[1] != capacity:
        # [x, y, valid] をゴール座標系で保持する
        env._gk_fit_buf = torch.zeros(n, capacity, 3, device=env.device)
        env._gk_fit_ptr = 0
        env._gk_fit_step = -1
        env._gk_fit_prev_active = torch.zeros(n, dtype=torch.bool, device=env.device)


def _gk_fit_tick(env: "ManagerBasedRLEnv", capacity: int) -> None:
    """位置履歴を 1 制御ステップ進める (冪等)。

    * **検出できたフレームだけ** 積む。``vp.ball_pos_b`` は mask=0 のとき 0 を返すので、
      それを混ぜるとフィットが原点へ引っ張られる。
    * 新しい球が発射された瞬間 (``ball_active`` の立ち上がり) に履歴を捨てる。
      前の球の軌道が残っていると直線フィットが致命的にずれる。
    """
    _gk_fit_buffers(env, capacity)
    step = int(env.common_step_counter)
    if env._gk_fit_step == step:
        return
    env._gk_fit_step = step

    bufs = gk_buffers(env)
    active = bufs["ball_active"]
    launched = active & (~env._gk_fit_prev_active)
    env._gk_fit_prev_active = active.clone()
    if bool(launched.any()):
        env._gk_fit_buf[launched] = 0.0

    pos, _ = _gk_perceived_goal_state(env)
    mask = _gk_perception(env).ball_mask            # 1 = 今フレーム検出できている
    slot = env._gk_fit_buf[:, int(env._gk_fit_ptr)]
    slot[:, 0] = pos[:, 0]
    slot[:, 1] = pos[:, 1]
    slot[:, 2] = mask * active.float()
    env._gk_fit_ptr = (int(env._gk_fit_ptr) + 1) % env._gk_fit_buf.shape[1]


def _gk_fitted_goal_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """位置ノイズを平均化して求めた現在位置と、既存の速度推定 (ゴール座標系, 各 (N,3))。

    方式: **速度推定で各サンプルを現在時刻へ引き戻してから平均する**::

        pos_now = Σ w_i (z_i − v_est · t_i) / Σ w_i        (t_i ≤ 0, 最新が t=0)

    ``z_i = pos_now + v·t_i + noise`` なので引き戻すと全サンプルが ``pos_now + noise`` に
    揃い、単純平均で分散が厳密に σ²/N になる。

    ★ 2 パラメータの直線フィットではなくこの形にした理由 (実測、白色ノイズ σ、25Hz):

        窓 0.5s   生 0.505m → 直線フィット 0.266m → **引き戻して平均 0.141m**
        窓 0.8s   生 0.514m → 直線フィット 0.229m → **引き戻して平均 0.117m**

      直線フィットは窓の**端** (= 現在時刻) で評価するため分散が中央の 4 倍になり、
      せっかくの平均化効果を半分捨てていた。

    ★ 速度は **フィットせず既存の推定をそのまま使う**。直線フィットから出る速度は
      窓 0.5s / σ=0.51 で誤差 0.95 m/s あり、既存推定 (真値 + perc_vel_bias 0.05〜0.15)
      より 1 桁悪い。実機の CVKF は運動モデル付きでもっと長い窓を使っているので、
      それを素朴な線形回帰で置き換えるのは改悪になる。

    ``ball_fit_window_s <= 0`` なら従来どおり生値を返す。有効サンプルが 2 未満の env
    (発射直後・見失い中) も生値へフォールバックする。
    """
    p = env.cfg.goalkeeper
    win_s = float(getattr(p, "ball_fit_window_s", 0.0))
    raw_pos, raw_vel = _gk_perceived_goal_state(env)
    if win_s <= 0.0:
        return raw_pos, raw_vel

    dt = float(env.step_dt)
    capacity = max(3, int(round(win_s / dt)))
    _gk_fit_tick(env, capacity)

    buf = env._gk_fit_buf
    ptr = int(env._gk_fit_ptr)
    ordered = buf if ptr == 0 else torch.cat([buf[:, ptr:], buf[:, :ptr]], dim=1)
    x, y, w = ordered[:, :, 0], ordered[:, :, 1], ordered[:, :, 2]

    # 最新フレームを t=0 とする相対時刻 (古いほど負)
    t = torch.arange(-(capacity - 1), 1, device=env.device, dtype=x.dtype) * dt

    s0 = w.sum(dim=1)
    ok = s0 >= 2.0
    safe_s0 = torch.where(ok, s0, torch.ones_like(s0))

    # 速度で現在時刻へ引き戻してから平均する
    px = (w * (x - raw_vel[:, 0:1] * t)).sum(dim=1) / safe_s0
    py = (w * (y - raw_vel[:, 1:2] * t)).sum(dim=1) / safe_s0

    zero = torch.zeros_like(px)
    pos = torch.stack(
        [torch.where(ok, px, raw_pos[:, 0]), torch.where(ok, py, raw_pos[:, 1]), zero], dim=1
    )
    return pos, raw_vel


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
    drive = torch.where(post_save_hold(env).unsqueeze(1), torch.zeros_like(drive), drive)

    # ★ 2026-08-15: 1 次ローパスで平滑化する。
    #
    #   この指令は自己位置 (MCL) から作られるので、MCL が跳ぶと不連続に飛ぶ。
    #   しかも dy = ずれ / 0.15s なので **0.195m 以上のずれで常に全力**になる設定で、
    #   MCL の跳び (±0.5m) は必ず「全力で横へ動け」に化ける。歩行中にこれが入ると
    #   急激な方向転換になり、位相を安定させても崩れる余地が残る。
    #   MuJoCo でランドマーク認識がカクついたときにロボットが揺れる件の、
    #   task_drive_phase_obs 側の修正では塞ぎきれない残りの経路。
    #
    #   本物のボールの動きは連続なのでローパスをほぼ素通りするが、MCL の跳びは
    #   1 フレームの不連続なので大きく減衰する。実機の vision_filter が CVKF で
    #   位置を平滑化しているのと同じ考え方で、むしろ実機の挙動に近づく。
    #
    #   ★ 実機の C++ 側にも同じフィルタを同じ時定数で実装すること。入れないと
    #     sim で学習した前提と挙動が変わる。
    #   ★ 状態を持つので reset_gk_buffers でクリアすること (前エピソードの指令が
    #     残ると開始直後に存在しない指令が出る)。
    tau = float(getattr(_gk_params_local(env), "drive_filter_tau_s", 0.12))
    if tau > 0.0:
        prev = getattr(env, _DRIVE_FILT_ATTR, None)
        if prev is None or prev.shape != drive.shape:
            prev = drive.clone()
            setattr(env, _DRIVE_FILT_ATTR, prev)
        # 同一ステップ内の複数回呼び出し (policy/critic/phase) では 1 回だけ更新する
        if getattr(env, _DRIVE_FILT_STEP_ATTR, -1) != int(env.common_step_counter):
            alpha = min(1.0, float(env.step_dt) / tau)
            prev = prev + alpha * (drive - prev)
            setattr(env, _DRIVE_FILT_ATTR, prev)
            setattr(env, _DRIVE_FILT_STEP_ATTR, int(env.common_step_counter))
        drive = getattr(env, _DRIVE_FILT_ATTR)

    return drive


_TASK_PHASE_ATTR = "_gk_task_gait_phase"
_TASK_PHASE_STEP_ATTR = "_gk_task_gait_phase_step"

# task_drive_vector の 1 次ローパス用の状態 (MCL の跳びが指令に化けるのを抑える)
_DRIVE_FILT_ATTR = "_gk_drive_filtered"
_DRIVE_FILT_STEP_ATTR = "_gk_drive_filtered_step"

# --- 歩行ゲートの sim2real ギャップを埋めるための状態 (2026-08-16) ---
#
# ★ 実機で振動した件への対処。歩行判定 (walking) の入力はシムでは
#   ``root_lin_vel_b`` = 物理エンジンの**真値**で、静止していれば厳密に 0 が返る。
#   実機は InEKF の推定値、しかも k1_odom_velocity_node が planar_odom の**位置を
#   有限差分**して作った値なので、静止していてもノイズで振れる。微分はノイズを増幅する。
#
#   閾値 0.15 にヒステリシスが無いため、推定値がその付近でディザすると
#   walk/stand がノイズの周期でトグルし、位相がゼロ埋めと再開を繰り返す
#   (= 脚が出かけては止まる = 振動)。真値で学習していると、この状況を一度も経験しない。
#
#   ここでは (a) 相関ノイズを載せて実機相当の入力にし、(b) ヒステリシスを入れて
#   トグル自体を起きにくくし、(c) **遅延**を入れる。
#   ★ (b) は実機の C++ 側にも同じ値で実装すること。
#
# ★ (c) の遅延がなぜ要るか (2026-08-16 追加。ノイズだけ入れたのは片手落ちだった)
#
#   このゲートは **閉ループの中**にある:
#       ゲート → gait_phase → 方策 → 関節 → ロボットの速度 → ゲート
#   出力が方策を動かし、方策がロボットを動かし、その速度がゲートに戻る。
#   しかも判定はハードしきい値 (0/1) なので、跨いだ瞬間に入力が飛ぶ。
#   **閉ループ + ハードしきい値 + 遅延 = 自励振動の典型条件**で、ノイズがゼロでも
#   振動する (歩き始めと止まり際は真値でも必ずしきい値を跨ぐため)。
#
#   シムで振動しないのは ``root_lin_vel_b`` が真値かつ **遅延ゼロ**だから。跨いだ
#   瞬間に方策が反応するので往復しない。実機のゲート入力は
#       InEKF 200Hz (5ms) + 差分の半サンプル (2.5ms) + LPF 5Hz の群遅延 (31.8ms)
#       + executor の遅延 = 40〜60ms = 制御 2〜3 tick
#   遅れている。遅延ゼロで学習した方策は「今の状態を見て今すぐ強く直す」制御則を
#   獲得するので、そこに遅延を入れると必ず行き過ぎて振動する (シャワーの温度調節と同じ)。
#
#   ★ 既存の DelayedPDActuatorCfg(min_delay=2, max_delay=7) とは **別の経路**。
#     あちらは「方策の出力 → 関節トルク」= ループの出力側。ここは
#     「ロボットの速度 → 歩行判定」= ループの入力側。片側だけ模擬しても
#     ループ全体の位相遅れが実機より小さいままで、sim では安定・実機では振動になる。
#
#   ★ DR は遅延を **消さない**。「遅れた情報しか来ない世界で安定する制御則」を
#     学ばせるためのもの。遅延の除去は推論側の仕事
#     (/k1_inekf/velocity → /k1_inekf/planar_odom の twist に替えて LPF の 31.8ms を省く)。
_BASE_VEL_NOISE_ATTR = "_gk_base_vel_noise"        # 相関ノイズの状態 (N, 2)
_BASE_VEL_NOISE_STEP_ATTR = "_gk_base_vel_noise_step"
_BASE_VEL_NOISE_SCALE_ATTR = "_gk_base_vel_noise_scale"  # env ごとの倍率 (N,)
_BASE_VEL_HIST_ATTR = "_gk_base_vel_hist"          # 遅延用リングバッファ (D, N, 2)
_BASE_VEL_HIST_POS_ATTR = "_gk_base_vel_hist_pos"  # リングバッファの書き込み位置 (int)
_BASE_VEL_HIST_STEP_ATTR = "_gk_base_vel_hist_step"
_BASE_VEL_DELAY_ATTR = "_gk_base_vel_delay"        # env ごとの遅延段数 (N,) long
_WALK_GATE_ATTR = "_gk_walk_gate"                  # ヒステリシスの保持状態 (N,) bool


def _gk_params_local(env: "ManagerBasedRLEnv"):
    """env cfg の GoalkeeperParamsCfg を返す (events からの import 循環を避けるため再定義)。"""
    return env.cfg.goalkeeper


def _delayed_base_vel(env: "ManagerBasedRLEnv", vel_b: torch.Tensor, delay_range: tuple) -> torch.Tensor:
    """ベース速度を env ごとの段数だけ遅らせて返す (N, 2)。

    実機のゲート入力は 40〜60ms (制御 2〜3 tick) 遅れている。シムは遅延ゼロなので、
    しきい値を跨いだ瞬間に方策が反応でき、閉ループが往復しない。この差を埋めないと
    「遅れた情報で判断する」状況を学習中に一度も経験しないまま実機に出ることになる。
    詳細は _BASE_VEL_* 定数群のコメント参照。

    リングバッファ長は max(delay_range)+1。段数は env ごとに固定 (個体差として扱い、
    エピソードリセットでは変えない)。0 段を含めると「遅延なし」の env も混ざるので、
    既定は 1〜3 段 = 20〜60ms にしてある。
    """
    lo, hi = int(delay_range[0]), int(delay_range[1])
    if hi <= 0:
        return vel_b
    depth = hi + 1

    hist = getattr(env, _BASE_VEL_HIST_ATTR, None)
    if hist is None or hist.shape != (depth, *vel_b.shape):
        # 立ち上がりは現在値で埋める (ゼロ埋めだと開始直後だけ「静止していた」ことになる)
        hist = vel_b.unsqueeze(0).repeat(depth, 1, 1).clone()
        setattr(env, _BASE_VEL_HIST_ATTR, hist)
        setattr(env, _BASE_VEL_HIST_POS_ATTR, 0)
        setattr(env, _BASE_VEL_HIST_STEP_ATTR, -1)

    delay = getattr(env, _BASE_VEL_DELAY_ATTR, None)
    if delay is None or delay.shape != (env.num_envs,):
        delay = torch.randint(lo, hi + 1, (env.num_envs,), device=env.device)
        setattr(env, _BASE_VEL_DELAY_ATTR, delay)

    pos = int(getattr(env, _BASE_VEL_HIST_POS_ATTR, 0))
    # 同一ステップ内の複数回呼び出し (policy/critic) では 1 回だけ書き込む
    if getattr(env, _BASE_VEL_HIST_STEP_ATTR, -1) != int(env.common_step_counter):
        pos = (pos + 1) % depth
        hist[pos] = vel_b
        setattr(env, _BASE_VEL_HIST_POS_ATTR, pos)
        setattr(env, _BASE_VEL_HIST_STEP_ATTR, int(env.common_step_counter))

    # env ごとに (pos - delay) の行を引く
    idx = (pos - delay) % depth                      # (N,)
    return hist[idx, torch.arange(env.num_envs, device=env.device)]


def _measured_base_speed(
    env: "ManagerBasedRLEnv",
    noise_amp: float,
    noise_tau_s: float,
    noise_scale_range: tuple[float, float],
    delay_range: tuple = (1, 3),
) -> torch.Tensor:
    """歩行ゲートの入力にする「実機相当のベース速度推定値」(N,)。

    シムの ``root_lin_vel_b`` は物理エンジンの真値で、静止していれば厳密に 0 になる。
    実機は InEKF の推定値で、しかも位置の有限差分なので静止していても振れる。
    その差を埋めないと「ノイズでゲートがトグルする」状況を学習中に一度も経験しない。

    ノイズは **時間相関** させる。実機側の推定値は 5Hz の LPF を通っており
    相関時間は τ = 1/(2π·5) ≒ 32ms ≒ 1.6 制御ステップある。白色ノイズだと方策が
    平均化して簡単に無視できてしまい、実機より易しい問題になる。

    ★ ノイズ幅は **実測ではない**。実機の静止時ノイズは未計測 (推論側リポジトリにも
      数値が存在しない)。既定 0.1 は「閾値 0.15 に対し、静止時の大きさ
      hypot(0.1, 0.1)=0.141 が閾値を超えない上限」として選んだ値。実機を 60 秒
      静止させて /k1_inekf/velocity を記録したら、その実測に置き換えること。
      env ごとに 0.5〜1.5 倍して振ってあるのは、実測が無いぶん幅を持たせるため。

    ★ 遅延 (``delay_range``) はノイズとは別の効き方をする。ノイズは「しきい値を
      跨ぎやすくする」だけだが、遅延は「跨いだ後の判断を狂わせる」ので、閉ループの
      振動に直結する。ノイズだけ入れても遅延起因の振動には耐性が付かない。

    Args:
        noise_amp: ノイズの振幅 [m/s]。0 で無効 (真値をそのまま返す)。
        noise_tau_s: 相関の時定数 [s]。
        noise_scale_range: env ごとの倍率の範囲。
        delay_range: ゲート入力の遅延段数の範囲 [制御 step]。実機実測 40〜60ms に
            対して 1〜3 step (20〜60ms)。(0, 0) で無効。
    """
    robot: Articulation = env.scene["robot"]
    vel_b = _delayed_base_vel(env, robot.data.root_lin_vel_b[:, :2], delay_range)
    if noise_amp <= 0.0:
        return torch.norm(vel_b, dim=1)

    scale = getattr(env, _BASE_VEL_NOISE_SCALE_ATTR, None)
    if scale is None or scale.shape != (env.num_envs,):
        lo, hi = float(noise_scale_range[0]), float(noise_scale_range[1])
        scale = torch.empty(env.num_envs, device=env.device).uniform_(lo, hi)
        setattr(env, _BASE_VEL_NOISE_SCALE_ATTR, scale)

    noise = getattr(env, _BASE_VEL_NOISE_ATTR, None)
    if noise is None or noise.shape != vel_b.shape:
        noise = torch.zeros_like(vel_b)
        setattr(env, _BASE_VEL_NOISE_ATTR, noise)
        setattr(env, _BASE_VEL_NOISE_STEP_ATTR, -1)

    # 同一ステップ内の複数回呼び出し (policy/critic) では 1 回だけ進める
    if getattr(env, _BASE_VEL_NOISE_STEP_ATTR, -1) != int(env.common_step_counter):
        alpha = min(1.0, float(env.step_dt) / max(noise_tau_s, 1e-6))
        # 1 次フィルタは振幅を sqrt(alpha/(2-alpha)) 倍に落とすので、その逆数を
        # 入力に掛けて出力の広がりが noise_amp 相当に保たれるようにする。
        gain = math.sqrt((2.0 - alpha) / alpha)
        white = (torch.rand_like(noise) * 2.0 - 1.0) * (noise_amp * gain)
        noise = noise + alpha * (white - noise)
        setattr(env, _BASE_VEL_NOISE_ATTR, noise)
        setattr(env, _BASE_VEL_NOISE_STEP_ATTR, int(env.common_step_counter))
    noise = getattr(env, _BASE_VEL_NOISE_ATTR)

    return torch.norm(vel_b + noise * scale.unsqueeze(1), dim=1)


def _walk_gate(env: "ManagerBasedRLEnv", speed: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """ヒステリシス付きの歩行判定 (N,) bool。

    停止中は ``speed >= hi`` で歩き出し、歩行中は ``speed < lo`` で止まる。
    単一しきい値だと推定ノイズがその付近でディザしたときに walk/stand が
    ノイズの周期でトグルし、位相のゼロ埋めと再開が繰り返されて振動になる。
    サーモスタットと同じ考え方。

    ★ 実機の C++ 側にも同じ lo/hi で実装すること。片側だけだと意味がない。
    """
    gate = getattr(env, _WALK_GATE_ATTR, None)
    if gate is None or gate.shape != speed.shape:
        gate = torch.zeros_like(speed, dtype=torch.bool)
    gate = torch.where(gate, speed >= lo, speed >= hi)
    # リセット直後は停止から始める (位相も 0 から始まる規約に合わせる)
    gate = torch.where(env.episode_length_buf <= 1, torch.zeros_like(gate), gate)
    setattr(env, _WALK_GATE_ATTR, gate)
    return gate


def _task_gait_phase_accum(
    env: "ManagerBasedRLEnv",
    phase_freq: float,
    walking: torch.Tensor,
    step_jitter: float = 0.0,
) -> torch.Tensor:
    """歩行位相を **積算** して返す φ∈[0,2π) (per-env)。

    ★ 2026-08-15 追加。旧実装は φ = 2π·f·t (エピソード開始からの絶対時間) で毎回
      計算し直していた。停止中もこの t は進み続けるので、ゼロ埋めから復帰したとき
      **位相が飛んだ位置から再開する**。1.6Hz なら 0.3 秒止まっただけで 0.48 周期
      ≒ 170° 飛ぶ。片足が空中にある最中にこれが起きると支持脚の切り替えが噛み合わず
      崩れる。MuJoCo でランドマーク認識がカクついたとき、ボールが静止していても
      ロボットが揺れて転倒したのはこれが原因。

      本関数は「歩いている間だけ位相を進める」ので、停止 → 再開が連続する。
      around_ball の :func:`~...around_ball.mdp.observations._high_action_gait_phase`
      と同じ方式 (あちらは速度依存周波数、こちらは固定 1.6Hz)。

    Args:
        walking: 歩行中フラグ (N,)。False の env は位相を進めない。
        step_jitter: 1 ステップあたりの位相増分に掛ける揺らぎの幅 (0 で無効)。
            ★ 2026-08-16 追加。実機の推論ループは SingleThreadedExecutor で 50Hz タイマ・
              ビジョン・twist・pose の全コールバックが直列化されており、ONNX 推論も
              タイマ内でインライン実行される。しかも wall_timer は遅れた tick を
              取り戻さない。にもかかわらず位相の積算は **定数 kDt = 1/50** を使うので、
              ループが伸びると位相が実時間からずれる。シムは決定的に回るので
              この状況を経験しない。ここで増分自体を揺らして耐性を付ける。
    """
    from ...locomotion.mdp.events import get_phase_freq

    phase = getattr(env, _TASK_PHASE_ATTR, None)
    if phase is None or phase.shape != (env.num_envs,):
        phase = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _TASK_PHASE_ATTR, phase)
        setattr(env, _TASK_PHASE_STEP_ATTR, -1)

    # 同一ステップ内の複数回呼び出し (policy/critic) では 1 回だけ積算する
    if getattr(env, _TASK_PHASE_STEP_ATTR, -1) != int(env.common_step_counter):
        # get_phase_freq は per-env テンソルを返すが、randomize_phase_freq イベントが
        # 未登録だと float を返すので、どちらでも動くようテンソル化する。
        pf = get_phase_freq(env, phase_freq)
        step = 2.0 * math.pi * float(env.step_dt) * (
            pf if torch.is_tensor(pf) else torch.full_like(phase, float(pf))
        )
        if step_jitter > 0.0:
            step = step * (1.0 + (torch.rand_like(step) * 2.0 - 1.0) * float(step_jitter))
        phase = (phase + torch.where(walking, step, torch.zeros_like(step))) % (2.0 * math.pi)
        # リセット直後の env は位相 0 から (locomotion と同じ規約)
        phase = torch.where(env.episode_length_buf <= 1, torch.zeros_like(phase), phase)
        setattr(env, _TASK_PHASE_ATTR, phase)
        setattr(env, _TASK_PHASE_STEP_ATTR, int(env.common_step_counter))
    return getattr(env, _TASK_PHASE_ATTR)


def task_drive_phase_obs(
    env: "ManagerBasedRLEnv",
    phase_freq: float = 1.6,
    cmd_threshold: float = 0.12,
    max_y: float = 1.3,  # = GOAL_HALF_WIDTH (ゴール幅 2.6m)
    vx_scale: float = 1.0,
    vy_scale: float = 1.3,
    use_perceived: bool = True,
    use_measured_speed: bool = True,
    speed_threshold: float = 0.15,
    speed_gate_lo: float | None = 0.12,
    speed_gate_hi: float | None = 0.18,
    speed_noise: float = 0.1,
    speed_noise_tau_s: float = 0.08,
    speed_noise_scale_range: tuple = (0.5, 1.5),
    speed_delay_range: tuple = (1, 3),
    phase_step_jitter: float = 0.1,
) -> torch.Tensor:
    """ステージ2/3 の ``gait_phase`` スロット (4 次元)。タスク駆動の歩行位相。

    locomotion の :func:`phase_obs` と同一フォーマット (左右 sin/cos、停止時はゼロ埋め)
    だが、停止判定を ``base_velocity`` コマンドではなく :func:`task_drive_vector` の
    **並進成分 (dx, dy)** で行う。

    ★ 向き成分 (dyaw) は判定に含めない。足踏みでは向きは直らない (その場旋回が要る) のに、
      yaw drift は実測で恒常的に 7〜12° あり、それだけでしきい値を超えて
      「定位置にいるのに歩き続ける」状態になっていたため。

    ★ 2026-08-15: **歩行を自己位置推定から切り離した**。修正は 2 点:

      (a) 停止判定を実測ベース速度にする (``use_measured_speed``)
          旧実装は task_drive_vector の並進成分で判定していたが、あれは自己位置
          (MCL) から計算される量なので、**歩行リズムが自己位置に直結していた**。
              dy = (target_y - pos_y) / 0.15   ← MCL が揺れると dy が跳ねる
          MuJoCo でランドマーク認識がカクつくと MCL 推定が細かく跳び、しきい値
          0.12 を高頻度でまたいで「歩行 ⇔ 停止」がトグルする。**ボールが静止して
          いてもロボットが揺れて転倒する**のはこれが原因。実測速度なら MCL が
          どう揺れても跳ねない (実機でも IMU + 脚オドメトリから取れる量)。

      (b) 位相を積算にする (:func:`_task_gait_phase_accum`)
          旧実装は φ = 2π·f·t の絶対時間ベースで、停止中も t が進むため復帰時に
          位相が飛ぶ (1.6Hz なら 0.3 秒の停止で約 170°)。片足が空中のときにこれが
          起きると支持脚の切り替えが噛み合わず崩れる。積算なら停止→再開が連続する。

      (a) がトグルを止め、(b) がトグル時の飛びを消す。両方で歩容の破綻を防ぐ。
      ``use_measured_speed=False`` で旧挙動 (既存 ckpt の再生・比較用)。

    ★ 2026-08-16: **実機で振動したので (a) の sim2real ギャップを埋めた**。
      (a) は「MCL に依存しない実測速度で判定する」ところまでは正しかったが、
      シムの ``root_lin_vel_b`` は物理エンジンの真値で、静止していれば厳密に 0 になる。
      実機は InEKF の推定値 (しかも位置の有限差分) なので静止していても振れる。
      閾値が 1 本しかないため、推定値がその付近でディザすると walk/stand が
      ノイズの周期でトグルし、位相のゼロ埋めと再開が繰り返されて振動になる。
      **MCL 起因のトグルを潰したら、速度推定起因の同じトグルが別口から復活した**形。

      対策は 3 つ:
        * ``speed_noise``: ゲートの入力に相関ノイズを載せて実機相当にする
        * ``speed_delay_range``: ゲートの入力を 1〜3 step 遅らせる (実機 40〜60ms 相当)
        * ``speed_gate_lo`` / ``speed_gate_hi``: ヒステリシスでトグル自体を起きにくくする
      ★ ヒステリシスは実機の C++ 側にも同じ値で実装すること。片側だけでは意味がない。
      ★ このゲートは閉ループ (ゲート → 位相 → 方策 → 関節 → 速度 → ゲート) の中にあり、
        判定がハードしきい値なので、**遅延だけでも自励振動する**。ノイズより遅延の方が
        機構的に直結している (_BASE_VEL_* 定数群のコメント参照)。

    Args:
        use_measured_speed: True で実測ベース速度、False で旧来の task_drive_vector 判定。
        speed_threshold: ヒステリシスを使わないとき (lo/hi が None) のしきい値 [m/s]。
            旧 cmd_threshold (0.12) は「位置ずれ [m] を 0.15s で割った速度」に対する
            閾値で単位の意味が違うため、stage1_speed_tol (0.15) と揃えてある。
        speed_gate_lo / speed_gate_hi: ヒステリシスの下降/上昇しきい値 [m/s]。
            どちらかが None なら ``speed_threshold`` の単一しきい値に戻る。
        speed_noise: ゲート入力に載せるノイズの振幅 [m/s]。0 で無効。**実測ではない**
            (:func:`_measured_base_speed` の注記参照)。
        speed_delay_range: ゲート入力の遅延段数 [制御 step]。(0, 0) で無効。
        phase_step_jitter: 位相増分の揺らぎ幅 (:func:`_task_gait_phase_accum` 参照)。
    """
    if use_measured_speed:
        speed = _measured_base_speed(
            env, speed_noise, speed_noise_tau_s, speed_noise_scale_range, speed_delay_range
        )
        if speed_gate_lo is None or speed_gate_hi is None:
            walking = speed >= speed_threshold
        else:
            walking = _walk_gate(env, speed, float(speed_gate_lo), float(speed_gate_hi))
    else:
        drive = task_drive_vector(
            env, max_y=max_y, vx_scale=vx_scale, vy_scale=vy_scale, use_perceived=use_perceived
        )
        walking = torch.norm(drive[:, :2], dim=1) >= cmd_threshold

    phase_left = _task_gait_phase_accum(env, phase_freq, walking, step_jitter=phase_step_jitter)
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    return torch.where(walking.unsqueeze(1), phase, torch.zeros_like(phase))


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
