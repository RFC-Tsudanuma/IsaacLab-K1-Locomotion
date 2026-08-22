# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from ...locomotion.mdp.commands import DiscreteVelocityCommand, DiscreteVelocityCommandCfg
from .kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class KickDirectionCommand(UniformVelocityCommand):
    """蹴り方向 + 目標ボール速度コマンド。

    エピソードごとにランダムな角度 θ と目標ボール速度 v をサンプリングし、
    command = [sin θ, cos θ, v] を返す。
    kick_state は command[:, :2] を蹴り方向ベクトル (cos θ, sin θ) として、
    target_kick_velocity 観測と kick_velocity_scaled 報酬は command[:, 2] を目標速度として使用する。

    メトリクス (walk_kick / walk_pass / walk_loop 共通、Metrics/kick_direction/ 以下に出る):

    * ``kick_vel_ratio``: 指令速度に対する実測ボール速度の追従率 (v_ball / v_target)。
    * ``kick_vel_error``: 同じく絶対誤差 [m/s]。
    * ``kick_rate``: キックが成立した (値 latch した) エピソードの割合。
    * ``kick_dir_error_deg``: 方向誤差 |τ_direction| [deg]。latch 時に凍結した
      ボール飛翔方向と指令蹴り方向の水平角度差。「誤差 ±10°」要件の直接の指標。
      これも「キックが成立した env」だけで平均している (``kick_rate`` で割り戻すこと)。
    * ``kick_dir_error_signed_deg``: 同じ誤差の **符号付き** [deg]。**正 = ボールが
      指令方向より右**。``kick_dir_error_deg`` は絶対値なので「±4° のランダム誤差」と
      「常に右へ 4°」を区別できない。系統的な左右バイアスの有無はこちらで見ること。
      0 付近ならバイアス無し、``kick_dir_error_deg`` と同じ大きさまで振れていれば
      誤差はほぼ全部が一方向のバイアス。
    * ``kick_foot_right_frac``: 蹴った足が右足だった割合 (0 = 常に左、1 = 常に右)。
      観測に左足裏しか入っておらず (``_sole()``)、歩行位相もエピソード開始時に
      必ず 0 から始まり、さらに mirror loss が無効 (``symmetry_cfg = None``) なので、
      このタスク群には左右どちらかへ偏る構造的な理由がある。0.5 から大きく外れて
      いれば片足でしか蹴っていない。
    * ``kick_elevation_deg``: 射出仰角 φ [deg]。ループシュートが出ているかの直接の指標。
    * ``kick_apex_height``: latch 後にボールが到達した最高高度 [m]。
    * ``ball_touch_count``: エピソード中に足がボールに触れた回数。1.0 が理想。
    * ``sole_height_at_kick``: latch を起こした接触 (= キック本体) の瞬間の足裏高さ [m]。
      射出仰角を決めている量なので、``kick_elevation_deg`` が出ない原因の切り分けは
      ここを見る。R=0.11 のとき 0.019 で 30°、0.036 で 20°、0.055 で 10°、0.074 で 0°。
      「キックが成立した env」だけで平均している (``kick_rate`` で割り戻すこと)。
    * ``plant_lon`` / ``plant_lat``: 蹴った瞬間の軸足 (蹴っていない方の足) の配置 [m]。
      キック方向フレームでボール中心から測った前後 (+ = ボールより前) と左右 (絶対値)。
      「軸足がボールの真横」= plant_lon ≈ −0.03 (足首基準)、plant_lat ≈ 0.19。
      軸足がボール後方に残っていると蹴り足が高い位置に当たるので、
      ``sole_height_at_kick`` が下がらない原因の切り分けに使う。
      これも「キックが成立した env」だけで平均している。
    * ``foot_vz``: 蹴った瞬間の **蹴り足** のワールド鉛直速度 [m/s]。**正 = すくい上げ**。
      ボールが浮いた原因が足の上向き運動 (反発係数に依存しない) なのか、地面との反発
      (Isaac では e≈0.6 だが MuJoCo・実機では ≈0) なのかを切り分ける指標。
      ``kick_apex_height`` が出ているのにここが 0 付近なら、その解は反発に依存しており
      実機へ転移しない。:func:`~.rewards.kick_foot_lift` の ``vz_foot_sat`` は
      この実測値を見て決める (飽和しきっているなら上げる)。
      これも「キックが成立した env」だけで平均している。

    ``cfg.log_contact_geometry`` が True のときだけ、当たり所の幾何が 3 つ増える
    (既定 False。既存タスクの TB タグ集合を変えないため):

    * ``foot_kick_dot``: latch を起こした接触の瞬間に凍結した、蹴り足のつま先方向
      (足のローカル +x の水平成分) と指令蹴り方向の内積。1 に近い = つま先が前 =
      トーキック、0 に近い = 足が真横 = 側面で当てている、−1 に近い = かかと側。
    * ``ball_side``: 同じ瞬間の、蹴り足のローカル座標で見たボール中心の y [m]。
      **正 = 右足の内側 (インサイド) の面の側**。足箱の半幅は 0.035 なので、
      0.035 付近まで来ていれば面の中央で当たっている。
      :func:`~.rewards.kick_inside_contact` の f_side がそのまま見える値で、
      「インサイドで当てられているか」はこの 2 つを並べて読む。
    * ``plant_yaw_dot``: 同じ瞬間の **軸足** (蹴っていない方の足) のつま先方向
      (上から見た水平成分) と指令蹴り方向の内積。1 = 軸足のつま先が蹴り方向、
      0 = 真横、−1 = かかとが蹴り方向。角度に直すなら ``acos``
      (0.87 = 30°、0.71 = 45°、0.50 = 60°)。
      ``plant_lon`` / ``plant_lat`` が軸足を **どこに置いたか** なのに対し、これは
      **どちらを向けたか**。:func:`~.rewards.kick_plant_yaw` が直接引っ張る値なので、
      その項が効いているかの判定はここを見る。胴体の向き (p_style) は帯で
      30-45° のずれを許しているので、**胴体と軸足がどれだけ別々の向きを向けている
      かを見る指標** でもある (両者を分ける自由度は Hip_Yaw)。
      これも「キックが成立した env」だけで平均している (``kick_rate`` で割り戻すこと)。

    ``kick_rate`` 以外は未キックの env を 0 として平均するため、キック成立分だけの値が
    欲しいときは ``kick_rate`` で割り戻すこと。
    """

    cfg: "KickDirectionCommandCfg"

    def __init__(self, cfg: "KickDirectionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 親 (UniformVelocityCommand) の速度追従メトリクスは、このコマンドが速度指令では
        # ないため意味を持たない。ログを汚さないように捨てる。
        self.metrics.pop("error_vel_xy", None)
        self.metrics.pop("error_vel_yaw", None)
        # 指令キック速度 v_target に対する実測ボール速度の追従メトリクス。
        # いずれも「キックが成立した (値 latch した) env」でのみ値を持ち、
        # 未キックの env は 0 のまま。CommandManager はエピソード終了時に
        # リセット対象 env の平均を Metrics/kick_direction/<name> として記録する。
        self.metrics["kick_vel_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_vel_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_rate"] = torch.zeros(self.num_envs, device=self.device)
        # 方向誤差 [deg]。「誤差 ±10°」要件 (walk_long_pass) の直接の指標。
        self.metrics["kick_dir_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        # 符号付きの方向誤差 [deg] (正 = 右) と、右足で蹴った割合。
        # 左右の系統バイアスを切り分けるための 2 つ。
        self.metrics["kick_dir_error_signed_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_foot_right_frac"] = torch.zeros(self.num_envs, device=self.device)
        # ループシュート (walk_loop) 用。他タスクでも「意図せず浮いていないか」の監視に使える。
        self.metrics["kick_elevation_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_apex_height"] = torch.zeros(self.num_envs, device=self.device)
        # エピソード中に足がボールに触れた回数。1.0 が理想 (1 回で蹴り切る)。
        # 未キックの env も含めた全 env 平均なので、kick_rate と併せて読むこと。
        self.metrics["ball_touch_count"] = torch.zeros(self.num_envs, device=self.device)
        # latch を起こした接触 (= キック本体) の瞬間の足裏高さ [m]。仰角を決める量。
        self.metrics["sole_height_at_kick"] = torch.zeros(self.num_envs, device=self.device)
        # 蹴った瞬間の軸足の配置 (キック方向フレーム、ボール中心基準) [m]。
        self.metrics["plant_lon"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["plant_lat"] = torch.zeros(self.num_envs, device=self.device)
        # 蹴った瞬間の蹴り足の鉛直速度 [m/s]。+ = すくい上げ。vz_foot_sat のチューニング用。
        self.metrics["foot_vz"] = torch.zeros(self.num_envs, device=self.device)
        # 低指令域 (v_target < low_speed_threshold) だけを切り出した内訳。
        # 全 env 平均の kick_rate / kick_vel_ratio では「弱い指令だけ蹴れていない/
        # 飛びすぎている」が高指令域の成績に埋もれて見えないため (walk_weak_kick 用)。
        self.metrics["kick_low_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_rate_low"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_vel_ratio_low"] = torch.zeros(self.num_envs, device=self.device)
        # 当たり所の幾何 (walk_inside_kick 用)。**キー自体を作らない** ことで、
        # 既定 False の他タスクでは TensorBoard のタグ集合が一切変わらない。
        if self.cfg.log_contact_geometry:
            self.metrics["foot_kick_dot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["ball_side"] = torch.zeros(self.num_envs, device=self.device)
            # 軸足の **向き** (つま先方向 · kick_dir)。位置の plant_lon / plant_lat とは
            # 別物なので分けて出す。同じフラグに相乗りさせているのは、これも
            # walk_inside_kick 専用の指標で、他タスクの TB タグ集合を変えないため。
            self.metrics["plant_yaw_dot"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        # kick_state は termination / reward 側が同じステップで計算済みのものを読むだけ
        # (ここで再計算するとパラメータの二重管理になる)。まだ無ければ何もしない。
        state = getattr(self._env, "_kick_latch_state", None)
        if state is None:
            return

        kick_done = state["kick_done"].float()
        v_ball = state["v_ball_frozen"]
        v_target = state["v_target"]

        # 追従率 = 実測ボール速度 / 指令速度。1.0 で指令どおり、>1 で蹴りすぎ。
        ratio = v_ball / torch.clamp(v_target, min=1e-6)

        self.metrics["kick_vel_ratio"] = ratio * kick_done
        self.metrics["kick_vel_error"] = torch.abs(v_ball - v_target) * kick_done
        self.metrics["kick_rate"] = kick_done
        self.metrics["kick_dir_error_deg"] = torch.rad2deg(state["tau_direction_frozen"]) * kick_done
        self.metrics["kick_dir_error_signed_deg"] = torch.rad2deg(state["tau_signed_frozen"]) * kick_done
        self.metrics["kick_foot_right_frac"] = state["kick_foot_frozen"] * kick_done
        self.metrics["kick_elevation_deg"] = torch.rad2deg(state["phi_frozen"]) * kick_done
        self.metrics["kick_apex_height"] = state["apex_height"] * kick_done
        # 接触回数は kick_done でマスクしない。「触ったが蹴れていない」エピソードこそ
        # 見たい対象なので、未接触の env も含めて数える。
        self.metrics["ball_touch_count"] = state["touch_count"]
        # キックが成立した env だけで平均する (未キックを 0 として混ぜると誤読する)。
        self.metrics["sole_height_at_kick"] = state["sole_height_at_kick"] * kick_done
        self.metrics["plant_lon"] = state["plant_lon_frozen"] * kick_done
        self.metrics["plant_lat"] = state["plant_lat_frozen"] * kick_done
        self.metrics["foot_vz"] = state["foot_vz_frozen"] * kick_done

        # 低指令域の内訳。読み方:
        #   低指令域のキック成功率 = kick_rate_low / kick_low_frac
        #   低指令域の追従率       = kick_vel_ratio_low / kick_rate_low
        # (どちらも分母を別メトリクスとして出すので、割り戻して読むこと)
        is_low = (v_target < self.cfg.low_speed_threshold).float()
        self.metrics["kick_low_frac"] = is_low
        self.metrics["kick_rate_low"] = kick_done * is_low
        self.metrics["kick_vel_ratio_low"] = ratio * kick_done * is_low

        # 当たり所の幾何。plant_lon / plant_lat と同じ流儀で、latch した env だけを
        # 見る (未キックを 0 として混ぜると誤読する) ため kick_done を掛ける。
        if self.cfg.log_contact_geometry:
            self.metrics["foot_kick_dot"] = state["foot_kick_dot_frozen"] * kick_done
            self.metrics["ball_side"] = state["ball_side_frozen"] * kick_done
            self.metrics["plant_yaw_dot"] = state["plant_yaw_dot_frozen"] * kick_done

    def _resample_command(self, env_ids: torch.Tensor):
        n = len(env_ids)
        low, high = self.cfg.ranges.heading

        # ロボットの現在ヨー角を取得し、そこからの相対オフセットとしてサンプリング
        robot_quat = self.robot.data.root_quat_w[env_ids]
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        offset = torch.empty(n, device=self.device).uniform_(low, high)
        theta = robot_yaw + offset

        speed_low, speed_high = self.cfg.target_speed_range

        self.command[env_ids, 0] = torch.sin(theta)
        self.command[env_ids, 1] = torch.cos(theta)
        self.command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(speed_low, speed_high)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "kick_dir_visualizer"):
                self.kick_dir_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            self.kick_dir_visualizer.set_visibility(True)
        else:
            if hasattr(self, "kick_dir_visualizer"):
                self.kick_dir_visualizer.set_visibility(False)

    def _debug_vis_callback(self, _event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # command = [sin θ, cos θ, 0] → world frame angle θ
        kick_x = self.command[:, 1]  # cos θ
        kick_y = self.command[:, 0]  # sin θ
        theta = torch.atan2(kick_y, kick_x)  # = θ (world frame)

        zeros = torch.zeros_like(theta)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, theta)

        default_scale = self.kick_dir_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)

        self.kick_dir_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)


@configclass
class KickDirectionCommandCfg(UniformVelocityCommandCfg):
    """蹴り方向コマンドの設定クラス。

    ranges.heading で蹴り方向のサンプリング範囲を指定する。
    ranges.lin_vel_x/y, ang_vel_z は使用しない（0 に設定すること）。
    """

    class_type: type = KickDirectionCommand

    target_speed_range: tuple[float, float] = (1.0, 4.0)
    """目標ボール速度 [m/s] のサンプリング範囲。command[:, 2] に格納される。"""

    low_speed_threshold: float = 0.8
    """``kick_*_low`` メトリクスで「低指令域」とみなす v_target の上限 [m/s]。

    既定 0.8 は kick_state の既定 v_thresh と同じ。**この値未満の指令は、既定の
    latch 閾値のままだと「指令どおり蹴っても latch が発火しない」領域**なので、
    そこだけを切り出して見られるようにしてある (メトリクスのみ。挙動には影響しない)。
    """

    log_contact_geometry: bool = False
    """当たり所の幾何 (``foot_kick_dot`` / ``ball_side``) をメトリクスに出すか。

    **既定 False = 既存タスクの TensorBoard のタグ集合を変えないため。** False の
    ときは ``metrics`` 辞書にキー自体を作らないので、出力されるタグが 1 つも増えない
    (過去の run と同じ並びで比較できる)。

    True にするのは当たり所そのものが目的のタスク (walk_inside_kick) だけ。
    メトリクスのみで、挙動には一切影響しない。
    """


class BallFollowVelocityCommand(DiscreteVelocityCommand):
    """目標終端 G へ向かう速度コマンド。walk phase では通常の歩行コマンドに切り替わる。

    ``cfg.follow_ball`` が True (kick phase) のとき、毎ステップ速度コマンド (vx, vy, wz) を
    「G へのロボット相対位置」と「kick_direction への角度誤差」で上書きする。

    G はボールそのものではなく :mod:`.kick_state` が計算する理想キック立ち位置側の点
    (後方レイ R 上をボール側へ滑る点で、P_kick で下限クランプされる)。ボール中心へ直行
    させると walk_speed 報酬 (G へ引く) と kick_pose_overshoot 罰 (後方レイ R の左右跨ぎ)
    の両方と衝突するため、指令も G を向ける。latch 後は kick_state 側で G が P_kick に
    固定されるので、飛翔したボールを追いかけることもない。

    False (walk phase) のときは何も上書きせず、親の :class:`DiscreteVelocityCommand`
    そのもの、つまり K1FlatEnvCfg で使っている通常の歩行コマンド（離散格子からの
    ランダムサンプリング）として振る舞う。lin_vel_command / command_resampling_time_range
    カリキュラムもこのときだけ意味を持つ。
    """

    cfg: "BallFollowVelocityCommandCfg"

    def _resample_command(self, env_ids):
        if not self.cfg.follow_ball:
            # walk phase: 通常の歩行コマンドを離散格子からサンプリングする
            super()._resample_command(env_ids)
        # kick phase: 毎ステップ _update_command で上書きするためリサンプルは不要

    def _update_command(self):
        if not self.cfg.follow_ball:
            # walk phase: 親 (UniformVelocityCommand) の standing-env ゼロ化などに任せる
            super()._update_command()
            return

        robot = self._env.scene["robot"]

        # NOTE: kick_state はステップ単位でキャッシュされる。CommandManager は
        #       TerminationManager / RewardManager より後に走るので、ここで得られるのは
        #       同じステップに確定済みの状態 (再計算されない)。
        state = kick_state(
            self._env,
            r_stance=self.cfg.r_stance,
            alpha=self.cfg.alpha,
            v_thresh=self.cfg.v_thresh,
            command_name=self.cfg.kick_direction_command_name or "kick_direction",
            track_ball=self.cfg.track_ball,
            v_thresh_target_frac=self.cfg.v_thresh_target_frac,
            v_thresh_floor=self.cfg.v_thresh_floor,
            r_max=self.cfg.r_max,
            orbit_beta=self.cfg.orbit_beta,
            overshoot_margin=self.cfg.overshoot_margin,
            lateral_band=self.cfg.lateral_band,
            style_halfwidth=self.cfg.style_halfwidth,
        )

        robot_pos_w = robot.data.root_pos_w[:, :2]

        # G - ロボット のワールドフレームベクトルをロボットフレームに変換
        to_G_w = torch.zeros(self.num_envs, 3, device=self.device)
        to_G_w[:, :2] = state["G"] - robot_pos_w
        to_G_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), to_G_w)

        max_vel = self.cfg.max_vel
        self.vel_command_b[:, 0] = torch.clamp(to_G_b[:, 0], -max_vel, max_vel)
        self.vel_command_b[:, 1] = torch.clamp(to_G_b[:, 1], -max_vel, max_vel)

        # wz: ロボットのヨー角と kick_direction の角度誤差
        if self.cfg.kick_direction_command_name:
            kick_cmd = self._env.command_manager.get_command(self.cfg.kick_direction_command_name)
            kick_theta = torch.atan2(kick_cmd[:, 0], kick_cmd[:, 1])

            quat = robot.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            ang_error = kick_theta - robot_yaw
            ang_error = torch.atan2(torch.sin(ang_error), torch.cos(ang_error))
            self.vel_command_b[:, 2] = torch.clamp(ang_error, -self.cfg.max_ang_vel, self.cfg.max_ang_vel)
        else:
            self.vel_command_b[:, 2] = 0.0


@configclass
class BallFollowVelocityCommandCfg(DiscreteVelocityCommandCfg):
    """ボール追従速度コマンドの設定クラス。

    ``follow_ball=False`` にすると DiscreteVelocityCommandCfg（通常の歩行コマンド）として
    振る舞うので、ranges / *_resolution も引き継いで設定しておくこと。
    """

    class_type: type = BallFollowVelocityCommand

    follow_ball: bool = True
    """True (kick phase) で G 追従、False (walk phase) で通常の歩行コマンド。"""

    max_vel: float = 1.0
    """速度コマンドの上限 [m/s]。follow_ball=True のときのみ使用。"""

    max_ang_vel: float = 1.0
    """角速度コマンドの上限 [rad/s]。follow_ball=True のときのみ使用。"""

    kick_direction_command_name: str | None = None
    """角速度コマンドの参照先となる kick_direction コマンド名。None なら wz=0。"""

    # -- kick_state (G の計算) に渡すパラメータ。キック報酬側と同じ値にすること。
    r_stance: float = 0.25
    """P_kick 半径 [m]。follow_ball=True のときのみ使用。"""

    alpha: float = 0.5
    """G の追従係数。follow_ball=True のときのみ使用。"""

    v_thresh: float = 0.8
    """値 latch のトリガー速度 [m/s]。follow_ball=True のときのみ使用。"""

    track_ball: bool = False
    """True で latch 前の P_kick をボールに追従させる (転がるボール用)。

    :func:`..kick_state.kick_state` の同名引数を参照。``terminations.kick_finished`` の
    params にも **同じ値** を渡すこと (先に呼ばれた方でその step の状態が確定するため)。
    """

    v_thresh_target_frac: float = 0.0
    """>0 で latch 閾値を指令速度に追従させる。0.0 (既定) はスカラー v_thresh のまま。

    :func:`..kick_state.kick_state` の同名引数を参照。``track_ball`` と同じく
    ``terminations.kick_finished`` にも **同じ値** を渡すこと。
    """

    v_thresh_floor: float = 0.0
    """指令追従の閾値の切片 = 下限 [m/s]。``v_thresh_target_frac`` とセットで使う。"""

    r_max: float | None = None
    """None 以外で目標終端 G を回り込み型にする。ボールから G までの最大距離 [m]。

    :func:`..kick_state.kick_state` の同名引数を参照。G は kick_state が 1 ステップに
    一度だけ計算するので、**キック報酬項・``terminations.kick_finished`` にも同じ値**を
    配ること (先に呼ばれた項の値でその step の G が確定する)。
    """

    orbit_beta: float = 0.6
    """回り込み型 G の先読み係数 (0 < beta < 1)。``r_max`` を入れたときだけ使う。"""

    overshoot_margin: float = 0.0
    """overshoot 判定 (キック線 R の左右跨ぎ) の遊び [m]。0.0 (既定) で従来どおり。

    :func:`..kick_state.kick_state` の同名引数を参照。``r_max`` と同じく全ての
    kick_state 利用項へ同じ値を配ること。
    """

    lateral_band: tuple[float, float] | None = None
    """終端の構え位置に持たせる横方向のあそび (帯) ``(下端, 上端)`` [m]。正 = 右。

    None (既定) で従来どおり横ずれ 0 の一点。:func:`..kick_state.kick_state` の
    同名引数を参照。``r_max`` と同じく全ての kick_state 利用項へ同じ値を配ること
    (G と P_kick はその step で最初に呼ばれた項の値で確定するため)。
    """

    style_halfwidth: float | None = None
    """p_style (胴体の向きの正対度) を **帯** で採点する半幅 [rad]。

    None (既定) で従来どおり ``clamp(forward·kick_dir, 0, 1)``。値を入れると
    角度差がこの半幅以内なら 1、超えた分だけ緩やかに減衰する。
    :func:`..kick_state.kick_state` の同名引数を参照。``r_max`` と同じく
    **全ての kick_state 利用項へ同じ値を配ること**。

    このコマンド自身は p_style を読まないが、``follow_ball=True`` のとき
    :func:`..kick_state.kick_state` を呼ぶ側になりうるので、他の項と食い違わない
    よう同じ値を保持する。
    """

