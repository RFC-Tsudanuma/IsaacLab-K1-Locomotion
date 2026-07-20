# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー用の歩行 (サイドステップ重視) 環境設定。

K1FlatEnvCfg を継承し、差分のみを上書きする薄い設定。
Flat 側の報酬・カリキュラム改善はそのまま自動で追従する。

狙い:
  - 体の向き (yaw) はほぼ正面 (ゴールライン方向) に固定したまま、左右に素早く移動する。
  - そのため lin_vel_y のコマンド範囲を広く・lin_vel_x と ang_vel_z を狭くする。
  - コマンドのリサンプリング間隔を短くし、左右の切り返しに素早く反応する歩容を学ばせる。
"""

import math

from isaaclab.utils import configclass

from .flat_env_cfg import K1FlatEnvCfg


@configclass
class K1KeeperWalkEnvCfg(K1FlatEnvCfg):
    """サイドステップ重視のキーパー歩行環境。"""

    def __post_init__(self):
        super().__post_init__()

        # --- コマンド: 横移動を主役にする ---
        # 前後は「ボールに詰める / 下がる」程度の狭い範囲に留め、左右を広くとる。
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (-0.6, 0.6)
        cmd.ranges.lin_vel_y = (-1.5, 1.5)
        # yaw は正面維持が基本なので旋回コマンドは弱める。
        cmd.ranges.ang_vel_z = (-0.5, 0.5)
        # heading_command=True (velocity_env_cfg の既定) なので ang_vel_z は heading 追従から生成される。
        # heading 範囲を正面付近に絞ることで「常にゴールライン正面を向く」挙動になる。
        cmd.ranges.heading = (-math.pi / 6, math.pi / 6)
        # 左右の切り返しに素早く反応させるため、コマンド更新を短周期にする。
        cmd.resampling_time_range = (0.8, 3.0)
        # 構え (静止) の割合を Flat より増やし、無コマンド時にその場で安定して立てるようにする。
        cmd.rel_standing_envs = 0.1
        # 横方向は細かい速度指令に追従できるよう分解能を上げる。
        cmd.lin_vel_y_resolution = 0.025

        # --- カリキュラム: y 方向を段階的に広げる ---
        # Flat の lin_vel_command カリキュラムは x を大きく広げる設定なので、
        # keeper 用に x/y の役割を入れ替えた stage 構成に差し替える。
        # 横 1.5 m/s は Flat の到達値 (0.9) を大きく超える挑戦的な目標なので、
        # 1.0 以降を細かく刻んで各ステージの飛び幅を抑える (5 ステージ構成)。
        if self.curriculum.lin_vel_command is not None:
            self.curriculum.lin_vel_command.params["stages_x"] = [
                (-0.3, 0.3),
                (-0.4, 0.4),
                (-0.5, 0.5),
                (-0.6, 0.6),
                (-0.6, 0.6),
            ]
            self.curriculum.lin_vel_command.params["stages_y"] = [
                (-0.4, 0.4),
                (-0.7, 0.7),
                (-1.0, 1.0),
                (-1.25, 1.25),
                (-1.5, 1.5),
            ]
            # 横移動は前進より追従が難しいので、閾値は Flat より少し緩めに設定する。
            # 高速域ほど絶対誤差が出やすいため後段は更に緩めるが、緩めすぎると
            # 「狭い範囲を習得した時点で次の緩い閾値も満たして一気に遷移」するので、
            # 前ステージの到達誤差より必ず厳しい値になるよう刻み幅を 0.05 に抑える。
            self.curriculum.lin_vel_command.params["error_threshold"] = [0.30, 0.35, 0.40, 0.45, 0.50]

        # Flat 側にある「コマンド更新間隔を段階的に短くする」カリキュラムは、
        # ここでは最初から短周期にしているため無効化する。
        self.curriculum.command_resampling_time_range = None

        # --- 報酬: 横方向の追従と姿勢維持を重視する ---
        # 横移動は前進に比べて報酬が得にくく足踏みの局所最適に落ちやすいので、
        # 速度追従 (鋭い項/粗い項の両方) を Flat より強めにする。
        self.rewards.track_lin_vel_xy_exp.weight = 4.5
        self.rewards.track_lin_vel_xy_coarse.weight = 1.5
        # 旋回コマンドは弱いので yaw 追従の重みは下げ、代わりに姿勢を崩さないことを重視する。
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.ang_vel_xy_l2.weight = -0.35
        self.rewards.flat_orientation_l2.weight = -25.0

        # サイドステップでは足がクロスしたり擦れたりしやすいので、
        # 足同士の接近ペナルティと滑りペナルティを強める。
        self.rewards.feet_close_penalty.weight *= 1.5
        self.rewards.feet_slide.weight *= 1.5

        # 素早い切り返しを許すため、動作平滑化ペナルティは Flat よりわずかに緩める。
        self.rewards.action_rate_l2.weight = -0.3


@configclass
class K1KeeperWalkEnvCfg_PLAY(K1KeeperWalkEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # カリキュラムは __init__ で _apply_stage(0) を呼び、cfg.ranges を直接
        # 上書きする。PLAY で範囲を固定するには先に無効化しないと必ず負ける。
        self.curriculum.lin_vel_command = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.lin_vel_y = (-1.5, -1.5)
        cmd.ranges.heading = (0.0, 0.0)   # ang_vel_z ではなくこちらを止める

        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
