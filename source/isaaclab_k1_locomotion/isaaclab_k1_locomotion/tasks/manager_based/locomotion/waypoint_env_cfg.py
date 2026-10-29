# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""経由点追従環境 (Isaac-Waypoint-K1-v0, 2026-10-29)。

凍結した歩行 expert (frozen_policies/best_walk) に歩行コマンドを出す上位方策を学習する。
環境は歩行 expert の学習環境 `K1FlatStancePlanePolishCfg` をそのまま継承し (観測ノイズ・
地形・摩擦 DR・位相パラメータが expert の学習分布と一致する)、以下だけを差し替える:

* action: ``joint_pos`` (12, expert の関節指令) に ``walk_command`` (3, 上位方策のコマンド) を追加。
  expert が見る ``actions`` 観測は ``joint_pos`` 項のみ (49 次元レイアウトを維持)。
* command: ``base_velocity`` = `ExternalVelocityCommandCfg` (``walk_command`` のコピー)、
  ``waypoints`` = `WaypointCommandCfg` (経由点 + 最終目標姿勢 + モード)。
* observation: 上位方策用の ``nav_policy`` / ``nav_critic`` グループを追加
  (歩行 expert 用の ``command`` / ``policy`` / ``critic`` は据え置き)。
* reward: 歩行用の報酬は全て捨て、経路進捗・ゴール姿勢・コマンド変化率・転倒のみ。
* curriculum: 歩行コマンド系のカリキュラムは全て無効。

学習は通常の train.py (``--task Isaac-Waypoint-K1-v0``) で行う。方策・PPO は
`agents/waypoint_policy.py` (`K1WaypointPPORunnerCfg`)。
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import PLANE_HEAVY_TERRAIN_CFG, K1FlatObservationsCfg, K1FlatStancePlanePolishCfg
from .velocity_env_cfg import ActionsCfg
from .mdp.waypoint import (
    ExternalVelocityCommandCfg,
    WalkCommandActionCfg,
    WaypointCommandCfg,
    goal_heading_exp,
    goal_position_exp,
    path_progress,
    walk_command_rate_l2,
)


@configclass
class K1WaypointActionsCfg(ActionsCfg):
    """joint_pos (12, expert) の後ろに上位方策の歩行コマンド (3) を並べる。"""

    walk_command = WalkCommandActionCfg(asset_name="robot")


@configclass
class K1WaypointNavPolicyCfg(ObsGroup):
    """上位 actor の観測。実機では自己位置推定 (経由点の相対位置) と IMU / オドメトリ速度で構成できる。"""

    waypoints = ObsTerm(func=mdp.generated_commands, params={"command_name": "waypoints"})
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1WaypointNavCriticCfg(ObsGroup):
    """上位 critic の観測 (ノイズなし)。"""

    waypoints = ObsTerm(func=mdp.generated_commands, params={"command_name": "waypoints"})
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1WaypointObservationsCfg(K1FlatObservationsCfg):
    nav_policy: K1WaypointNavPolicyCfg = K1WaypointNavPolicyCfg()
    nav_critic: K1WaypointNavCriticCfg = K1WaypointNavCriticCfg()


@configclass
class K1WaypointCommandsCfg:
    base_velocity = ExternalVelocityCommandCfg(asset_name="robot", action_name="walk_command")
    waypoints = WaypointCommandCfg(asset_name="robot", debug_vis=False)


@configclass
class K1WaypointRewardsCfg:
    """上位方策用の報酬 (歩行用の報酬は継承しない)。値は RewardManager が dt (0.02) を掛ける。"""

    # 残り経路長の減少 [m] × 50 → 1 m 進むごとに +1.0。ポテンシャル形式なので総和は初期経路長で頭打ち。
    path_progress = RewTerm(func=path_progress, weight=50.0, params={"command_name": "waypoints"})
    # ゴール姿勢に居続けると毎秒 +2.0 (位置) +2.0 (向き)。早く着くほど多く貰える = 最速到達の動機。
    # 全経由点を通過した env にしか出ないので、経由点を飛ばしても得をしない。
    goal_position = RewTerm(func=goal_position_exp, weight=2.0, params={"command_name": "waypoints", "std": 0.25})
    goal_heading = RewTerm(
        func=goal_heading_exp, weight=2.0, params={"command_name": "waypoints", "std_yaw": 0.3, "std_pos": 0.25}
    )
    # 50 Hz でコマンドが暴れると歩行 expert の学習分布 (区分一定コマンド) から外れるので軽く抑える。
    walk_command_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.05, params={"action_name": "walk_command"})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class K1WaypointEnvCfg(K1FlatStancePlanePolishCfg):
    observations: K1WaypointObservationsCfg = K1WaypointObservationsCfg()
    actions: K1WaypointActionsCfg = K1WaypointActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 親の __post_init__ は歩行コマンド cfg (ranges 等) を触るので、コマンドはその後で差し替える
        self.commands = K1WaypointCommandsCfg()

        # expert が見る actions 観測は関節指令 (12) のみ (action 空間が 12 + 3 になるため)
        self.observations.policy.actions.params = {"action_name": "joint_pos"}
        self.observations.critic.actions.params = {"action_name": "joint_pos"}

        # 報酬は上位方策用に丸ごと置き換える
        self.rewards = K1WaypointRewardsCfg()

        # 歩行コマンド系のカリキュラムは全て無効 (base_velocity は ExternalVelocityCommand)
        self.curriculum.lin_vel_command = None
        self.curriculum.extreme_commands = None
        self.curriculum.command_resampling_time_range = None
        self.curriculum.push_robot_stage1 = None

        # 外乱は歩行 expert が経験した穏やかな範囲に固定 (Polish の強い push は到達時間の評価を乱す)
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
        }

        # 経路は最長 5 セグメント × 2 m なのでロボットが地形の端から落ちないよう平面の縁を広げる
        self.scene.terrain.terrain_generator = PLANE_HEAVY_TERRAIN_CFG.replace(border_width=15.0)

        self.episode_length_s = 20.0


@configclass
class K1WaypointEnvCfg_PLAY(K1WaypointEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.nav_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.waypoints.debug_vis = True
