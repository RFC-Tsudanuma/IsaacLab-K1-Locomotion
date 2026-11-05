# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""経路追従 + 姿勢到達 / 見ながら到達環境 (Isaac-Tracking-K1-v0, 2026-11-05)。

凍結ゲート (歩行 expert + 回転 expert + ゲート網) に 5 次元コマンドを出す上位方策を学習する。
環境はゲート学習環境 `K1FlatGatingCfg` を継承し (物理・観測レイアウト・外乱がゲートの学習分布と
一致する)、以下だけを差し替える:

* action: ``joint_pos`` (12) に ``walk_command`` (5, 上位の出力) を追加。
* command: ``base_velocity`` = `ExternalTransitionCommandCfg` (上位の出力で mode / コマンドを決める)、
  ``tracking`` = `TrackingGoalCommandCfg` (経由点 + ゴール + 見つめる対象 + task_mode)。
* observation: 上位用の ``nav_policy`` / ``nav_critic`` グループを追加。
* reward: 両 expert の報酬は全て捨て、経路進捗・ゴール・見つめ・コマンド変化率・切替・転倒のみ。
* reset: 親 (関節 ±30°, 初速 ±0.5) のまま。v3 で立位スタートにしたら頑健性が落ちた (rl_traking.md §6)。
  評価は立位スタートで行う。

設計は rl_traking.md を参照。方策・PPO は `agents/tracking_policy.py` (`K1TrackingPPORunnerCfg`)。
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .flat_env_cfg import PLANE_HEAVY_TERRAIN_CFG, K1FlatGatingCfg, K1FlatTransitionObservationsCfg
from .mdp.tracking import (
    TRACK_ACTION_DIM,
    ExternalTransitionCommandCfg,
    TrackingGoalCommandCfg,
    heading_progress,
    look_at_penalty,
    lowlevel_state,
    mode_switch,
    track_command_rate_l2,
    track_goal_heading_exp,
    track_goal_position_exp,
)
from .mdp.waypoint import WalkCommandActionCfg, path_progress
from .velocity_env_cfg import ActionsCfg


@configclass
class K1TrackingActionsCfg(ActionsCfg):
    """joint_pos (12, ゲート混合の関節指令) の後ろに上位の出力 (5) を並べる。"""

    walk_command = WalkCommandActionCfg(asset_name="robot", dim=TRACK_ACTION_DIM)


@configclass
class K1TrackingNavPolicyCfg(ObsGroup):
    """上位 actor の観測 (36 次元)。実機では自己位置推定 + IMU + 自分の直前出力で構成できる。"""

    tracking = ObsTerm(func=mdp.generated_commands, params={"command_name": "tracking"})
    last_command = ObsTerm(func=mdp.last_action, params={"action_name": "walk_command"})
    lowlevel = ObsTerm(func=lowlevel_state, params={"command_name": "base_velocity"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1TrackingNavCriticCfg(ObsGroup):
    tracking = ObsTerm(func=mdp.generated_commands, params={"command_name": "tracking"})
    last_command = ObsTerm(func=mdp.last_action, params={"action_name": "walk_command"})
    lowlevel = ObsTerm(func=lowlevel_state, params={"command_name": "base_velocity"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1TrackingObservationsCfg(K1FlatTransitionObservationsCfg):
    nav_policy: K1TrackingNavPolicyCfg = K1TrackingNavPolicyCfg()
    nav_critic: K1TrackingNavCriticCfg = K1TrackingNavCriticCfg()


@configclass
class K1TrackingCommandsCfg:
    base_velocity: ExternalTransitionCommandCfg = None  # __post_init__ でゲート cfg から作る
    tracking = TrackingGoalCommandCfg(asset_name="robot", debug_vis=False)


@configclass
class K1TrackingRewardsCfg:
    """上位方策用の報酬 (rl_traking.md §5)。値は RewardManager が dt (0.02) を掛ける。"""

    path_progress = RewTerm(func=path_progress, weight=50.0, params={"command_name": "tracking"})
    goal_position = RewTerm(func=track_goal_position_exp, weight=2.0, params={"command_name": "tracking", "std": 0.15})
    goal_heading = RewTerm(
        func=track_goal_heading_exp, weight=2.0, params={"command_name": "tracking", "std_yaw": 0.15, "std_pos": 0.15}
    )
    look_at = RewTerm(func=look_at_penalty, weight=-1.0, params={"command_name": "tracking"})
    # 向き誤差の減少 [rad] × 100 → 1 rad 早く向くごとに +2.0 (pivot の価値を見えやすくする, v4)
    heading_progress = RewTerm(func=heading_progress, weight=100.0, params={"command_name": "tracking"})
    walk_command_rate_l2 = RewTerm(func=track_command_rate_l2, weight=-0.05, params={"action_name": "walk_command"})
    mode_switch = RewTerm(func=mode_switch, weight=-0.5, params={"command_name": "base_velocity"})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class K1TrackingEnvCfg(K1FlatGatingCfg):
    observations: K1TrackingObservationsCfg = K1TrackingObservationsCfg()
    actions: K1TrackingActionsCfg = K1TrackingActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 下位コマンドはゲート学習の TransitionCommandCfg (walk / turn 生成器の cfg) を引き継ぎ、遷移だけ外部駆動にする
        gate_cmd = self.commands.base_velocity
        self.commands = K1TrackingCommandsCfg(
            base_velocity=ExternalTransitionCommandCfg(
                asset_name=gate_cmd.asset_name,
                walk_cfg=gate_cmd.walk_cfg,
                turn_cfg=gate_cmd.turn_cfg,
                initial_turn_prob=0.35,  # turn 始まり (目標 = 既定目標) で pivot の価値を学ばせる (v3)
                zero_command_history_on_switch=True,
            )
        )

        # expert / ゲートが見る actions 観測は関節指令 (12) のみ (action 空間が 12 + 5 になるため)
        self.observations.policy.actions.params = {"action_name": "joint_pos"}
        self.observations.critic.actions.params = {"action_name": "joint_pos"}

        self.rewards = K1TrackingRewardsCfg()

        # 経路は最長 5 セグメント × 2 m なので平面の縁を広げる
        self.scene.terrain.terrain_generator = PLANE_HEAVY_TERRAIN_CFG.replace(border_width=15.0)
        self.episode_length_s = 20.0


@configclass
class K1TrackingEnvCfg_PLAY(K1TrackingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.nav_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.tracking.debug_vis = True
