# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import K1RoughEnvCfg
from .velocity_env_cfg import CurriculumCfg
import math
from .mdp.commands import DiscreteVelocityCommandCfg
from .mdp.curriculums import lin_vel_command_curriculum, modify_command_resampling_time_range
from .mdp.events import randomize_phase_freq
from .rough_env_cfg import _PHASE_FREQ


@configclass
class K1FlatCurriculumCfg(CurriculumCfg):
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (1.0, 7.0),
            "num_steps": 8000,
        },
    )
    '''
    lin_vel_command = CurrTerm(
        func=lin_vel_command_curriculum,
        params={
            "command_name": "base_velocity",
            "stages": [(-0.3, 0.3), (-0.6, 0.6)],
            "error_threshold": 0.35,
            "asset_name": "robot",
            "ema_alpha": 0.026,
            "min_updates": 50,
        },
    )
    '''


@configclass
class K1FlatEnvCfg(K1RoughEnvCfg):
    curriculum: K1FlatCurriculumCfg = K1FlatCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_phase_freq = EventTerm(
            func=randomize_phase_freq,
            mode="startup",
            params={
                "base_phase_freq": _PHASE_FREQ,
                "offset_range": (-0.05, 0.05),
            },
        )

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # No height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -0.17
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Ankle_.*"]
        )
        self.commands.base_velocity.rel_standing_envs = 0.02
        prev = self.commands.base_velocity
        self.commands.base_velocity = DiscreteVelocityCommandCfg(
            asset_name=prev.asset_name,
            resampling_time_range=prev.resampling_time_range,
            rel_standing_envs=prev.rel_standing_envs,
            rel_heading_envs=prev.rel_heading_envs,
            heading_command=prev.heading_command,
            heading_control_stiffness=prev.heading_control_stiffness,
            debug_vis=prev.debug_vis,
            ranges=DiscreteVelocityCommandCfg.Ranges(
                lin_vel_x=(0.4, 0.4),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(-math.pi, math.pi),
            ),
            lin_vel_x_resolution=0.2,
            lin_vel_y_resolution=0.1,
            ang_vel_z_resolution=0.2,
        )


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        

        #前見る用コマンド
        """
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        """
