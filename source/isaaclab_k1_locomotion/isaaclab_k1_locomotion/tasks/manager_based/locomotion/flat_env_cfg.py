# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .rough_env_cfg import K1RoughEnvCfg, _PHASE_FREQ
from .velocity_env_cfg import CurriculumCfg
import math
from .mdp.commands import DiscreteVelocityCommandCfg
from .mdp.events import randomize_phase_freq
from .mdp.rewards import feet_landing_impact, feet_landing_vel
from .mdp.curriculums import (
    modify_command_resampling_time_range,
    lin_vel_command_curriculum,
    modify_push_robot,
)


# 段差・坂道なし、ランダムノイズのみの軽く凹凸した地面
NOISY_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.9,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
    },
)


@configclass
class K1FlatCurriculumCfg(CurriculumCfg):
    """K1 Flat 環境用のカリキュラム設定。"""

    # ステップ数が5000を超えたら、コマンドのリサンプリング時間分布の範囲を (1.0, 5.0) に変更
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (1.0, 7.0),
            "num_steps": 8000,
        },
    )

    # より細かいコマンド変動に対応
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (0.5, 7.0),
            "num_steps": 14000,
        },
    )

    # 線速度コマンド範囲を段階的に拡げるカリキュラム
    # 追従誤差(EMA)が threshold を下回るとステージが進む: ±0.3 → ±0.6 → ±1.0
    lin_vel_command = CurrTerm(
        func=lin_vel_command_curriculum,
        params={
            "command_name": "base_velocity",
            "stages_x": [(-0.6, 0.6), (-1.2, 1.2)],
            "stages_y": [(-0.5, 0.5), (-0.8, 0.8)],
            "error_threshold": 0.35,
            "asset_name": "robot",
            "ema_alpha": 0.026,
            "min_updates": 50,
        },
    )

    # push_robot を段階的に強くするカリキュラム
    # 初期値 (EventCfg): interval 7-10s, vel ±0.5 → ±0.5
    push_robot_stage1 = CurrTerm(
        func=modify_push_robot,
        params={
            "term_name": "push_robot",
            "num_steps": 6000,
            "interval_range_s": (4.0, 8.0),
            "velocity_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2)},
        },
    )
    # push_robot_stage2 = CurrTerm(
    #     func=modify_push_robot,
    #     params={
    #         "term_name": "push_robot",
    #         "num_steps": 16000,
    #         "interval_range_s": (3.0, 8.0),
    #         "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3)},
    #     },
    # )

@configclass
class K1FlatEnvCfg(K1RoughEnvCfg):
    curriculum: K1FlatCurriculumCfg = K1FlatCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # 環境毎に歩行周波数 _PHASE_FREQ を ±0.1 Hz の範囲でランダム化 (startup で1度だけ)。
        # phase_obs / feet_phase / foot_clearance_ji_pen がこの per-env 値を自動で参照する。
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
        # 軽い凹凸のみの地面 (段差・坂道なし)
        # self.scene.terrain.terrain_type = "generator"
        # self.scene.terrain.terrain_generator = NOISY_FLAT_TERRAIN_CFG
        # self.scene.terrain.max_init_terrain_level = None
        # No height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.ang_vel_xy_l2.weight = -0.32
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.action_rate_l2.weight = -0.6
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Ankle_.*"]
        )
        # 着地時の衝撃力ペナルティ: 接地瞬間の力ノルムが大きいほどペナルティを与える。
        # 単位は [N]・両足合計なので、過大ペナルティにならないよう重みは小さめにする。
        self.rewards.feet_landing_impact = RewTerm(
            func=feet_landing_impact,
            weight=-1.5e-2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "contact_threshold": 1.0,
            },
        )
        # 着地時の速度ペナルティ: 接地した瞬間の足の(鉛直)速度が大きいほどペナルティを与える。
        # 硬い踏みつけ(下向き速度が大きい着地)を抑制し、柔らかい接地を促す。
        # 単位は [m/s]・両足合計 (着地イベント時のみ非0) なので、重みは衝撃力より大きめにとる。
        self.rewards.feet_landing_vel = RewTerm(
            func=feet_landing_vel,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
                "contact_threshold": 1.0,
                "vertical_only": False,
            },
        )
        # 速度コマンドを離散格子からサンプリングする版に差し替える
        # lin_vel_x / lin_vel_y は lin_vel_command カリキュラムが段階的に拡張する
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
                lin_vel_x=prev.ranges.lin_vel_x,
                lin_vel_y=prev.ranges.lin_vel_y,
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
            lin_vel_x_resolution=0.2,
            lin_vel_y_resolution=0.1,
            ang_vel_z_resolution=0.2,
        )

@configclass
class K1FlatEnvLearnStandingCfg(K1FlatEnvCfg):
    """追加学習で立ち姿勢を覚えるための環境設定。これは予め普通のFlatで学習したポリシーに追加学習する用途"""
    def __post_init__(self):
        super().__post_init__()
        # Rewards
        self.commands.base_velocity.resampling_time_range = (1.0, 5.0)  # コマンドのリサンプリング時間の範囲を変更
        self.commands.base_velocity.rel_standing_envs = 0.3

@configclass
class K1FlatImproveSteadynessCfg(K1FlatEnvCfg):
    """学習済のポリシーに対して安定化のための追加学習を行う際の環境設定"""
    def __post_init__(self):
        super().__post_init__()
        # Rewards
        self.commands.base_velocity.resampling_time_range = (1.0, 4.0)  # コマンドのリサンプリング時間の範囲を変更
        self.rewards.ang_vel_xy_l2.weight = -0.30 * 1.7
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.action_rate_l2.weight = -0.6 * 1.3
        self.rewards.dof_acc_l2.weight = -1.2e-6
        self.rewards.dof_torques_l2.weight = -1.0e-5

@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
