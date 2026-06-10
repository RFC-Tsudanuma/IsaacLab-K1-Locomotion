# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draft env cfg for hierarchical dribble training.

Design:
- Robot / terrain / actions are inherited from ``K1FlatEnvCfg``.
- The scene adds a soccer ball + two foot-vs-ball contact sensors, copied from
  the kick env (``kick_env_cfg.py``).
- Commands: a ``kick_direction`` command holds a world-frame xy unit vector
  representing where the high-level policy should drive the ball. The original
  ``base_velocity`` command from FlatEnv is kept so the ``low_level`` obs group
  retains its ``velocity_commands`` slot (whose value is overwritten by the
  hierarchical wrapper before reaching the frozen policy anyway).
- Observations:

    * ``policy`` group  = FlatEnv K1PolicyCfg - ``velocity_commands`` + ``ball_pos_rel`` + ``kick_direction_b``
    * ``critic`` group  = FlatEnv K1CriticCfg - ``velocity_commands`` + ``ball_pos_rel`` + ``ball_vel`` + ``kick_direction_b``
    * ``low_level`` group = FlatEnv K1PolicyCfg as-is                   (for frozen)

  The ``low_level`` group is kept structurally identical to what the frozen
  walking policy was trained on so it can be consumed directly by the frozen
  net. ``scripts/rsl_rl/train_dribble.py`` should be launched with
  ``--low_level_obs_group low_level``.

- Rewards (in addition to inherited FlatEnv rewards):
    * ``ball_velocity_along_kick`` — ball velocity projected onto the world-frame kick direction
    * ``ball_speed`` — magnitude of the ball's world-frame xy velocity
    * ``robot_velocity_toward_ball`` — small shaping reward: robot velocity component
      toward the ball (zeroed out when very close)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import K1FlatEnvCfg
from .rough_env_cfg import K1CriticCfg, K1ObservationsCfg, K1PolicyCfg
from .velocity_env_cfg import CommandsCfg, MySceneCfg
from .mdp.commands import KickDirectionCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from .mdp.curriculums import randomize_ball_init_velocity
from .mdp.events import reset_prev_high_action
from .mdp.observations import ball_pos_rel, ball_vel, kick_direction_b, last_high_action
from .mdp.terminations import time_after_ball_contact
from .mdp.rewards import (
    ball_speed,
    ball_velocity_along_kick,
    com_jerk_l2,
    high_action_rate_l2,
    high_action_smoothness_l2,
    robot_facing_ball,
    robot_velocity_toward_ball,
)


@configclass
class K1DribbleSceneCfg(MySceneCfg):
    """Scene: FlatEnv scene + soccer ball + foot-vs-ball contact sensors."""

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,  # 11cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),  # 450g
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.7,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.11)),
    )

    contact_balls_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    contact_balls_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )


@configclass
class K1DribblePolicyCfg(K1PolicyCfg):
    """High-level policy obs: FlatEnv K1PolicyCfg + ball position + kick direction (in base frame).

    ``velocity_commands`` の代わりに **前回の高レベル action (= 歩行コマンド)** を
    観測する。位置 (term の並び順) は親の ``velocity_commands`` と同じ。
    """

    # K1PolicyCfg.velocity_commands を同名の項で上書き → 並び順は親と同じ。
    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    ball_pos_rel = ObsTerm(func=ball_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
    kick_direction_b = ObsTerm(func=kick_direction_b, params={"command_name": "kick_direction"})


@configclass
class K1DribbleCriticCfg(K1CriticCfg):
    """High-level critic obs: FlatEnv K1CriticCfg + ball pos/vel (privileged) + kick direction."""

    # 上位 policy と同様に velocity_commands を前回 high-level action に差し替え。
    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    ball_pos_rel = ObsTerm(func=ball_pos_rel)
    ball_vel = ObsTerm(func=ball_vel)
    kick_direction_b = ObsTerm(func=kick_direction_b, params={"command_name": "kick_direction"})


@configclass
class K1DribbleObservationsCfg(K1ObservationsCfg):
    """Dribble obs groups.

    ``low_level`` is structurally identical to the FlatEnv ``K1PolicyCfg`` that
    the frozen walking policy was trained on. The hierarchical wrapper reads
    this group and overwrites its ``velocity_commands`` slice with the
    high-level action before feeding the frozen policy.
    """

    policy: K1DribblePolicyCfg = K1DribblePolicyCfg()
    critic: K1DribbleCriticCfg = K1DribbleCriticCfg()
    low_level: K1PolicyCfg = K1PolicyCfg()


@configclass
class K1DribbleCommandsCfg(CommandsCfg):
    """FlatEnv の ``base_velocity`` に加えて、ワールド座標系のキック方向コマンドを追加。"""

    kick_direction = KickDirectionCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
    )


@configclass
class K1DribbleRewardsCfg:
    """Dribble 専用の報酬。K1Rewards (歩行用) は継承せずに丸ごと置き換える。"""

    # --- ボール関連の主報酬 ---
    # キック方向 (ワールド座標) に沿ったボール速度成分が大きいほど高い報酬。
    ball_velocity_along_kick = RewTerm(
        func=ball_velocity_along_kick,
        weight=5.0,
        params={"command_name": "kick_direction", "max_speed": 3.0},
    )
    # ボール速度の大きさ (方向問わず)。max_speed で正規化、上限 1.0。
    ball_speed = RewTerm(
        func=ball_speed,
        weight=1.0,
        params={"max_speed": 3.0},
    )

    # --- Shaping ---
    # ロボットがボールに向かって進んでいる成分。
    robot_velocity_toward_ball = RewTerm(
        func=robot_velocity_toward_ball,
        weight=0.5,
        params={"max_speed": 1.3, "min_distance": 0.15},
    )
    # ロボット Trunk の正面 (base +x) がボール方向を向いているほど大きい [0, 1]。
    robot_facing_ball = RewTerm(
        func=robot_facing_ball,
        weight=0.5,
        params={"min_distance": 0.15},
    )

    # --- ペナルティ ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    # 上位ポリシーが出力する 3D 歩行コマンドに対する平滑性ペナルティ。
    # 元の action_smoothness_l2 は env.action_manager.action (= frozen の 22D 関節指令)
    # を見ており上位ポリシーが直接制御できないので、上位 action 用に差し替え。
    action_smoothness_l2 = RewTerm(func=high_action_smoothness_l2, weight=-0.08)
    # 上位ポリシー action の 1 ステップ差分 (action rate) ペナルティ。
    # コマンドが急変するとロボットが追従しきれず歩行が乱れるので軽く抑える。
    action_rate_l2 = RewTerm(func=high_action_rate_l2, weight=-0.3)
    # ロボット root body COM の jerk (加速度の時間微分) ペナルティ。
    # 値域が大きくなりやすいので重みは非常に小さめから始める。
    com_jerk_l2 = RewTerm(func=com_jerk_l2, weight=-1e-6)


@configclass
class K1DribbleEnvCfg(K1FlatEnvCfg):
    """K1FlatEnv + a soccer ball, used as the dribble training env."""

    scene: K1DribbleSceneCfg = K1DribbleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1DribbleObservationsCfg = K1DribbleObservationsCfg()
    commands: K1DribbleCommandsCfg = K1DribbleCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 報酬は dribble 用に丸ごと置き換える。
        # ※ K1FlatEnvCfg.__post_init__ が K1Rewards の各項を弄っている分は捨てて構わない。
        self.rewards = K1DribbleRewardsCfg()

        # ※ 上位 policy / critic の ``velocity_commands`` 項は K1DribblePolicyCfg /
        # K1DribbleCriticCfg 側で ``last_high_action`` に上書き済み (位置はそのまま)。
        # ``low_level`` 観測グループは K1PolicyCfg のまま残してあるので、wrapper が
        # 入力時に同スロットを現ステップの高レベル action で塗り潰す。

        # ロボットの初期 yaw を固定。ボールの reset 範囲 (x=(0.15, 1.5)) は
        # env-frame で評価されるため、yaw をランダム化するとロボットの真後ろにも
        # スポーンしうる。dribble では「ボールは常に正面 (x > 0)」を保ちたいので yaw=0 に固定する。
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        # リセット時に ``_prev_high_action`` バッファ (HierarchicalVecEnvWrapper が
        # 用意) を 0 にする。これがないと新エピソード最初の観測が前エピソードの
        # 高レベル action を引きずる。
        self.events.reset_prev_high_action = EventTerm(
            func=reset_prev_high_action,
            mode="reset",
        )

        # FlatEnv 由来のカリキュラム (base_velocity 追従/push_robot 強化) は
        # 高レベルタスクには不要なので全て無効化。
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.push_robot_stage1 = None

        # # DEBUG: 学習不調の切り分けのため一旦無効化。
        # # ボールに初めて接触してから 4 秒後にエピソード終了。
        # # time_out=True としているので termination_penalty の対象外 (時間切れ扱い)。
        # self.terminations.after_ball_contact_timeout = DoneTerm(
        #     func=time_after_ball_contact,
        #     time_out=True,
        #     params={"time_after_contact": 3.0},
        # )

        # Spawn the ball in a random position around the robot on reset.
        # 初期は静止 (velocity_range は全て 0)。総ステップ数の半分を超えたら
        # 下の randomize_ball_init_velocity カリキュラムでランダム初速に切り替わる。
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "pose_range": {
                    "x": (0.15, 1.5),
                    "y": (-1.5, 1.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("soccer_ball"),
            },
        )

        # self.curriculum.randomize_ball_init_velocity = CurrTerm(
        #     func=randomize_ball_init_velocity,
        #     params={
        #         "term_name": "reset_ball",
        #         "num_steps": 3000,
        #         "velocity_range": {
        #             "x": (-0.2, 0.2),
        #             "y": (-0.2, 0.2),
        #             "z": (0.0, 0.0),
        #             "roll": (0.0, 0.0),
        #             "pitch": (0.0, 0.0),
        #             "yaw": (0.0, 0.0),
        #         },
        #     },
        # )


@configclass
class K1DribbleEnvCfg_PLAY(K1DribbleEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
