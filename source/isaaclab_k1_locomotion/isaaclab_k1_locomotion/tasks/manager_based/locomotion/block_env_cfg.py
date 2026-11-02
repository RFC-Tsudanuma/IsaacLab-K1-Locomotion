# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール阻止 (キーパー) 環境 (Isaac-Block-K1-v0, 2026-11-01)。

凍結した歩行 expert (frozen_policies/best_walk) を事前分布として、飛んでくるボール
(0.4〜6.0 m/s, 自分の左右 ±1.0 m を狙って来る, 転がり or 20 cm 程度のバウンド) を
自分の後ろへ通さない (止める or 前に弾く) 方策を学習する。

環境は歩行 expert の学習環境 `K1FlatStancePlanePolishCfg` を継承し (waypoint と同じ流儀)、以下を差し替える:

* scene: サッカーボール + 足とボールの接触センサ (dribble と同じ) を追加。ボールは反発・減衰あり。
* action: ``joint_pos`` (12, expert + 残差の関節指令) + ``walk_command`` (3, 歩行コマンド) +
  ``joint_residual`` (12, 残差の保持用)。方策は 15 次元 (コマンド + 残差) を出し、
  `BlockPPO.act` が 27 次元に組み立てる。
* command: ``base_velocity`` = `ExternalVelocityCommand` (walk_command のコピー)、
  ``ball_launch`` = `BallLaunchCommand` (射出 + ゴールライン + 判定)。
* observation (方策用に追加、expert 用の ``command`` / ``policy`` / ``critic`` は据え置き):
    - ``block_policy``: 内界センサ (歩行 K1PolicyCfg から線速度を除いたもの、ノイズあり) + 残差 + ゴールライン
    - ``block_critic``: 同上 + 線速度 (ノイズなし)
    - ``ball_noisy``:   遅延 + 30 fps + ノイズ付きボール観測の履歴 (actor の encoder 入力)
    - ``ball_true``:    真値ボール観測の履歴 (critic の encoder 入力 / latent 蒸留の教師)
* reward: 阻止 (+) / 失点 (−) / 転倒 (−) / 足を予測到達点へ (shaping) / 残差・コマンドの大きさと変化率 (−)
* termination: 失点・阻止で終了 (どちらも time_out=False)。
* reset: 立位スタート (関節 ±5°, 初速 ±0.1)。

学習は train.py (``--task Isaac-Block-K1-v0``)。方策・PPO は `agents/block_policy.py`、
runner cfg は `K1BlockPPORunnerCfg`。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .dribble_env_cfg import K1DribbleSceneCfg
from .flat_env_cfg import PLANE_HEAVY_TERRAIN_CFG, K1FlatObservationsCfg, K1FlatStancePlanePolishCfg
from .rough_env_cfg import K1CriticCfg, K1PolicyCfg
from .velocity_env_cfg import ActionsCfg
from .mdp.block import (
    BALL_HISTORY,
    JOINT_DIM,
    BallLaunchCommandCfg,
    DelayedBallObs,
    ball_conceded,
    ball_resolved,
    ball_state_rel,
    foot_on_intercept,
    joint_residual_l2,
)
from .mdp.waypoint import ExternalVelocityCommandCfg, WalkCommandActionCfg, walk_command_rate_l2


@configclass
class K1BlockSceneCfg(K1DribbleSceneCfg):
    """dribble のシーン (ボール + 足接触センサ) のボールだけ反発・減衰付きに差し替える。"""

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # 転がり抵抗・空気抵抗の代わり (PhysX の球は無減衰で永久に転がる)
                linear_damping=0.1,
                angular_damping=0.1,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.7),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            # 地形側は restitution 0 × multiply なので、球側を max にして反発を有効にする
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, restitution=0.6, restitution_combine_mode="max"
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.11)),
    )


@configclass
class K1BlockActionsCfg(ActionsCfg):
    """joint_pos (12, expert + 残差) / walk_command (3) / joint_residual (12, 残差の保持用)。"""

    walk_command = WalkCommandActionCfg(asset_name="robot")
    joint_residual = WalkCommandActionCfg(asset_name="robot", dim=JOINT_DIM)


@configclass
class K1BlockPolicyCfg(K1PolicyCfg):
    """方策の内界観測: 歩行 K1PolicyCfg (線速度なし・ノイズあり) + 前回残差 + ゴールライン。履歴なし。"""

    actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    last_residual = ObsTerm(func=mdp.last_action, params={"action_name": "joint_residual"})
    goal_line = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_launch"})


@configclass
class K1BlockCriticCfg(K1CriticCfg):
    """critic の内界観測: 歩行 K1CriticCfg (線速度・ZMP 込み・ノイズなし) + 前回残差 + ゴールライン。"""

    actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    last_residual = ObsTerm(func=mdp.last_action, params={"action_name": "joint_residual"})
    goal_line = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_launch"})


@configclass
class K1BallTrueCfg(ObsGroup):
    """真値ボール観測 (相対位置 xy + 速度 xy) の履歴。"""

    ball = ObsTerm(func=ball_state_rel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = BALL_HISTORY
        self.flatten_history_dim = True


@configclass
class K1BallNoisyCfg(ObsGroup):
    """カメラ検出を模したボール観測 (遅延 + 30 fps + 検出欠け + ガウスノイズ) の履歴。ノイズは項の中で乗せる。"""

    ball = ObsTerm(
        func=DelayedBallObs,
        params={
            "delay_range": (0.16, 0.24),  # 認識まで含めた遅延 ≈ 200 ms (env ごとに抽選)
            "camera_fps": 30.0,
            "pos_noise": 0.05,  # 位置 σ=5 cm (±10 cm ≈ 2σ)
            "vel_noise": 1.0,  # 速度 σ = 1.0 + 0.3·|v| m/s (カルマンフィルタ推定: 速いほど誤差大)
            "vel_noise_rel": 0.3,
            "drop_prob": 0.1,  # 検出欠け
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = BALL_HISTORY
        self.flatten_history_dim = True


@configclass
class K1BlockObservationsCfg(K1FlatObservationsCfg):
    block_policy: K1BlockPolicyCfg = K1BlockPolicyCfg()
    block_critic: K1BlockCriticCfg = K1BlockCriticCfg()
    ball_noisy: K1BallNoisyCfg = K1BallNoisyCfg()
    ball_true: K1BallTrueCfg = K1BallTrueCfg()


@configclass
class K1BlockCommandsCfg:
    base_velocity = ExternalVelocityCommandCfg(asset_name="robot", action_name="walk_command")
    ball_launch = BallLaunchCommandCfg(asset_name="robot")


@configclass
class K1BlockRewardsCfg:
    """阻止タスクの報酬 (歩行用の報酬は継承しない)。値は RewardManager が dt (0.02) を掛ける。"""

    # 主目的: 阻止 (+2 実効) / 失点 (−4 実効)。転倒も失点と同程度に罰する。
    blocked = RewTerm(func=mdp.is_terminated_term, weight=100.0, params={"term_keys": "ball_resolved"})
    conceded = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "ball_conceded"})
    fell = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": ["base_contact", "base_height"]})
    # shaping: ボール接近中、予測到達点 (ライン上の y) に足を置く
    foot_on_intercept = RewTerm(func=foot_on_intercept, weight=0.5, params={"std": 0.15})
    # 事前分布 (歩行 expert) からのずれと、その変化率・コマンド変化率は軽く抑える
    joint_residual_l2 = RewTerm(func=joint_residual_l2, weight=-0.05)
    joint_residual_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.02, params={"action_name": "joint_residual"})
    walk_command_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.05, params={"action_name": "walk_command"})


@configclass
class K1BlockEnvCfg(K1FlatStancePlanePolishCfg):
    scene: K1BlockSceneCfg = K1BlockSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1BlockObservationsCfg = K1BlockObservationsCfg()
    actions: K1BlockActionsCfg = K1BlockActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 親の __post_init__ は歩行コマンド cfg を触るので、コマンドはその後で差し替える
        self.commands = K1BlockCommandsCfg()

        # expert が見る actions 観測は関節指令 (12) のみ (action 空間が 12 + 3 + 12 になるため)
        self.observations.policy.actions.params = {"action_name": "joint_pos"}
        self.observations.critic.actions.params = {"action_name": "joint_pos"}

        self.rewards = K1BlockRewardsCfg()

        self.terminations.ball_conceded = DoneTerm(func=ball_conceded)
        self.terminations.ball_resolved = DoneTerm(func=ball_resolved)

        # 歩行コマンド系のカリキュラムは全て無効 (base_velocity は ExternalVelocityCommand)
        self.curriculum.lin_vel_command = None
        self.curriculum.extreme_commands = None
        self.curriculum.command_resampling_time_range = None
        self.curriculum.push_robot_stage1 = None

        # 立位スタート (waypoint と同じ: 親の ±30° / ±0.5 m/s は開始直後の転倒を増やすだけ)
        self.events.reset_robot_joints.params["position_range"] = (-math.radians(5.0), math.radians(5.0))
        self.events.reset_base.params["velocity_range"] = {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
        }

        # ボールは最長 5 m 先から来て弾かれて飛ぶので平面の縁を広げる
        self.scene.terrain.terrain_generator = PLANE_HEAVY_TERRAIN_CFG.replace(border_width=15.0)

        # 到達 ≤ 3 s + 止まる/弾く時間。阻止・失点で早期終了するので大半はこれより短い。
        self.episode_length_s = 8.0


@configclass
class K1BlockEnvCfg_PLAY(K1BlockEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.block_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
