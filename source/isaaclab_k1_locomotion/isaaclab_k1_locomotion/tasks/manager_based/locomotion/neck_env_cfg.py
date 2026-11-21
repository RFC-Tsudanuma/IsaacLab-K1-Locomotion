# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""首 (ヨー・ピッチ) 安定化環境 (Isaac-Neck-K1-v0, 2026-11-20)。設計は scripts/rsl_rl/neck_compensate.md。

凍結した歩行 expert (frozen_policies/best_walk) が乱数の歩行コマンドで歩いている上で、首 2 軸の
関節目標への残差を学習する。環境は歩行 expert の学習環境 `K1FlatStancePlaneWideCfg` を継承し
(歩行コマンドの分布・地形・摩擦 DR・外乱・位相が expert の学習分布と一致する)、以下だけを変える:

* robot: 首 2 関節を可動にした URDF (K1_locomotion_neck.urdf) + 首のアクチュエータ。
* action: ``joint_pos`` (12, expert) に ``neck`` (2, 残差) を追加。
* command: ``neck`` (`NeckCommandCfg`: 安定化器 + q_cmd 生成 + センサ遅延) を追加。``base_velocity`` は親のまま。
* observation: ``neck_policy`` / ``neck_critic`` グループを追加 (歩行 expert 用の 3 グループは据え置き)。
* reward: 歩行用の報酬は全て捨て、ブラー (主)・指令方向からのずれ・行動変化率・トルクのみ。

学習は ``scripts/rsl_rl/train_neck.sh``。方策・PPO は `agents/neck_policy.py` (`K1NeckPPORunnerCfg`)。
比較用: ``env.actions.neck.scale=0`` で解析 IK のみ、``env.commands.neck.stabilize=false`` も足すと安定化なし。
"""

import os

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import K1FlatObservationsCfg, K1FlatStancePlaneWideCfg
from .mdp import phase_obs
from .mdp.neck import (
    NECK_JOINTS,
    NeckActionCfg,
    NeckCommandCfg,
    neck_blur_speed,
    neck_blur_speed_sq,
    neck_eval,
    neck_pointing_error,
)
from .mdp.waypoint import walk_command_rate_l2
from .rough_env_cfg import _K1_URDF_PATH
from .velocity_env_cfg import ActionsCfg

_NECK_URDF_PATH = os.path.join(os.path.dirname(_K1_URDF_PATH), "K1_locomotion_neck.urdf")
_NECK_CFG = SceneEntityCfg("robot", joint_names=NECK_JOINTS, preserve_order=True)


@configclass
class K1NeckActionsCfg(ActionsCfg):
    """joint_pos (12, expert の関節指令) の後ろに首の残差 (2) を並べる。"""

    neck = NeckActionCfg(asset_name="robot")


@configclass
class K1NeckPolicyCfg(ObsGroup):
    """首方策の観測 (1 step 26 次元 × 履歴 5)。実機では IMU・首エンコーダ・歩行プロセスの内部値で構成できる。"""

    neck = ObsTerm(func=mdp.generated_commands, params={"command_name": "neck"})  # ノイズ・遅延は NeckCommand 内で付く
    last_action = ObsTerm(func=mdp.last_action, params={"action_name": "neck"})
    gait_phase = ObsTerm(func=phase_obs)  # params は env cfg が歩行 expert と同じ値を入れる
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 5
        self.flatten_history_dim = True


@configclass
class K1NeckCriticCfg(K1NeckPolicyCfg):
    """critic は actor の観測 + 真値 (遅延・ノイズなし)。"""

    neck_eval = ObsTerm(func=neck_eval)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    neck_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": _NECK_CFG})
    neck_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": _NECK_CFG})


@configclass
class K1NeckObservationsCfg(K1FlatObservationsCfg):
    neck_policy: K1NeckPolicyCfg = K1NeckPolicyCfg()
    neck_critic: K1NeckCriticCfg = K1NeckCriticCfg()


@configclass
class K1NeckRewardsCfg:
    """設計書 4.4。値は RewardManager が dt (0.02) を掛ける。blur / pointing は 1 step 遅れ (mdp/neck.py)。"""

    # 主: 目標方向に対する光軸の相対角速度の 2 乗 [(rad/s)²]。1 乗 (blur) だと、制御できない 6 Hz 以上の揺れが誤差の
    # 向きを決めてしまい低周波の改善が報酬に出にくく、学習が頭打ちになった (2026-11-20: 2 乗 0.246 / 1 乗 0.254 で停滞)。
    blur_sq = RewTerm(func=neck_blur_speed_sq, weight=-10.0)
    blur = RewTerm(func=neck_blur_speed, weight=0.0)  # ログ用 (1 乗)
    pointing = RewTerm(func=neck_pointing_error, weight=-2.0)  # 弱め: 光軸と目標方向のなす角 [rad]
    action_rate = RewTerm(func=walk_command_rate_l2, weight=-0.05, params={"action_name": "neck"})
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.01, params={"asset_cfg": _NECK_CFG})


@configclass
class K1NeckEnvCfg(K1FlatStancePlaneWideCfg):
    observations: K1NeckObservationsCfg = K1NeckObservationsCfg()
    actions: K1NeckActionsCfg = K1NeckActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 首を可動にする。ゲインは実機 (booster_k1_locomotion の HEAD_KP / HEAD_KD)、armature は MuJoCo モデルの値。
        # 指令の遅延は 0〜1 物理 step (≤ 5 ms、env ごと。2026-11-20 ユーザー指定)。脚と同じ 2〜7 step (10〜35 ms) にすると
        # 先読み付き IK が 6 Hz 以上の揺れを増幅して効果がほぼ消える (安定化の効きはこの遅延で決まる)。
        robot = self.scene.robot
        self.scene.robot = robot.replace(
            spawn=robot.spawn.replace(asset_path=_NECK_URDF_PATH),
            actuators={
                **robot.actuators,
                "neck": DelayedPDActuatorCfg(
                    joint_names_expr=NECK_JOINTS,
                    effort_limit=6.0,
                    velocity_limit=18.0,
                    stiffness=10.0,
                    damping=0.0,  # Kd は NeckActionCfg.damping (PhysX の陰的な関節ダンピング) で入れる
                    armature=0.002,
                    min_delay=0,
                    max_delay=1,
                ),
            },
        )

        # expert が見る actions 観測は関節指令 (12) のみ (action 空間が 12 + 2 になるため)
        self.observations.policy.actions.params = {"action_name": "joint_pos"}
        self.observations.critic.actions.params = {"action_name": "joint_pos"}

        # 位相・歩行周波数は歩行 expert と同じ式 (平滑化の窓長 = 歩行 1 周期)
        phase = dict(self.observations.policy.gait_phase.params)
        self.observations.neck_policy.gait_phase.params = phase
        self.observations.neck_critic.gait_phase.params = phase
        self.commands.neck = NeckCommandCfg(phase_freq_params={k: v for k, v in phase.items() if k != "cmd_threshold"})

        self.rewards = K1NeckRewardsCfg()


@configclass
class K1NeckEnvCfg_PLAY(K1NeckEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
