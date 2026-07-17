# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークキック環境。

K1FlatEnvCfg を継承し、ボール蹴りタスク向けの観測・報酬・コマンドを追加する。

観測 (policy, 55次元) は B-Human "A Modular Ball Kicking Behavior with
Reinforcement Learning" (Reichenberg & Frese) の構成を K1 向けに移植したもの。
Flags (3次元) は使わない。歩行タスク (K1PolicyCfg) の観測とは独立に定義しており、
locomotion 側には影響しない。
"""

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from ..locomotion.rough_env_cfg import _COMMAND_THRESHOLD, _PHASE_FREQ
from ..locomotion.velocity_env_cfg import JOINT_NAMES_K1, ObservationsCfg
from . import mdp

_BALL_RADIUS = 0.11

# --------------------------------------------------------------------------- #
# キック報酬の共有パラメータ
#
# kick_state (latch 状態) はこの 3 つで決まる。全てのキック報酬項と kick_finished
# termination に **同じ値** を渡すこと（先に呼ばれた項の値でその step の状態が確定するため、
# 項ごとに違う値を渡すと結果が評価順に依存してしまう）。
# --------------------------------------------------------------------------- #
_KICK_STATE_PARAMS = {
    "r_stance": 0.25,  # P_kick 半径: ボール後方どれだけの点を理想キック立ち位置とするか [m]
    "alpha": 0.5,      # G の追従係数 (<1 でないとロボットが詰めても G が逃げる)
    "v_thresh": 0.8,   # 値 latch のトリガー速度 (ball release の近似) [m/s]
}
# τ_direction のシェイピング係数 [rad]。項1-3 で共有する。
_SIGMA_DIRECTION = 0.35

# --------------------------------------------------------------------------- #
# キック後の猶予窓
#
# latch から何秒エピソードを続けるか。蹴った直後に転ぶ挙動を罰するには、転倒が完成する
# (胴体が接地して base_contact が発火する) まで窓が届いている必要がある。K1 は重心高
# 0.6m 程度で、片足を振ってバランスを崩してから接地するまで概ね 1-1.5 秒かかるため
# 2.0 秒とる。0.6 秒では転倒の前にエピソードが終わり、転倒罰も、姿勢を支える dense な
# ペナルティ (base_height_penalty, flat_orientation_l2 など) も効かなかった。
#
# NOTE: RewardManager は value = func * weight * dt で払うので、post-latch に dense で
#       払う項1-3 の累積は「weight × 凍結値 × 窓の秒数」= 窓の長さに比例する。窓を延ばす
#       ぶん weight を割り戻さないとキック報酬だけが膨らみ、転倒罰 (-100 * dt = -2.0 の
#       一度きり) や歩行の安定ペナルティが相対的に薄まってしまう。
#       → _KICK_W_SCALE で割り戻し、窓を変えてもキック 1 回あたりの収益を一定に保つ。
# --------------------------------------------------------------------------- #
_KICK_WINDOW_S = 2.0
_CTRL_DT = 0.02  # decimation(4) * sim.dt(0.005) = 50Hz
_KICK_DELAY_STEPS = int(_KICK_WINDOW_S / _CTRL_DT)  # 100 steps
# 仕様書の weight は窓 0.6 秒 (30 steps) 前提の配分なので、その比で割り戻す。
_KICK_W_SCALE = 0.6 / _KICK_WINDOW_S


def _leg_joints() -> SceneEntityCfg:
    """脚 12 関節を JOINT_NAMES_K1 の順で参照する cfg。

    SceneEntityCfg は resolve() で joint_ids を書き込む可変オブジェクトなので、
    項ごとに新しいインスタンスを作って共有しない。
    """
    return SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)


def _sole() -> SceneEntityCfg:
    """足裏。論文の評価表が "Left Sole" 基準なので左足を使う。"""
    return SceneEntityCfg("robot", body_names="left_foot_link")


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickPolicyCfg(ObsGroup):
    """Actor 用 55 次元観測。実機で得られる情報のみ（真の線速度は含めない）。

    順序は論文の表と一致させること (concatenate_terms は宣言順に連結する):
      gravity(3) + ang_vel(3) + sole(3) + gait_phase(2) + joint_pos(12)
      + joint_vel(12) + prev_joint_request(12) + phase_factor_offset(1)
      + kick_direction(2) + target_kick_velocity(1) + ball_vel(2)
      + prev_ball_pos(2) = 55
    """

    # 1. Gravity (3)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    # 2. Angular Velocity (3)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    # 3. Current Sole 3D Position (3)
    sole_pos = ObsTerm(
        func=mdp.sole_pos_b,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": _sole()},
    )
    # 4. Gait Phase (2)
    gait_phase = ObsTerm(
        func=mdp.gait_phase_sincos,
        params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD},
    )
    # 5. Joint Position (12)
    joint_pos = ObsTerm(
        func=loco_mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.03, n_max=0.03),
        params={"asset_cfg": _leg_joints()},
    )
    # 6. Joint Velocity (12)
    joint_vel = ObsTerm(
        func=loco_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": _leg_joints()},
    )
    # 7. Previous Joint Request (12) — 前ステップのアクション（目標関節角）
    prev_joint_request = ObsTerm(func=loco_mdp.last_action)
    # 8. Gait Phase Factor Offset (1)
    gait_phase_factor_offset = ObsTerm(
        func=mdp.gait_phase_factor_offset,
        params={"base_phase_freq": _PHASE_FREQ},
    )
    # 9. Kick Direction (2) — base 相対の単位ベクトル
    kick_direction = ObsTerm(func=mdp.kick_dir_b, params={"command_name": "kick_direction"})
    # 10. Target Kick Velocity (1)
    target_kick_velocity = ObsTerm(
        func=mdp.target_kick_velocity,
        params={"command_name": "kick_direction"},
    )
    # 11. Ball 2D Velocity (2) — base 相対
    ball_vel = ObsTerm(func=mdp.ball_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1))
    # 12. Previous Ball 2D Position (2) — base 相対
    prev_ball_pos = ObsTerm(func=mdp.prev_ball_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1WalkKickCriticCfg(ObsGroup):
    """Critic 用：policy と同じ項をノイズ無しで並べ、特権情報を追加する。

    特権情報は真の base 線速度と、遅延なしのボール現在位置。
    """

    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel)
    sole_pos = ObsTerm(func=mdp.sole_pos_b, params={"asset_cfg": _sole()})
    gait_phase = ObsTerm(
        func=mdp.gait_phase_sincos,
        params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD},
    )
    joint_pos = ObsTerm(func=loco_mdp.joint_pos_rel, params={"asset_cfg": _leg_joints()})
    joint_vel = ObsTerm(func=loco_mdp.joint_vel_rel, params={"asset_cfg": _leg_joints()})
    prev_joint_request = ObsTerm(func=loco_mdp.last_action)
    gait_phase_factor_offset = ObsTerm(
        func=mdp.gait_phase_factor_offset,
        params={"base_phase_freq": _PHASE_FREQ},
    )
    kick_direction = ObsTerm(func=mdp.kick_dir_b, params={"command_name": "kick_direction"})
    target_kick_velocity = ObsTerm(
        func=mdp.target_kick_velocity,
        params={"command_name": "kick_direction"},
    )
    ball_vel = ObsTerm(func=mdp.ball_vel_b)
    prev_ball_pos = ObsTerm(func=mdp.prev_ball_pos_b)

    # -- 特権情報
    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
    ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1WalkKickObservationsCfg(ObservationsCfg):
    policy: K1WalkKickPolicyCfg = K1WalkKickPolicyCfg()
    critic: K1WalkKickCriticCfg = K1WalkKickCriticCfg()


@configclass
class K1WalkKickEnvCfg(K1FlatEnvCfg):
    observations: K1WalkKickObservationsCfg = K1WalkKickObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # Scene: サッカーボール + 足-ボール接触センサー
        # ------------------------------------------------------------------ #
        self.scene.soccer_ball = RigidObjectCfg(
            prim_path="/World/envs/env_.*/SoccerBall",
            spawn=sim_utils.SphereCfg(
                radius=_BALL_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                mass_props=MassPropertiesCfg(mass=0.45),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0),
                    metallic=0.0,
                    roughness=0.7,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, _BALL_RADIUS)),
        )
        self.scene.contact_balls_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
        )
        self.scene.contact_balls_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
        )

        # ------------------------------------------------------------------ #
        # Commands: base_velocity をボール追従コマンドに置き換え
        #   vx = ロボットフレームでのボール相対 x 位置（クランプ済み）
        #   vy = ロボットフレームでのボール相対 y 位置（クランプ済み）
        #   wz = 0
        # ------------------------------------------------------------------ #
        # follow_ball=False にすると通常の歩行コマンドに戻るので (walk phase)、
        # K1FlatEnvCfg が設定した DiscreteVelocityCommandCfg の中身をそのまま引き継ぐ。
        # こうしておくと walk phase は元の歩行タスクと同一の速度コマンド分布になり、
        # lin_vel_command / command_resampling_time_range カリキュラムもそのまま効く。
        prev = self.commands.base_velocity
        self.commands.base_velocity = mdp.BallFollowVelocityCommandCfg(
            asset_name=prev.asset_name,
            resampling_time_range=prev.resampling_time_range,
            rel_standing_envs=prev.rel_standing_envs,
            rel_heading_envs=prev.rel_heading_envs,
            heading_command=prev.heading_command,
            heading_control_stiffness=prev.heading_control_stiffness,
            debug_vis=False,
            ranges=prev.ranges,
            lin_vel_x_resolution=prev.lin_vel_x_resolution,
            lin_vel_y_resolution=prev.lin_vel_y_resolution,
            ang_vel_z_resolution=prev.ang_vel_z_resolution,
            follow_ball=True,
            max_vel=1.0,
            max_ang_vel=1.0,
            kick_direction_command_name="kick_direction",
            # G の計算はキック報酬と同じ kick_state を共有するので、値も必ず揃える
            **_KICK_STATE_PARAMS,
        )

        # 蹴り方向 + 目標ボール速度コマンドを追加
        self.commands.kick_direction = mdp.KickDirectionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1e9, 1e9),
            heading_command=False,
            debug_vis=True,
            target_speed_range=(1.0, 4.0),  # 目標ボール速度 [m/s]
            ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(-math.pi / 4, math.pi / 4),  # ロボット正面から左右 45° のオフセット
            ),
        )

        # ------------------------------------------------------------------ #
        # Events: ボール位置リセット
        # ------------------------------------------------------------------ #
        # NOTE: perturb_ball（定期的にボールをずらすイベント）は無効化した。
        #       P_kick はエピソード開始時のボール位置に固定するため、途中でボールが
        #       動くと理想キック立ち位置が実際のボールとずれてしまう。
        # NOTE: dist_range の下限は「リセット直後にロボットの足がボールに触れない距離」。
        #       base 中心からの距離なので、ball_radius (0.11m) と足の張り出しを考えると
        #       0.3m では正面に湧いたときに接触する。r_stance (0.25m) より十分外側でもある。
        self.events.reset_ball = EventTerm(
            func=mdp.reset_ball_in_front_of_robot,
            mode="reset",
            params={
                "dist_range": (0.5, 0.8),
                "half_angle": 1.047,
                "ball_radius": _BALL_RADIUS,
            },
        )

        # ------------------------------------------------------------------ #
        # Terminations: 1 エピソード = 1 キック
        # ------------------------------------------------------------------ #
        # latch (kick_done) 成立から delay_steps 後にエピソード終了。
        # NOTE: この項は kick_state の毎ステップ更新も担っている（rewards.py 冒頭の NOTE 参照）。
        self.terminations.kick_finished = DoneTerm(
            func=mdp.kick_finished,
            params={**_KICK_STATE_PARAMS, "delay_steps": _KICK_DELAY_STEPS},
        )

        # ------------------------------------------------------------------ #
        # Rewards: キック関連を B-Human の報酬テーブルに全面置換
        # ------------------------------------------------------------------ #
        # 速度追従（BallFollowVelocityCommand によるボール追従）は残し、Walk Speed を足す。
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # 項7. Termination: 転倒・低姿勢のみを罰する。
        # kick_finished（キック成功による終了）まで罰してしまわないよう、
        # is_terminated ではなく is_terminated_term で対象項を明示する。
        self.rewards.termination_penalty = RewTerm(
            func=loco_mdp.is_terminated_term,
            weight=-100.0,
            params={"term_keys": ["base_contact", "base_height"]},
        )

        # 項1-6。weight=0 で開始し、カリキュラムで Phase 2 に立ち上げる。
        self.rewards.kick_direction = RewTerm(
            func=mdp.kick_direction,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION},
        )
        self.rewards.kick_velocity_scaled = RewTerm(
            func=mdp.kick_velocity_scaled,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION, "sigma_velocity": 1.0},
        )
        self.rewards.kick_velocity_strong = RewTerm(
            func=mdp.kick_velocity_strong,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION},
        )
        self.rewards.walk_speed = RewTerm(
            func=mdp.walk_speed,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_walk": 0.5, "sigma_walk_potential": 0.5},
        )
        self.rewards.approach_penalty = RewTerm(
            func=mdp.approach_penalty,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_sole": 0.35, "sigma_pose": 0.3},
        )
        self.rewards.kick_pose_overshoot = RewTerm(
            func=mdp.kick_pose_overshoot,
            weight=0.0,
            params={**_KICK_STATE_PARAMS},
        )

        # ------------------------------------------------------------------ #
        # Curriculum: キック報酬のフェードイン
        #
        # このタスクは K1WalkKickWalkPhaseEnvCfg (walk phase) の checkpoint から
        # 再開する前提なので、歩行はすでに獲得済み。よって 0 から立ち上げてよく、
        # ランプ (0 → 500 iteration) 自体が段階的導入になる。
        # walk phase を挟まず 0 から学習する場合は start_step を大きく取ること。
        # ------------------------------------------------------------------ #
        # steps_per_iteration = num_steps_per_env (PPO config)
        # start_step / end_step は iteration 数で指定する
        _spi = 24
        _phase2 = {"start_step": 0, "end_step": 500, "steps_per_iteration": _spi}

        # Phase 1→2: 速度追従報酬をフェードアウト
        self.curriculum.track_lin_vel_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "track_lin_vel_xy_exp", "start_weight": 1.0, "end_weight": 0.5, **_phase2},
        )
        self.curriculum.track_ang_vel_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "track_ang_vel_z_exp", "start_weight": 1.0, "end_weight": 0.5, **_phase2},
        )

        # Phase 2: キック報酬を一斉フェードイン（end_weight は仕様書の重み範囲の中央付近）
        # 項1-3 は post-latch に dense で払うので、猶予窓の長さに比例して累積が増える。
        # _KICK_W_SCALE で割り戻し、窓を変えてもキック 1 回あたりの収益を一定に保つ。
        self.curriculum.kick_direction_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "kick_direction", "start_weight": 0.0,
                    "end_weight": 6.0 * _KICK_W_SCALE, **_phase2},
        )
        self.curriculum.kick_velocity_scaled_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "kick_velocity_scaled", "start_weight": 0.0,
                    "end_weight": 4.0 * _KICK_W_SCALE, **_phase2},
        )
        self.curriculum.kick_velocity_strong_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "kick_velocity_strong", "start_weight": 0.0,
                    "end_weight": 3.0 * _KICK_W_SCALE, **_phase2},
        )
        self.curriculum.walk_speed_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "walk_speed", "start_weight": 0.0, "end_weight": 1.5, **_phase2},
        )
        self.curriculum.approach_penalty_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "approach_penalty", "start_weight": 0.0, "end_weight": -3.0, **_phase2},
        )
        self.curriculum.kick_pose_overshoot_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "kick_pose_overshoot", "start_weight": 0.0, "end_weight": -50.0, **_phase2},
        )

        # ------------------------------------------------------------------ #
        # その他
        # ------------------------------------------------------------------ #
        self.scene.env_spacing = 100.0
        # 1 エピソード = 1 キック。ボールは 0.3-0.8m 前方に湧くので、歩いて蹴るには十分な長さ。
        # 蹴れなければ time_out で終了する。
        self.episode_length_s = 10.0


@configclass
class K1WalkKickWalkPhaseEnvCfg(K1WalkKickEnvCfg):
    """Stage 1: 通常の歩行だけを学習する。

    論文の 3-stage training の 1 段目 (「まず通常の歩行ポリシーを学習する」) にあたる。
    観測は K1WalkKickEnvCfg と同じ 55 次元のまま揃えてあるので、この run の checkpoint を
    そのまま K1WalkKickEnvCfg (stage 2) に引き継げる::

        # stage 1
        _labpython2 train.py --task Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0 --headless --num_envs 4096
        # stage 2 (stage 1 の checkpoint から再開)
        _labpython2 train.py --task Isaac-Velocity-Flat-K1-Walk-Kick-v0 --headless --num_envs 4096 \
            --checkpoint logs/rsl_rl/k1_walk_kick_walk_phase/<run>/model_<N>.pt

    観測は 55 次元・同じ並びのまま、ボール/キック由来のスロットの中身だけを歩行コマンドに
    差し替える (:func:`mdp.walk_command_xy` / :func:`mdp.walk_command_yaw_dir`)。
    55 次元の観測には速度指令が含まれないため、スロットを流用せずランダムな歩行コマンドを
    出すと、policy から見て指令が観測不能になり追従を学習できない (期待値 = 静止が最適解に
    なってしまう)。スロットを流用することで指令が可観測になり、かつ「スロットが指す方へ歩く」
    という入力→挙動の対応が kick phase と共通になる。

    ボールはこの phase では観測にも報酬にも一切現れないので、シーンごと取り除く。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 通常の歩行コマンド（K1FlatEnvCfg と同じ離散速度コマンド）に戻す
        self.commands.base_velocity.follow_ball = False
        self.commands.base_velocity.kick_direction_command_name = None
        self.commands.kick_direction = None

        # -- 観測: 次元と並びを保ったまま、ボール/キック由来のスロットを歩行コマンドに置換
        for _group in (self.observations.policy, self.observations.critic):
            _group.prev_ball_pos.func = mdp.walk_command_xy
            _group.prev_ball_pos.params = {"command_name": "base_velocity"}
            _group.kick_direction.func = mdp.walk_command_yaw_dir
            _group.kick_direction.params = {"command_name": "base_velocity"}
            _group.ball_vel.func = mdp.zero_obs
            _group.ball_vel.params = {"dim": 2}
            _group.target_kick_velocity.func = mdp.zero_obs
            _group.target_kick_velocity.params = {"dim": 1}
        # critic の特権情報のボール位置もゼロ埋め (次元は 3 のまま)
        self.observations.critic.ball_pos_rel.func = mdp.zero_obs
        self.observations.critic.ball_pos_rel.params = {"dim": 3}

        # -- ボールをシーンから取り除く（観測にも報酬にも現れないため）
        self.scene.soccer_ball = None
        self.scene.contact_balls_left = None
        self.scene.contact_balls_right = None
        self.events.reset_ball = None

        # -- 速度追従の重みを K1FlatEnvCfg (= 元の歩行タスク) の値に戻す
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # -- キック関連の報酬とそのカリキュラムを全て無効化する
        for _term in (
            "kick_direction",
            "kick_velocity_scaled",
            "kick_velocity_strong",
            "walk_speed",
            "approach_penalty",
            "kick_pose_overshoot",
        ):
            setattr(self.rewards, _term, None)
            setattr(self.curriculum, f"{_term}_weight", None)
        self.curriculum.track_lin_vel_weight = None
        self.curriculum.track_ang_vel_weight = None

        # -- キック成立による終了も無効化する。
        # ボールが動かない以上 latch は発火しないが、kick_state を毎ステップ回す意味が
        # 無いので項ごと外す。エピソードは time_out か転倒で終わる。
        self.terminations.kick_finished = None

        # 歩行学習なので、キック用に切り詰めた 10 秒ではなく元の歩行タスク相当に戻す
        self.episode_length_s = 20.0


@configclass
class K1WalkKickWalkPhaseEnvCfg_PLAY(K1WalkKickWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKickEnvCfg_PLAY(K1WalkKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
