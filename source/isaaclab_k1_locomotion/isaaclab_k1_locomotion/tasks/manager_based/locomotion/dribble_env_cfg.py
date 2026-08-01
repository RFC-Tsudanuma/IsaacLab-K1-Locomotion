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
    * ``ball_velocity_along_kick`` — directional alignment (cosine) of ball velocity with the world-frame kick direction (magnitude ignored)
    * ``ball_speed`` — magnitude of the ball's world-frame xy velocity
    * ``robot_velocity_toward_ball`` — small shaping reward: robot velocity component
      toward the ball (zeroed out when very close)
"""

import math

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

from .flat_env_cfg import K1FlatEnvCfg, K1CriticCfg, K1ObservationsCfg, K1PolicyCfg
from .velocity_env_cfg import CommandsCfg, MySceneCfg
from .mdp.commands import KickDirectionCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from .mdp.curriculums import (
    kick_angle_range_curriculum,
    modify_reward_weight_linear,
    randomize_ball_init_velocity,
    randomize_ball_init_pose,
)
from .mdp.events import reset_prev_high_action
from .mdp.observations import ball_pos_rel, ball_vel, kick_direction_b, last_high_action
from .mdp.terminations import time_after_ball_contact
from .mdp.rewards import (
    approach_ball_progress,
    ball_speed_along_kick,
    ball_velocity_along_kick,
    com_jerk_l2,
    high_action_rate_l2,
    high_action_smoothness_l2,
    high_action_xy_coactivation,
    robot_behind_ball,
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
        resampling_time_range=(6.0, 10.0),
        # 学習初期は正面付近のみ出題し、カリキュラム (kick_angle_curriculum) で
        # 段階的に全周へ広げる。ここの値はカリキュラム初段で上書きされる初期値。
        angle_range=(-0.5, 0.5),
        debug_vis=True,
    )


@configclass
class K1DribbleRewardsCfg:
    """Dribble 専用の報酬。K1Rewards (歩行用) は継承せずに丸ごと置き換える。"""

    # --- ボール関連の主報酬 ---
    # ボール速度の「方向」がキック方向 (ワールド座標) と揃っているほど高い報酬 (速度の大きさは見ない)。
    ball_velocity_along_kick = RewTerm(
        func=ball_velocity_along_kick,
        weight=4.5,
        params={"command_name": "kick_direction"},
    )
    # ボール速度の「キック方向への射影成分」 [0, 1]。max_speed で正規化、逆方向は 0。
    # 方向無視の ball_speed と違い「速く かつ 正しい方向に」蹴ったときだけ報酬が出るので、
    # 方向を無視してとにかく蹴る局所解を資金提供しない (主報酬 ball_velocity_along_kick と協調)。
    # 方向射影なので farm 不可・目的 (正しい方向に速く蹴る) に直結する。robot_velocity_toward_ball
    # の支配を崩すため、接近系の weight を下げる代わりにこの目的直結項を 0.8→1.5 に上げる。
    ball_speed_along_kick = RewTerm(
        func=ball_speed_along_kick,
        weight=1.5,
        params={"command_name": "kick_direction", "max_speed": 2.0},
    )

    # --- Shaping ---
    # ロボット→ボール距離の減少量 (ポテンシャル形式) で接近を誘引する。
    # 速度射影 (robot_velocity_toward_ball) と違い上限クランプ・近接ゼロが無く、
    # 常時密な勾配がかかるので「動かない」局所解に陥りにくい。
    # weight の目安: 制御ステップ dt=0.02s・接近速度~0.9m/s → 約0.018m/step なので、
    # 旧 shaping ([0,1]×1.0) と同オーダーにするには ~40。
    # ターゲットを「キック方向から見てボール後ろ側 behind_offset[m] の staging 地点」にする。
    # これで「ボールに寄る」と「正しい側へ回り込む」が同一勾配になり、最短直線接近で
    # 間違った向きに蹴る挙動を抑える (= ball_velocity_along_kick への追従を促す)。
    approach_ball_progress = RewTerm(
        func=approach_ball_progress,
        weight=40.0,
        params={"command_name": "kick_direction", "behind_offset": 0.5},
    )
    # ロボット速度のボール方向成分 [0, 1] を毎ステップ密に与える接近報酬。
    # approach_ball_progress (ポテンシャル形式 = 最適方策を変えない) と違い、
    # 非ポテンシャルなので「立ち止まる」局所解を実際に崩して移動側へ最適方策を寄せる。
    # 速度ベースなので立っていると 0 → parking で farm できない。が、移動し続ける限り毎ステップ
    # 稼げる (≒ farmable) ので、これが大きいと「接近 (手段)」が「正しく蹴る (目的)」を上回り続ける。
    # ロボットは既に十分動くようになったので anti-parking としての役目は薄れた。支配を崩すため
    # 2.5→1.2 に下げ、接近の主役は farm 不可なポテンシャル形式 approach_ball_progress に委ねる。
    robot_velocity_toward_ball = RewTerm(
        func=robot_velocity_toward_ball,
        weight=1.2,
        params={
            "max_speed": 1.0,
            "min_distance": 0.05,
            "command_name": "kick_direction",
            "behind_offset": 0.5,
        },
    )
    # ロボット Trunk の正面 (base +x) がボール方向を向いているほど大きい [0, 1]。
    robot_facing_ball = RewTerm(
        func=robot_facing_ball,
        weight=0.4,
        params={"min_distance": 0.15},
    )
    # キック方向から見てボールの後ろ側 かつ ボールに近いほど大きい [0, 1]。
    # 密な非ポテンシャル形式で「早く正しい配置に着く」挙動を誘導しつつ、
    # 近接ゲート (engage_distance) で「遠くの後ろ側に居座って稼ぐ」局所最適を防ぐ。
    # stage 1 で along_kick が頭打ち & behind_ball がほぼ 0 (裏取りできていない) だったため、
    # engage_distance 0.45→1.2 で「遠くから裏取りを形成」、weight 1.2→2.5 で接近系に勝てる
    # オーダーにして回り込みを本気で誘導する (横〜後方キックの精度を上げる)。
    robot_behind_ball = RewTerm(
        func=robot_behind_ball,
        weight=2.5,
        params={"command_name": "kick_direction", "min_distance": 0.15, "engage_distance": 1.2},
    )

    # --- ペナルティ ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.5 * 0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.6 * 0.5)
    # 上位ポリシーが出力する 3D 歩行コマンドに対する平滑性ペナルティ。
    # 元の action_smoothness_l2 は env.action_manager.action (= frozen の 22D 関節指令)
    # を見ており上位ポリシーが直接制御できないので、上位 action 用に差し替え。
    action_smoothness_l2 = RewTerm(func=high_action_smoothness_l2, weight=-0.12)
    # 上位ポリシー action の 1 ステップ差分 (action rate) ペナルティ。
    # コマンドが急変するとロボットが追従しきれず歩行が乱れるので軽く抑える。
    action_rate_l2 = RewTerm(func=high_action_rate_l2, weight=-0.4 * 0.8)
    # 上位 action の並進 vx・vy 共活性ペナルティ (|vx|*|vy|)。
    # x+theta / y+theta のどちらかのモードに寄せ、xy 同時入力による不安定を抑える。
    # weight はカリキュラム (high_action_xy_coactivation_curriculum) で 0 → -3.0 に
    # 線形ランプするので、ここの値は据え置き時のフォールバック。
    high_action_xy_coactivation = RewTerm(func=high_action_xy_coactivation, weight=-2.6)
    # ロボット root body COM の jerk (加速度の時間微分) ペナルティ。
    # 値域が大きくなりやすいので重みは非常に小さめから始める。
    com_jerk_l2 = RewTerm(func=com_jerk_l2, weight=-2e-6 * 0.5)


@configclass
class K1DribbleEnvCfg(K1FlatEnvCfg):
    """K1FlatEnv + a soccer ball, used as the dribble training env."""

    scene: K1DribbleSceneCfg = K1DribbleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1DribbleObservationsCfg = K1DribbleObservationsCfg()
    commands: K1DribbleCommandsCfg = K1DribbleCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0

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
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.7, 0.7)
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)

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
                    "x": (0.10, 1.5),
                    "y": (-1.5, 1.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("soccer_ball"),
            },
        )

        self.curriculum.randomize_ball_init_velocity = CurrTerm(
            func=randomize_ball_init_velocity,
            params={
                "term_name": "reset_ball",
                "num_steps": 5000,
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
        self.curriculum.randomize_ball_init_pose = CurrTerm(
            func=randomize_ball_init_pose,
            params={
                "term_name": "reset_ball",
                "num_steps": 5000,
                "pose_range": {
                    "x": (0.10, 3.0),
                    "y": (-2.5, 2.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

        # vx・vy 共活性ペナルティの weight を段階的に強める。
        # 学習初期 (~3000 step) はペナルティ 0 で自由に探索させ、ドリブルの基礎を
        # 学んでから 3000→8000 step で weight を 0 → -3.0 に線形ランプして
        # x+theta / y+theta のモード分離を徐々に強制する。
        self.curriculum.high_action_xy_coactivation_curriculum = CurrTerm(
            func=modify_reward_weight_linear,
            params={
                "term_name": "high_action_xy_coactivation",
                "start_weight": 0.0,
                "end_weight": -3.0,
                "start_step": 3000,
                "end_step": 8000,
            },
        )

        # キック方向の角度レンジを性能ベースで段階的に拡げるカリキュラム。
        # 初期は正面付近 (±0.5rad) のみ → 直線接近がそのまま正解になるので「正しい方向に
        # 蹴る」を先に獲得させ、動くボールの方向誤差 1-cos が閾値を下回るたびに全周 (±π) へ
        # 広げて回り込みが必要なケースを徐々に導入する。
        self.curriculum.kick_angle_curriculum = CurrTerm(
            func=kick_angle_range_curriculum,
            params={
                "command_name": "kick_direction",
                # ±1.2→±2.0 のような大きなジャンプは「崖」になり、回り込み能力が
                # 追いつかず停滞する。間隔を細かくして徐々に広げる。
                "stages": [
                    (-0.4, 0.4),
                    (-0.8, 0.8),
                    (-1.2, 1.2),
                    (-1.6, 1.6),
                    (-2.0, 2.0),
                    (-2.6, 2.6),
                    (-math.pi, math.pi),
                ],
                # ステージ別閾値: 広いレンジほど本質的に精度が落ちるので緩める (永久停滞を防ぐ)。
                # 0.25→cos>0.75(≒41°) ... 0.5→cos>0.5(≒60°)。最後の値は最終段なので未使用。
                "error_threshold": [0.32, 0.36, 0.39, 0.42, 0.45, 0.50, 0.55],
                "min_speed": 0.5,
                "ema_alpha": 0.02,
                "min_updates": 100,
                # 閾値を 3 ウィンドウ連続で下回ったときのみ進行 (1 回の偶発低誤差で
                # 駆け抜けるのを防ぐデバウンス)。
                "confirm_windows": 3,
                # 各ステージに最低 8000 env-step (= num_steps_per_env 24 で約 333 iter) 留まる
                # ハードフロア。狭いレンジは「前進 ≒ 正方向キック」で誤差が偶発的に低くなり
                # 性能ゲートだけだと即通過するため、実際に方向制御を学ぶ時間を強制確保する。
                # 7 ステージ × 333 iter ≒ 最短でも全周到達まで ~2300 iter。
                "min_stage_steps": 8000,
            },
        )


@configclass
class K1DribbleEnvCfg_PLAY(K1DribbleEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
