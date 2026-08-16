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

原典との差分を個別に検証するための ablation タスクを 2 系統持つ。どちらも観測 55 次元・
同じ並びのままなので、既存の checkpoint をそのまま出発点にできる:

* **地形** (:class:`K1WalkKickWalkPhaseRoughEnvCfg` → :class:`K1WalkKickRoughEnvCfg`
  → :class:`K1WalkKick360RoughEnvCfg`): 平坦 → 数 cm の凹凸。歩容ごと変わるので
  0 から学習し直す。平坦版の 3 段レシピをそのまま rough でなぞる。
* **転がるボール** (:class:`K1WalkKick360MovingBallEnvCfg`): 地形は平坦のまま、
  リセット時にボールへランダム水平初速を与える。walk_kick_360 からの fine-tune。
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from ..locomotion.rough_env_cfg import _COMMAND_THRESHOLD, _PHASE_FREQ, _apply_play_viewer
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

# --------------------------------------------------------------------------- #
# Terrain ablation 用の凹凸地形
#
# B-Human のポスターとの差分のうち「bumpy な地面」だけを切り出して検証するためのもの。
# ``..locomotion.flat_env_cfg.NOISY_FLAT_TERRAIN_CFG`` (段差・坂道なしのランダムノイズ)
# と同じレシピだが、ablation なので **平坦なサブ地形を混ぜない** (あちらは 10% が plane)。
#
# * noise_range=(0.0, 0.04): 起伏は 0-4cm。**下向きの起伏を作らない** ことが重要で、
#   地面が z=0 以上にしかないので base_height 終了判定 (root z < 0.35m、絶対高さ基準)
#   が凹凸で誤爆しない。ロボットの立ち高さ 0.6m 台に対して 4cm はキック動作の
#   軸足接地に効く程度の外乱で、歩行を壊すほどではない。
# * curriculum=False: 難易度固定・ランダム配置。段階的に難しくするのが目的ではなく、
#   「常に凹凸がある」条件と平坦条件を比べるのが目的。
# * num_rows/num_cols=10: サブ地形の実現値を 100 通り用意して、env 間で凹凸パターンが
#   偏らないようにする。terrain generator を使うと env origin は地形原点になるので、
#   ``env_spacing`` (walk_kick は 100.0) は無視され、複数 env が同じ原点を共有する。
#   env 間の衝突は InteractiveScene 側でフィルタされるので物理的な干渉は起きない。
#
# height_scanner は **追加しない**。観測 55 次元・並びを walk_kick 系と同一に保ち、
# 外受容なし (proprioception のみ) で凹凸に対応させるのがこの ablation の趣旨で、
# checkpoint の相互流用もそのまま効く。
# --------------------------------------------------------------------------- #
WALK_KICK_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)

# ボールを地面から浮かせて落とす量 [m]。凹凸の振幅 (最大 4cm) より上に取る。
# terrain generator の env origin z は「サブ地形中央付近の最大高さ」なので、そこから
# 離れた位置では地面がそれより低い。浮かせずに置くとめり込んだ状態から弾かれてしまう。
_BALL_SPAWN_CLEARANCE = 0.05


def _apply_rough_terrain(cfg: "K1WalkKickEnvCfg") -> None:
    """flat 版のタスク cfg を「地形だけ」凹凸に差し替える。

    観測・報酬・コマンド・終了条件には一切触らない。terrain ablation の 3 タスク
    (walk phase / kick / 360) で共通の処理をここに集約し、二重定義を避ける。
    ``__post_init__`` の最後 (基底クラスの設定が全部済んだ後) に呼ぶこと。

    ``reset_ball.params`` は差し替えではなくキー追加で触るので、360 版が入れた
    ``half_angle`` / ``dist_range`` の全方位設定はそのまま生き残る。
    """
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = WALK_KICK_ROUGH_TERRAIN_CFG
    # terrain curriculum は使わない (K1FlatEnvCfg が curriculum.terrain_levels を
    # 既に None にしてある)。max_init_terrain_level=None で全 row を初期から使う。
    cfg.scene.terrain.max_init_terrain_level = None

    # ボールを置くタスク (kick phase) では、地形の起伏ぶんだけ浮かせてから落とす。
    # walk phase は reset_ball ごと None なので触らない。
    if cfg.events.reset_ball is not None:
        cfg.events.reset_ball.params["spawn_clearance"] = _BALL_SPAWN_CLEARANCE


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
    ball_vel = ObsTerm(func=mdp.ball_vel_b, noise=Unoise(n_min=-0.4, n_max=0.4))
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
            target_speed_range=(0.25, 2.0),  # 目標ボール速度 [m/s]
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


@configclass
class K1WalkKick360EnvCfg(K1WalkKickEnvCfg):
    """全方位版ウォークキック。ボール出現・蹴り方向とも 360°、距離 0.5-1.5 m。

    :class:`K1WalkKickEnvCfg` (ボール ±60° / 蹴り ±45°, 0.5-0.8 m) を全方位に広げたもの。
    質的に新しいのは「ボールの正面側から半周回り込んで蹴る」エピソード。段階カリキュラムは
    使わず **walk_kick の checkpoint からの fine-tune を実質カリキュラムとする**
    (蹴り方は既知・回り込みだけ新規)::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-360-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick/<run>/model_<N>.pt

    変更点は :class:`~..walk_loop_pass.walk_loop_pass_env_cfg.K1WalkLoopPass360EnvCfg`
    と同一。回り込みは誘導 (G の迂回経路) ではなく **報酬の抑止で成立させる** (B-Human 方式)。
    approach_penalty (遠いほど罰 = 接近圧) を ball_avoidance (姿勢ができていないのに
    近いほど罰) に差し替えることで、ボール正面からまっすぐ突っ込む行動が罰され、
    距離を保って回り込む解に寄る。差し替え理由の詳細は
    :func:`.mdp.rewards.ball_avoidance` の docstring を参照。

    init_side の確定遅延 (kick_state._INIT_SIDE_COMMIT_DIST) は共有コード側の修正で、
    ボール正面スタート (s≈0) の回り込みがコイントスで overshoot 罰されるのを防ぐ。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. サンプリングを全方位・遠距離へ
        self.events.reset_ball.params["half_angle"] = math.pi
        self.events.reset_ball.params["dist_range"] = (0.5, 1.5)
        self.commands.kick_direction.ranges.heading = (-math.pi, math.pi)

        # -- 2. approach_penalty → ball_avoidance
        #
        # approach_penalty は「姿勢が悪くても足をボールに近づけるだけで罰が減る」ので、
        # 正面からの突っ込み (→ 衝突で latch が悪い値で凍結 → エピソード終了) を
        # むしろ後押ししてしまう。ball_avoidance は同じ weight で向きだけ逆。
        self.rewards.approach_penalty = None
        self.curriculum.approach_penalty_weight = None
        self.rewards.ball_avoidance = RewTerm(
            func=mdp.ball_avoidance,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_sole": 0.35, "sigma_pose": 0.3},
        )
        self.curriculum.ball_avoidance_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "ball_avoidance",
                "start_weight": 0.0,
                "end_weight": -3.0,
                "start_step": 0,
                "end_step": 500,
                # 基底の _spi (= num_steps_per_env) と同じ値。あちらは __post_init__ の
                # ローカル変数なので参照できず、リテラルで持つ。
                "steps_per_iteration": 24,
            },
        )

        # -- 3. 移動距離が伸びるぶんエピソードを延長
        #
        # 1.5 m + 半周回り込みで移動 2.5-3 m。学習初期の遅い歩きだと 10 秒では
        # 蹴る前に time_out するエピソードが増える。
        self.episode_length_s = 15.0


@configclass
class K1WalkKick360EnvCfg_PLAY(K1WalkKick360EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# --------------------------------------------------------------------------- #
# Ablation 1: 凹凸地形 (terrain ablation)
#
# B-Human のポスターとの差分のうち「bumpy な地面」だけを検証する系統。地形以外は
# 元タスクと完全に同一 (観測・報酬・コマンド・終了条件・カリキュラム) なので、
# 平坦版との比較がそのまま地形の効果になる。
#
# こちらは fine-tune ではなく **0 から学習し直す**。地形が変わると歩容そのものが
# 変わるため、平坦で獲得した歩行を出発点にすると「平坦向けの歩容から抜けられない」
# 交絡が入る。既存の 3 段レシピ (walk phase → kick → 360) をそのまま rough でなぞる。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickWalkPhaseRoughEnvCfg(K1WalkKickWalkPhaseEnvCfg):
    """Stage 1 (rough): 凹凸地形で通常の歩行だけを学習する。

    :class:`K1WalkKickWalkPhaseEnvCfg` との差は地形だけ。観測は 55 次元・同じ並びのまま
    なので、この run の checkpoint をそのまま :class:`K1WalkKickRoughEnvCfg` (stage 2)
    に引き継げる::

        # stage 1 (rough)
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Rough-K1-Walk-Kick-Walk-Phase-v0 \
            --headless --num_envs 4096
        # stage 2 (rough, stage 1 の checkpoint から)
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Rough-K1-Walk-Kick-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase_rough/<run>/model_<N>.pt
        # stage 3 (rough, stage 2 の checkpoint から) → K1WalkKick360RoughEnvCfg
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Rough-K1-Walk-Kick-360-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_rough/<run>/model_<N>.pt

    3 段の通し実行は :file:`scripts/rsl_rl/train_walk_kick_360.sh` を rough 用の
    タスク名で上書きして回す (同スクリプトの変数定義のコメント参照)。

    ``--resume`` ではなく ``--load_pretrained`` を使う理由は平坦版と同じ
    (:file:`scripts/rsl_rl/train_walk_kick.sh` の冒頭コメント参照)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_rough_terrain(self)


@configclass
class K1WalkKickWalkPhaseRoughEnvCfg_PLAY(K1WalkKickWalkPhaseRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _apply_play_viewer(self)


@configclass
class K1WalkKickRoughEnvCfg(K1WalkKickEnvCfg):
    """Stage 2 (rough): 凹凸地形でボール追従とキックを学習する。

    :class:`K1WalkKickEnvCfg` との差は地形だけ (:func:`_apply_rough_terrain`)。
    ボールのリセット高さだけは terrain generator に合わせて浮かせる必要があるので、
    ``reset_ball`` の ``spawn_clearance`` が入る。観測・報酬・コマンドは平坦版と同一。

    学習は :class:`K1WalkKickWalkPhaseRoughEnvCfg` の checkpoint から (docstring 参照)。
    キック報酬のフェードイン (0 → 500 iteration) も平坦版と同じ設定を継承する。
    出発点が「歩けるだけのポリシー」であることも平坦版と同じなので、流儀を変える理由がない。

    NOTE: 凹凸の上ではボールが自分から転がる余地がある。転がるボールへの対応は
          :class:`K1WalkKick360MovingBallEnvCfg` が扱う別の ablation なので、
          こちらは P_kick 固定 (``track_ball=False``) のまま。両者の交絡を避けるため、
          地形の振幅は「ボールがほぼ止まっている」範囲に収めてある
          (:data:`WALK_KICK_ROUGH_TERRAIN_CFG` の noise_range)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_rough_terrain(self)


@configclass
class K1WalkKickRoughEnvCfg_PLAY(K1WalkKickRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _apply_play_viewer(self)


@configclass
class K1WalkKick360RoughEnvCfg(K1WalkKick360EnvCfg):
    """Stage 3 (rough): 凹凸地形の全方位版。平坦の 3 段レシピを rough でなぞる最終段。

    :class:`K1WalkKick360EnvCfg` との差は地形だけ (:func:`_apply_rough_terrain`)。
    360 版固有の設定は基底の ``__post_init__`` で全て済んでおり、
    :func:`_apply_rough_terrain` は地形と ``reset_ball`` の ``spawn_clearance`` しか
    触らないので、そのまま生き残る:

    * ボール配置の全方位化 (``half_angle=π``) と遠距離化 (``dist_range=(0.5, 1.5)``)
      — ``reset_ball.params`` を差し替えではなくキー追加で触っているため無傷。
    * ``approach_penalty`` → ``ball_avoidance`` の差し替えとそのカリキュラム。
    * 蹴り方向の全方位化 (``kick_direction.ranges.heading``) と ``episode_length_s=15.0``。

    学習は :class:`K1WalkKickRoughEnvCfg` (rough の stage 2) の checkpoint から。
    平坦版と同じく **checkpoint の継承そのものが実質カリキュラム** で、段階カリキュラムは
    追加しない (蹴り方は rough で獲得済み・回り込みだけが新規)::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Rough-K1-Walk-Kick-360-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_rough/<run>/model_<N>.pt

    3 段の通し実行は :file:`scripts/rsl_rl/train_walk_kick_360.sh` を rough 用の
    タスク名で上書きして回す (同スクリプトの変数定義のコメント参照)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_rough_terrain(self)


@configclass
class K1WalkKick360RoughEnvCfg_PLAY(K1WalkKick360RoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _apply_play_viewer(self)


# --------------------------------------------------------------------------- #
# Ablation 2: 転がるボール (ball velocity ablation)
#
# ボール初速のサンプリング範囲 [m/s]。方向は 0-2π の一様乱数。
#
# 上限 0.5 は kick_state の v_thresh (0.8) より下でなければならない。超えると
# ロボットが触る前に値 latch が成立し、「蹴っていないのに τ_direction が凍結される」
# ことになる (reset_ball_in_front_of_robot の NOTE 参照)。
# 実効的には摩擦で減速するので、ロボットが到達する頃には 0.5 より遅い。
# --------------------------------------------------------------------------- #
_MOVING_BALL_SPEED_RANGE = (0.0, 0.5)


@configclass
class K1WalkKick360MovingBallEnvCfg(K1WalkKick360EnvCfg):
    """転がるボールを蹴る全方位版。地形は平坦のまま、リセット時にボールへ初速を与える。

    B-Human のポスターとの差分のうち「転がるボール」だけを検証する系統。
    :class:`K1WalkKick360EnvCfg` の checkpoint からの fine-tune 前提で、
    段階カリキュラムは使わない (360 が walk_kick から fine-tune するのと同じ流儀)::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Moving-Ball-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_360/<run>/model_<N>.pt

    観測は 55 次元・同じ並びのまま。ボール速度 (``ball_vel``, 2 次元) は元から観測に
    入っているので、静止ボールで学習した checkpoint でもそのスロットが「常に 0」から
    「意味のある値」に変わるだけで、入力層も obs normalizer もそのまま使える。

    P_kick の追従 (このタスクの本体)
    --------------------------------
    :mod:`.mdp.kick_state` は「理想キック立ち位置」P_kick をエピソード開始時のボール位置に
    固定する。ボールが動くとこの点が実際のボールから外れ、``approach_penalty`` /
    ``ball_avoidance`` (``d_to_P_kick`` を見る) が的外れな点への整列を要求し、
    latch 後に G を戻す先もずれる。そこで ``track_ball=True`` で **latch 前は P_kick を
    毎ステップ引き直す**。latch 後の凍結 (飛翔評価) は従来どおり。

    他の項が動くボールで意味を保つかは個別に確認済み:

    * ``walk_speed`` / ``BallFollowVelocityCommand``: 目標終端 G は元から現在のボール位置
      から計算されており (latch 後だけ P_kick に固定)、追従は既に成立している。
    * ``kick_direction`` / ``kick_velocity_*``: latch 時のボール飛翔方向と速度で採点する。
      初速 0.5 m/s が乗るぶん τ_direction / v_ball に誤差が入るが、それは「転がるボールを
      狙って蹴るのは難しい」という **このタスクが測りたい難しさそのもの**。
    * ``kick_pose_overshoot``: 判定に使う後方レイ R は元から現在のボール位置を基準に
      引かれているので、ボールと一緒に平行移動する。追加の対処は不要。
    * ``extra_ball_touch`` (このタスクでは未使用) が使う接触カウントは、リセット直後の
      偽の速度跳ね上がりを ``just_reset`` ガードで既に除いてある。

    NOTE: ``track_ball`` は kick_state を **最初に呼ぶ項** の値でその step の状態が確定する
          (r_stance / alpha / v_thresh と同じ)。P_kick を作るのは kick_state だけで報酬項は
          結果を読むだけなので、毎ステップ最初に走る ``kick_finished`` と、同じ状態を読む
          ``base_velocity`` コマンドの 2 か所に配れば足りる。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. リセット時にボールへランダム水平初速を与える
        self.events.reset_ball.params["speed_range"] = _MOVING_BALL_SPEED_RANGE

        # -- 2. latch までは P_kick をボールに追従させる
        self.terminations.kick_finished.params["track_ball"] = True
        self.commands.base_velocity.track_ball = True


@configclass
class K1WalkKick360MovingBallEnvCfg_PLAY(K1WalkKick360MovingBallEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# --------------------------------------------------------------------------- #
# Ablation 3: 知覚ノイズ (perception noise ablation)
#
# 実機で walk_kick_360 の方策を回すと、蹴り損ねがそこそこの確率で起きる。原因は
# ボール認識の誤差 (計測済み: 約 3cm、prev_ball_pos の Unoise ±2cm より広い) と
# 認識遅延 (制御 50Hz に対し vision 30Hz + 未計測の処理・通信遅延)。
# シム側のボール位置観測を実機の認識パイプライン寄りにして fine-tune する。
#
# * _BALL_OBS_DELAY_STEP_RANGE: エピソードごとに一様サンプルする遅延 [制御ステップ]。
#   (2, 6) = 40-120ms @50Hz。実機の遅延計測ができたら「計測値 + マージン」に絞ること。
# * _BALL_OBS_CAMERA_HZ: 実機の vision レート。ホールドのぶん実効遅延はさらに
#   0〜1 フレーム (0-33ms) 伸びる。
# * _BALL_OBS_JITTER: フレーム更新時に引き直す一様ノイズ [m]。ホールド中は同じ実現値が
#   保持されるので、毎ステップ独立な Unoise より濾しにくい (実機の誤差系列に近い)。
# --------------------------------------------------------------------------- #
_BALL_OBS_DELAY_STEP_RANGE = (2, 6)
_BALL_OBS_CAMERA_HZ = 30.0
_BALL_OBS_JITTER = 0.05


def _apply_noisy_ball_obs(cfg: "K1WalkKickEnvCfg") -> None:
    """policy のボール位置観測だけを実機の認識パイプライン寄りに差し替える。

    固定 1 ステップ遅延 + 毎ステップ独立 Unoise(±2cm) を
    :func:`.mdp.observations.noisy_ball_pos_b` (エピソードごとランダム遅延 +
    30Hz サンプル&ホールド + フレーム同期ジッタ) に置き換える。

    観測の次元・並び・他の全ての項に触れないので、既存の checkpoint をそのまま
    ``--load_pretrained`` できる。weak / middle など他系統の 360 タスクからも呼べる
    (レシピ側が触るのは報酬・カリキュラム・コマンド帯・DR だけで、観測は無関係)。
    ``__post_init__`` の最後 (基底クラスの設定が全部済んだ後) に呼ぶこと。

    触らないもの:

    * **critic の ``prev_ball_pos``**: 従来どおり :func:`.mdp.prev_ball_pos_b`
      (ノイズ無し 1 ステップ遅延)。critic は特権情報 (``ball_pos_rel``) も持つ
      asymmetric 構成なので、ここを汚す理由がない。
    * **``ball_vel``**: 静止ボールではほぼ 0 で、遅延・ホールドを掛けても実質変わらない。
      転がるボール (moving_ball ablation) と組むときは同じパイプラインを通すこと。
    """
    term = cfg.observations.policy.prev_ball_pos
    term.func = mdp.noisy_ball_pos_b
    term.params = {
        "delay_step_range": _BALL_OBS_DELAY_STEP_RANGE,
        "camera_hz": _BALL_OBS_CAMERA_HZ,
        "jitter": _BALL_OBS_JITTER,
    }
    # ジッタは関数内でフレーム同期に付与する (ホールド中は同じ実現値を保持する) ので、
    # 毎ステップ独立の Unoise は外す。残すと二重にノイズが乗る。
    term.noise = None


def _disable_ball_obs_jitter(cfg: "K1WalkKickEnvCfg") -> None:
    """PLAY 用: 関数内ジッタを切る。遅延 + サンプル&ホールドは残す。

    ``enable_corruption = False`` は ObsTerm の ``noise`` しか切らないので、
    :func:`_apply_noisy_ball_obs` が関数側に移したジッタはそれに合わせて別途切る。
    遅延とサンプル&ホールドは観測パイプラインの構造 (= PLAY で見たいもの) なので残す。
    """
    cfg.observations.policy.prev_ball_pos.params["jitter"] = 0.0


@configclass
class K1WalkKick360NoisyBallEnvCfg(K1WalkKick360EnvCfg):
    """知覚ノイズ+遅延つき全方位版。ボール位置観測だけを実機の認識パイプライン寄りにする。

    :class:`K1WalkKick360EnvCfg` との差は policy の ``prev_ball_pos`` 項だけ:
    固定 1 ステップ遅延 + 毎ステップ独立 Unoise(±2cm) を、
    :func:`.mdp.observations.noisy_ball_pos_b` (エピソードごとランダム遅延 2-6 ステップ +
    30Hz サンプル&ホールド + フレーム同期ジッタ ±5cm) に差し替える。
    walk_kick_360 の checkpoint からの fine-tune 前提::

        _labpython2 scripts/rsl_rl/train.py \\
            --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Noisy-Ball-v0 \\
            --headless --num_envs 4096 \\
            --load_pretrained logs/rsl_rl/k1_walk_kick_360/<run>/model_<N>.pt

    観測は 55 次元・並びとも同一なので checkpoint はそのまま載る。
    差し替えの詳細と「触らないもの」は :func:`_apply_noisy_ball_obs` を参照。

    同じ観測差し替えを weak / middle の帯に適用したものが
    :class:`~..walk_weak_kick.walk_weak_kick_env_cfg.K1WalkKick360WeakNoisyBallEnvCfg` /
    :class:`~..walk_middle_kick.walk_middle_kick_env_cfg.K1WalkMiddleKick360NoisyBallEnvCfg`。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_noisy_ball_obs(self)


@configclass
class K1WalkKick360NoisyBallEnvCfg_PLAY(K1WalkKick360NoisyBallEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)
