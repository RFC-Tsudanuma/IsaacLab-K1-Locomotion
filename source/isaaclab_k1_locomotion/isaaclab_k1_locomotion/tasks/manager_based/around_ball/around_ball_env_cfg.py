# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスクの環境定義。

frozen 歩行ポリシーの上に載せる高レベルポリシーを PPO で学習するための階層タスク。
dribble と同じく ``HierarchicalVecEnvWrapper`` (scripts/rsl_rl/dribble_helpers.py) で
学習する前提で、上位 action は 3D 歩行コマンド (vx, vy, wz)。

タスク:
    1. ワールド座標のキック方向コマンド (``kick_direction``) が与えられる。
    2. ロボットはボール (視野 ±60° でしか見えない) を動かさないように回り込み、
       ボール後方 (キック方向の反対側) 0.5m の目標点に体の向きを揃えて到達する。
    3. 揃ったら止まらずにボールへ突進し、歩いたままボールに突っ込んで
       キック方向へボールを動かす (蹴りモーションは不要、体当たりで良い)。

観測 (dribble と同じ構造):
    * ``policy``    = 歩行 K1PolicyCfg の velocity_commands を前回上位 action に差し替え
                      + FOV マスク付きボール相対位置 + 可視フラグ + キック方向 (base frame)
    * ``critic``    = 同様 + 特権情報 (真のボール位置・速度)
    * ``low_level`` = 歩行 K1PolicyCfg そのまま (frozen ポリシー用)
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# 歩行 (locomotion) パッケージの共通部品は import で参照のみ (locomotion 側は変更しない)
from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from ..locomotion.rough_env_cfg import K1CriticCfg, K1ObservationsCfg, K1PolicyCfg
from ..locomotion.velocity_env_cfg import CommandsCfg, MySceneCfg
from ..locomotion.mdp.commands import KickDirectionCommandCfg
from ..locomotion.mdp.events import reset_prev_high_action
from ..locomotion.mdp.observations import ball_pos_rel, ball_vel, kick_direction_b, last_high_action
from ..locomotion.mdp.rewards import (
    ball_velocity_along_kick,
    com_jerk_l2,
    robot_facing_ball,
)

from .mdp.curriculums import modify_kick_angle_range
from .mdp.events import (
    reset_ball_in_front_cone,
    reset_ball_last_seen,
    reset_ball_perception,
    reset_base_forward_velocity,
)
from .mdp.terminations import ball_kicked
from .mdp.observations import (
    ball_in_fov,
    ball_pos_rel_perceived,
    high_action_phase_obs,
    kick_direction_b_perceived,
)
from .mdp.rewards import (
    ball_disturbance_when_misaligned,
    ball_out_of_fov,
    charge_to_ball_when_aligned,
    gated_high_action_rate_l2,
    gated_high_action_smoothness_l2,
    gated_high_action_xy_coactivation,
    misaligned_ball_proximity,
    standoff_point_progress,
)

# --- タスクの幾何パラメータ ---
FOV_HALF_ANGLE_DEG = 60.0  # 視野: 左右 ±60°
STANDOFF = 0.5             # 回り込み目標点: ボール後方 0.5m (ユーザ要件)
ALIGN_ANGLE_TOL = 0.5      # 「揃った」とみなす配置角の閾値 [rad] (≈29°)。接触許可と障害物判定で共有


@configclass
class K1AroundBallSceneCfg(MySceneCfg):
    """Scene: 歩行シーン + サッカーボール (dribble と同じ物性値だが独立に定義)。"""

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.11)),
    )


@configclass
class K1AroundBallPolicyCfg(K1PolicyCfg):
    """上位 policy 観測。

    ``velocity_commands`` は前回の上位 action (歩行コマンド) に差し替え (dribble と同じ)。
    ``gait_phase`` は base_velocity (階層構成では未使用のダミー) ではなく
    上位 action 駆動の位相に差し替え。
    ボール位置は FOV マスク付き (視野外では最後に見えた値を保持) + 可視フラグ。
    """

    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    # fixed_freq=1.6: frozen に使う 0524_walk.pt は旧規約 (固定 1.6Hz, φ=2πft) で
    # 学習されている。速度依存アキュムレータ規約の新しい歩行 pt に差し替えるときは
    # fixed_freq を None にすること。
    gait_phase = ObsTerm(func=high_action_phase_obs, params={"cmd_threshold": 0.05, "fixed_freq": 1.6})
    # ボール位置は認識チーム出力を模擬した知覚DR付き (レイテンシ/更新レート/ドロップ/
    # 距離依存ノイズ/バイアス)。ノイズは関数内で付加するので ObsTerm 側の noise は付けない。
    # 実測スペックが出たら下の params を差し替える。critic は真値を見る (非対称)。
    ball_pos_rel = ObsTerm(
        func=ball_pos_rel_perceived,
        params={
            "fov_half_angle_deg": FOV_HALF_ANGLE_DEG,
            "max_latency": 8,
            "dropout_prob": 0.1,
            "noise_along_sigma": 0.04,
            "noise_along_per_m": 0.03,
            "noise_lat_sigma": 0.02,
            "noise_lat_per_m": 0.015,
        },
    )
    # 認識トラッキングの生死フラグ (タイムアウト方式)。フレーム単位ではなく
    # 「直近 timeout_s 以内に検出更新があったか」= teammate kick policy の
    # BALL_TIMEOUT_S=0.2s と同一セマンティクス。max_latency は ball_pos_rel と揃える。
    ball_in_fov = ObsTerm(func=ball_in_fov, params={"timeout_s": 0.2, "max_latency": 8})
    # キック方向はヨー推定誤差付き (per-episode 固定。reset_ball_perception で採番)。
    kick_direction_b = ObsTerm(func=kick_direction_b_perceived, params={"command_name": "kick_direction"})


@configclass
class K1AroundBallCriticCfg(K1CriticCfg):
    """上位 critic 観測 (特権情報: FOV に関係ない真のボール位置・速度)。"""

    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    # fixed_freq=1.6: frozen に使う 0524_walk.pt は旧規約 (固定 1.6Hz, φ=2πft) で
    # 学習されている。速度依存アキュムレータ規約の新しい歩行 pt に差し替えるときは
    # fixed_freq を None にすること。
    gait_phase = ObsTerm(func=high_action_phase_obs, params={"cmd_threshold": 0.05, "fixed_freq": 1.6})
    ball_pos_rel = ObsTerm(func=ball_pos_rel)
    ball_vel = ObsTerm(func=ball_vel)
    kick_direction_b = ObsTerm(func=kick_direction_b, params={"command_name": "kick_direction"})


@configclass
class K1AroundBallLowLevelCfg(K1PolicyCfg):
    """frozen 歩行ポリシー用観測。

    構造 (項の並び・次元) は歩行学習時の K1PolicyCfg と同一。ただし ``gait_phase`` は
    上位 action 駆動の位相に差し替える。frozen は「velocity_commands の速度」と
    「gait_phase のテンポ」の対応を学習しているが、階層構成で velocity_commands
    スロットには wrapper が上位 action を書き込むため、位相も同じ上位 action から
    作らないと学習時に見たことのない矛盾した入力になる (特に base_velocity を
    固定値にした歩行設定では常に最速テンポ・停止なしになってしまう)。
    """

    # fixed_freq=1.6: frozen に使う 0524_walk.pt は旧規約 (固定 1.6Hz, φ=2πft) で
    # 学習されている。速度依存アキュムレータ規約の新しい歩行 pt に差し替えるときは
    # fixed_freq を None にすること。
    gait_phase = ObsTerm(func=high_action_phase_obs, params={"cmd_threshold": 0.05, "fixed_freq": 1.6})


@configclass
class K1AroundBallObservationsCfg(K1ObservationsCfg):
    """観測グループ。``low_level`` は gait_phase 差し替え以外 K1PolicyCfg と同一構造。"""

    policy: K1AroundBallPolicyCfg = K1AroundBallPolicyCfg()
    critic: K1AroundBallCriticCfg = K1AroundBallCriticCfg()
    low_level: K1AroundBallLowLevelCfg = K1AroundBallLowLevelCfg()


@configclass
class K1AroundBallCommandsCfg(CommandsCfg):
    """歩行の ``base_velocity`` (low_level 観測のスロット維持用) + キック方向コマンド。

    resampling_time_range をエピソード長より長くして、1 エピソード内では
    キック方向が固定になるようにする (リセット時には必ず再サンプルされる)。
    """

    kick_direction = KickDirectionCommandCfg(
        asset_name="robot",
        resampling_time_range=(30.0, 30.0),
        # 学習初期はほぼ正面 (±60°) のみ。modify_kick_angle_range カリキュラムで
        # ±180° まで徐々に広げる (回り込みの難易度カリキュラム)。
        angle_range=(-math.pi / 3.0, math.pi / 3.0),
        debug_vis=True,
    )


@configclass
class K1AroundBallRewardsCfg:
    """回り込み専用の報酬。歩行用の K1Rewards は継承せず丸ごと置き換える (dribble と同じ方針)。"""

    # --- タスク主報酬 ---
    # ボール後方 standoff [m] の目標点への接近 (ポテンシャル形式)。
    # weight の目安は dribble の approach_ball_progress と同じ ~40
    # (dt=0.02s・接近 0.9m/s → 約 0.018m/step × 40 ≈ 0.7/step)。
    standoff_point_progress = RewTerm(
        func=standoff_point_progress,
        weight=40.0,
        params={"command_name": "kick_direction", "standoff": STANDOFF},
    )
    # 【2段目】回り込めた状態でのみ開く「ボールへの突進」報酬 [0, 1]。
    # ボールの真後ろに揃う (gate≈1) と、ボール方向への前進速度がそのまま報酬になる。
    # 速度そのものを見るので「速く突っ込む」ほど高い。min_distance=0 なので
    # ボールに接触するまで報酬が続く = ボール前で止まらず歩いたまま突っ込む。
    charge_to_ball_when_aligned = RewTerm(
        func=charge_to_ball_when_aligned,
        weight=18.0,
        params={
            "command_name": "kick_direction",
            # 姿勢ゲートを厳しめに: 体がキック方向を向き (heading_std 0.5≈29°)、
            # かつボールの真後ろに正確にいる (pos_std 0.4) ほど突進報酬が開く。
            # 甘いと「体が横を向いたまま突っ込む」「斜めから当てる」が残る。
            "gate_pos_std": 0.4,
            "gate_heading_std": 0.5,
            # 突進速度の正規化上限。--high_action_clip の vx と揃える (1.2 m/s で報酬最大)。
            "max_speed": 1.2,
            "min_distance": 0.0,
        },
    )
    # 【成果】突進の結果、ボールがキック方向 (ワールド座標) に動くほど報酬。
    # 方向のコサイン一致 [0, 1] × 速度ゲート (dribble と同じ既存関数)。
    # 弾いたボールが転がっている間ずっと入るので「正しい方向に強く当てる」を教える。
    ball_moved_along_kick = RewTerm(
        func=ball_velocity_along_kick,
        weight=4.5,
        params={"command_name": "kick_direction"},
    )

    # --- 制約 (ペナルティ / shaping) ---
    # 回り込み中にボールを突っ切るショートカットの禁止:
    # 真後ろ以外の方向から min_clearance 未満に入ったら罰。
    misaligned_ball_proximity = RewTerm(
        func=misaligned_ball_proximity,
        weight=-4.0,
        params={"command_name": "kick_direction", "min_clearance": 0.45, "angle_tol": ALIGN_ANGLE_TOL},
    )
    # 「揃っていない」状態でボールを動かしたら罰。揃った後の突進でボールを
    # 弾くのは無罪 (angle_tol は misaligned_ball_proximity と共有)。
    ball_disturbance = RewTerm(
        func=ball_disturbance_when_misaligned,
        weight=-3.0,
        params={"command_name": "kick_direction", "angle_tol": ALIGN_ANGLE_TOL, "max_speed": 0.5},
    )
    # ボールへ正対しているほど報酬 → 視野維持を助ける shaping。
    robot_facing_ball = RewTerm(
        func=robot_facing_ball,
        weight=1.0,
        params={"min_distance": 0.15},
    )
    # 視野 (±60°) からボールを外したら罰 (観測の hold-last-seen と対)。
    ball_out_of_fov = RewTerm(
        func=ball_out_of_fov,
        weight=-0.5,
        params={"fov_half_angle_deg": FOV_HALF_ANGLE_DEG},
    )

    # 時間ペナルティ: 生存している毎ステップ -0.2。「時間そのものがコスト」になるので
    # 回り込み・突進を最短で済ませるほど得。12 秒完走で合計 -120 相当だが、
    # 転倒 (-500) でエピソードを切り上げて逃れる抜け道は依然成立しない。
    time_penalty = RewTerm(func=mdp.is_alive, weight=-0.2)

    # --- 姿勢・滑らかさペナルティ (dribble の値をそのまま流用) ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.5 * 0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0 * 0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.6 * 0.5)
    # 上位 action (3D 歩行コマンド) の平滑性・変化率ペナルティ。
    # コマンドのジッター (毎ステップの細かい振り回し) は frozen が追従しきれず
    # 足のバタつき・転倒になる。特に近距離では wz が ±全開で反転する激しい振動
    # (実機ログで 1tick 変化 max|dcmd|=2.11 を観測) が転倒の主因なので、そこを
    # 直接抑えるために強化した (rate -0.6→-1.2, smoothness -0.25→-0.5)。
    # ※ 強くしすぎると方向転換が鈍って回り込みが遅くなる。まずここから始めて、
    #   回り込みが鈍いようなら緩める。
    # ※ ハンドオフ中 (上位 action が通常歩行コマンドに差し替わっている間) は
    #    ゲートで 0 になる版を使う。差し替え後のコマンドはポリシーの出力ではないので、
    #    その滑らかさで罰する/褒めるのは筋が通らない。
    action_smoothness_l2 = RewTerm(func=gated_high_action_smoothness_l2, weight=-0.5)
    action_rate_l2 = RewTerm(func=gated_high_action_rate_l2, weight=-1.2)
    # vx・vy 共活性ペナルティ。回り込みの円弧移動は「ボールに正対しながら横歩き
    # (vy) + 旋回 (wz)」で実現でき、wz は対象外なので干渉しない。
    # 実機ログで「深い回り込み中に vx=1.20 と vy=-1.00 を同時全開 (斜め全速) で回り、
    # 外に膨らんでボールを視野外に失って転倒」を観測。-1.0 では斜め全速を抑えきれて
    # いなかったので強化 (-1.0→-2.5)。これで「vy+wz の周回」か「vx+wz の突進」の
    # どちらかのモードに分離させ、ボールに正対したままキレイに回らせる。
    # 突進 (vx+wz) は罰対象外なので蹴りの勢いは削がない。
    high_action_xy_coactivation = RewTerm(func=gated_high_action_xy_coactivation, weight=-2.5)
    # 重心 jerk ペナルティ: 体のカクつき (コマンド急変で frozen が乱れた瞬間) を
    # 直接罰する。バタつき抑制のため dribble の値の 5 倍に強化。
    com_jerk_l2 = RewTerm(func=com_jerk_l2, weight=-5e-6)


@configclass
class K1AroundBallEnvCfg(K1FlatEnvCfg):
    """K1FlatEnv + サッカーボール。ボール回り込みの学習環境。"""

    scene: K1AroundBallSceneCfg = K1AroundBallSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1AroundBallObservationsCfg = K1AroundBallObservationsCfg()
    commands: K1AroundBallCommandsCfg = K1AroundBallCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0

        # 完全平面に戻す。歩行の K1FlatEnvCfg は芝対策で 1〜4cm のランダム凹凸地形
        # (NOISY_FLAT_TERRAIN_CFG) を使うが、凹凸の上ではボールが触れていないのに
        # 勝手に転がり、ball_disturbance / ball_moved_along_kick の学習信号を汚す。
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 報酬は回り込み用に丸ごと置き換える。
        self.rewards = K1AroundBallRewardsCfg()

        # ロボットの初期位置は env 原点固定・yaw は全周ランダム。
        # ボールは reset_ball_in_front_cone がロボットの向き基準の扇形に配置するので、
        # yaw をどう振っても「開始時にボールが視野内」は保証される。
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi, math.pi)
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)

        # リセット時に上位 action バッファを 0 にする。
        self.events.reset_prev_high_action = EventTerm(
            func=reset_prev_high_action,
            mode="reset",
        )

        # 「動きながら回り込み開始」を学習分布に含める: リセット時にロボットへ正面方向
        # (≒ボール方向) への前進初速を与える。実運用の Approach→AroundBall 引き渡し
        # (前進中に回り込みへ突入) を再現し、引き渡し遷移のコマンド跳ねを抑える。
        # ★ reset_base より後に登録すること (heading をリセット後の姿勢で読むため)。
        self.events.reset_base_forward_velocity = EventTerm(
            func=reset_base_forward_velocity,
            mode="reset",
            params={"speed_range": (0.0, 0.6)},
        )

        # ボールは「リセット後のロボットの向き」基準の正面扇形に静止状態でスポーン
        # (walk_kick の reset_ball_in_front_of_robot と同方式)。half_angle=1.0rad (≈57°)
        # ≤ FOV 60° なので、ロボット yaw が全周ランダムでも開始時は必ず視野内。
        # 距離の下限 0.6m は misaligned_ball_proximity のペナルティ圏 (0.45m) の外。
        self.events.reset_ball = EventTerm(
            func=reset_ball_in_front_cone,
            mode="reset",
            params={
                "dist_range": (0.6, 3.0),
                "half_angle": 1.0,
                "ball_radius": 0.11,
            },
        )

        # 知覚DR (ball_pos_rel_perceived / kick_direction_b_perceived) の per-episode
        # パラメータを採番し、履歴・認識出力バッファを現在の真値で初期化する。
        # ★ reset_ball より後に登録すること (配置後のボール真値で初期化するため)。
        # 実測スペックが出たら latency_range / update_period_range / *_sigma を差し替える。
        self.events.reset_ball_perception = EventTerm(
            func=reset_ball_perception,
            mode="reset",
            params={
                "max_latency": 8,
                # ★ ビジョンは実機で約30Hz が上限 (これ以上速くならない)。制御は50Hz
                #   (step_dt=0.02s) なので 30Hz = 1.67 tick。tick は整数しか取れないので
                #   「30Hz より速い」設定 (period=1) は実機に存在しない = 学習してはいけない。
                #   period 2 = 25Hz, 3 = 16.7Hz, 4 = 12.5Hz。
                #   このレンジは「間隔の下限・上限」で、実際の間隔は毎更新でこの範囲から
                #   引き直される (observations 側)。30Hz が常時出るとは限らず負荷で遅く
                #   なる状況を、エピソード内で速い/遅いフレームが混ざる形で再現する。
                "update_period_range": (2, 4),    # 25Hz〜12.5Hz でフレーム間隔がジッタ
                # ★ レイテンシ: ビジョン担当と確認 (2026-07)、カメラ30Hzだが認識出力までの
                #   レイテンシは実測なし・大きさ不明。分かっているのは下限だけ (撮影1フレーム
                #   33ms ≈ 2 tick は物理的に切れない)。不明な間は広めに 40〜160ms をDRして
                #   「どのレイテンシでも動く」ポリシーにしておく (per-episode 固定サンプル)。
                #   実測が出たらこのレンジを実測±マージンに絞ること。max_latency (履歴長) は
                #   上限 8 tick に合わせて全箇所 8 で揃えている。
                "latency_range": (2, 8),          # 40〜160ms @ 50Hz
                "bias_sigma": 0.03,               # 位置系統バイアス [m]
                "yaw_err_sigma": 0.05,            # ヨー推定誤差 [rad] ≈ 3°
            },
        )

        # ボールを蹴れたら (スポーン位置から 0.3m 以上動いたら) 1 秒後にエピソード終了。
        # 終了すると通常リセットが全部走るので「ボールもロボットも」再配置される。
        # time_out=True: 成功による区切りなので termination_penalty (-500) の対象外。
        #
        # delay_steps 150 (3秒) → 50 (1秒):
        #   * ボール速度は接触直後が最大なので ball_moved_along_kick は 1 秒でほぼ取り切れる。
        #   * この猶予はハンドオフ (突進→通常歩行の引き渡し) 後に転倒しないかを見る時間でもある。
        #     ハンドオフ中の転倒は time_out ではない通常終了なので -500 が乗り、GAE で遡って
        #     「どんな速度・姿勢で接触したか」に信用割当される = 生き延びられる接触を学ぶ。
        #   * 3 秒のままだと、その間ずっとチェイスに報酬が入り続けて転倒しても黒字になる
        #     (charge のゲーティングと合わせてここを塞ぐ)。
        self.terminations.ball_kicked = DoneTerm(
            func=ball_kicked,
            time_out=True,
            params={"kick_dist_threshold": 0.3, "delay_steps": 50},
        )

        # 歩行 (FlatEnv) 由来のカリキュラムは高レベルタスクには不要なので無効化。
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.push_robot_stage1 = None

        # キック方向の難易度カリキュラム: ±60° (ほぼ正面・回り込みほぼ不要) から
        # ±180° (真後ろ・全周回り込み) へ徐々に広げる。真後ろを最初から混ぜると
        # 立ち上がりが遅いので、易しい状況で「揃えたら突っ込む」を先に学ばせる。
        self.curriculum.kick_angle_range = CurrTerm(
            func=modify_kick_angle_range,
            params={
                "command_name": "kick_direction",
                "start_deg": 60.0,
                "end_deg": 180.0,
                "start_step": 2000,
                "end_step": 10000,
            },
        )


@configclass
class K1AroundBallEnvCfg_PLAY(K1AroundBallEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 知覚DR は enable_corruption では切れない (関数内でノイズ付加するため)。
        # play では挙動確認のためクリーン化する。ロバスト性を実機同等で見たいときは
        # このブロックをコメントアウトすれば学習時と同じ知覚ノイズで再生できる。
        self.observations.policy.ball_pos_rel.params.update({
            "dropout_prob": 0.0,
            "noise_along_sigma": 0.0,
            "noise_along_per_m": 0.0,
            "noise_lat_sigma": 0.0,
            "noise_lat_per_m": 0.0,
        })
        # 更新レートと遅延は「ノイズ」ではなくビジョン(約30Hz上限)というハード制約なので
        # play でもクリーン化しない。ここを 1 tick に戻すと実機に無い 50Hz 認識で
        # 再生することになり、実機で効いてくる遅延の影響が見えなくなる。
        # レイテンシは実測が無い (ビジョン担当も不明) ので、学習レンジ 40〜160ms の
        # 中央 80ms を代表値として固定再生する。実測が出たらその値に差し替えること。
        self.events.reset_ball_perception.params.update({
            "latency_range": (4, 4),          # 80ms 固定 (学習レンジ 40〜160ms の中央)
            "update_period_range": (2, 2),    # 25Hz 固定
            "bias_sigma": 0.0,
            "yaw_err_sigma": 0.0,
        })
