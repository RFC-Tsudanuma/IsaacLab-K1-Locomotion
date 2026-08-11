# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスクの環境定義。

frozen 歩行ポリシー (0524_walk.pt) の上に載せる高レベルポリシーを PPO で学習する
階層タスク。around_ball と同じく ``HierarchicalVecEnvWrapper``
(scripts/rsl_rl/dribble_helpers.py) で学習する前提で、上位 action は
3D 歩行コマンド (vx, vy, wz)。

タスク (RoboCup HSL 2026 Middle ディビジョン想定):
    ロボットはゴール中央 (ゴールライン上) に立ち、転がってくるボールが
    ゴールラインを越える前に横ステップ移動で遮る。ダイブはしない。

フィールド仕様 (ルールブック Table 2/3/4 の Middle 値、シーンは簡易プリミティブ):
    * ゴール幅 (ポスト内側間) 2.6 m / クロスバー高さ 1.7 m / ポスト太さ 0.10 m
    * ボール: FIFA サイズ4相当 (直径 0.20 m, 質量 0.37 kg)
    * 失点: ボール全体がポスト間のゴールラインを越えたとき
      (ボール中心 x < −半径)。ポストは物理コリジョン有り (跳ね返りは失点扱いに
      しない。跳ね返り後にラインを越えれば失点)。

座標系: env origin = ゴール中央 (ゴールライン上)。+x がフィールド側。

3 ステージのカリキュラム (別 gym ID + シェルスクリプト + ckpt 受け渡し):
    * Stage 1 (Isaac-Goalkeeper-Stage1-K1-v0): ボールなし。ゴール幅内のランダム
      目標 y への速い到達と停止。目標到達で再サンプル。
    * Stage 2 (Isaac-Goalkeeper-K1-v0): 遅いボール。スポーン距離・角度・狙い先を
      ランダム化。
    * Stage 3 (Isaac-Goalkeeper-Stage3-K1-v0): セーブ成功率に応じて初速上限を
      連続的に引き上げる適応カリキュラム (mdp.adaptive_ball_speed)。
      初速上限 (ball_speed_cap) は Stage 1 の実効横移動速度から
      「セーブ可能な限界初速 × 0.9」を逆算して --override_json で与えること。

観測はステージ間で次元固定 (ステージ1 はボール観測にダミー 0 を入れる)。
可変パラメータは ``GoalkeeperParamsCfg`` に集約し、
``--override_json '{"env": {"goalkeeper.ball_speed_cap": 2.0}}'`` の形で
設定ファイルから制御できる。
"""

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# 歩行 (locomotion) パッケージの共通部品は import で参照のみ (locomotion 側は変更しない)
from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from ..locomotion.rough_env_cfg import K1CriticCfg, K1ObservationsCfg, K1PolicyCfg
from ..locomotion.velocity_env_cfg import MySceneCfg
from ..locomotion.mdp.events import reset_prev_high_action
from ..locomotion.mdp.observations import last_high_action
from ..locomotion.mdp.rewards import (
    com_jerk_l2,
    high_action_rate_l2,
    high_action_smoothness_l2,
)

# frozen 連携の位相観測は around_ball の実装 (0524_walk.pt 対応済み) を再利用
from ..around_ball.mdp.observations import high_action_phase_obs

from .mdp.curriculums import adaptive_ball_speed
from .mdp.events import (
    reset_ball_perception,
    reset_ball_shot,
    reset_gk_buffers,
    reset_stage1_target_and_park,
    stage1_target_tick,
)
from .mdp.observations import (
    gk_ball_active,
    gk_ball_pos_rel,
    gk_ball_pos_rel_perceived,
    gk_ball_vel,
    gk_ball_vel_perceived,
    gk_self_state,
    gk_target_y,
)
from .mdp.rewards import (
    face_field,
    hold_at_target,
    return_to_center_after_save,
    save_touch_bonus,
    stay_on_goal_line,
    target_reach_velocity,
    track_target_y,
)
from .mdp.terminations import goal_conceded, robot_out_of_bounds, save_success

# --- ゴール・ボールの幾何パラメータ (ルールブック Middle ディビジョン = M-Field) ---
# ★ 2026-08-08: ゴール幅 2.5m → 2.6m に変更。ポスト位置・クロスバー長・ゴールライン
#   マーカー・観測のクランプ (max_y) はすべてこの定数から導出されるので、ここだけ直せば
#   追従する。ただし mdp/observations.py・mdp/rewards.py の引数デフォルト値だけは
#   import 循環を避けるため直値で持っているので、変えるときは併せて直すこと。
GOAL_HALF_WIDTH = 1.3      # ゴール幅 2.6m の半分 (ポスト内側)
POST_RADIUS = 0.05         # ポスト/クロスバー太さ 0.10m (許容 0.07〜0.12)
CROSSBAR_HEIGHT = 1.7      # クロスバー下端の目安 (許容 1.5〜1.9。横セーブのみなので判定未使用)
BALL_RADIUS = 0.10         # FIFA サイズ4相当 (直径約 0.20m)
BALL_MASS = 0.37           # 同 (質量 0.35〜0.39 kg)
LINE_WIDTH = 0.05          # ゴールライン白線の幅 (ルールブック 0.05〜0.12m の下限)


@configclass
class GoalkeeperParamsCfg:
    """goalkeeper タスクの可変パラメータ (イベント・終了・カリキュラムが実行時に参照)。

    ``--override_json`` の ``{"env": {"goalkeeper.<field>": value}}`` で上書きできる。
    """

    # 幾何 (判定用。シーン側の定数と一致させること)
    ball_radius: float = BALL_RADIUS
    goal_half_width: float = GOAL_HALF_WIDTH
    # 守備面: ゴールラインから guard_x [m] フィールド側にロボットを置く。
    guard_x: float = 0.4
    # ボール接近中に「位置ずれ [m] → 速度 [m/s]」へ換算する除数 [s]。
    # ★ 2026-08-08: 到達猶予時間で割る方式をやめ、この固定値にした (最速で向かう)。
    #   ずれ > drive_t_fast × 1.3 [m] で常に全力。0.15 なら 0.195m 以上で全力になる。
    #   小さくするほど全力域が広がる。0 にすると停止できず振動するので下げすぎないこと。
    drive_t_fast: float = 0.15

    # ステージ1: ボールのパーク位置 (ゴール座標系 x, y) とランダム目標
    park_pos: tuple = (5.0, 0.0)
    stage1_target_range: float = 1.3    # 目標 y ∈ ±この値 [m] (= GOAL_HALF_WIDTH)
    stage1_reach_tol: float = 0.15      # 到達判定の許容誤差 [m]
    stage1_cmd_tol: float = 0.08        # 到達判定: 上位コマンドノルム上限 (足踏み対策)
    stage1_speed_tol: float = 0.15      # 到達判定: ベース並進速度上限 [m/s]
    stage1_hold_steps: int = 25         # 静止保持ステップ数 (0.5s @ 50Hz) → 目標再サンプル
    # 目標サンプリングの分布制御 (速度学習の圧を保つ):
    stage1_min_move: float = 0.5        # 現在位置からの最低移動距離 [m]。近距離帯は除外して採る
    stage1_far_prob: float = 0.3        # 「反対側ポスト際ゾーン」を目標にする確率 (長距離スプリント保証)
    stage1_far_zone: tuple = (0.95, 1.3)  # ポスト際ゾーンの |y| 範囲 [m]。ポスト間 2.6m の往復を練習させる

    # ステージ2/3: ボールのスポーンと初速
    spawn_dist_range: tuple = (1.5, 5.0)
    spawn_half_angle: float = 1.1          # スポーン方位 ±[rad] (+x 正面基準, ≈63°)
    aim_y_range: float = 1.1               # 狙い先 y ∈ ±この値 [m] (ポスト内側)
    # 適応カリキュラム (mdp.adaptive_difficulty) が段階的に広げる狙い先 y の範囲 [m]。
    aim_y_stages: tuple = (0.4, 0.6, 0.8, 1.1)
    ball_speed_min: float = 0.5            # 初速下限 [m/s]
    ball_speed_max: float = 1.0            # 初速上限 [m/s] (ステージ2 固定 / ステージ3 初期値)
    ball_speed_cap: float = 3.0            # 適応カリキュラムの上限 [m/s]。
    #   Stage1 の実効横移動速度 v_lat から eval_goalkeeper_speed.py が逆算した値に
    min_time_to_line: float = 1.2

    # --- 到達不能球 (2026-08-11 追加) ---
    # ★ min_time_to_line のクランプは「取れない球」を訓練から完全に除外するため、
    #   ポリシーは **間に合わないときにどう振る舞うか** を一度も学んでいない。実測では
    #   3.5m から 4.0 m/s (到達 0.875s < 1.2s) の球を与えると転倒が 1 回 → 56 回に増える。
    #   歩幅も測ったが、転倒直前の踏み出しは通常の全力横移動より **小さく**、歩幅制限では
    #   直らないことが分かった (eval_gk_stride.py)。素直に「取れない球」を一定割合で混ぜ、
    #   既存の転倒ペナルティ経由で「届かなくても姿勢を保つ」を学ばせる。
    #   既定 0.0 なので、override JSON で有効化しない限り従来の挙動は変わらない。
    hard_ball_prob: float = 0.0        # 到達不能球にする確率
    hard_ball_speed_mult: float = 1.6  # 初速を上限 hi の何倍まで出すか
    hard_ball_min_time: float = 0.5    # 到達不能球に適用する緩和クランプ [s]。
    #   0 にしないのは、到達 0.4s 級だと知覚 (遅延 116ms + 更新 40ms) が間に合わず
    #   学習信号がゼロのノイズになるため。

    # 知覚DR (policy のボール観測に掛かる。critic は真値):
    #
    # ★ 下記のうち **実際に読まれているのは perc_update_rate_hz と perc_vel_bias_range
    #   だけ**。位置側のレイテンシ・ノイズ・検出率は VirtualPerception が持っており、
    #   値は mdp/perception.py の soccer_vision_train_cfg() が決めている
    #   (レイテンシ 116ms 固定 / σ(d) = 0.124d + 0.149 [m] / 検出率 90%)。
    #   perc_latency_range 以下の 5 つは **どこからも参照されていない (dead)**。
    #   触っても効かないので、値を変えたいときは soccer_vision_train_cfg() 側か
    #   mdp/observations.py の _gk_perception() での上書きを見ること。
    perc_latency_range: tuple = (2, 4)              # ← dead (未参照)
    # ビジョンの更新レート [Hz]。_gk_perception() で VirtualPerception に流し込む。
    # ★ 2026-08-08: 上限を 25Hz に下げた。カメラ自体は 30fps 出るが、実機では
    #   検出処理の取りこぼしと後段の遅れがあるので、名目 fps ではなく
    #   「ポリシーに新しい値が届く実効レート」で見る。ここに per-env の
    #   ガウスジッタ (update_hz_std = 1.06Hz) が乗るので、上端の env は 26Hz 台に届く。
    perc_update_rate_hz: tuple = (20.0, 25.0)
    perc_dropout_prob: float = 0.1            # ← dead (未参照)
    perc_noise_sigma: float = 0.03            # ← dead (未参照)
    perc_noise_per_m: float = 0.02            # ← dead (未参照)
    perc_vel_noise_sigma: float = 0.1         # ← dead (未参照)
    perc_bias_sigma: float = 0.03             # ← dead (未参照)
    # 速度のエピソード固定バイアス (x, y 各軸独立)。遅延由来の系統誤差を模擬。
    perc_vel_bias_range: tuple = (0.5, 1.0)

    # ------------------------------------------- 自己位置推定 (実機は MCL) の誤差
    # 実機の MCL は白色ノイズではなく「バイアス + ドリフト + 不連続な跳び」を出す。
    # 平滑化は意図的に OFF で、1 フレームあたり 0.5 m / 0.5 rad まで動く設定。
    # キーパーは常時ボールを見て下を向くのでランドマークが入らず、odometry のみに
    # 落ちている時間が長いと想定される。跳びは学習で経験しないと実機で未知入力になる。
    #
    # ★ y 方向の誤差はボールと自分の両方に同じだけ乗って相対関係が保たれるため、
    #   横移動の指令にはほぼ効かない (往復で相殺される)。効くのは
    #   (a) 守備面までの前後距離 x → 定位置がじわじわずれる
    #   (b) ヨー → 相対/ゴール座標の変換に残る
    #   の 2 つ。実測ではセーブ成立 96.7% → 93.7% に収まる。
    loc_bias_xy_m: float = 0.20        # 位置バイアスの一様サンプル幅 [±m]
    loc_bias_yaw_deg: float = 6.0      # ヨーバイアスの一様サンプル幅 [±deg]
    loc_drift_xy_mps: float = 0.03     # 位置ドリフト速度 [±m/s]
    loc_drift_yaw_dps: float = 1.0     # ヨードリフト速度 [±deg/s]
    # 跳びの頻度。再収束イベントはゴール前でランドマークが見える状況なら
    # 数秒〜数十秒に 1 回 (毎秒 1 回は MCL の挙動ではない)。
    loc_jump_hz_range: tuple = (0.0, 0.15)  # 跳びの発生頻度 [回/秒]
    loc_jump_m: float = 0.5            # 跳びの大きさ [±m] (MCL の 1 フレーム補正上限)
    loc_jump_rad: float = 0.2          # 跳びの大きさ [±rad] (同上)
    # ★ 再収束の時定数。ランドマークが視野に入ると誤差が戻る挙動を表す。0 で無効。
    #   これが無いと誤差が上限に張り付き、ロボットが誤った自己位置を信じ続けて徘徊する。
    #   定常誤差 ≈ ドリフト速度 × tau = 0.03 × 5 = 0.15m。跳びは数秒で解消される。
    loc_recover_tau_s: float = 5.0
    loc_max_err_m: float = 0.6         # 累積誤差の上限 [±m] (跳びの直後だけ触れる保険)
    loc_max_err_rad: float = 0.3       # 累積誤差の上限 [±rad]

    # 知覚DR (VirtualPerception + 速度バイアス) を全部切ってクリーン観測にするフラグ。
    perception_clean: bool = False

    # セーブ判定
    touch_force_threshold: float = 0.1  # 足-ボール接触力のしきい値 [N]
    touch_proximity: float = 0.5        # タッチ判定フォールバックの近傍距離 [m]
    save_delay_steps: int = 100         # セーブ確定までの保持時間 (2s @ 50Hz)

    # ステージ3: 適応カリキュラム (セーブ成功率 EMA → 初速上限)
    adaptive_success_threshold: float = 0.85
    adaptive_fail_threshold: float = 0.55
    adaptive_speed_delta: float = 0.05      # 1 回の調整量 [m/s]
    adaptive_ema_alpha: float = 0.01        # エピソード 1 件あたりの EMA 更新率
    adaptive_warmup_episodes: int = 2000    # 調整開始までのウォームアップ件数
    # 難易度を 1 段動かした後、次の判定を再開するまでに必要なエピソード件数。
    adaptive_cooldown_episodes: int = 3000


# ---------------------------------------------------------------------------

@configclass
class K1GoalkeeperSceneCfg(MySceneCfg):
    """歩行シーン + サッカーボール + 簡易ゴール (ポスト×2 + クロスバー) + 接触センサ。

    ゴールはリポジトリ内にコード参照済みのアセットが無いため、ルールブック仕様の
    プリミティブ (静的コライダ) で構築する。ゴールライン x=0、+x がフィールド側。
    """

    # FIFA サイズ4相当のボール (既存タスクのサイズ5相当 0.11m/0.45kg とは意図的に変える)
    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=MassPropertiesCfg(mass=BALL_MASS),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.7,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, BALL_RADIUS)),
    )

    # ゴールポスト (静的コライダ。rigid_props を付けないので不動)。
    # ポスト中心は内側面が ±GOAL_HALF_WIDTH になるよう ±(GOAL_HALF_WIDTH + POST_RADIUS)。
    goal_post_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalPostLeft",
        spawn=sim_utils.CylinderCfg(
            radius=POST_RADIUS,
            height=CROSSBAR_HEIGHT + 2 * POST_RADIUS,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.5),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.4,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, GOAL_HALF_WIDTH + POST_RADIUS, 0.5 * (CROSSBAR_HEIGHT + 2 * POST_RADIUS)),
        ),
    )
    goal_post_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalPostRight",
        spawn=sim_utils.CylinderCfg(
            radius=POST_RADIUS,
            height=CROSSBAR_HEIGHT + 2 * POST_RADIUS,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.5),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.4,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -(GOAL_HALF_WIDTH + POST_RADIUS), 0.5 * (CROSSBAR_HEIGHT + 2 * POST_RADIUS)),
        ),
    )
    # クロスバー (横セーブのみなので視覚+コリジョンの飾りに近いが、仕様どおり置く)
    goal_crossbar: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalCrossbar",
        spawn=sim_utils.CylinderCfg(
            radius=POST_RADIUS,
            height=2 * (GOAL_HALF_WIDTH + 2 * POST_RADIUS),
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.5),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.4,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, CROSSBAR_HEIGHT + POST_RADIUS)),
    )
    # ゴールライン (視覚のみ・コリジョンなし)。ルールブックの線幅 0.05m・白。
    goal_line: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalLine",
        spawn=sim_utils.CuboidCfg(
            size=(LINE_WIDTH, 2 * (GOAL_HALF_WIDTH + 2 * POST_RADIUS), 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.001)),
    )

    # 足-ボール接触センサ (ball_kick と同方式)。セーブのタッチ検出が使う。
    contact_balls_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    contact_balls_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )


# ---------------------------------------------------------------------------

@configclass
class K1GoalkeeperPolicyCfg(K1PolicyCfg):
    """上位 policy 観測 (59 次元 = 歩行 49 + タスク 10)。

    先頭 49 は歩行 K1PolicyCfg と同一スロット構造で、
    ``velocity_commands`` → 前回上位 action、``gait_phase`` → 上位 action 駆動位相
    (0524_walk.pt の旧規約 fixed_freq=1.6) に差し替え (around_ball と同じ)。

    追加のタスク観測 (ステージ間で次元固定):
        ball_pos_rel(2) + ball_vel(2) + ball_active(1) + target_y(1) + self_state(4)
    ステージ1 ではボール観測はダミー 0 (ball_active=0)、target_y はランダム目標。
    ステージ2 以降は target_y = ボール到達予測点 (compute_target_y)。
    """

    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    # fixed_freq=1.6: frozen に使う 0524_walk.pt は旧規約 (固定 1.6Hz, φ=2πft)。
    # 新規約の歩行 pt に差し替えるときは None にすること。
    gait_phase = ObsTerm(func=high_action_phase_obs, params={"cmd_threshold": 0.05, "fixed_freq": 1.6})

    # ボール観測は知覚DR版 (レイテンシ・更新レート・ドロップ・距離依存ノイズ・バイアス。
    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel_perceived)
    ball_vel = ObsTerm(func=gk_ball_vel_perceived)
    ball_active = ObsTerm(func=gk_ball_active)
    # 到達予測も知覚DR後のボール状態から計算 (実機では認識出力から同じ計算をする)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True})
    self_state = ObsTerm(func=gk_self_state, noise=Unoise(n_min=-0.02, n_max=0.02))


@configclass
class K1GoalkeeperCriticCfg(K1CriticCfg):
    """上位 critic 観測 (特権情報つき、ノイズなし)。"""

    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    gait_phase = ObsTerm(func=high_action_phase_obs, params={"cmd_threshold": 0.05, "fixed_freq": 1.6})

    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel)
    ball_vel = ObsTerm(func=gk_ball_vel)
    ball_active = ObsTerm(func=gk_ball_active)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH})
    self_state = ObsTerm(func=gk_self_state)


@configclass
class K1GoalkeeperLowLevelCfg(K1PolicyCfg):
    """frozen 歩行ポリシー用観測 (49 次元、around_ball の low_level と同一方針)。

    構造は歩行学習時の K1PolicyCfg と同一。``velocity_commands`` スロットは
    wrapper が上位 action で上書きする。``gait_phase`` のみ上位 action 駆動に
    差し替える (base_velocity ダミーから位相を作ると frozen が壊れる)。
    """

    gait_phase = ObsTerm(func=high_action_phase_obs, params={"cmd_threshold": 0.05, "fixed_freq": 1.6})


@configclass
class K1GoalkeeperObservationsCfg(K1ObservationsCfg):
    policy: K1GoalkeeperPolicyCfg = K1GoalkeeperPolicyCfg()
    critic: K1GoalkeeperCriticCfg = K1GoalkeeperCriticCfg()
    low_level: K1GoalkeeperLowLevelCfg = K1GoalkeeperLowLevelCfg()


# ---------------------------------------------------------------------------

@configclass
class K1GoalkeeperRewardsCfg:
    """ゴールキーパー専用の報酬。歩行用 K1Rewards は継承せず丸ごと置き換える。"""

    # --- タスク主報酬 ---
    track_target_coarse = RewTerm(
        func=track_target_y, weight=1.5, params={"std": 0.6, "max_y": GOAL_HALF_WIDTH},
    )
    track_target_fine = RewTerm(
        func=track_target_y, weight=1.5, params={"std": 0.15, "max_y": GOAL_HALF_WIDTH},
    )
    # 目標方向への横移動速度 (遠くても勾配が一定に出る密報酬)。
    target_reach_velocity = RewTerm(
        func=target_reach_velocity,
        weight=5.0,
        params={"deadband": 0.12, "v_cap": 0.8, "cmd_scale": 0.5, "max_y": GOAL_HALF_WIDTH},
    )
    # 目標到達後の「コマンド 0 で静止」報酬 (足踏み局所最適対策込み)。
    hold_at_target = RewTerm(
        func=hold_at_target,
        weight=2.0,
        params={
            "pos_std": 0.25,
            "cmd_std_coarse": 0.35,
            "cmd_std_fine": 0.1,
            "lin_vel_std": 0.4,
            "yaw_rate_weight": 0.25,
            "max_y": GOAL_HALF_WIDTH,
        },
    )
    # セーブ (タッチして弾いた) 瞬間の一回限りボーナス。
    save_touch_bonus = RewTerm(func=save_touch_bonus, weight=100.0)
    # 弾いた後のゴール中央への復帰 (小さめ)。
    return_to_center = RewTerm(func=return_to_center_after_save, weight=1.0, params={"std": 0.5})

    # --- 位置・姿勢の shaping ---
    # ゴールライン近傍に留まる (前後方向の定位置維持)。
    stay_on_goal_line = RewTerm(func=stay_on_goal_line, weight=1.0, params={"std": 0.3})
    # フィールド側を向き続ける (vy 横ステップの意味論維持)。
    face_field = RewTerm(func=face_field, weight=1.0, params={"std": 0.5})

    # --- 失敗ペナルティ ---
    # 失点・転倒・場外 (time_out=False の終了) に対する大きな負報酬。
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

    # --- 姿勢・滑らかさペナルティ (around_ball 系と同じ値) ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.5 * 0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0 * 0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.6 * 0.5)
    # 上位コマンドの急変ペナルティ (frozen が追従できないジッターの抑制)。
    action_smoothness_l2 = RewTerm(func=high_action_smoothness_l2, weight=-0.5)
    action_rate_l2 = RewTerm(func=high_action_rate_l2, weight=-1.2)
    com_jerk_l2 = RewTerm(func=com_jerk_l2, weight=-5e-6)


# ---------------------------------------------------------------------------

@configclass
class K1GoalkeeperEnvCfg(K1FlatEnvCfg):
    """K1FlatEnv + ゴール + ボール。ゴールキーパー学習環境 (ステージ2 = 遅いボール)。"""

    scene: K1GoalkeeperSceneCfg = K1GoalkeeperSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: K1GoalkeeperObservationsCfg = K1GoalkeeperObservationsCfg()
    goalkeeper: GoalkeeperParamsCfg = GoalkeeperParamsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0

        # 完全平面 (凹凸の上ではボールが勝手に転がり判定・予測が汚れる)。
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 報酬はゴールキーパー用に丸ごと置き換える。
        self.rewards = K1GoalkeeperRewardsCfg()

        # ロボットは守備面 (ゴールラインの guard_x=0.4m 前) 付近・フィールド側 (+x)
        self.events.reset_base.params["pose_range"]["x"] = (0.3, 0.5)
        self.events.reset_base.params["pose_range"]["y"] = (-0.5, 0.5)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        # 静止に近い状態から開始 (キーパーは構えて待つ)。
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }

        # リセット時: 上位 action バッファ → goalkeeper 状態バッファ → ボール発射
        self.events.reset_prev_high_action = EventTerm(func=reset_prev_high_action, mode="reset")
        self.events.reset_gk_buffers = EventTerm(func=reset_gk_buffers, mode="reset")
        self.events.reset_ball = EventTerm(func=reset_ball_shot, mode="reset")
        self.events.reset_ball_perception = EventTerm(func=reset_ball_perception, mode="reset")

        # ボールの摩擦・反発ドメインランダム化 (ball_kick と同じ値)。
        self.events.ball_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "static_friction_range": (0.5, 1.0),
                "dynamic_friction_range": (0.35, 0.8),
                "restitution_range": (0.0, 0.3),
                "num_buckets": 64,
            },
        )

        # 終了条件: 失点 (失敗) / セーブ成功 (成功) / 守備範囲逸脱 (失敗)。
        # 転倒 (base_contact) とタイムアウトは K1FlatEnvCfg から継承。
        self.terminations.goal_conceded = DoneTerm(func=goal_conceded)
        self.terminations.save_success = DoneTerm(func=save_success, time_out=True)
        # ラインより後ろ (ゴール内) に下がる守り方は許さない (x 下限 -0.1)。
        self.terminations.out_of_bounds = DoneTerm(
            func=robot_out_of_bounds,
            params={"x_range": (-0.1, 2.5), "y_abs_max": 2.2},
        )

        # 歩行 (FlatEnv) 由来のカリキュラムは高レベルタスクには不要なので無効化。
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.push_robot_stage1 = None


# ---------------------------------------------------------------------------

@configclass
class K1GoalkeeperStage1EnvCfg(K1GoalkeeperEnvCfg):
    """ステージ1: ボールはパーク (観測ダミー 0)。±1.3m のランダム目標への往復。"""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0

        # ボールは発射せず遠方にパーク + ランダム目標を採番。
        self.events.reset_ball = EventTerm(func=reset_stage1_target_and_park, mode="reset")
        # 毎ステップ: 目標到達 (コマンド 0 + 静止) を保持できたら目標を再サンプル。
        self.events.stage1_target_tick = EventTerm(
            func=stage1_target_tick,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # 毎制御ステップ (dt=0.02s)
            is_global_time=True,
        )

        # ボールが無いのでセーブ関連の項は落とす。
        self.terminations.save_success = None
        self.rewards.save_touch_bonus = None
        self.rewards.return_to_center = None
        # 到達後の停止をステージ1 の主課題にする (足踏み対策の勾配を強く)。
        self.rewards.hold_at_target.weight = 8.0


# ---------------------------------------------------------------------------

@configclass
class K1GoalkeeperStage3EnvCfg(K1GoalkeeperEnvCfg):
    """ステージ3: セーブ成功率 (EMA) が閾値を超えるたびに初速上限を引き上げる。"""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.ball_speed_adaptive = CurrTerm(func=adaptive_ball_speed)


# ---------------------------------------------------------------------------

def _make_play_clean(cfg: K1GoalkeeperEnvCfg) -> None:
    """PLAY 環境の共通クリーン化: 外乱と知覚DRを切って挙動確認しやすくする。

    知覚DRは enable_corruption では切れない (関数内でノイズ付加するため)、
    GoalkeeperParamsCfg の perc_* をクリーン値に上書きする。学習時と同じ
    知覚ノイズで再生したいときはこの呼び出しをコメントアウトする。
    """
    cfg.scene.num_envs = 32
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # VirtualPerception + 速度バイアスをクリーン化 (真値・遅延なし・見失いなし)。
    cfg.goalkeeper.perception_clean = True


@configclass
class K1GoalkeeperEnvCfg_PLAY(K1GoalkeeperEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)


@configclass
class K1GoalkeeperStage1EnvCfg_PLAY(K1GoalkeeperStage1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)
