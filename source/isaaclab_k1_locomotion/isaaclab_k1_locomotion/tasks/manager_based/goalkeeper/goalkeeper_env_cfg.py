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
    # ★ 2026-08-12: 0.4 → 0.9 (ユーザー指示)。前に出るほどシュートコースが狭まるが、
    #   ボールが守備面に届くまでの距離が 0.5m 縮むぶん反応時間も減る。
    # ★ 2026-08-16: 0.9 → 0.6 (ユーザー指示)。ゴールラインに近づくぶん体で塞げる
    #   角度は減るが、ボールが守備面に届くまでの距離が伸びて反応時間が増える。
    #   ball_speed_cap を 6.0 に引き上げたので、速い球に対する猶予を優先する側に振る。
    #   ※ 初期配置 (K1GKDirectEnvCfg.__post_init__) はこの値から導出しているが、
    #     導出は cfg 構築時に走るので **override JSON で guard_x を変えても
    #     初期配置は追従しない** (train.py は cfg 構築後に override を適用する)。
    #     したがって guard_x は JSON ではなく **この既定値を直接変えること**。
    #   ※ この値は直接版・階層版・dualhist・横移動特化で共有される。
    guard_x: float = 0.6
    # ボール接近中に「位置ずれ [m] → 速度 [m/s]」へ換算する除数 [s]。
    # ★ 2026-08-08: 到達猶予時間で割る方式をやめ、この固定値にした (最速で向かう)。
    #   ずれ > drive_t_fast × 1.3 [m] で常に全力。0.15 なら 0.195m 以上で全力になる。
    #   小さくするほど全力域が広がる。0 にすると停止できず振動するので下げすぎないこと。
    drive_t_fast: float = 0.15
    # ★ 2026-08-15: task_drive_vector の 1 次ローパスの時定数 [s]。0 で無効。
    #   この指令は自己位置 (MCL) から作られ、しかも drive_t_fast=0.15 のせいで
    #   0.195m 以上のずれで常に全力になる。MCL の跳び (±0.5m) は必ず「全力で横へ」
    #   に化けるので、歩行中に入ると急激な方向転換になって崩れる。
    #   MuJoCo でランドマーク認識がカクつくとロボットが揺れる件の、位相修正
    #   (task_drive_phase_obs) では塞ぎきれない残りの経路。
    #   本物のボールの動きは連続なのでほぼ素通りし、1 フレームの跳びだけ減衰する。
    #   ★ 実機の C++ 側にも同じ時定数で実装すること。
    drive_filter_tau_s: float = 0.12

    # --- 待機保持 (idle hold) のデッドバンド (2026-08-17) ---
    #
    # ★ 「脅威が無く、かつ定位置の近くに居る」ときは指令を **厳密ゼロ** にして、
    #   初期姿勢のままじっと立たせる (:func:`~.mdp.observations.is_idle_hold`)。
    #   セーブ後の post_save_hold と同じ状態を、待機全般へ広げたもの。
    #
    #   これが無いと「歩けないほど小さいが、ゼロでもない指令」が出続ける帯域が残る:
    #     * 位相の停止しきい値 (0.12) 未満なので足を踏み替えられない
    #     * しかし指令はゼロでないので方策は寄せようとする → 上体だけが揺れる
    #     * ずれは踏み替えないと直らないので、この状態が永久に続く (リミットサイクル)
    #   MuJoCo で「ボールを置いているのに小刻みに震える」のはこれ。実測でも待機中に
    #   静止ブースト (_stand_still_boost) が開いていた割合は 1.1% しかなく、
    #   action ペナルティの weight を上げても待機中には圧が届いていなかった。
    #
    #   指令が厳密ゼロになると、(a) 位相がゼロ埋めされ (b) 静止ブーストが必ず開き
    #   (c) 学習中に毎エピソード通る post_save_hold と同一状態になる、の 3 つが揃う。
    #
    # ★ 入る/出るでしきい値を変える (ヒステリシス)。同値だと境界でトグルして、
    #   歩行の開始・停止を繰り返す別の振動になる。
    # ★ enter は「中央復帰の効果が効き始める 0.6m」より十分内側に置くこと。
    #   広げすぎると定位置からずれたまま待機して、次の球への到達性能が落ちる
    #   (中央から 0.6m ずれるだけで到達不能球が 33% → 42% に増える実測がある)。
    # ★ 実機の C++ 側にも同じ判定を同じしきい値で実装すること。
    # --- 脅威判定の頑健化 (2026-08-18) ---
    #
    # ★ 実機で「静止したボールが見えていると横移動する」不具合が出た。原因は
    #   知覚したボール速度のノイズが「接近中」に化けること:
    #
    #     approaching = vx < -0.05          ← ノイズで頻繁に成立してしまう
    #     t      = (ball_x - guard_x) / (-vx)   ← vx が 0 に近いと発散
    #     y_pred = (ball_y + vy * t).clamp(±max_y)
    #
    #   静止ボールが 3m 先で、ノイズで vx = -0.06 になった瞬間に t = 40 秒となり、
    #   わずかな vy でも y_pred が発散して **±max_y (ゴールポスト際) にクランプ**される。
    #   目標が飛ぶので全力の横移動指令が出て、同時に待機保持も解除される。
    #   ボールを隠すと止まるのは、未検出なら ball_active=false で脅威判定に入らないため。
    #
    # 対策は 2 つのゲート:
    #   * approach_speed_min: これ未満の接近速度は脅威とみなさない
    #   * arrival_t_max: 到達予測時間がこれを超えるなら脅威とみなさない (発散を直接塞ぐ)
    #
    # ★ 実際のシュートには影響しない。スポーン距離は速度に比例して決まる
    #   (d ∈ [v × spawn_time_near(0.55), max(spawn_dist_floor(2.0), v × spawn_time_far(1.4))])
    #   ので、**発射時の到達予測時間は常に 0.55〜1.4 秒**。3.0 秒には十分な余裕がある。
    #   approach_speed_min = 0.3 も ball_speed_min = 0.5 より下なので全ての球が通る。
    #   したがって塞いでいるのは「ノイズ起因の偽の脅威」だけ。
    # ★ 実機の C++ 側にも同じ判定を同じしきい値で実装すること。
    approach_speed_min: float = 0.3    # 脅威とみなす最低接近速度 [m/s]
    arrival_t_max: float = 3.0         # 脅威とみなす到達予測時間の上限 [s]
    #
    # ★ 2026-08-18 追加: 脅威判定の **持続要求 (デバウンス)**。
    #   しきい値だけだと、ノイズが一瞬でも approach_speed_min を超えた時点で
    #   脅威が立ってしまう。実機のボール速度推定ノイズはまだ未計測なので、
    #   しきい値の値だけに頼るのは危険。
    #
    #   ノイズ起因の脅威は **単発** (次のフレームには消える) だが、本物のシュートは
    #   接近している限り **連続して** 成立する。この差を使う。
    #   遅らせるのは脅威の立ち上がり 3 フレーム = 60ms @50Hz だけで、
    #   本物の球は発射時点で到達まで 0.55〜1.4 秒あるので実害は無い。
    #
    #   解除側を長め (5 フレーム) にしてあるのは、接近中に検出が 1 フレーム抜けても
    #   脅威が落ちないようにするため (落ちると目標が中央に戻って指令が反転する)。
    # ★ 実機の C++ 側にも同じ段数で実装すること。
    threat_on_frames: int = 3          # 連続でこの回数成立したら脅威に入る
    threat_off_frames: int = 5         # 連続でこの回数不成立なら脅威を抜ける
    #
    # ★ 2026-08-18 追加: 「実際に動いたか」の判定。
    #
    #   arrival_t_max は **遠い球にしか効かない**。ボールが守備面 (guard_x) の近くに
    #   あると t = (ball_x − guard_x) / closing の分子が小さく、ノイズで closing が
    #   approach_speed_min を超えただけで t < arrival_t_max も同時に成立してしまう。
    #   例: ボール 0.7m・closing 0.31 → t = 0.32s で両ゲート通過。しかも近い球では
    #   y_pred ≈ ball_y なので、目標がボールの y になりロボットが正対しに動く。
    #
    #   根本の弱点は「速度が位置の微分でノイズが乗る」こと。**位置は直接の観測値**
    #   なので桁違いに頑健。そこで「基準位置からの変位」を追加条件にする:
    #
    #     ref ← ref + (dt/tau) * (pos − ref)     # 遅い追従の基準点
    #     moved = |pos − ref| > threat_min_travel_m
    #
    #   静止ボールは ref が現在位置へ収束するので **推定速度がどう暴れても
    #   絶対に成立しない**。動く球は ref が遅れるので即座に成立する。
    # ★ 実機の C++ 側にも同じ判定を同じ値で実装すること。
    threat_min_travel_m: float = 0.15  # 基準位置からこれ以上動いていること [m]
    threat_travel_tau_s: float = 0.5   # 基準位置の追従時定数 [s]

    # ★ 2026-08-18: ボール履歴版で、方策に見せる速度指令を隠す確率。
    #   1.0 = 完全に隠す (手書きの指令を方策から見せない)。
    #   0 から 1 へ段階的に上げて移行する (詳細は ballhist/observations.py の
    #   ballhist_velocity_commands)。直接版では読まれない。
    cmd_dropout_p: float = 1.0

    # --- 状況の多様化 (2026-08-18) ---
    #
    # ★ [静止, 横移動, 枠外, ボールなし] の比率。残りがシュート。
    #   既定 [0.15, 0.10, 0.10, 0.05] = シュート 60%。
    #
    #   従来は 100% シュートで、「ボールは見えているが脅威ではない」状況を
    #   方策が一度も経験していなかった。実機の不具合 (静止ボールを置いたら
    #   横移動する) はまさにその分布外の状態で起きていた。
    #
    #   ★ シュートの比率を下げるぶん、同じ学習量に達するのに iteration が
    #     増える (60% なら約 1.7 倍)。セーブ率の立ち上がりは遅くなるが、
    #     到達できる上限は変わらない。
    #   ★ 全部 0 にすれば従来どおり 100% シュートに戻る。
    situation_probs: tuple = (0.15, 0.10, 0.10, 0.05)

    idle_hold_enter_m: float = 0.25    # このずれ未満で待機保持に入る [m]
    idle_hold_exit_m: float = 0.30     # このずれ以上で待機保持を抜ける [m]
    idle_hold_enter_yaw: float = 0.15  # 向きのずれがこの未満で入る [rad] (≒8.6°)
    idle_hold_exit_yaw: float = 0.22   # 向きのずれがこの以上で抜ける [rad] (≒12.6°)

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
    #
    # ★ 2026-08-14: サンプリング順を「距離 → 速度クランプ」から
    #   **「速度 → 距離」** に反転した (reset_ball_shot 参照)。距離は
    #       d ∈ [ v × spawn_time_near ,  max(spawn_dist_floor, v × spawn_time_far) ]
    #   から引かれる。spawn_dist_range と min_time_to_line は**もう使われない**
    #   (旧ラン/直接制御版との互換のためフィールドだけ残してある)。
    #
    #   旧方式の問題:
    #     * 「近い = 必ず遅い」を強制するので、速い球が近距離から来ない (実機では起きる)
    #     * 遅い球が最遠から来る。位置ノイズ σ(d)=0.124d+0.149 は 6.5m で 0.96m
    #       (ゴール幅の 3/4) に達し、到達点予測が無意味になる。知覚クリーン実験で
    #       これが学習速度を約 10 倍遅くしていることを確認した。
    spawn_dist_range: tuple = (1.5, 5.0)   # ← 未使用 (互換のため残置)
    # 距離の上限を決める時間 [s]。d_max = v × これ。v=3 → 4.2m / v=6 → 8.4m。
    spawn_time_far: float = 1.4
    # 距離の下限を決める時間 [s]。「反応が成立する最短距離」。v=6 → 3.3m で、
    # 守備面まで 0.40s・知覚レイテンシ 0.156s を引いて反応 0.24s、その場から 6cm 移動 +
    # タッチ判定 0.5m = 0.56m 幅をカバーできる。0.4 まで詰めると反応時間がほぼ消え、
    # 立っていた場所に偶然来たときだけ止まる運任せの球になり学習信号がノイズ化する。
    spawn_time_near: float = 0.55
    # 遅い球が至近距離に湧くのを防ぐ距離の下限 [m] (v=1 だと上限 1.4m で近すぎるため)。
    spawn_dist_floor: float = 2.0
    # スポーン距離そのものの床 [m]。v × spawn_time_near だけだと 0.5 m/s の球が 0.28m =
    # ロボットの足元に湧く。後段に「0.6m 未満なら押し出す」補正はあるが、あれは狙い方向を
    # 変えてしまうので最初から湧かせない。
    spawn_dist_near_floor: float = 1.0
    # スポーン点を守備面のどれだけ前に出すかの最小値 [m]。距離だけで下限を決めると
    # 広角の球が **キーパーの背後** に湧く (ang=±1.1rad では sx = 0.45d なので、
    # d=1.5m でも sx=0.68m < guard_x)。実測: この制約が無いと最易段でも 40% が
    # 到達不能判定になり、入れると 10.4% に落ちる。
    spawn_ahead_min: float = 0.7

    # --- 到達可能性の判定に使う下位ポリシーのエンベロープ (_mark_unreachable) ---
    # ★ 下位を差し替えたら実測して更新すること。現在の値は 07-28
    #   (k1_gk_direct_stage1/2026-07-28_17-13-15) を eval_gk_direct_lateral.py で
    #   計測した結果。
    reach_v_max: float = 1.278     # 定常横速度 [m/s] (指令 1.3 に対する実測)
    reach_t_acc: float = 0.6       # 静止 → 定常の立ち上がり [s]
    reach_latency_s: float = 0.156  # 知覚レイテンシ 116ms + 更新間隔 40ms
    spawn_half_angle: float = 1.1          # スポーン方位 ±[rad] (+x 正面基準, ≈63°)
    aim_y_range: float = 1.1               # 狙い先 y ∈ ±この値 [m] (ポスト内側)
    # 適応カリキュラム (mdp.adaptive_difficulty) が段階的に広げる狙い先 y の範囲 [m]。
    aim_y_stages: tuple = (0.4, 0.6, 0.8, 1.1)
    ball_speed_min: float = 0.5            # 初速下限 [m/s]
    ball_speed_max: float = 1.0            # 初速上限 [m/s] (ステージ2 固定 / ステージ3 初期値)
    # ★ 2026-08-16: 3.0 → 6.0 (ユーザー指示)。3.0 は「適応カリキュラムが到達できる上限」
    #   であって難易度の頭打ちではない。実測 (2026-08-15_11-31-55) では iter 25186 で
    #   3.000 に達したあと 23000 iter そこに張り付いていた ＝ 設定した天井に当たっていた。
    #   6.0 まで開ければ続きを登れる。
    ball_speed_cap: float = 6.0            # 適応カリキュラムの上限 [m/s]。
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
    # ★ 2026-08-14: 距離サンプリング反転に伴い、到達不能球は「距離の下限をさらに詰める」
    #   ことで作る。v=6 なら 6×0.35 = 2.1m から飛んでくる (守備面まで 0.2s、知覚を引くと
    #   反応時間ほぼゼロ = 届かない)。0 にしないのは、届かないことを認識する余地は
    #   残すため (知覚が何も届かないと学習信号がただのノイズになる)。
    hard_ball_time_near: float = 0.35
    hard_ball_min_time: float = 0.5    # ← 未使用 (互換のため残置)

    # --- 到達不能球の自動有効化 (mdp.adaptive_hard_ball、2026-08-13 追加) ---
    # ★ 到達不能球は「初速上限 hi が十分上がってから」でないと意味がない。hard ball の
    #   初速は ball_speed_cap ではなく **その時点の hi の hard_ball_speed_mult 倍** から
    #   引かれるので、学習初期 (hi=1.0) に有効化しても 1.0〜1.6 m/s = 全然「不能」に
    #   ならず、立ち上がりを遅くするだけになる。かといって手動で有効化する運用は、
    #   忘れると「取れない球を一度も経験していないポリシー」がそのまま実機へ行く
    #   (実測: そのポリシーに到達不能球を与えると転倒が 1 回 → 56 回)。
    #   そこで難易度カリキュラムの進行を監視して自動で入れる。
    hard_ball_auto: bool = False           # 自動有効化を使うか (既定 OFF = 従来の挙動)
    hard_ball_prob_max: float = 0.1        # 最終的に到達させる混入率
    hard_ball_step: float = 0.02           # 1 段の増分
    hard_ball_ramp_episodes: int = 3000    # 段を 1 つ上げる間隔 [エピソード]
    # 有効化のトリガ (いずれかを満たしたら開始):
    #   1. ball_speed_hi が cap に到達した (これ以上難しくならない)
    #   2. ball_speed_hi が hard_ball_plateau_episodes の間まったく動かない (頭打ち)
    #   3. 総エピソード数が hard_ball_force_episodes を超えた (保険。1・2 が成立しない
    #      まま学習が終わるのを防ぐ = 「静かにスキップされる」ことが無いようにする)
    # ★ 2026-08-14: 実測ペースに合わせて桁を修正。4096 env / 25s エピソードでは
    #   **1 iter あたり約 63 エピソード**進む (12500 iter で 786,000 エピソード)。
    #   旧値 50000 / 400000 は iter 800 / iter 6000 相当で、早すぎた。
    #   新値は iter 換算でおよそ 4800 / 24000 iter 相当。
    hard_ball_plateau_episodes: int = 300000
    hard_ball_force_episodes: int = 1500000
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

    # --- 自己位置の跳び → ボール速度への漏れ込み (2026-08-14 追加) ---
    # 実機のボール速度は観測値ではなく、フィールド座標系のボール位置
    # (= 自己位置 + 相対位置) を CVKF で微分した推定値。自己位置が跳ぶとボールが
    # 動いたのと区別が付かず、偽の速度になる。CVKF に自機移動の補償項は無い。
    #
    # 下の 3 つは vision_filter の CVKF の**確定パラメータから数値的に導出**した値で、
    # 実測値ではない (measurement_noise_std=0.25m / process_acceleration_std=0.8m/s^2 /
    # NIS 閾値 9.21 から定常カルマンゲインを解いた結果)。したがって実機を計測しなくても
    # この形は再現できる。未知なのは「跳びの頻度と大きさ」の方で、そちらは
    # loc_jump_hz_range / loc_jump_m で DR 済み。
    #
    # ★ インパルスではなく時定数 0.79s の「山」であることが本質。1 フレームのスパイクを
    #   入れると「1 フレームだけ無視する」という実機に転移しない対処を学習してしまう。
    #   2 秒続く偽の「接近中」信号は、キーパーをポストまで走らせるのに十分な長さ。
    loc_vel_leak_coef: float = 0.82        # ピーク偽速度 [m/s] = coef × 跳び幅 [m]
    loc_vel_leak_tau_s: float = 0.79       # 山の減衰時定数 [s] (約 2 秒で収まる)
    loc_vel_leak_nis_gate_m: float = 0.80  # これを超える跳びは NIS ゲートで棄却され漏れない

    # 知覚DR (VirtualPerception + 速度バイアス) を全部切ってクリーン観測にするフラグ。
    perception_clean: bool = False

    # セーブ判定
    touch_force_threshold: float = 0.1  # 足-ボール接触力のしきい値 [N]
    touch_proximity: float = 0.5        # タッチ判定フォールバックの近傍距離 [m]
    save_delay_steps: int = 100         # セーブ確定までの保持時間 (2s @ 50Hz)

    # --- ボール位置の時系列フィット (2026-08-16) ---
    # 到達点予測 (compute_target_y) に渡す位置・速度を、生の 1 フレーム観測ではなく
    # 直近ウィンドウの最小二乗フィットから作る。窓の長さ [s]。**0 で無効 (既定)**。
    #
    # なぜ効くか: VirtualPerception の位置ノイズ sigma = 0.124d + 0.149 は
    # ``randn_like`` でビジョン更新ごとに独立サンプルされる **白色雑音** なので、
    # 平均化で sqrt(N) 分の 1 に減る。飛翔時間 0.7s × 20〜25Hz = 14〜17 サンプルあるのに、
    # 従来は 1 フレームしか使っていなかった:
    #     速度上限 3.0 → sigma(p90) 0.51m   ← タッチ判定半径 0.50m と同等
    #     速度上限 6.0 → sigma(p90) 0.86m
    # 適応カリキュラムが 2.07 m/s で停滞したのは、予測誤差がタッチ半径に達したため。
    #
    # 方式は「速度推定で各サンプルを現在時刻へ引き戻してから平均」(_gk_fitted_goal_state)。
    # 実測 (白色ノイズ σ、ビジョン 25Hz):
    #     窓 0.5s  生 0.505m → 引き戻して平均 0.141m  (3.6 倍改善)
    #     窓 0.8s  生 0.514m → 引き戻して平均 0.117m  (4.4 倍改善)
    # 2 パラメータの直線フィットだと窓の端で評価するぶん分散が 4 倍になり 0.27m 止まり。
    #
    # ★ **速度はフィットしない。** 直線フィットから出る速度は窓 0.5s / σ=0.51 で誤差
    #   0.95 m/s あり、既存推定 (真値 + perc_vel_bias 0.05〜0.15) より 1 桁悪い。
    # ★ 有効にしたら **実機の C++ 側にも同じ平均化を実装すること**。
    #   シムだけ賢くすると、そのぶんがまるごと sim-to-real ギャップになる。
    # ★ 窓を長くするほど平均化は効くが、転がり減速で等速の仮定から外れる。0.5s 前後が目安。
    ball_fit_window_s: float = 0.0
    # ★ 2026-08-15: セーブ後の「その場保持」を次の球の発射まで続けるか。
    #
    #   True  = 従来 (2026-08-11 のユーザー指示)。``touched`` が立ってから次の球が
    #           発射されるまで指令を全成分ゼロにして、止めた地点に立たせ続ける。
    #           転倒しないかを目視・数値で確認するのが目的だった。
    #   False = セーブ確定 (save_cd の countdown) までで保持を終え、次の球までの
    #           待ち時間を**中央への復帰**に充てる。
    #
    #   False を推奨する理由 (2026-08-15 に到達可能性モデルで定量化):
    #     継続モードでは 2 球目以降が必ず「前の球を止めた場所」から始まる。球が来た
    #     瞬間のキーパー y と「物理的に取れない球」の割合は
    #         y=0.0 → 33.0% / y=0.4 → 37.3% / y=0.6 → 42.1% / y=0.8 → 47.9%
    #     で、中央から 0.6m ずれるだけで +9pt。これは下位を横 2.0 m/s に作り直した
    #     場合の改善 (-2.8pt) の 3 倍以上で、しかも上位ポリシーが完全に制御できる量。
    #     保持を切れば ``compute_target_y`` が自動で目標 0 (中央) を返すので、
    #     報酬の変更なしに復帰指令が出る。
    #
    #   既定を True のままにしてあるのは既存タスク (直接制御版・階層版 v2) の挙動を
    #   変えないため。新しく回すタスクでは False を検討すること。
    post_save_hold_until_relaunch: bool = True

    # ステージ3: 適応カリキュラム (セーブ成功率 EMA → 初速上限)
    adaptive_success_threshold: float = 0.85
    adaptive_fail_threshold: float = 0.55
    # ★ 2026-08-15: 難易度を変えた直後に EMA を戻す「中立値」。
    #   None なら従来どおり success と fail の**中点** (= 既存タスクの挙動そのまま)。
    #
    #   分離した理由: 降格を減らしたくて fail_threshold を下げると、中点も一緒に下がり、
    #   昇格に必要な EMA の伸び (中立値 → success_threshold) が長くなって**昇格が遅く
    #   なる**。「降格は非常口だけにしたい、でも昇格の速さは落としたくない」を両立できない。
    #   ここを明示すれば 2 つを独立に決められる。
    #
    #   実測 (DH 版 Stage2, 2026-08-15): success 0.80 / fail 0.55 の帯に真の成功率
    #   (約 0.70) が乗り、1200 iter で約 10 往復した。昇格・降格のたびに EMA リセットと
    #   クールダウンを払うので、その間は前へ進まない。fail を 0.35 に下げつつ中立値を
    #   0.675 (元の中点) に据え置けば、往復だけが消える。
    adaptive_neutral_ema: float | None = None
    adaptive_speed_delta: float = 0.05      # 1 回の調整量 [m/s] (加算方式のとき)
    # ★ 2026-08-15: 初速の刻みを **乗算** にする倍率。1.0 以下なら従来の加算方式。
    #   加算 0.05 は上限が 3.0 だった頃の設定で、cap 6.0 では 1.0 → 6.0 に 100 回の
    #   昇格が必要になる。実測 1 昇格 ≈ 3400 iter なので 340,000 iter (約 8 日) かかり
    #   実用にならない。×1.2 なら 10 回で到達する。
    #   速い球ほど 0.05 m/s の差は相対的に小さいので、比率で刻む方が難易度として素直。
    #   昇格直後の落ち込みが大きすぎるようなら 1.15 (13 回) に下げる。
    adaptive_speed_ratio: float = 1.2
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
