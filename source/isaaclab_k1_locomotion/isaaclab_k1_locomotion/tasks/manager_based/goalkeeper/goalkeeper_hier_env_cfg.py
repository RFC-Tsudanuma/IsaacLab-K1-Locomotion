# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (階層版 v2) の環境定義。

**凍結する下位ポリシーが違う**点だけが旧階層版 (``goalkeeper_env_cfg.py``) との本質的な差。

    旧: 0524_walk.pt (汎用歩行)          横移動 0.66 m/s が天井 → セーブ率が頭打ち
    新: k1_gk_direct_stage1/2026-07-28   横移動 1.28 m/s (指令 1.3 に対し追従誤差 2%)

07-28 は「gk_direct の Stage1」= 横重視の速度コマンド追従だけを学習した歩行ポリシーで、
**実機にデプロイして良好に動いた実績がある**。つまり「50Hz で (vx, vy, wz) を投げると
横に走る」インターフェースがハード上で検証済みで、上位 MLP を 1 個足すだけで実機に載る。

位置づけ:
    直接制御版 (``goalkeeper_direct_env_cfg.py``) の Stage2 は ``velocity_commands``
    スロットに手書きの P 制御 (:func:`mdp.observations.task_drive_vector`) を流し込んで
    いる。本タスクはそこを**学習した上位ポリシーで置き換える**。手書き法則が構造的に
    持っていた弱点が、そのまま上位の伸びしろになる:

      1. 後手に回る: 手書きは「ずれが 0.195m を超えてから全力」。静止から横 1.3 m/s に
         乗るまで約 0.6s (その間に進むのは約 0.39m) かかるので、必要横移動量 0.3〜0.8m の
         大半の球で加速し切る前に到着 = 常に出遅れる。上位はボールの位置・速度を直接
         見ているので、予測点が確定する前に動き出せる。
      2. guard_x 固定: 手書きは常に守備面へ戻る。上位は vx を状況で使える。
      3. heading 固定: 手書きは常に正面。ただし後述の通り「体を傾けて前進を流用」は
         この下位でも損なので、wz は**姿勢維持専用**と割り切ってよい。

07-28 の実測エンベロープ (eval_gk_direct_lateral.py、指令 0.5〜1.5 を掃引):
    * 横追従: 0.5→0.460 / 0.9→0.878 / 1.2→1.182 / 1.3→1.278 (誤差 2〜8%、飽和なし)
    * yaw ドリフト: 指令 vy によらず約 10°/s。放っておくと円を描いて横移動する。
      上位は wz ≈ -0.175 rad/s の定常オフセットで打ち消す (権限 ±1.0 の 17.5%)。
      1 回のセーブ動作 (0.25〜0.6s) 中のズレは 3〜6° = 速度損失 0.5% なので実害は小さく、
      効いてくるのは往復を繰り返したときの累積。
    * 後退ドリフト: 指令 vy ≥ 0.9 で約 -0.10 m/s。上位は vx ≈ +0.10 で打ち消す。
    * 体を θ 傾けたときのゴールライン方向速度は θ=0 が最良 (fwd が負なので傾けるほど悪化)。
      → **wz を移動速度の足しに使う余地は無い。姿勢維持専用。**

観測 (次元・順序は gk_direct と完全に同一 = policy 59 / critic 64):
    ``mdp/symmetry.py`` のスライス定義をそのまま使えるようにするため、gk_direct の
    観測クラスを継承して**中身だけ**差し替える (スロットの順序・次元は変えない)。
        * policy   = gk_direct Stage2 の policy から velocity_commands → 前回上位 action、
                     gait_phase → 上位 action 駆動位相 に差し替え
        * critic   = 同上 (真値版)
        * low_level = gk_direct Stage1 の policy そのまま (ボール系はゼロ 10)。
                     gait_phase だけ上位 action 駆動に差し替える。
                     **59 次元でなければ凍結ポリシーが読めない。**

ステージ:
    * Stage 1 (``Isaac-GoalkeeperHier-Stage1-K1-v0``)
        ボールは遠方にパーク (観測ダミー 0)。ランダム目標 y への到達と停止。
        ランダムなのは **y だけ** (x は守備面 guard_x 固定、向きは正面固定)。
        目標範囲は **±1.35m (幅 2.7m)** = ゴール幅 + 0.1m のマージン。
        併せて「横に動きながら姿勢 (face_field) と前後位置 (stay_on_goal_line) を保つ」が
        主課題になる — 下位に上記の定常ドリフトがあるため、これを能動的に潰すことを
        ここで学ばせる必要がある。
        ★ このステージは**実機検証のマイルストーン**でもある。ボールもシュータもビジョンも
          要らずに「指定 y へ行って止まる」を実機で確認でき、そこで実機側の実効横速度・
          立ち上がり時間・停止距離を測って Stage2 の ball_speed_cap を決める。
    * Stage 2 (``Isaac-GoalkeeperHier-Stage2-K1-v0``)
        ゴール + ボール。gk_direct Stage2 の資産 (エピソード継続 + relaunch、適応カリキュラム
        ``adaptive_difficulty``、save_clearance、実機準拠の知覚モデル、到達不能球) を継承。
        Stage1 の ckpt から ``--resume``。

学習・再生 (階層エンジンは goalkeeper 専用の train/play_goalkeeper.py):
    train_gk_hier_stage1.sh / train_gk_hier_stage2.sh / play_gk_hier.sh 参照。
    ``--high_action_clip 1.0 1.3 1.0`` = 07-28 の学習コマンドレンジそのまま。
    1.5 でも実測上は綺麗に追従する (1.474 m/s) が、学習分布の外なので採らない。
    上げたくなったら Stage2 ckpt から --resume して clip だけ変えればよい
    (clip は wrapper 側の引数でネットワーク構造には焼き込まれていない)。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ..locomotion.rough_env_cfg import K1ObservationsCfg
from ..locomotion.mdp.events import reset_prev_high_action
from ..locomotion.mdp.observations import last_high_action
from ..locomotion.mdp.rewards import (
    com_jerk_l2,
    high_action_rate_l2,
    high_action_smoothness_l2,
)

# frozen 連携の位相観測は around_ball の実装を再利用 (旧規約 fixed_freq 対応済み)
from ..around_ball.mdp.observations import high_action_phase_obs

from .goalkeeper_env_cfg import GOAL_HALF_WIDTH
from .goalkeeper_direct_env_cfg import (
    LATERAL_TARGET_SPEED,
    K1GKDirectCriticCfg,
    K1GKDirectEnvCfg,
    K1GKDirectPolicyCfg,
    K1GKDirectStage1PolicyCfg,
)
from .mdp.curriculums import adaptive_difficulty, adaptive_hard_ball
from .mdp.events import reset_stage1_target_and_park, stage1_target_tick
from .mdp.rewards import (
    face_field,
    hold_at_target,
    save_clearance_bonus,
    save_touch_bonus,
    stay_on_goal_line,
    target_reach_velocity,
    track_target_y,
)

# 上位 action 駆動の歩行位相のパラメータ。
#   fixed_freq=1.6: 07-28 (= locomotion と同じ規約) は φ = 2π·f·t の固定 1.6Hz で学習
#     されている。速度依存アキュムレータ規約の下位に差し替えるときは None にすること。
#     なお学習時は randomize_phase_freq による env ごとの ±0.05Hz ランダム化が乗っていたが、
#     ここは固定 1.6 で与える。そのランダム化の内側なので分布外にはならない。
#   cmd_threshold=0.05: 下位の停止規約 (rough_env_cfg の _COMMAND_THRESHOLD) と同値。
_PHASE_PARAMS = {"cmd_threshold": 0.05, "fixed_freq": 1.6}

# Stage 1 の目標 y の範囲 [±m]。**ゴール幅 (GOAL_HALF_WIDTH = 1.3、幅 2.6m) より少し広い。**
# 幅 2.7m = ゴール幅 + 0.1m のマージン (ユーザー指示 2026-08-13)。
# Large フィールド (ゴール幅 3.1m) まで広げる案もあったが、横移動距離が伸びるぶん
# 1 目標あたりの所要時間が増えて学習が遅くなるため、今回は時間を優先してマージンのみ。
# Large 対応が必要になったらここを 1.55 に変えるだけでよい (Stage 2 は実ゴール幅のまま)。
#
# ★ この値がゴール幅を超えていても壊れない理由: 目標 y の ±max_y クランプは
#   :func:`mdp.observations.compute_target_y` の中で **ボール到達点の外挿 (y_pred) に
#   しか掛かっていない**。Stage 1 はボール非アクティブなのでバッファの目標値が
#   そのまま素通りする (観測 gk_target_y も報酬の誤差計算も同じ経路)。
#   したがって報酬・観測側の max_y は Stage 1 では効かないので触らなくてよい。
# ★ ゴールポストは x=0 にあり、ロボットは守備面 (guard_x=0.9) にいるので、
#   |y| = 1.35 まで出てもポストと接触しない。out_of_bounds の |y| 上限も 2.2 で余裕がある。
STAGE1_TARGET_HALF_WIDTH = 1.35
# 「反対側のポスト際ゾーン」(長距離スプリントを学習分布に必ず入れるための枝) も
# 同じ比率で広げる (元は 0.95〜1.3 = ゴール幅基準の 0.73〜1.0)。
STAGE1_FAR_ZONE = (1.0, STAGE1_TARGET_HALF_WIDTH)


# ---------------------------------------------------------------------------
# 観測 (スロットの順序・次元は gk_direct と同一。中身だけ差し替える)
# ---------------------------------------------------------------------------

@configclass
class K1GKHierPolicyCfg(K1GKDirectPolicyCfg):
    """上位 policy 観測 (59 次元)。gk_direct Stage2 の policy から 2 項だけ差し替え。

    ``velocity_commands`` は gk_direct では手書きの ``task_drive_vector`` が入っていた
    スロット。階層版ではそこが「前回自分が出した歩行コマンド」になる (around_ball と同じ)。
    """

    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    gait_phase = ObsTerm(func=high_action_phase_obs, params=dict(_PHASE_PARAMS))


@configclass
class K1GKHierCriticCfg(K1GKDirectCriticCfg):
    """上位 critic 観測 (64 次元、特権情報つき・ノイズなし)。

    actor と位相の定義がズレると価値推定が actor の見ている状態と食い違うので、
    gait_phase は policy と必ず同じ関数・同じ params にすること。
    """

    velocity_commands = ObsTerm(func=last_high_action, params={"action_dim": 3})
    gait_phase = ObsTerm(func=high_action_phase_obs, params=dict(_PHASE_PARAMS))


@configclass
class K1GKHierLowLevelCfg(K1GKDirectStage1PolicyCfg):
    """凍結する下位 (07-28) 用の観測 (59 次元)。

    gk_direct Stage1 の policy 観測そのもの = 歩行 49 + ボール系ゼロ 10。
    ``velocity_commands`` スロットは wrapper が上位 action で上書きするので定義は据え置く。
    ``gait_phase`` だけ上位 action 駆動に差し替える: 下位は「velocity_commands の速度」と
    「gait_phase のテンポ」の対応を学習しているので、コマンドを上位 action に差し替える
    なら位相も同じ上位 action から作らないと、学習時に見たことのない矛盾した入力になる。
    """

    gait_phase = ObsTerm(func=high_action_phase_obs, params=dict(_PHASE_PARAMS))


@configclass
class K1GKHierObservationsCfg(K1ObservationsCfg):
    policy: K1GKHierPolicyCfg = K1GKHierPolicyCfg()
    critic: K1GKHierCriticCfg = K1GKHierCriticCfg()
    low_level: K1GKHierLowLevelCfg = K1GKHierLowLevelCfg()


# ---------------------------------------------------------------------------
# 報酬 (歩行用 K1Rewards は継承せず丸ごと置き換える)
# ---------------------------------------------------------------------------

@configclass
class K1GKHierRewardsCfg:
    """階層版ゴールキーパーの報酬。

    下位が凍結されているので、歩容まわりの報酬 (foot_clearance / stance_foot_flat /
    feet_phase など) は**勾配を持たない**ため一切入れない。上位が制御できるのは
    「毎ステップどんな歩行コマンドを出すか」だけなので、報酬もその粒度で書く。

    ★ 直接制御版に対する構造的な利点がここに出る。直接版はセーブ報酬 (touch 100 /
      clearance 50) が歩容報酬を桁で上回るため、学習が進むほど「歩容を削ってでも y に
      速く着く」方向に押され、実測で foot_clearance が劣化し続けていた
      (2026-08-12 の run で 0.311、直近 3000iter で -0.039)。下位凍結ならこの綱引きが
      そもそも発生しない。
    """

    # --- タスク主報酬 ---
    # 目標 y への距離。σ の違う 2 項でマルチスケール (遠距離の誘導 + 最後の押し込み)。
    track_target_coarse = RewTerm(
        func=track_target_y, weight=1.5, params={"std": 0.6, "max_y": GOAL_HALF_WIDTH},
    )
    track_target_fine = RewTerm(
        func=track_target_y, weight=1.5, params={"std": 0.15, "max_y": GOAL_HALF_WIDTH},
    )
    # 目標方向への横移動速度 (遠くても勾配が一定に出る密報酬)。
    # ★ v_cap は下位の実力に合わせること。旧階層版は 0524_walk.pt に合わせた 0.8 だったが、
    #   07-28 は指令 1.3 に対し 1.278 m/s 出るので LATERAL_TARGET_SPEED (=1.3) に上げる。
    #   ここが低いと「上限に達したら勾配が消える」ので速度が伸びない。
    target_reach_velocity = RewTerm(
        func=target_reach_velocity,
        weight=5.0,
        params={
            "deadband": 0.12,
            "v_cap": LATERAL_TARGET_SPEED,
            "cmd_scale": 0.5,
            "max_y": GOAL_HALF_WIDTH,
        },
    )
    # 目標到達後の「コマンドを 0 に落として静止」報酬。足踏み局所最適への本命対策。
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
    # セーブ (触れて弾いた) 瞬間の一回限りボーナスと、その「質」への上乗せ。
    save_touch_bonus = RewTerm(func=save_touch_bonus, weight=100.0)
    save_clearance = RewTerm(func=save_clearance_bonus, weight=50.0)

    # --- 下位の定常ドリフトを打ち消すための姿勢・定位置維持 ---
    # ★ ここは 07-28 の実測を踏まえて旧階層版から強化した項。
    #   下位は横移動中に「yaw 約 10°/s」「後退 約 0.10 m/s」の定常ドリフトを持つので、
    #   上位が小さな定常オフセット (wz, vx) を出し続けて能動的に潰す必要がある。
    #   単一の緩いガウス (σ=0.5) だと 10° ズレても報酬が 0.885 までしか落ちず勾配が弱いので、
    #   track_target と同じくマルチスケールにして細かい側で押し込む。
    #   (σ=0.15 なら 10°=0.175rad で 0.26 まで落ちる = しっかり効く)
    face_field_coarse = RewTerm(func=face_field, weight=1.0, params={"std": 0.5})
    face_field_fine = RewTerm(func=face_field, weight=1.5, params={"std": 0.15})
    # ★ 2026-08-13: weight 1.0 → 2.5 (coarse/fine とも)。1 回目の Stage1 学習
    #   (3000 iter) で、他の報酬項が横ばいの中 **stay_on_goal_line_fine だけが
    #   低下し続けた** (0.406 → 0.368)。eval でも守備面からのずれが平均 0.32m /
    #   p90 0.61m と大きく、guard_x=0.9 に対して 0.6m 下がるとほぼゴールライン上
    #   =「前に出てシュートコースを狭める」という守備面の意図が失われる。
    #   y の課題 (target_reach_velocity=5.0) に対して相対的に弱すぎたのが原因。
    #   ※ 自己位置推定のバイアス (loc_bias_xy_m=0.20) の分は観測できないので
    #     原理的に消せない。0.2m 程度の残差は許容範囲と考える。
    stay_on_goal_line_coarse = RewTerm(func=stay_on_goal_line, weight=2.5, params={"std": 0.3})
    stay_on_goal_line_fine = RewTerm(func=stay_on_goal_line, weight=2.5, params={"std": 0.1})

    # --- 失敗ペナルティ ---
    # 失点・転倒・場外 (time_out=False の終了)。
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

    # --- 姿勢・滑らかさペナルティ (around_ball 系と同じ値) ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.5 * 0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0 * 0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.6 * 0.5)
    # 上位コマンドのジッター抑制。下位は 0.5〜7.0s 間隔で再サンプルされるコマンドで
    # 学習しているので、50Hz で振り回されると追従しきれず足がバタつく。
    # ★ この 2 項が罰するのは「変化量」であって大きさではないので、ドリフト補正のための
    #   定常オフセット (wz ≈ -0.175, vx ≈ +0.10) は無罰で出し続けられる。
    action_smoothness_l2 = RewTerm(func=high_action_smoothness_l2, weight=-0.5)
    action_rate_l2 = RewTerm(func=high_action_rate_l2, weight=-1.2)
    # 体のカクつき (コマンド急変で下位が乱れた瞬間) を直接罰する。
    com_jerk_l2 = RewTerm(func=com_jerk_l2, weight=-5e-6)


# ---------------------------------------------------------------------------
# 環境
# ---------------------------------------------------------------------------

@configclass
class K1GKHierEnvCfg(K1GKDirectEnvCfg):
    """階層版の土台。gk_direct Stage2 (ゴール + ボール + 知覚 + セーブ判定) から、
    行動の出し方だけを「12 関節直接」→「歩行コマンド + 凍結下位」に差し替える。

    直接版から引き継ぐもの: シーン、ボール発射、知覚モデル、エピソード継続 (relaunch)、
    セーブ判定、終了条件、GoalkeeperParamsCfg。
    差し替えるもの: 観測グループ (3 種)、報酬 (丸ごと)、task_drive_vector の同期イベント。
    """

    observations: K1GKHierObservationsCfg = K1GKHierObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 報酬は階層版用に丸ごと置き換える (歩容系は下位凍結で勾配を持たないため)。
        self.rewards = K1GKHierRewardsCfg()

        # 手書きの移動要求を velocity_commands スロットへ毎ステップ流し込むイベントは、
        # そのスロットを上位ポリシーが担当するので不要 (これが本タスクの主眼)。
        self.events.sync_task_command = None

        # リセット時に上位 action バッファを 0 にする。
        self.events.reset_prev_high_action = EventTerm(func=reset_prev_high_action, mode="reset")

        # --- 物理 DR を startup → reset に変える (2026-08-13) ---
        # locomotion 由来の物理 DR は全部 mode="startup"、つまり **シミュレーション開始時に
        # env ごとに 1 回だけ** 引かれ、以後エピソードをまたいでも二度と変わらない。
        # 4096 env で 3000 iter 回しても、ロボットの物理パラメータは 4096 通りのままで、
        # 反復回数を増やしても経験の幅は一切広がらない (1 回目の Stage1 学習で
        # iter 1000 以降すべての指標が横ばいだったのはこれも一因)。
        # reset にすると毎エピソード引き直しになり、20s エピソード × 3000 iter で
        # 約 29.5 万通り = **70 倍以上**の物理条件を経験する。
        #
        # ★ 累積しないことは IsaacLab 実装で確認済み: randomize_rigid_body_mass /
        #   randomize_actuator_gains はどちらも「適用前に default 値へ戻してから」
        #   ランダム化する (envs/mdp/events.py のコメント "randomization is applied on
        #   the default values and not the previously randomized values")。
        #   material も上書き代入なので add/scale でも値が積み上がらない。
        #
        # ☠ base_com は **絶対に reset にしないこと** (2026-08-13、学習 1 本を潰した)。
        #   randomize_rigid_body_com だけは default に戻さず現在値に加算する:
        #
        #       coms = asset.root_physx_view.get_coms().clone()
        #       coms[env_ids[:, None], body_ids, :3] += rand_samples   # ← += である
        #
        #   本家が startup 専用を想定しているため。reset にすると CoM がランダムウォークし、
        #   ±0.05m 一様 (σ=0.0289m) なのでリセット 100 回で σ≈0.29m、1000 回で σ≈0.91m。
        #   胴体の重心が体外へ飛んでバランスが物理的に取れなくなる。
        #   実測 (10000 iter の Stage1): 平均エピソード長 998 → 33 step、
        #   base_height による終了 97.2%、mean_reward 226 → -29 で完全崩壊した。
        #   エピソードが短くなるほどリセット頻度が上がるので自己増強する。
        #   CoM も per-episode で振りたくなったら、既定値をキャッシュして「既定 + 乱数」を
        #   **代入**する自作関数を書くこと (本家の関数をそのまま使ってはいけない)。
        # ★ 速度: material / mass / com は CPU テンソル経由なので IsaacLab は startup 限定を
        #   推奨しているが、このタスクは 20s エピソード = 1000 step ごとの一斉リセットで、
        #   24 step/iter なら **約 42 iter に 1 回**しか呼ばれないので償却コストは小さい。
        #   actuator gains は K1 が explicit actuator (BoosterDelayedPDActuator) なので
        #   CPU 経路 (ImplicitActuator 限定) に入らず GPU で完結する。
        # ★ randomize_phase_freq は対象外。階層版の gait_phase は
        #   high_action_phase_obs(fixed_freq=1.6) で per-env オフセットを読まないため、
        #   reset にしてもこのタスクでは一切効かない (無駄な呼び出しになるだけ)。
        for _term in ("physics_material", "add_base_mass", "randomize_actuator_gains"):
            _cfg = getattr(self.events, _term, None)
            if _cfg is not None:
                _cfg.mode = "reset"

        # base_velocity コマンドは low_level 観測のスロット確保用に残るだけで、
        # 中身は wrapper が上位 action で上書きする。残留値が悪さをしないようゼロ固定する。
        # (親クラスで既に heading_command=False / resampling 無効化 / standing 0% 済み)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class K1GKHierStage1EnvCfg(K1GKHierEnvCfg):
    """Stage 1: ボールはパーク (観測ダミー 0)。ランダム目標 y への到達と停止。

    実機検証のマイルストーンを兼ねる。ここで実機側の実効横速度・立ち上がり時間・
    停止距離を測り、Stage2 の ball_speed_cap を実機基準で決める。
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0

        # 目標 y の範囲を 2.7m 幅 (±1.35) に広げる = ゴール幅 + 0.1m のマージン。
        # ゴール幅 2.6m より広いが、Stage 1 はボール非アクティブで目標クランプが
        # 効かない経路なので問題なく通る (STAGE1_TARGET_HALF_WIDTH のコメント参照)。
        self.goalkeeper.stage1_target_range = STAGE1_TARGET_HALF_WIDTH
        self.goalkeeper.stage1_far_zone = STAGE1_FAR_ZONE

        # ボールは発射せず遠方にパーク + ランダム目標を採番する。
        # ★ reset_gk_buffers より後に登録すること (目標を上書きするため)。親の
        #   __post_init__ で reset_gk_buffers → reset_ball の順に登録済みなので、
        #   同名フィールドを差し替えれば順序は保たれる。
        self.events.reset_ball = EventTerm(func=reset_stage1_target_and_park, mode="reset")
        # 毎制御ステップ: 目標到達 (コマンド 0 + 静止) を保持できたら目標を再サンプル。
        _dt = self.sim.dt * self.decimation
        self.events.stage1_target_tick = EventTerm(
            func=stage1_target_tick,
            mode="interval",
            interval_range_s=(_dt, _dt),
            is_global_time=True,
        )
        # セーブ後に次の球を撃つイベントはボールが無いので無効化する。
        self.events.relaunch_ball = None

        # ボールが無いのでセーブ関連の項と失点終了は落とす。
        self.terminations.goal_conceded = None
        self.rewards.save_touch_bonus = None
        self.rewards.save_clearance = None
        # 到達後の停止を Stage1 の主課題にする (足踏み対策の勾配を強く)。
        self.rewards.hold_at_target.weight = 8.0


@configclass
class K1GKHierStage2EnvCfg(K1GKHierEnvCfg):
    """Stage 2: ゴール + ボール。セーブ成功率 (EMA) に応じて難易度を上げる適応カリキュラム。

    難易度は「狙い先の広さ → ボール初速」の順に上がる (:func:`mdp.adaptive_difficulty`)。
    Stage 1 の ckpt から ``--resume`` すること。
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.difficulty = CurrTerm(func=adaptive_difficulty)
        # ★ difficulty より後に登録すること。adaptive_hard_ball は difficulty が書いた
        #   _gk_speed_hi / _gk_episode_count を読むだけで、CurriculumManager は cfg の
        #   定義順に実行するため、先に置くと 1 ステップ古い状態を見ることになる。
        self.curriculum.hard_ball = CurrTerm(func=adaptive_hard_ball)
        # 到達不能球の自動有効化を ON にする (既定は OFF なので直接制御版は影響を受けない)。
        # カリキュラムが cap に到達 / 頭打ち / 保険の上限エピソード数のいずれかで
        # 混入が始まり、以後 3000 エピソードごとに +0.02 して 0.1 で頭打ちになる。
        self.goalkeeper.hard_ball_auto = True

        # --- 初期配置: 守備面かつゴール中央にピンポイント (ユーザー指示 2026-08-13) ---
        # 親 (K1GKDirectEnvCfg) は x = guard_x ± 0.1 / y = ±0.5 のランダムだが、
        # 「ゴールラインから 0.9m 前、ポスト間の真ん中」に固定する。
        #
        # ★ これで初期分布が狭くなるのは **各エピソードの 1 球目だけ**。本タスクは
        #   エピソード継続モード (relaunch_ball_after_save) で、セーブ後はロボットを
        #   リセットせず止めた地点から次の球に備えるため、2 球目以降の開始位置は
        #   前の球をどこで止めたかで自然にばらける。実戦の「キックオフから始まって
        #   連続でシュートを受ける」流れとも一致する。
        _gx = float(self.goalkeeper.guard_x)
        self.events.reset_base.params["pose_range"]["x"] = (_gx, _gx)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        # ★ yaw と初速のランダム化は残す。実機で毎回きっちり正対して静止している
        #   保証は無いので、ここまで固定すると「完全に整った状態からしか始められない」
        #   方策になる。位置だけ指定どおり固定し、姿勢の微小なばらつきは残す。
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }

        # --- ボール初速の上限 5.0 m/s (ユーザー指示 2026-08-13) ---
        # ★ cap を上げるだけでは 5 m/s は実現しない。reset_ball_shot には実現可能性
        #   クランプ speed <= (スポーン点→狙い先の距離) / min_time_to_line があり、
        #   spawn_dist_range 上限 5.0m・min_time_to_line 1.2s では
        #       v_feasible <= 5.2 / 1.2 ≈ 4.3 m/s
        #   で頭打ちになり、cap 5.0 は一度も届かない。
        #
        #   そこで **スポーン距離の上限を広げる** ことで 5 m/s を可能にする:
        #       6.0m / 1.2s = 5.0 m/s  → 6.0m 以遠のスポーンで cap に到達できる
        #   上限を 6.5m にして、5 m/s が出る球がそれなりの割合で混ざるようにした。
        #
        #   min_time_to_line を下げる (1.2 → 1.0) という手もあるが、そちらは
        #   **近距離の球も含めて全部の反応時間を削る**ので難易度の上がり方が乱暴になる。
        #   「速い球は遠くから来る」という物理的に自然な相関を保つ本案を採る。
        #   知覚側も max_detection_range=7.0m なので 6.5m は視野内に収まる。
        self.goalkeeper.ball_speed_cap = 5.0
        self.goalkeeper.spawn_dist_range = (1.5, 6.5)


# ---------------------------------------------------------------------------
# PLAY
# ---------------------------------------------------------------------------

def _make_play_clean(cfg: K1GKHierEnvCfg) -> None:
    """PLAY 用: 外乱と知覚DR を切って挙動を見やすくする。"""
    cfg.scene.num_envs = 32
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # VirtualPerception + 速度バイアスをクリーン化 (真値・遅延なし・見失いなし)。
    cfg.goalkeeper.perception_clean = True


@configclass
class K1GKHierStage1EnvCfg_PLAY(K1GKHierStage1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)


@configclass
class K1GKHierStage2EnvCfg_PLAY(K1GKHierStage2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)
