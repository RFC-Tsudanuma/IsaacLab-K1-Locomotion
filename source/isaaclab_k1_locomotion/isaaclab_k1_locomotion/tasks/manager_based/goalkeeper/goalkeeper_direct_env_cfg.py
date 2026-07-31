# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (直接制御版) の環境定義。

階層版 (``goalkeeper_env_cfg.py``: 凍結歩行 + 速度コマンド) の後継。
**12 関節を直接制御する単一ポリシー**で、locomotion の歩行学習をベースに
「左右の横移動」に特化させる。

なぜ作り直したか (計測に基づく):
    凍結歩行ポリシー 0524_walk.pt の横移動は 0.66 m/s が上限で、指令を上げても
    伸びず (1.0 以上は転倒)、体を斜めに向けて前進を流用する案も旋回コストが
    利益を上回るため成立しなかった。一方でエンベロープ計測から、セーブ可否は
    ほぼ「必要横移動量」だけで決まり (0.7m で成功率が半減)、横移動を 2 倍に
    できればセーブ率は 61% → 85% 相当まで伸びる見込みが立った。
    そこで横移動そのものを学習対象にする。

2 ステージ構成 (観測レイアウトは全ステージ共通 = 59 次元):
    * Stage 1 (``Isaac-GoalkeeperDirect-Stage1-K1-v0``)
        locomotion そのままの速度コマンド追従タスク。ただしコマンド範囲を
        横重視 (vx ±1.0 / vy ±1.5) にし、横方向の追従・速度に追加報酬を掛ける。
        ボール系スロットはゼロのダミーで次元だけ確保する。
        ゴール・ボールはシーンに置かない (自由に走り回らせるため)。
    * Stage 2 (``Isaac-GoalkeeperDirect-Stage2-K1-v0``)
        ゴール + ボールを置き、ボール系スロットに実値を入れてセーブを学習。
        ``velocity_commands`` スロットはタスク由来の「移動要求」に差し替える
        (:func:`mdp.task_drive_vector`)。Stage 1 の ckpt から ``--resume``。
        セーブ成功率 (EMA) に応じて難易度を段階的に上げる適応カリキュラム付き
        (:func:`mdp.adaptive_difficulty`: 狙い先の広さ → ボール初速の順)。

観測スロット (順序厳守。変えると warmstart / ステージ間の重み引き継ぎが壊れる):
    先頭 49 = 歩行 K1PolicyCfg と完全同一 (base_ang_vel 3 / projected_gravity 3 /
    velocity_commands 3 / joint_pos 12 / joint_vel 12 / actions 12 / gait_phase 4)
    → ``train.py --warmstart_actor <歩行 ckpt>`` で歩行の重みを引き継げる。
    続く 10 = ball_pos_rel 2 / ball_vel 2 / ball_active 1 / target_y 1 / self_state 4

歩行周期は locomotion と同じ規約 (``_PHASE_FREQ`` = 1.6Hz 固定 +
``randomize_phase_freq`` による env ごとの ±0.05Hz ランダム化) をそのまま使う。
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
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from ..locomotion.mdp.rewards import foot_clearance_ji
from ..locomotion.rough_env_cfg import (
    _COMMAND_THRESHOLD,
    _PHASE_FREQ,
    _STANCE_RATIO,
    K1CriticCfg,
    K1ObservationsCfg,
    K1PolicyCfg,
)

from .goalkeeper_env_cfg import (
    BALL_RADIUS,
    GOAL_HALF_WIDTH,
    GoalkeeperParamsCfg,
    K1GoalkeeperSceneCfg,
)
from .mdp.curriculums import adaptive_difficulty
from .mdp.events import (
    relaunch_ball_after_save,
    reset_ball_perception,
    reset_ball_shot,
    reset_gk_buffers,
)
from .mdp.observations import (
    gk_ball_active,
    gk_ball_pos_rel,
    gk_ball_pos_rel_perceived,
    gk_ball_vel,
    gk_ball_vel_perceived,
    gk_self_state,
    gk_target_y,
    task_drive_phase_obs,
    task_drive_vector,
    zeros_obs,
)
from .mdp.rewards import (
    face_field,
    lateral_speed_bonus,
    return_to_center_after_save,
    save_clearance_bonus,
    save_touch_bonus,
    stance_foot_flat,
    stay_on_goal_line,
    target_reach_velocity_direct,
    track_lin_vel_y_exp,
)
from .mdp.terminations import goal_conceded, robot_out_of_bounds

# 横移動の目標速度 [m/s]。lateral_speed_bonus の正規化基準。
# ユーザー要件 (2026-07-21): 実効 1.0 m/s 必須・1.2 m/s あると理想。
# コマンド上限 (vy ±1.3) と揃えてあり、ここまで出せれば報酬が満額になる。
LATERAL_TARGET_SPEED = 1.3


# ---------------------------------------------------------------------------
# Observations (全ステージ共通レイアウト = 59 次元)
# ---------------------------------------------------------------------------

@configclass
class K1GKDirectStage1PolicyCfg(K1PolicyCfg):
    """Stage 1 の actor 観測。先頭 49 は歩行 K1PolicyCfg をそのまま継承する。

    ボール系スロットはゼロのダミー。ゼロ入力の列には勾配が流れないので、
    該当する重みは初期値のまま Stage 2 へ渡る (ball_kick と同じ次元一致方式)。

    **項の定義順 = スロット順。ここを変えると warmstart と
    ステージ間の重み引き継ぎが壊れる。**
    """

    ball_pos_rel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_vel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_active = ObsTerm(func=zeros_obs, params={"dim": 1})
    target_y = ObsTerm(func=zeros_obs, params={"dim": 1})
    self_state = ObsTerm(func=zeros_obs, params={"dim": 4})


@configclass
class K1GKDirectStage1CriticCfg(K1CriticCfg):
    """Stage 1 の critic 観測 (歩行 K1CriticCfg + 同じダミースロット)。"""

    ball_pos_rel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_vel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_active = ObsTerm(func=zeros_obs, params={"dim": 1})
    target_y = ObsTerm(func=zeros_obs, params={"dim": 1})
    self_state = ObsTerm(func=zeros_obs, params={"dim": 4})


@configclass
class K1GKDirectStage1ObservationsCfg(K1ObservationsCfg):
    policy: K1GKDirectStage1PolicyCfg = K1GKDirectStage1PolicyCfg()
    critic: K1GKDirectStage1CriticCfg = K1GKDirectStage1CriticCfg()


@configclass
class K1GKDirectPolicyCfg(K1GKDirectStage1PolicyCfg):
    """Stage 2/3 の actor 観測。スロットの順序・次元は Stage 1 と同一で中身だけ差し替える。

    ``velocity_commands`` スロットは、Stage 1 では外部から与えられる速度コマンド
    だったが、Stage 2/3 ではタスク由来の「移動要求」(守備面までの前後ずれ /
    目標 y までの横ずれ / 向きの誤差) に差し替える。意味が揃っているので
    「このスロットが大きい方向へ速く動く」という Stage 1 の学習がそのまま活きる。
    """

    velocity_commands = ObsTerm(
        func=task_drive_vector,
        params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    )
    # ★ gait_phase もタスク駆動に差し替える (2026-07-24)。
    #   既定の phase_obs は停止判定に base_velocity コマンドのノルムを使うが、
    #   Stage 2/3 では velocity_commands "スロットの中身" だけを task_drive_vector に
    #   差し替えており、base_velocity コマンド項自体は Stage 1 の設定
    #   (10 秒ごとにランダム再サンプル) のまま生き残っている。その結果、位相が
    #   タスクと無関係なランダムコマンドで駆動され、ボールを止めた後も
    #   「歩き続けろ」という位相が入り続けて **足踏みが止まらなかった**。
    #   階層版は high_action_phase_obs で解決済みだったが、直接制御版に
    #   移植されていなかった。観測スロットと同じ task_drive_vector でゲートする。
    gait_phase = ObsTerm(
        func=task_drive_phase_obs,
        params={
            "phase_freq": _PHASE_FREQ,
            # cmd_threshold は関数側の既定 (0.12 [m]) を使う。_COMMAND_THRESHOLD
            # (0.05) は速度 [m/s] 用のしきい値なので、位置ずれには流用しない。
            "max_y": GOAL_HALF_WIDTH,
            "use_perceived": True,
        },
    )
    # ボール系は知覚DR (レイテンシ/更新レート/ドロップ/距離依存ノイズ) 付きの実値。
    # ノイズは関数内で付加するので ObsTerm 側の noise は付けない。
    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel_perceived)
    ball_vel = ObsTerm(func=gk_ball_vel_perceived)
    ball_active = ObsTerm(func=gk_ball_active)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True})
    self_state = ObsTerm(func=gk_self_state, noise=Unoise(n_min=-0.02, n_max=0.02))


@configclass
class K1GKDirectCriticCfg(K1GKDirectStage1CriticCfg):
    """Stage 2/3 の critic 観測 (真値・ノイズなし)。"""

    velocity_commands = ObsTerm(
        func=task_drive_vector, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": False}
    )
    # policy と同じくタスク駆動位相 (真値版)。actor/critic で位相の定義がズレると
    # 価値推定が actor の見ている状態と食い違うので必ず揃える。
    gait_phase = ObsTerm(
        func=task_drive_phase_obs,
        params={
            "phase_freq": _PHASE_FREQ,
            # cmd_threshold は関数側の既定 (0.12 [m]) を使う。_COMMAND_THRESHOLD
            # (0.05) は速度 [m/s] 用のしきい値なので、位置ずれには流用しない。
            "max_y": GOAL_HALF_WIDTH,
            "use_perceived": False,
        },
    )
    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel)
    ball_vel = ObsTerm(func=gk_ball_vel)
    ball_active = ObsTerm(func=gk_ball_active)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": False})
    self_state = ObsTerm(func=gk_self_state)


@configclass
class K1GKDirectObservationsCfg(K1ObservationsCfg):
    policy: K1GKDirectPolicyCfg = K1GKDirectPolicyCfg()
    critic: K1GKDirectCriticCfg = K1GKDirectCriticCfg()


# ---------------------------------------------------------------------------
# Stage 1: locomotion ベースの横移動特化
# ---------------------------------------------------------------------------

@configclass
class K1GKDirectStage1EnvCfg(K1FlatEnvCfg):
    """Stage 1: 速度コマンド追従を横重視で学習する (ゴール・ボールなし)。

    報酬・イベント・終了条件は locomotion (K1FlatEnvCfg) をそのまま使い、
    横方向の追従報酬と速度ボーナスだけを上乗せする。歩容や姿勢の作り込みは
    歩行タスク側の資産をそのまま活かす。
    """

    observations: K1GKDirectStage1ObservationsCfg = K1GKDirectStage1ObservationsCfg()
    # ステージ 2/3 と同じフィールドを持たせておく (JSON 上書きのパス互換のため)
    goalkeeper: GoalkeeperParamsCfg = GoalkeeperParamsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- 横重視のコマンド範囲 ---
        # 前進は「歩容を健全に保つ + 定位置復帰に使う」ぶんだけ残し (±1.0)、
        # 横は歩行の学習上限 ±0.9 を超えて ±1.3 まで要求する。
        # 上限 1.3 の根拠 (2026-07-21 ユーザー要件): 実効横速度は 1.0 m/s 必須・1.2 で
        # 十分。±1.5 で学習したら 1.47 m/s 出たが、支持脚がつま先立ちで地面を蹴る
        # 無理な歩容が出た (かかとが浮く)。追従率 ~0.97 なので指令 1.3 → 実効 ~1.26 と
        # 要件を満たしつつ、過剰な速度要求によるつま先立ちの誘因を減らす。
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.3, 1.3)

        # カリキュラムも横重視の段階に差し替える (前進は早々に上限へ、横は細かく伸ばす)
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-0.6, 0.6), (-0.8, 0.8), (-1.0, 1.0), (-1.0, 1.0),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.6, 0.6), (-0.9, 0.9), (-1.1, 1.1), (-1.3, 1.3),
        ]

        # --- 横移動特化の追加報酬 ---
        # 1) 横方向だけの追従報酬。locomotion の track_lin_vel_xy_exp は前後と左右を
        #    合算するため、横が遅くても前進で取り返せてしまう。
        self.rewards.track_lin_vel_y = RewTerm(
            func=track_lin_vel_y_exp,
            weight=2.5,
            params={"command_name": "base_velocity", "std": 0.3},
        )
        # 2) 実速度そのものへの線形報酬。追従報酬 (ガウス) は達成不能な高速域で
        #    勾配が消えて「諦め」に落ちるので、上限付近でも勾配が残る項を足す。
        self.rewards.lateral_speed_bonus = RewTerm(
            func=lateral_speed_bonus,
            weight=2.0,
            params={
                "v_ref": LATERAL_TARGET_SPEED,
                "command_name": "base_velocity",
                "min_cmd": 0.6,
            },
        )
        # 3) 遊脚のクリアランス (足上げ)。目視で「足が上がりきらず擦って崩れかける」
        #    挙動が見られたため追加 (locomotion では未使用の項)。
        #    位相は get_phase_freq 経由で randomize_phase_freq に自動追従する
        #    (feet_height_bezier は位相を固定値で受けるため使わない)。
        #
        #    ★ 速度重み付きの `foot_clearance_ji_pen` は **使わない**。あちらのペナルティは
        #      「足の水平速度 × 高さ誤差²」で、足を上げずに **ゆっくり動かせば** 速度項が
        #      小さくなりペナルティを回避できてしまう。実際 weight=10 で試したところ
        #      「片足は擦ったまま (低速で逃げる) / もう片足はつま先だけ上げる」という
        #      2 つの抜け道に収束した。本項 (`foot_clearance_ji`) は速度を掛けない
        #      exp(-(目標−実際)²/σ²) の **報酬** なので、低速で逃げられない。
        #
        #    ★ target_clearance は「足リンク原点の絶対高さ」であって持ち上げ量ではない。
        #      接地時点で既に 0.035m あるので、持ち上げ 5cm = 0.035 + 0.05 = 0.085m。
        #      exp 型は目標で報酬が最大になるため、二乗ペナルティのように手前で釣り合わず
        #      目標値をそのまま狙わせられる。
        #
        #    実測 (2026-07-20, 4000iter 時点): 持ち上げ平均 3.1cm (指令0.9) / 3.6cm (指令1.5)。
        #      学習後は eval_frozen_lateral.py の「足の持ち上げ高さ」で再測し、
        #      5cm に届かなければ weight を上げ、横移動速度が 1.3 m/s を割ったら下げる。
        #    重みの履歴:
        #      2.0 (2026-07-22 実測): 持ち上げ 5.3〜5.4cm と目標超過 → 1.5 に減
        #      1.5 + joint_deviation_hip -0.4 (2026-07-23 実測): 3.8〜3.9cm に低下。
        #        外股矯正で股関節の可動を縛った分、遊脚を持ち上げる余地も削られた。
        #        5cm 回復のため 2.5 へ増強 (股関節制約と綱引きになるので旧値 2.0 より上)。
        #        ★ 外股・つま先立ちの再発と速度 (1.0 必須) を学習後に必ず確認すること。
        #      2.5 (2026-07-27 実測, 8000iter): 持ち上げ 3.9〜4.4cm と再び 5cm 割れ。
        #        つま先立ち対策で足首を 2 方向から締めた (stance_foot_flat -3.0→-8.0 +
        #        feet_parallel_to_ground を状態報酬化) ぶん、遊脚を持ち上げる余地が
        #        削られた。同じ綱引きが股関節制約で起きたのが上の 07-23 の記録。
        #      3.5 + 目標 6cm (2026-07-27 実測, 8000iter): **失敗**。目視で
        #        「支持脚で地面を蹴って跳び、その間に遊脚を上げる」歩容に退行した。
        #        実測でも lin_vel_z_l2 -0.028→-0.058 (上下動 2 倍)、
        #        dof_pos_limits_ankle -0.0027→-0.0085 (足首が可動限界まで底屈)、
        #        stance_foot_flat -0.10→-0.20 (つま先立ち再発) と裏付けられた。
        #        原因: foot_clearance_ji が見るのは足リンクの **ワールド z (絶対高さ)**
        #        なので、「遊脚を股関節・膝で上げる」以外に「体ごと持ち上げる」でも
        #        達成できてしまう。base_height_penalty は min_height を下回った時だけ
        #        罰するため体を高く上げるのは無罰で、跳ぶのが最安の解になった。
        #        weight と目標を同時に上げて現在地での勾配を 2.0 倍にしたのが行き過ぎ。
        #      → weight は 2.5 に戻し、**目標 6cm だけ**を変更する (勾配 1.43 倍)。
        #        誤差 2cm はガウス勾配の最大点 (σ/√2 = 2.1cm, σ=0.03) にほぼ一致するので、
        #        weight を上げずに引き上げ圧力を稼げる。
        #        ★ 学習後に eval_gk_direct_lateral.py の「足の持ち上げ高さ」で 5cm に
        #          届いたか、速度が 1.0 m/s を割っていないか、跳躍が出ていないかを確認。
        #      ★ 目標値 0.085 → 0.095 (持ち上げ 5cm → 6cm) に引き上げ (2026-07-27)。
        #        理由: 本項は exp(-(目標−実際)²/σ²) で **目標ちょうどで最大** になるため、
        #        「足を上げると損」な項 (dof_vel_limits / energy / 足首の締め付け) と
        #        釣り合う平衡点は **必ず目標より下** に来る。実際 weight 2.5・目標 5cm で
        #        平衡は 4.0cm だった。つまり目標 5cm のままでは weight をいくら上げても
        #        **平均 5cm には原理的に到達しない**。要件 (平均 5cm 以上) を満たすには
        #        目標自体を上に置く必要がある。
        #        6cm にするのは、現在地 4cm からの誤差 2cm がガウス勾配の最大点
        #        (σ/√2 = 2.1cm, σ=0.03) にほぼ一致するため。7cm 以上にすると誤差が
        #        ピークを過ぎて勾配が落ち始め、かつピーク持ち上げが 9〜10cm と
        #        身長 0.6m のロボットには過大になる。
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance_ji,
            weight=2.5,
            params={
                "command_name": "base_velocity",
                # 0.095 (6cm) → 0.105 (7cm) に引き上げ (2026-07-29)。
                # 実機の人工芝はパイル高さ 20〜30mm。支持脚は荷重でパイルを潰して沈むが、
                # 遊脚は芝の上を通るので **実効クリアランス ≒ 足上げ − パイル高さ**。
                # 6cm 目標で得られた実測 4.0〜5.0cm では、指令 1.2 (実用速度)・パイル 30mm
                # のとき残り 10mm しかなく余裕が無い。
                # 跳躍で稼ぐ抜け道は lin_vel_z_l2 = -2.5 で塞げることが実証済みなので
                # (目標 6cm + -0.8 は跳躍、6cm + -2.5 は跳ばず 4.6cm)、その状態で目標だけ上げる。
                # 2026-07-29: 0.105 (7cm) は跳躍でしか達成できないことが判明したため
                # 0.095 (6cm) に戻す。7cm 目標では
                #   跳躍を許すと 6.8cm (体ごと持ち上げているだけ)
                #   跳躍を封じると 2.6〜3.3cm (脚の関節だけでは届かない)
                # となり、実測 4.3cm を得られる 6cm 目標が最良だった。
                "target_clearance": 0.095,
                "phase_freq": _PHASE_FREQ,
                "stance_ratio": _STANCE_RATIO,
                "cmd_threshold": _COMMAND_THRESHOLD,
            },
        )
        # 上下動 (跳躍) のペナルティを強化する (flat_env_cfg の既定は -0.8)。
        # ★ 2026-07-27: 目標 6cm にすると weight を 2.5 に戻しても
        #   「支持脚で地面を蹴って跳び、その間に遊脚を上げる」歩容に退行した。
        #   実測: lin_vel_z_l2 -0.028 → -0.056 (上下動 2 倍)、
        #        dof_pos_limits_ankle -0.0027 → -0.0082 (足首が可動限界まで底屈)、
        #        stance_foot_flat -0.101 → -0.170 (つま先立ち再発)。
        #   原因: foot_clearance_ji が見るのは足リンクの **ワールド z (絶対高さ)** なので、
        #   「遊脚を股関節・膝で上げる」以外に「体ごと持ち上げる」でも達成できる。
        #   さらに base_height_penalty は min_height を **下回った時だけ** 罰するため、
        #   体を高くするのは無罰。足首を -8.0 で締めて脚だけでの足上げ限界が約 4cm に
        #   下がった結果、目標 6cm の残り 2cm を稼ぐ最安の手段が跳躍になった。
        #
        #   ただし地面は完全な平面なので「絶対高さ = 地面からのクリアランス」であり、
        #   つまずき防止という目的に対して指標自体は正しい。問題は達成手段が跳躍で
        #   あることなので、**測り方を変えるのではなく跳躍を直接罰する**。
        #   ★ 効かなければ (跳躍が止まらなければ) 足上げをベース相対で測る方式へ移行する。
        #   ★ 強すぎると着地の衝撃吸収や重心の上下動まで殺す。横速度 (1.0 必須) を再測すること。
        self.rewards.lin_vel_z_l2.weight = -2.5
        # ★ 2026-07-29: 跳躍だけを狙い撃つ flight_phase (両足同時浮きのペナルティ、
        #   mdp/rewards.py に実装あり) を weight -2.0 で試したが **採用しない**。
        #   跳躍は完全に止まった (上下動 raw 0.052 → 0.023、過去最良) が、
        #   **足上げが 2.6〜3.3cm まで低下**した (07-28 の 4.3cm より悪化)。
        #   目標 6cm では lin_vel_z_l2 -2.5 だけで跳躍を抑えられており (07-28 実績)、
        #   flight_phase を足すと跳躍防止が二重になって足上げだけが削られる。
        #   → 目標 7cm と併せて 07-28 構成に戻す。使うなら目標を上げた場合のみ。
        # 足裏を地面と平行に保つ報酬。**状態報酬に切り替える** (locomotion は差分形式 3.0)。
        # 上の足上げ報酬は「足リンク原点の高さ」しか見ないので、足首を背屈させて
        # つま先を上げるだけで高さを稼ぐ抜け道がある (実際に発生し、足裏が接地しなくなった)。
        #
        # ★ 2026-07-24: 既定の enable_potential=True はポテンシャル差分 (γΦ' − Φ) を返す。
        #   これはエピソード積算で Φ(終) − Φ(始) に畳まれるため、歩行のような**周期運動**では
        #   「傾く」と「戻る」が毎歩打ち消し合い、正味の圧力がほぼゼロになる。weight を
        #   3.0 → 6.0 に上げても効かなかったのはこのため。
        #   結果として **遊脚 (空中の足) の姿勢を罰する項が実質存在しない** 状態だった。
        #   着地時の足の向きは遊脚期に決まるので、接地後にしか効かない stance_foot_flat
        #   では「つま先から着地する」歩容を直せない (4000iter 時点で既に発生)。
        #   enable_potential=False で素の exp(-誤差/σ) を返す状態報酬にし、遊脚を含めて
        #   常時「足裏が水平なら得」という圧力をかける。
        #   ★ weight は 6.0 → 1.5 に下げること。定常報酬化すると常時最大 6.0 が入り、
        #     速度報酬 (2.5 + 2.0) を上回って「その場で足を水平にして立つ」諦め解に
        #     落ちる危険がある。1.5 なら水平/つま先立ちの差が約 1.2 で、速度を殺さずに効く。
        self.rewards.feet_parallel_to_ground.params["enable_potential"] = False
        self.rewards.feet_parallel_to_ground.weight = 1.5
        # 4) 足首 (Ankle_Pitch) の基準姿勢からの逸脱ペナルティ。
        #    ±1.5 学習で「支持脚がつま先立ちのまま横移動する」歩容が出た (かかとが浮き、
        #    つま先だけで接地)。つま先立ち = 足首の底屈なので、関節角の逸脱を直接罰する。
        #    feet_parallel_to_ground はポテンシャル形式のため定常的なつま先立ちに無力で、
        #    重みを上げても効かなかった。こちらは状態ペナルティなので維持し続ける限り効く。
        #    ★ 強すぎると足首が固まって着地の衝撃吸収や蹴り出しまで殺し、横移動速度が
        #      落ちる。速度要件 (1.0 必須 / 1.2 理想) を割ったら -0.1 台へ緩めること。
        self.rewards.ankle_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch"])},
        )
        # 5) 接地脚の足裏水平ペナルティ (つま先立ちの本命対策)。
        #    **接触センサで実際に接地している足** の足裏の傾き (pitch²+roll²) を状態
        #    ペナルティとして毎ステップ罰す。
        #    初版は位相ベースで接地脚を推定していたが効かなかった (weight=-1 で目視改善ゼロ)。
        #    原因: 位相モデルは前進歩行前提で、速い横移動では実際の足の動きと合わず
        #    「移動方向と反対の足がずっとつま先立ち」なのに位相上 swing 扱いされて
        #    罰を逃れていた (ユーザー目視で判明)。接触力ベースなら、つま先だけで接地
        #    している足も正しく捕捉できる。遊脚 (完全に浮いた足) は接触力ゼロで自動的に
        #    対象外になり、足上げ (foot_clearance) と干渉しない。
        #    ★ 強すぎると足首を固めて着地衝撃を吸収できず横移動速度が落ちる。
        #      速度が 1.0 m/s を割ったら緩めること。判定を直したので weight は据え置きで
        #      効くはず。効き不足なら -2.0 へ。
        # ★ sensor_cfg/asset_cfg は params で明示的に渡すこと。RewardManager は params の
        #   SceneEntityCfg しか body_ids を解決しないので、関数デフォルト値のままだと
        #   全 body を指してしまう (13 body vs 接触 2 body で shape 不一致になる)。
        #   2026-07-22 実測 (weight=-1.0): 学習中に Episode_Reward/stance_foot_flat が
        #   下がらず (-0.047→-0.060)、ポリシーが「ペナルティを払ってでもつま先立ちの方が
        #   得」と判断していた。速度 1.23 m/s と要件に余裕があるため -3.0 へ増強
        #   (foot_clearance を 2.0→1.5 に下げた分の圧をこちらに回す)。
        #   2026-07-24 実測 (weight=-3.0, 10000iter): 依然として単調悪化
        #   (-0.09 → -0.19)。4000iter 時点では着地で足裏が接地していたが、以降
        #   「進行方向と逆の足 (蹴り出し脚) だけがつま先立ち」に退行 (ユーザー目視)。
        #   横移動の蹴り出しは足首の底屈そのものなので速度報酬 (2.5+2.0) と正面衝突する。
        #   実測 1.49 m/s と要件 (1.0 必須 / 1.3 目標) に余裕があるぶんを姿勢に回し
        #   -8.0 へ増強する。★ 学習後に必ず eval_gk_direct_lateral.py で速度を再測し、
        #   1.0 m/s を割ったら -5.0 付近へ戻すこと。曲線が単調悪化のままならまだ不足。
        self.rewards.stance_foot_flat = RewTerm(
            func=stance_foot_flat,
            weight=-8.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
                "force_threshold": 1.0,
            },
        )
        # 6) 向き維持の強化。wz=0 指令でも横移動中に体が回り「円を描いて横移動」する
        #    症状が出た (実測 yaw drift 20〜26°)。locomotion 既定の track_ang_vel_z_exp
        #    (3.0) が横移動系報酬 (track_lin_vel_y 2.5 + lateral_speed_bonus 2.0) に
        #    負けていたため 5.0 へ引き上げ、直線的な横移動に矯正する。
        #    → 9000iter 実測で yaw drift 26°→14° に半減 (2026-07-23 確認)。
        #    2026-07-29: 15000iter 実測でも 10〜12° 残り、目視でも横移動が弧を描く。
        #      5.0 → 7.0 へさらに引き上げる。
        #      ★ 上げすぎると「回らないこと」を優先して横移動速度が落ちる
        #        (横移動系 track_lin_vel_y 2.5 + lateral_speed_bonus 2.0 = 4.5 との綱引き)。
        #        速度 1.0 m/s を割ったら 6.0 へ戻すこと。
        #      ★ なお本項が追従するのは角速度 wz であって heading そのものではないため、
        #        微小なバイアスは積分されてドリフトとして残る。学習時は heading_command=True
        #        で heading フィードバックが閉じているが、eval / play は wz=0 の開ループで
        #        駆動しているのでドリフトが補正されない。実タスク (Stage 2/3) では
        #        task_drive_vector の第3成分が dyaw = -heading を渡すので閉ループになる。
        #    2026-07-29 実測 (7.0, 4000iter): yaw drift は 10〜12° → 5.7〜8.8° と改善したが
        #      **代償が大きすぎた**。回転抑制の報酬が 3.78 と横移動系 (1.29 + 0.29 = 1.58) の
        #      2.4 倍になり、「体を回さずに横へ進む」ため股を開く歩容に流れた:
        #        joint_deviation_hip -0.147 → -0.445 (外股が 3 倍。weight は据え置きなのに違反増)
        #        横速度 (指令1.2)    1.182 → 0.628 (要件 1.0 を下回る)
        #        コマンド範囲カリキュラム stage 3/3 → 2/3 で停滞 (追従誤差が閾値を超えて進めない)
        #      外股・速度低下・yaw 改善はすべて同じ変更の表裏。外股対策
        #      (joint_deviation_hip) を強めるのは逆効果で、元の配分を戻すのが正しい。
        #    2026-07-29 実測 (6.0, 8000iter): 7.0 よりは改善したが **まだ不足**。
        #      追従誤差 EMA 0.425 で進級閾値 0.400 を越えられず、カリキュラムは
        #      stage 2/3 (lin_vel_y_max 1.1) で停滞。外股も -0.370 と 2.5 倍のまま、
        #      横速度も 0.762 と要件 1.0 を下回った。
        #      → 元の 5.0 に戻す。yaw drift は 10〜12° に悪化するが、実タスク
        #        (Stage 2/3) では task_drive_vector の第3成分が dyaw = -heading を渡して
        #        **閉ループで補正される**ため、開ループ計測でのドリフトは実性能への影響が小さい。
        #        一方 横速度 1.0 m/s はセーブ成否に直結する硬い要件なので、そちらを優先する。
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        # 7) 外股 (ガニ股) の矯正。9000iter 版の目視で「股関節を外に開いたまま横移動する」
        #    歩容が確認された (2026-07-23)。外股は支持基底面が広がり横安定には有利なので
        #    ポリシーが好んで使うが、実機では股関節に無理な角度が掛かり続ける。
        #    locomotion 既定の joint_deviation_hip (Hip_Yaw + Hip_Roll, -0.10) では
        #    外股のメリットに負けていたため 4 倍に強化する。
        #    ★ 横ステップは本質的に Hip_Roll の外転を使うので、強くしすぎると横移動
        #      そのものを殺す。速度が 1.0 m/s を割ったら -0.2 へ緩めること。
        self.rewards.joint_deviation_hip.weight = -0.4


@configclass
class K1GKDirectStage1EnvCfg_PLAY(K1GKDirectStage1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# Stage 2: ゴール + ボールでセーブを学習
# ---------------------------------------------------------------------------

@configclass
class K1GKDirectEnvCfg(K1GKDirectStage1EnvCfg):
    """Stage 2: Stage 1 の歩容の上に、ゴール・ボール・セーブ課題を載せる。"""

    scene: K1GoalkeeperSceneCfg = K1GoalkeeperSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: K1GKDirectObservationsCfg = K1GKDirectObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # エピソード継続モードなので長めに取る。ボール到達に最長 10s かかる
        # (spawn 最遠 5.0m / 初速下限 0.5 m/s) ため、10s のままだと 1 球で
        # 終わってしまい継続にする意味が出ない。25s なら平均 3〜5 球入る。
        self.episode_length_s = 25.0

        # 完全平面に戻す (凹凸の上ではボールが勝手に転がり判定を汚す)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # --- 初期配置: 守備面 (ゴールラインの guard_x=0.4m 前) 付近 ---
        self.events.reset_base.params["pose_range"]["x"] = (0.3, 0.5)
        self.events.reset_base.params["pose_range"]["y"] = (-0.5, 0.5)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }

        # --- リセット順序: 状態バッファ → ボール発射 → 知覚DR 初期化 ---
        self.events.reset_gk_buffers = EventTerm(func=reset_gk_buffers, mode="reset")
        self.events.reset_ball = EventTerm(func=reset_ball_shot, mode="reset")
        self.events.reset_ball_perception = EventTerm(func=reset_ball_perception, mode="reset")
        # エピソード継続モード: セーブ確定から 1.0s 後に次の球を撃つ (毎ステップ判定)。
        # ステージ1 の resample_stage1_target と同じ「成功したら切らずに次の課題」方式。
        _dt = self.sim.dt * self.decimation
        self.events.relaunch_ball = EventTerm(
            func=relaunch_ball_after_save,
            mode="interval",
            interval_range_s=(_dt, _dt),
            is_global_time=True,
            params={"respawn_delay_steps": 50},
        )
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

        # --- タスク報酬を上乗せ (歩容まわりの locomotion 報酬はそのまま残す) ---
        # 速度コマンドはもう外部から来ないので、コマンド追従系は無効化する。
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.track_lin_vel_y = None
        self.rewards.lateral_speed_bonus = None

        # ★ y 方向の dense 報酬 (2026-07-24 追加)。
        #   上の 4 項を無効化した結果、**横位置 (y) を評価する密報酬が一つも無く**、
        #   残る速度関連の項が feet_slide / dof_vel_limits / action_smoothness /
        #   energy という **ペナルティだけ** になっていた。つまり「速く動く」ことは
        #   それ自体が純損で、見返りは 3〜5 秒先の save_touch_bonus (+100) /
        #   termination_penalty (-200) のみ。gamma=0.99 @ 50Hz では 0.99^250 ≈ 0.08
        #   まで減衰するため、走り出す判断の時点でほぼ見えていない。
        #   → 「必要最小限しか動かない」が最適解になり、Stage 1 で獲得した
        #      横移動 (実測 1.49 m/s) が Stage 2/3 で使われず忘却されていた。
        #   weight は Stage 1 の速度報酬合計 (track_lin_vel_y 2.5 +
        #   lateral_speed_bonus 2.0 = 4.5) と同オーダーにして、ステージ遷移で
        #   「動く価値」が急落しないようにする。
        #   ★ 階層版の target_reach_velocity ではなく _direct 版を使うこと
        #     (階層版は停止判定に上位アクションを見るため、直接制御版では
        #      常に満額になり足踏みを許してしまう)。
        self.rewards.target_reach_velocity = RewTerm(
            func=target_reach_velocity_direct,
            weight=4.0,
            params={
                "deadband": 0.12,
                "v_cap": LATERAL_TARGET_SPEED,
                "stop_speed": 0.5,
                "max_y": GOAL_HALF_WIDTH,
            },
        )

        self.rewards.save_touch_bonus = RewTerm(func=save_touch_bonus, weight=100.0)
        # セーブの「質」への上乗せ (2026-07-24)。触れただけで満額だと、ゴール正面に
        # 転がったまま止めた結末と完全に弾き出した結末が同じ扱いになる (実戦では
        # 前者はそのまま押し込まれる)。ゴール枠から 1.5m 離せば満点。
        # weight は touch の 50% を上限とし、「まず届くこと」の優先度を崩さない
        # (質を重く見すぎると、確実に触れる球を丁寧に処理する方が得になり、
        #  遠い球へ飛びつかなくなる)。
        self.rewards.save_clearance = RewTerm(func=save_clearance_bonus, weight=50.0)
        self.rewards.return_to_center = RewTerm(
            func=return_to_center_after_save, weight=1.0, params={"std": 0.5}
        )
        self.rewards.stay_on_goal_line = RewTerm(func=stay_on_goal_line, weight=1.0, params={"std": 0.3})
        self.rewards.face_field = RewTerm(func=face_field, weight=1.0, params={"std": 0.5})

        # --- 終了条件 ---
        # ★ エピソード継続モード (2026-07-24): save_success は DoneTerm にしない。
        #   従来は「セーブ成功＝エピソード終了」だったため、1 エピソードに報酬
        #   イベントが 1 回しか無く、return_to_center_after_save は発火する間も
        #   なく終了していた。成功時は relaunch_ball_after_save が次の球を撃ち、
        #   ロボットはセーブした場所からそのまま次に備える。
        #   失敗 (失点・転倒・場外) だけがエピソードを切る。
        self.terminations.goal_conceded = DoneTerm(func=goal_conceded)
        self.terminations.out_of_bounds = DoneTerm(
            func=robot_out_of_bounds,
            params={"x_range": (-0.1, 2.5), "y_abs_max": 2.2},
        )

        # --- 歩行由来のカリキュラムは無効化 (コマンドが無いので意味を持たない) ---
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.push_robot_stage1 = None


@configclass
class K1GKDirectStage2EnvCfg(K1GKDirectEnvCfg):
    """Stage 2 本体: セーブ成功率 (EMA) に応じて難易度を段階的に上げる (2 軸)。

    ★ 2026-07-31: 旧 ``K1GKDirectStage3EnvCfg`` から改名。Stage 2/3 を統合済みで
      「Stage 3」という段は存在しないため、番号とログ出力先を実態に合わせた。
      親の :class:`K1GKDirectEnvCfg` は「ゴール + ボールを置くが難易度は固定」の
      土台で、Play 用 cfg も継承しているので残してある (タスク登録のみ廃止)。

    ★ 2026-07-31: 初速のみを動かす ``adaptive_ball_speed`` から、
      「狙い先の広さ → 初速」の順で上げる ``adaptive_difficulty`` に変更。
      難易度の主因は初速ではなく必要横移動量 (実測: 0.7m で成功率半減) であり、
      その主因を決める aim_y_range が最初から最大値固定だったため、成功率が
      62% 前後で頭打ちになりカリキュラムが不感帯で休眠していた。
      (階層版 goalkeeper_env_cfg.py は従来の adaptive_ball_speed のまま。)
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.difficulty = CurrTerm(func=adaptive_difficulty)


def _make_play_clean(cfg: K1GKDirectEnvCfg) -> None:
    """PLAY 用: 外乱と知覚DR を切って挙動を見やすくする。"""
    cfg.scene.num_envs = 32
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # VirtualPerception + 速度バイアスをクリーン化 (真値・遅延なし・見失いなし)。
    # キーパーの動きそのものを純粋に確認するため。
    cfg.goalkeeper.perception_clean = True


@configclass
class K1GKDirectEnvCfg_PLAY(K1GKDirectEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)
