# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ロングパス + 短期 I/O 履歴 (short history) 環境。

:class:`~..walk_long_pass.walk_long_pass_env_cfg.K1WalkLongPassEnvCfg` の
**短期履歴用** バリアント。行動空間を継承し、policy 観測の
本体状態 5 項に 0.1 秒ぶんの履歴を付ける。さらに、左足裏だけを見る 3 次元スロットを
ミラー可能なボール 3D 位置へ差し替える。蹴り方向は、報酬と同じ真のコマンド方向を
現在のbase座標へ変換してactorへ渡す。観測フィールド名と順序は継承元のままなので、
policy / critic の次元は 223 / 61 を維持する。

観測の意味変更に適応する間に「蹴らない」へ崩れないよう、継続学習用のカリキュラムだけ
親から変更する。Stage 1では目標方向を採点せず、足中心から見たボール方向が内側面法線
から30°以内になる接触を先に学ぶ。キック成立率と内側接触率のEMAがともに0.80以上に
なったら一方向にStage 2へ進み、面角度・直線スイング・逆方向罰則と、精度ゲート付きの
球速帯カリキュラムを有効化する。

内側接触報酬は両Stageで通常値の10倍にする。Stage 2では接触時の足内側面法線を
目標方向へ向ける報酬を10倍、接触直前の足速度を目標方向へ揃える報酬を3倍で有効化する。
球速と30度仰角は方向から独立して評価し、最終球方向は目標の反対半球だけを罰する。
キック成立前の探索を補助するため、エピソード最初の足とボールの接触には+2を払い、
2回目以降の接触は1回あたり-0.5として、一度の接触で蹴り切るよう誘導する。

歩行の前に、現在の立位から左右どちらかの足で直接蹴る前段を置く。ボールは蹴り方向に
合わせてbaseから``r_stance``だけ先へ置き、左右の足側を等確率で選ぶ。方向帯は
±10°→±30°→±45°→±60°→±90°、位置ジッタは0→1→2→3→4cmと広げる。
各段はキック成立率・インサイド接触率・方向誤差で昇格し、最後の±90°帯を達成した
次段で通常の歩行指令と前方半円0.5-1.5mのボール配置へ切り替える。報酬構成と
キック後2秒の回復区間は全段で共通とし、静止段階だけはリセット時の胴体yawからの
ずれを罰して、胴体の初期向きを保ったまま蹴るようにする。

arXiv:2401.16889 (Locomotion policy に短期の観測+行動履歴を与える) 相当。
ネットワーク構造 (MLP / 隠れ層) は変更しない。増えるのは入力層の幅だけ。

なぜ履歴か
----------
policy は feedforward MLP なので 1 ステップの観測しか見ていない。実機で効く情報の
多くは「単フレームでは観測できないが数フレームの差分には現れる」もの:

* アクチュエータの遅れ … 目標角 (``prev_joint_request``) と実現角 (``joint_pos``) の
  ずれの **時間発展** が、そのステップの実効ゲイン/遅延を語る。sim2real で最も
  ずれる部分なので、履歴があると「今の機体がどれだけ鈍いか」を暗黙同定できる。
* 接地/離地の遷移 … ``projected_gravity`` と ``base_ang_vel`` の 0.1 秒の軌跡には、
  接地衝撃・スリップ・押されの区別が出る。単フレームでは同じ値になる。
* 速度ノイズの平滑 … ``joint_vel`` は Unoise(±1.5) と大きなノイズを載せているので、
  5 フレームあれば実質的なローパスを policy 側で学習できる。

履歴長 = 5 ステップ (0.1 秒)
---------------------------
制御周期は ``sim.dt (0.005) * decimation (4) = 0.02 s`` (50 Hz) なので::

    0.1 s / 0.02 s = 5 ステップ

``_HISTORY_LEN`` は下限 4 を掛けて算出している (制御周期を上げたときに履歴が
2-3 フレームまで潰れると、差分が取れず履歴の意味が無くなるため)。

履歴を付ける項 / 付けない項
---------------------------
付ける (本体状態と自分の出力 = 「機体の I/O」):

======================  ====  =========================================
term                    dim   意味
======================  ====  =========================================
``projected_gravity``   3     胴体姿勢
``base_ang_vel``        3     胴体角速度
``joint_pos``           12    実現関節角
``joint_vel``           12    実現関節速度
``prev_joint_request``  12    **actions** (前ステップの目標関節角)
======================  ====  =========================================

付けない (タスク条件付け項):

* ``kick_direction`` … 現在の真のyawでbase座標へ変換した方向。履歴化せず現在値だけを渡す。
* ``target_kick_velocity`` … エピソード中ずっと定数。履歴を取ると同じ値が5回並ぶだけ。
* ``gait_phase`` / ``gait_phase_factor_offset`` … 位相は時刻の決定的関数なので、
  履歴は現在値から復元できる。
* ``ball_vel`` / ``prev_ball_pos`` … **既に履歴になっている**。``prev_ball_pos`` は
  遅延させたボール位置そのもの、``ball_vel`` はその差分。ここに history_length を
  重ねると同じ情報を二重に持つことになる (指示の「既存の ball history と重複しない」)。
* ``sole_pos`` … 順序保持のため属性名だけを残した、1 ステップ遅延の
  **ボール 3D 位置**。既存の ``prev_ball_pos`` / ``ball_vel`` と同様に履歴化しない。

次元
----
========  ======  =======  ==============================================
group     before  after    内訳
========  ======  =======  ==============================================
policy    55      **223**  履歴 5 × 42 = 210 + 非履歴 13
critic    61      61       変更なし
action    12      12       変更なし
========  ======  =======  ==============================================

critic に履歴は付けない。左足裏スロットだけは遅延なしボール位置へ
置換するが、既存の特権情報 ``ball_pos_rel`` も残し、61 次元と checkpoint の
形状契約を変えない。どちらもミラー写像では (x, y, z) → (x, −y, z) とする。

既存 checkpoint からの引き継ぎ (重要)
--------------------------------------
**``--load_pretrained`` に生の long_pass checkpoint を渡してはいけない。**
[train.py] は「形の合わないテンソルを捨てる」ので、``actor.0.weight`` (入力層) と
``actor_obs_normalizer.*`` が捨てられ、入力層がランダム初期化された死んだポリシーになる。

さらに **``expand_checkpoint_kick_flag.py`` も使えない**。あちらは「末尾にゼロを足す」
だけだが、履歴化は各項を **その場で 5 倍に展開** するので、55 次元の並びが 223 次元の
中に散らばる (``joint_pos`` は index 11-22 → 35-94 へ移動する)。専用の
:mod:`scripts.rsl_rl.expand_checkpoint_history` を使うこと::

    python scripts/rsl_rl/expand_checkpoint_history.py \\
        logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \\
        -o /tmp/long_pass_history_init.pt

このスクリプトは履歴ブロックの列を拡張して **形状上は** 読み込めるようにする。
ただし、元 checkpoint の左足裏スロットの重みと正規化統計は、新タスクでは
ボール位置に対して適用される。したがって **意味的に互換ではなく、挙動も一致しない**。
利用する場合は近似的な初期化として扱うこと。通し実行は
``./scripts/rsl_rl/train_walk_long_pass_history.sh`` だが、これも同じ近似初期化を使う。

``common_step_counter`` が 0 に戻る ``--load_pretrained`` でもキック報酬を消さないため、
500 iteration までに終わる報酬 weight のランプだけは終値に固定する。Stage 1では
内側接触だけを採点し、Stage 2昇格後に500 iterationの安定期間を置いて球速帯を進める。
球速帯はキック成立率 EMA が0.80以上かつ成功キックの方向平均誤差 EMA が25°以下なら
進み、成立率が0.50未満または方向誤差が35°以上なら2倍速で戻る。最終帯へ到達した後だけ
``non_kick_ball_touch`` を 500 iteration かけて 0 → -25 へ立ち上げる。
``--reset_noise_std`` は **付けないこと** (蹴り方を壊す)。

見るべきもの
------------
* ``Metrics/kick_direction/kick_rate`` … 観測の意味変更へ適応する間の成立率を監視する。
  旧 checkpoint を使う場合も iter 0 で親と同じ値になる保証はない。
* ``Metrics/kick_direction/kick_vel_ratio`` … 独立した球速追従が指令速度へ収束するか。
* ``Metrics/kick_direction/kick_dir_error_deg`` … 報酬から高精度方向一致を外した後も、
  インサイドフォームによって結果の方向精度が維持・改善するか。
* ``Curriculum/kick_speed_range/kick_dir_error_ema_deg`` … 速度帯ゲートが見る、
  成功キックだけの方向平均誤差 EMA。
* ``Curriculum/kick_speed_range/stage`` … 1は内側接触の獲得、2は方向フォームと球速帯。
* ``Curriculum/kick_speed_range/inside_contact_rate_ema`` … 成功キック中、足内側30°以内で
  接触できた割合のEMA。0.80以上がStage 2への昇格条件。
* ``Curriculum/kick_speed_range/first_touch_rate`` … 終了エピソード中、一度以上ボールへ
  接触した割合。
* ``Curriculum/kick_speed_range/extra_touch_count`` … 終了エピソードあたりの2回目以降の
  接触回数。
* ``Curriculum/kick_speed_range/touch_to_kick_rate`` … 接触した終了エピソードのうち
  ``kick_done``まで到達した割合。
* ``Train/mean_episode_length`` … 履歴は転倒直前の兆候 (角速度の発散) を見せるので、
  転倒が減れば伸びる。
* ``Policy/mean_noise_std`` … 入力層だけが広がった状態からの再学習なので、std が
  暴れないか。暴れるなら learning_rate を下げる。
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_long_pass.walk_long_pass_env_cfg import (
    _LONG_PASS_SPEED_RANGE,
    _LONG_PASS_SPEED_RANGE_START,
    _SPEED_RAMP_END_ITER,
    _SPEED_RAMP_START_ITER,
    K1WalkLongPassEnvCfg,
)
from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _KICK_W_SCALE,
    _SIGMA_DIRECTION,
    K1WalkKickCriticCfg,
    K1WalkKickObservationsCfg,
    K1WalkKickPolicyCfg,
)
from . import inside_rewards
from .observations import (
    BALL_POSITION_NOISE_MAX_M,
    SharedDelayedBallPosition,
)
from .pre_walk_curriculum import (
    pre_walk_initial_yaw_deviation,
    pre_walk_inside_kick_curriculum,
    reset_ball_for_pre_walk_kick,
)

# --------------------------------------------------------------------------- #
# 短期履歴の長さ
#
# 制御周期 = sim.dt * decimation。velocity_env_cfg.py が sim.dt=0.005 / decimation=4
# を設定しているので 0.02 s (50 Hz)。0.1 秒相当は 5 ステップ。
#
# NOTE: 下限 4 は「差分を取るのに最低限必要なフレーム数」。制御周期を 0.04 s 以上に
#       上げると 0.1 秒が 2-3 フレームになり、履歴が実質ただのノイズ平均になる。
# NOTE: この値を変えたら scripts/rsl_rl/expand_checkpoint_history.py の
#       --history-len も揃えること (既定値は 5)。
# --------------------------------------------------------------------------- #
_HISTORY_S = 0.1
_SIM_DT = 0.005
_DECIMATION = 4
_CTRL_DT = _SIM_DT * _DECIMATION  # 0.02 s = 50 Hz
_HISTORY_LEN = max(4, round(_HISTORY_S / _CTRL_DT))  # 5

# PPO runner の num_steps_per_env。カリキュラムの step を iteration へ換算する値なので、
# runner 側を変えた場合はここも同時に変えること。
_STEPS_PER_ITERATION = 48

# インサイドフォームへ配分する nominal weight。post-latch の実 weight は親と同じく
# _KICK_W_SCALE を掛け、2 秒の支払い窓を変えてもキック 1 回の収益を維持する。
_INSIDE_FACE_WEIGHT = 3.0
_STRAIGHT_SWING_WEIGHT = 3.0
_INSIDE_CONTACT_WEIGHT = 3.0
_ANKLE_CONTACT_WEIGHT = 3.0
_VELOCITY_TRACKING_WEIGHT = 5.0
_OPPOSITE_DIRECTION_WEIGHT = -2.0
_INSIDE_CONTACT_MULTIPLIER = 10.0
_INSIDE_FACE_MULTIPLIER = 10.0
_STRAIGHT_SWING_MULTIPLIER = 3.0
# Stage 1 の接触角報酬は、横向き接触からも内側面へ向かう勾配が残る幅にする。
# 昇格判定は下の 30 deg のままなので、Stage 2 条件自体は緩めない。
_INSIDE_CONTACT_REWARD_SIGMA_DEG = 90.0
_INSIDE_CONTACT_ANGLE_DEG = 30.0
# 足collisionの踵はfoot_link原点から約-64 mm。踵から60 mmを狙うためlocal X=-4 mm。
_ANKLE_CONTACT_TARGET_X = -0.004
_ANKLE_CONTACT_SIGMA_X = 0.025
_INSIDE_STAGE_PROMOTE_KICK_RATE = 0.80
_INSIDE_STAGE_PROMOTE_CONTACT_RATE = 0.80
_FIRST_TOUCH_WEIGHT = 100.0  # イベント1回 × dt 0.02 = +2.0
_EXTRA_TOUCH_WEIGHT = -25.0  # イベント1回 × dt 0.02 = -0.5

# 歩行前の静止インサイドキック段階。方向を左右各10°から90°まで広げ、同時に
# ボール位置の前後・左右ジッタも0→4cmへ広げる。最後の90°帯を達成した次段で
# 通常の歩行開始位置（前方半円、0.5-1.5m）へ切り替える。
_PRE_WALK_DIRECTION_HALF_ANGLES_DEG = (10.0, 30.0, 45.0, 60.0, 90.0)
_PRE_WALK_POSITION_JITTER_M = (0.0, 0.01, 0.02, 0.03, 0.04)
_PRE_WALK_SIDE_OFFSET_M = 0.096
_PRE_WALK_INITIAL_YAW_DEVIATION_WEIGHT = -5.0

# 球速帯を進退させるキック成立率と方向平均誤差のヒステリシス。どちらかが中間帯に
# ある間は停止し、成立率が崩れるか方向誤差が広がった場合は進行時の2倍速で戻す。
_SPEED_GATE_ADVANCE_ABOVE = 0.80
_SPEED_GATE_RETREAT_BELOW = 0.50
_SPEED_GATE_ADVANCE_ERROR_BELOW_DEG = 25.0
_SPEED_GATE_RETREAT_ERROR_ABOVE_DEG = 35.0
_SPEED_GATE_RETREAT_SCALE = 2.0

# --------------------------------------------------------------------------- #
# 履歴を付ける policy 観測項 (K1WalkKickPolicyCfg の項名)
#
# 「本体状態 + 自分の出力」だけ。タスク条件付け項 (kick_direction, gait_phase, ball 系)
# は入れない。理由はモジュール docstring の表を参照。
#
# NOTE: 順番は ObservationManager の連結順とは無関係 (連結順は PolicyCfg の宣言順で
#       決まる)。ここは「どの項に付けるか」の集合でしかない。
# NOTE: この集合を変えたら expand_checkpoint_history.py の _POLICY_TERMS も直すこと。
#       あちらは 55 → 223 の列写像を持っているので、片方だけ変えると checkpoint が
#       黙って壊れた並びで読まれる。
# --------------------------------------------------------------------------- #
_HISTORY_TERMS = (
    "projected_gravity",   # 3
    "base_ang_vel",        # 3
    "joint_pos",           # 12
    "joint_vel",           # 12
    "prev_joint_request",  # 12  ← actions (前ステップの目標関節角)
)


def _freeze_fade_in_curricula(cfg, before_iter: int) -> list[str]:
    """``before_iter`` までに終わる報酬 weight のランプだけを終値に固定する。

    継続学習の開始時にキック報酬を 0 へ戻すと、適応期間中に「蹴らない方が得」という
    収支を作ってしまう。一方、球速帯と後段の非キック接触罰は今回の復旧順序に必要なので、
    ``mdp.linear_reward_weight`` かつ早期に終わる項だけを対象にする。

    Returns:
        定数化した curriculum term の名前。
    """
    frozen: list[str] = []
    for name in dir(cfg.curriculum):
        if name.startswith("_"):
            continue
        term = getattr(cfg.curriculum, name, None)
        if term is None or getattr(term, "func", None) is not mdp.linear_reward_weight:
            continue
        params = term.params
        if "end_step" not in params or params["end_step"] > before_iter:
            continue
        if params["start_weight"] == params["end_weight"]:
            continue
        params["start_weight"] = params["end_weight"]
        frozen.append(name)
    return frozen


@configclass
class K1WalkLongPassHistoryPolicyCfg(K1WalkKickPolicyCfg):
    """223 次元 actor 観測の 1 フレーム分の定義。

    ``sole_pos`` は継承元の項順を保つための属性名。値は左足裏ではなく、
    実機 vision の遅延を表す1制御ステップ前のボール3D位置である。
    ``prev_ball_pos`` は同じ遅延・ノイズ標本を共有する。``kick_direction`` は
    自己位置の遅延・位置/yawバイアスを掛けず、報酬と同じ真のコマンド方向を使う。
    """

    sole_pos = ObsTerm(
        func=SharedDelayedBallPosition,
        params={"delay_steps": 1, "dim": 3, "noise_max": BALL_POSITION_NOISE_MAX_M},
    )
    kick_direction = ObsTerm(
        func=mdp.kick_dir_b,
        params={"command_name": "kick_direction"},
    )
    prev_ball_pos = ObsTerm(
        func=SharedDelayedBallPosition,
        params={"delay_steps": 1, "dim": 2, "noise_max": BALL_POSITION_NOISE_MAX_M},
    )


@configclass
class K1WalkLongPassHistoryCriticCfg(K1WalkKickCriticCfg):
    """61 次元 critic 観測。同じスロットを遅延なしボール位置に置換する。"""

    sole_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        params={"delay_steps": 0, "dim": 3},
    )


@configclass
class K1WalkLongPassHistoryObservationsCfg(K1WalkKickObservationsCfg):
    policy: K1WalkLongPassHistoryPolicyCfg = K1WalkLongPassHistoryPolicyCfg()
    critic: K1WalkLongPassHistoryCriticCfg = K1WalkLongPassHistoryCriticCfg()


@configclass
class K1WalkLongPassHistoryEnvCfg(K1WalkLongPassEnvCfg):
    """ロングパス + 短期 I/O 履歴 + ミラー可能な観測。"""

    observations: K1WalkLongPassHistoryObservationsCfg = K1WalkLongPassHistoryObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 0. 歩行前に、現在の立位から左右どちらかの足で蹴る技能を獲得する
        #
        # commandはbase yaw相対で引かれる。ボールをその方向へr_stanceだけ進め、
        # 左右どちらかへスタンス半幅だけずらすので、方向帯を広げても初期姿勢から
        # 直接インサイドキックできる。前段中はbase_velocity指令を0にし、最後の
        # ±90°帯を達成した次段で通常の歩行指令と0.5-1.5m配置へ戻す。
        _initial_half_angle = math.radians(_PRE_WALK_DIRECTION_HALF_ANGLES_DEG[0])
        self.commands.kick_direction.ranges.heading = (-_initial_half_angle, _initial_half_angle)
        self.commands.base_velocity.max_vel = 0.0
        self.commands.base_velocity.max_ang_vel = 0.0
        self.events.reset_ball = EventTerm(
            func=reset_ball_for_pre_walk_kick,
            mode="reset",
            params={
                "command_name": "kick_direction",
                "r_stance": self.terminations.kick_finished.params["r_stance"],
                "side_offset": _PRE_WALK_SIDE_OFFSET_M,
                "longitudinal_jitter": _PRE_WALK_POSITION_JITTER_M[0],
                "lateral_jitter": _PRE_WALK_POSITION_JITTER_M[0],
                "aligned_to_kick_direction": True,
                "walk_dist_range": (0.5, 1.5),
                "walk_half_angle": math.pi / 2,
                "ball_radius": 0.11,
            },
        )
        self.rewards.pre_walk_initial_yaw_deviation = RewTerm(
            func=pre_walk_initial_yaw_deviation,
            weight=_PRE_WALK_INITIAL_YAW_DEVIATION_WEIGHT,
        )

        # -- 1. 最終球方向中心の報酬をインサイドフォーム中心へ置換
        #
        # 旧 kick_direction (nominal 6) は、接触時のインサイド面角度 (3) と
        # 接触直前の足速度方向 (3) に分ける。球速と30度仰角からは _r_direction を外し、
        # 最終球方向は反対半球だけを飽和ペナルティ (-2) で禁止する。
        # 既存の方向誤差 metric は報酬から独立して残るので、結果の精度は引き続き観測できる。
        kick_state_params = dict(self.rewards.kick_direction.params)
        for name in ("sigma_direction", "v_gate_frac", "sigma_gate"):
            kick_state_params.pop(name, None)

        velocity_params = dict(self.rewards.kick_velocity_scaled.params)
        velocity_params.pop("sigma_direction", None)
        elevation_params = dict(self.rewards.kick_elevation.params)
        elevation_params.pop("sigma_direction", None)

        self.rewards.kick_direction = None
        self.curriculum.kick_direction_weight = None

        self.rewards.kick_inside_contact = RewTerm(
            func=inside_rewards.kick_inside_contact,
            weight=_INSIDE_CONTACT_WEIGHT * _INSIDE_CONTACT_MULTIPLIER * _KICK_W_SCALE,
            params={
                **kick_state_params,
                "sigma_contact": math.radians(_INSIDE_CONTACT_REWARD_SIGMA_DEG),
            },
        )

        self.rewards.kick_ankle_contact = RewTerm(
            func=inside_rewards.kick_ankle_contact,
            weight=_ANKLE_CONTACT_WEIGHT * _INSIDE_CONTACT_MULTIPLIER * _KICK_W_SCALE,
            params={
                **kick_state_params,
                "target_x": _ANKLE_CONTACT_TARGET_X,
                "sigma_x": _ANKLE_CONTACT_SIGMA_X,
            },
        )

        self.rewards.first_ball_touch = RewTerm(
            func=inside_rewards.first_ball_touch,
            weight=_FIRST_TOUCH_WEIGHT,
            params={
                **kick_state_params,
                # 接触だけなら +0.5、latch 閾値まで加速できれば +20.0。
                # 接触率は上がったが kick へ変換されない Stage 1 の局所解を崩す。
                "base_fraction": 0.25,
                "speed_bonus_scale": 9.75,
            },
        )

        self.rewards.kick_inside_face_alignment = RewTerm(
            func=inside_rewards.kick_inside_face_alignment,
            weight=0.0,
            params={**kick_state_params, "sigma_angle": _SIGMA_DIRECTION},
        )

        self.rewards.kick_straight_swing = RewTerm(
            func=inside_rewards.kick_straight_swing,
            weight=0.0,
            params={**kick_state_params, "sigma_angle": _SIGMA_DIRECTION},
        )

        self.rewards.kick_velocity_scaled.func = inside_rewards.kick_velocity_independent
        self.rewards.kick_velocity_scaled.params = velocity_params
        self.curriculum.kick_velocity_scaled_weight.params["end_weight"] = (
            _VELOCITY_TRACKING_WEIGHT * _KICK_W_SCALE
        )

        self.rewards.kick_elevation.func = inside_rewards.kick_elevation_independent
        self.rewards.kick_elevation.params = elevation_params

        self.rewards.kick_opposite_direction = RewTerm(
            func=inside_rewards.kick_opposite_direction,
            weight=0.0,
            params=kick_state_params,
        )
        self.curriculum.kick_opposite_direction_weight = None

        # -- 2. 継続学習用の復旧カリキュラム
        #
        # 観測の意味変更へ適応する間は、親の最終速度帯を最初から要求しない。
        # 親の早期報酬 weight は終値に保ち、成立率を見ながら球速帯を進退させる。
        # 最初の接触には定額報酬を払い、2回目以降は弱く罰して、一度の接触で
        # kick_done まで到達するよう誘導する。回り込み中の接触姿勢罰は、従来どおり
        # 最終速度帯へ到達した後にだけ立ち上げる。
        non_kick_touch_params = dict(self.rewards.extra_ball_touch.params)
        non_kick_touch_params["sigma_pose"] = self.rewards.ball_avoidance.params["sigma_pose"]
        self.rewards.extra_ball_touch.weight = _EXTRA_TOUCH_WEIGHT
        self.curriculum.extra_ball_touch_weight = None
        self.rewards.non_kick_ball_touch = RewTerm(
            func=mdp.non_kick_ball_touch,
            weight=0.0,
            params=non_kick_touch_params,
        )
        self.curriculum.non_kick_ball_touch_weight = CurrTerm(
            func=mdp.linear_reward_weight_after_speed_gate,
            params={
                "term_name": "non_kick_ball_touch",
                "start_weight": 0.0,
                # RewardManager は dt=0.02 を掛けるので、最大で1接触あたり-0.5。
                "end_weight": -25.0,
                "command_name": "kick_direction",
                "ramp_iterations": 500,
                "steps_per_iteration": _STEPS_PER_ITERATION,
            },
        )

        _freeze_fade_in_curricula(self, before_iter=_SPEED_RAMP_START_ITER)

        self.curriculum.kick_speed_range = CurrTerm(
            func=inside_rewards.inside_kick_stage_curriculum,
            params={
                "command_name": "kick_direction",
                "inside_contact_term_name": "kick_inside_contact",
                "inside_face_term_name": "kick_inside_face_alignment",
                "straight_swing_term_name": "kick_straight_swing",
                "opposite_direction_term_name": "kick_opposite_direction",
                "inside_contact_weight": (
                    _INSIDE_CONTACT_WEIGHT * _INSIDE_CONTACT_MULTIPLIER * _KICK_W_SCALE
                ),
                "stage2_inside_face_weight": (
                    _INSIDE_FACE_WEIGHT * _INSIDE_FACE_MULTIPLIER * _KICK_W_SCALE
                ),
                "stage2_straight_swing_weight": (
                    _STRAIGHT_SWING_WEIGHT * _STRAIGHT_SWING_MULTIPLIER * _KICK_W_SCALE
                ),
                "stage2_opposite_direction_weight": _OPPOSITE_DIRECTION_WEIGHT * _KICK_W_SCALE,
                "inside_contact_angle_deg": _INSIDE_CONTACT_ANGLE_DEG,
                "promote_kick_rate": _INSIDE_STAGE_PROMOTE_KICK_RATE,
                "promote_inside_contact_rate": _INSIDE_STAGE_PROMOTE_CONTACT_RATE,
                "start_range": _LONG_PASS_SPEED_RANGE_START,
                "end_range": _LONG_PASS_SPEED_RANGE,
                "speed_start_step": _SPEED_RAMP_START_ITER,
                "speed_end_step": _SPEED_RAMP_END_ITER,
                "steps_per_iteration": _STEPS_PER_ITERATION,
                "speed_advance_above": _SPEED_GATE_ADVANCE_ABOVE,
                "speed_retreat_below": _SPEED_GATE_RETREAT_BELOW,
                "speed_advance_error_below_deg": _SPEED_GATE_ADVANCE_ERROR_BELOW_DEG,
                "speed_retreat_error_above_deg": _SPEED_GATE_RETREAT_ERROR_ABOVE_DEG,
                "speed_retreat_scale": _SPEED_GATE_RETREAT_SCALE,
            },
        )

        self.curriculum.pre_walk_inside_kick = CurrTerm(
            func=pre_walk_inside_kick_curriculum,
            params={
                "command_name": "kick_direction",
                "reset_event_name": "reset_ball",
                "direction_half_angles_deg": _PRE_WALK_DIRECTION_HALF_ANGLES_DEG,
                "position_jitter_m": _PRE_WALK_POSITION_JITTER_M,
                "promote_kick_rate": _INSIDE_STAGE_PROMOTE_KICK_RATE,
                "promote_inside_contact_rate": _INSIDE_STAGE_PROMOTE_CONTACT_RATE,
                "promote_direction_error_deg": _SPEED_GATE_ADVANCE_ERROR_BELOW_DEG,
                "inside_contact_angle_deg": _INSIDE_CONTACT_ANGLE_DEG,
            },
        )

        # -- 3. policy 観測の本体状態 5 項に履歴を付ける
        #
        # ObsGroup 側の history_length は **使わない**。グループに設定すると
        # ObservationManager が全項に一括で配ってしまい (observation_manager.py の
        # 「check group history params and override terms」)、kick_direction や
        # ball_vel まで 5 倍になる。項ごとに設定すること。
        #
        # flatten_history_dim=True で (num_envs, H, dim) → (num_envs, H*dim) に潰す。
        # 並びは CircularBuffer.buffer の仕様どおり **古い順 → 最新が末尾**。
        # checkpoint 拡張スクリプトはこの並びに依存している。
        #
        # NOTE: 履歴に積まれるのは **ノイズを載せた後** の値 (ObservationManager は
        #       modifier → noise → clip → scale の後に append する)。フレームごとに
        #       独立なノイズが乗るので、policy は平滑化も学習できる = 狙いどおり。
        # NOTE: エピソード開始直後は CircularBuffer が 0 埋めではなく
        #       **最初の 1 フレームで全スロットを埋める** ので、リセット直後に
        #       「過去 = 0」という嘘の遷移を見せることはない。
        _policy = self.observations.policy
        for _name in _HISTORY_TERMS:
            _term = getattr(_policy, _name)
            _term.history_length = _HISTORY_LEN
            _term.flatten_history_dim = True


@configclass
class K1WalkLongPassHistoryEnvCfg_PLAY(K1WalkLongPassHistoryEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 帯カリキュラムを外して終点に固定する。CurriculumManager は PLAY でも
        # 毎リセットで走り、common_step_counter=0 から始まるので、残すと
        # --kick_speed の指定が上書きされる (K1WalkLongPassEnvCfg_PLAY と同じ理由)。
        # 親が K1WalkLongPassEnvCfg_PLAY ではないので、ここで同じ処理を行う。
        self.curriculum.kick_speed_range = None
        self.curriculum.pre_walk_inside_kick = None
        self.commands.kick_direction.target_speed_range = _LONG_PASS_SPEED_RANGE
        self.commands.kick_direction.ranges.heading = (-math.pi / 2, math.pi / 2)
        self.commands.base_velocity.max_vel = 1.0
        self.commands.base_velocity.max_ang_vel = 1.0
        self.events.reset_ball.params["aligned_to_kick_direction"] = False
        self.rewards.kick_inside_face_alignment.weight = (
            _INSIDE_FACE_WEIGHT * _INSIDE_FACE_MULTIPLIER * _KICK_W_SCALE
        )
        self.rewards.kick_straight_swing.weight = (
            _STRAIGHT_SWING_WEIGHT * _STRAIGHT_SWING_MULTIPLIER * _KICK_W_SCALE
        )
        self.rewards.kick_opposite_direction.weight = _OPPOSITE_DIRECTION_WEIGHT * _KICK_W_SCALE
