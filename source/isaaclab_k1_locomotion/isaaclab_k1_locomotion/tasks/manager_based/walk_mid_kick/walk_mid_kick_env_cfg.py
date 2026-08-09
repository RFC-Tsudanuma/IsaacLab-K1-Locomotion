# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークミドルキック環境（5-10 m の転がしキック / walk_kick を弱める側）。

:class:`K1WalkKick360EnvCfg` を継承し、**全力キック (実測 v ≈ 6.0 m/s / 実機 20 m) を
5-10 m 相当の帯まで絞る**タスク。戦略側の要求「5-10 m 飛ぶ・誤差 ±10° のキック」に対する
2 つ目のアプローチで、:mod:`..walk_long_pass` (loop_pass_360 を強くする側) と同じ目的地を
反対側から狙う。どちらが先に要件を満たすか分からないので、両方を並行して残す。

転がり減速 a の較正（実機 2 点）
-------------------------------
飛距離は d = v² / 2a。実機の観測が 2 点あり、速度で 3 倍離れているのに整合する:

======================  ==================  ============  ============
ポリシー                 v_ball [m/s]        実機飛距離     逆算した a
======================  ==================  ============  ============
loop_pass_360           ~1.7-2.0            2 m           0.76
walk_kick_360           ~6.0                20 m          0.91
======================  ==================  ============  ============

(walk_kick_360 の v_ball は run 2026-08-08_16-48-06 の kick_vel_ratio 7.164 から。
walk_kick 系の指令帯は (0.25, 2.0) なので E[1/v_target] = ln(8)/1.75 = 1.188、
6.0 × 1.188 = 7.13 で実測と一致する。)

**a ≈ 0.85 m/s²** を採ると 5 m → 2.9 m/s、10 m → 4.1 m/s。帯は余裕を見て (3.0, 4.5)。

なぜ「下げる」方が安全か
------------------------
最大重みの ``kick_direction`` (6.0) が持つ速度ゲートは **片側** である::

    g(v) = sigmoid((v_ball − v_gate_frac·v_target) / sigma_gate)   弱いと削る / 強すぎても削らない

したがって帯の中央 3.75 で採点すると:

* **下げる側 (このタスク, v=6.0)**: gate = sigmoid((6.0−2.25)/0.3) ≈ **1.00**。
  最大の項が満額のまま残り、失うのは ``kick_velocity_scaled`` (4.0) だけ。
  **キックは常に黒字**なので「蹴らない」へ落ちる経路が構造的に無い。
* 上げる側 (walk_long_pass, v=2.4): gate = sigmoid((2.4−2.25)/0.3) ≈ 0.62、
  帯上限では 0.12 まで落ちる。最大の項が消えるとキックが赤字になり、
  ``kick_finished`` がエピソードを終わらせるぶん「蹴らずに time_out まで歩く」方が
  得になる。walk_long_pass で実際に 2 回この収束を踏んでいる。

物理的にも、6.0 → 4 m/s は既にある動作を絞るだけ (探索不要) なのに対し、
2.4 → 4 m/s はハード壁 (足先速度 ≈ 6.5 m/s) に向かって新しい動作を発見する必要がある。

起点としての walk_kick_360 の質
-------------------------------
======================  ================  ================
metric                  loop_pass_360     walk_kick_360
======================  ================  ================
kick_rate               0.997             0.998
ball_touch_count        1.295             **1.054**
kick_elevation_deg      26.8 (すくい)      **7.9 (フラット)**
======================  ================  ================

``ball_touch_count`` 1.05 は「ほぼ一発で蹴り切る」ことを意味し、±10° 要件に直接効く
(多重接触 = 1 回目でボールが動いてから蹴る = 方向誤差の最大の発生源)。仰角 7.9° も
転がし距離には理想的で、**すくい型を解きほぐす必要がない**。そのため walk_long_pass で
必要だった ``kick_elevation`` の扱いと ``extra_ball_touch`` はどちらも不要。

Walk-Kick-360 からの変更点は 4 つだけ
-------------------------------------
1. ``kick_velocity_strong`` を項ごと撤去。r_dir × v_ball (青天井・速いほど得) が
   残っていると常に全力キックが最適になり、**威力指令が意味を持たない**。
   walk_kick で威力指令が効かないのはこの項が原因。loop_pass がやったのと同じ 1 手。
2. 目標ボール速度帯をカリキュラムで (5.0, 7.0) → (3.0, 4.5) へ降ろす。
   開始点を今のポリシーの出力 (v ≈ 6.0) の上に置くので、**iter 0 から全項が満額**。
3. 項1 に片側速度ゲート (``v_gate_frac``) を掛ける。
4. ``sigma_velocity`` を帯幅に合わせる。

学習手順 (walk_kick_360 の checkpoint から fine-tune)::

    ./scripts/rsl_rl/train_walk_mid_kick.sh

--resume ではなく --load_pretrained を使うこと（理由は walk_pass_env_cfg の docstring 参照）。

NOTE: **``--reset_noise_std`` を使わないこと。** walk_long_pass の 2 回の失敗
      (run 2026-08-09_09-56-21 / 11-03-31) はどちらも std を継承値 0.078 から 0.3 へ
      戻したことが原因で、iter 1-5 の時点で既に kick_rate が 0.12-0.19 まで落ちていた
      (同じ条件で std をいじらなかった loop_pass_360 は同時点で 0.52-0.60)。
      std 0.3 は 12 個の脚関節目標に乗る大きなノイズで、精密なスイングを壊す。
      帯カリキュラムが緩やかなら追加の探索は要らない。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import K1WalkKick360EnvCfg

# --------------------------------------------------------------------------- #
# 目標ボール速度レンジ [m/s]（カリキュラムの終点）
#
# 実機 2 点から較正した a ≈ 0.85 m/s² で d = v²/2a を解くと:
#   3.0 m/s → 5.3 m,  3.75 m/s → 8.3 m,  4.5 m/s → 11.9 m
# 要件の 5-10 m を帯の中に収め、上側にだけ少し余裕を持たせた形。
#
# NOTE: a の不確かさは 0.76-0.91 (実機 2 点の逆算値の幅)。a=0.76 なら 4.5 m/s で
#       13.3 m、a=0.91 なら 11.1 m。下限側 3.0 m/s は 4.9-5.9 m でどちらでも要件を満たす。
# --------------------------------------------------------------------------- #
_MID_KICK_SPEED_RANGE = (3.0, 4.5)

# --------------------------------------------------------------------------- #
# 帯カリキュラムの開始点と区間
#
# 開始点 (5.0, 7.0) は **継承元 walk_kick_360 の実出力 v ≈ 6.0 m/s を挟む帯**。
# ここを起点にすると fine-tune の 1 iteration 目から
#   gate = sigmoid((6.0 − 0.6·6.0)/0.3) ≈ 1.00,  f_vel = exp(−((6.0−6.0)/0.9)²) = 1.00
# となり全項が満額で払われる。「継承元が満点を取れる設定から始める」という原則
# (walk_long_pass の失敗から得た教訓) をここでも守る。
#
# 継承元の帯 (0.25, 2.0) をそのまま開始点にしてはいけない。ポリシーは
# kick_velocity_strong のせいで指令を無視して全力で蹴るので、v=6.0 に対して
# f_vel = exp(−((6.0−1.1)/0.9)²) ≈ 0 となり、速度追従の勾配が最初から死ぬ。
#
# 区間 500 → 3000 iteration:
#   * 500 まで待つのはキック報酬 weight 自体のランプ (0 → 500) と重ねないため。
#   * 2500 iteration かけて 6.0 → 3.75 (中央値) を降りる。
#   * 終点の後に仕上げの余地を残すので ITER は 5000 以上で回すこと。
# --------------------------------------------------------------------------- #
_MID_KICK_SPEED_RANGE_START = (5.0, 7.0)
_SPEED_RAMP_START_ITER = 500
_SPEED_RAMP_END_ITER = 3000

# --------------------------------------------------------------------------- #
# 値 latch のトリガー速度 [m/s]
#
# 継承値は 0.8。帯が (3.0, 4.5) まで降りると、かすり当て (v_ball ≈ 1) でも latch が
# 不可逆に成立して 2 秒後にエピソードが終わる余地が出る (walk_pass が踏んだ機序)。
# 1.5 に上げてもこのタスクでは副作用が無い: 開始時の実蹴りは 6.0 m/s、終点でも 3.0 m/s
# 以上なので、**正当なキックは全て 1.5 を大きく超える**。純粋な安全マージン。
#
# NOTE: kick_state はステップ単位でキャッシュされ、最初に呼んだ項の v_thresh で
#       その step の状態が確定する。全ての項に同じ値を配る必要があるため、
#       :func:`_apply_v_thresh` でまとめて上書きする (walk_pass と同じ手順)。
# --------------------------------------------------------------------------- #
_MID_KICK_V_THRESH = 1.5

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# 継承値 1.0 は帯 (0.25, 2.0) 用。終点の帯幅 1.5 に対して 0.9 なら、帯の両端
# (3.0 vs 4.5) が 1.7σ 離れるので指令の識別が効く。開始帯 (5.0, 7.0) でも端で
# f_vel = 0.29 と勾配が残る。
# --------------------------------------------------------------------------- #
_MID_KICK_SIGMA_VELOCITY = 0.9

# --------------------------------------------------------------------------- #
# 項1 (kick_direction) の片側速度ゲート
#
# 項1 は重み最大 (6.0) なのに素の定義では速度非依存なので、v_thresh をぎりぎり超える
# だけの弱い接触でも方向さえ合っていれば満点が出る。g(v) = sigmoid((v_ball −
# 0.6·v_target) / 0.3) で弱すぎる蹴りを削る。**片側なので蹴りすぎは削らない** ——
# これがこのタスクで「降りる側が安全」である理由そのものなので、両側ゲートにしないこと。
# (蹴りすぎを見るのは kick_velocity_scaled の仕事。)
# --------------------------------------------------------------------------- #
_MID_KICK_V_GATE_FRAC = 0.6
_MID_KICK_SIGMA_GATE = 0.3

# kick_state を参照する報酬項（v_thresh を配る対象）。
# この cfg で None の項 (kick_velocity_strong / approach_penalty) も名前だけ挙げておき、
# getattr の None ガードで飛ばす (親の構成が変わっても取りこぼさないように)。
_KICK_STATE_REWARD_TERMS = (
    "kick_direction",
    "kick_velocity_scaled",
    "kick_velocity_strong",
    "kick_elevation",
    "walk_speed",
    "approach_penalty",
    "ball_avoidance",
    "kick_pose_overshoot",
    "extra_ball_touch",
)


def _apply_v_thresh(cfg: "K1WalkMidKickEnvCfg", v_thresh: float) -> None:
    """kick_state を共有する全ての項に同じ v_thresh を配る。

    1 つでも取りこぼすと、その step で最初に評価された項の値で latch 状態が確定し、
    結果が報酬項の評価順に依存してしまう (walk_pass の同名ヘルパーと同じ)。
    報酬項の構成を変え終えた後に呼ぶこと。
    """
    cfg.commands.base_velocity.v_thresh = v_thresh
    cfg.terminations.kick_finished.params["v_thresh"] = v_thresh
    for _name in _KICK_STATE_REWARD_TERMS:
        _term = getattr(cfg.rewards, _name, None)
        if _term is not None:
            _term.params["v_thresh"] = v_thresh


@configclass
class K1WalkMidKickEnvCfg(K1WalkKick360EnvCfg):
    """中距離キック専用。Walk-Kick-360 と観測・行動空間・蹴り方の報酬は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. kick_velocity_strong を撤去（威力指令を意味あるものにする本体）
        #
        # r_dir * v_ball（水平速度、上限なし・速いほど得）が残っていると、どんな指令でも
        # 全力で蹴るのが最適になる。walk_kick で威力指令が効かない原因はこの項。
        # 報酬項とカリキュラムは必ず対で消すこと (項だけ消すと linear_reward_weight が
        # 存在しない term を触りにいく)。
        self.rewards.kick_velocity_strong = None
        self.curriculum.kick_velocity_strong_weight = None

        # -- 2. 目標ボール速度帯（初期値はカリキュラムの開始点）
        #
        # 終点 _MID_KICK_SPEED_RANGE へはカリキュラム (下の -- 5) が降ろす。
        # ここに終点を直接書くと、継承元の v ≈ 6.0 が f_vel ≈ 0 で採点されて
        # 速度追従の勾配が最初から死ぬ。
        self.commands.kick_direction.target_speed_range = _MID_KICK_SPEED_RANGE_START

        # -- 3. 速度シェイピングを帯幅に合わせる
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _MID_KICK_SIGMA_VELOCITY

        # -- 4. 項1 に片側速度ゲートを掛けて、弱い接触で満点が出る穴を塞ぐ
        self.rewards.kick_direction.params["v_gate_frac"] = _MID_KICK_V_GATE_FRAC
        self.rewards.kick_direction.params["sigma_gate"] = _MID_KICK_SIGMA_GATE

        # -- 5. latch のトリガー速度を上げる（全項に一括配布）
        _apply_v_thresh(self, _MID_KICK_V_THRESH)

        # -- 6. 目標ボール速度の帯を (5.0,7.0) → (3.0,4.5) へ滑らかに降ろす
        #
        # このタスクの成否を決めるカリキュラム。進捗は
        # Curriculum/kick_speed_range/speed_{min,max} で追える。
        self.curriculum.kick_speed_range = CurrTerm(
            func=mdp.linear_command_speed_range,
            params={
                "command_name": "kick_direction",
                "start_range": _MID_KICK_SPEED_RANGE_START,
                "end_range": _MID_KICK_SPEED_RANGE,
                "start_step": _SPEED_RAMP_START_ITER,
                "end_step": _SPEED_RAMP_END_ITER,
                # 基底の _spi (= num_steps_per_env) と同じ値。あちらは __post_init__ の
                # ローカル変数なので参照できず、リテラルで持つ。
                "steps_per_iteration": 24,
            },
        )


@configclass
class K1WalkMidKickEnvCfg_PLAY(K1WalkMidKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # -- 帯カリキュラムを外し、帯を終点に固定する
        #
        # CurriculumManager は PLAY でも毎リセットで走る。残しておくと
        # common_step_counter が 0 から始まるので alpha=0、つまり帯が開始点
        # (5.0, 7.0) に巻き戻され、**--kick_speed の指定も上書きされる**
        # (play.py の _apply_kick_speed は env 生成前に cfg を書くだけなので、
        #  カリキュラムが後から潰してしまう)。
        self.curriculum.kick_speed_range = None
        self.commands.kick_direction.target_speed_range = _MID_KICK_SPEED_RANGE
