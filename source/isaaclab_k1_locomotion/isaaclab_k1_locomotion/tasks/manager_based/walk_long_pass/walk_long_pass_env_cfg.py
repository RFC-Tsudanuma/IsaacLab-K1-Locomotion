# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークロングパス環境（5-10 m の強い転がしパス専用ポリシー）。

:class:`K1WalkLoopPass360EnvCfg` を継承し、目標ボール速度の帯を (2.0, 3.0) →
(3.2, 5.0) に張り替えたもの。戦略側の要求「5-10 m 飛ぶ・誤差 ±10° のキック」の
飛距離側を担う。

距離⇔速度の換算は実機較正の 1 点 (指令 2.0 m/s → 実測 2 m ⇒ 転がり減速
a ≈ 1.0 m/s²) に基づく::

    d = v² / 2a  →  3.2 m/s → 5 m,  4.0 m/s → 8 m,  4.5 m/s → 10 m

帯の下限 3.2 は要件下限 5 m 相当。2-4.5 m の弱いパスは loop_pass_360 が引き続き
担い、戦略側が距離帯でポリシーを選ぶ 2 本立てとする（帯を重ねない理由は
walk_pass と同じ: それぞれが狭い速度域に集中できる）。

すくい型のスイングは維持する
----------------------------
kick_elevation (φ_target=30°) は sim で φ≈27° を現役で誘導しており
(loop_pass_360 run の実測: kick_elevation_deg ≈ 26.8°, 報酬 ≈ 0.157)、この
「浮き球報酬による意図的な威力劣化」が実機で威力指令が意味を持つ土台になっている。
実機では仰角が sim より目減りしてほぼ浮かない柔らかいパスになる、という sim2real
特性まで含めて検証済みの動作ファミリーなので、**帯を上げるだけで蹴り方には触らない**:

* kick_elevation (Gaussian 30°) … そのまま
* kick_velocity_scaled の use_3d_speed=True … そのまま（elevation との綱引き防止）
* ball_avoidance / 360° サンプリング / episode_length 15 s … そのまま

5 m/s はすくい型の実証済み射程内 (loop_shoot が同型のスイングで v3d 6.3-6.5 m/s)。
学習後に kick_vel_ratio が帯上限側だけ垂れるようなら、そのとき初めて elevation の
weight を下げる等を検討する。

帯は一段で上げてはいけない（失敗記録）
--------------------------------------
初版は ``target_speed_range`` をいきなり (2.0, 3.0) → (3.2, 5.0) に張り替えていたが、
**1200 iteration で「蹴らずに 15 秒歩き回る」に完全収束した**
(run 2026-08-09_09-56-21: kick_rate 0.997 → 0.037、time_out 0.944、
ball_touch_count 1.30 → 0.91、Episode_Reward/kick_direction 0.213 → 0.001)。

機序は報酬の収支逆転:

* ``kick_finished`` は latch の 2 秒後にエピソードを終わらせるので、キックは常に
  **「残りの歩行報酬 (feet_phase など) を捨てる」コスト**を払っている。
* 継承元の実蹴りは約 2.4 m/s。新しい帯で採点すると、v_gate × f_vel は
  指令 3.2 で 0.38、4.1 で 0.009、**5.0 で ~0**。帯の大半でキック報酬が消える。
* 結果、蹴らない方が得になる (mean_reward はむしろ 12.1 → 21.5 に増えた)。
  探索 std も 0.3 → 0.08 に潰れ、iteration を増やしても抜けられない局所最適。

そこで帯は :func:`~..walk_kick.mdp.curriculums.linear_command_speed_range` で
継承元が満点を取れる (2.0, 3.0) から目標の (3.2, 5.0) へ滑らかに動かす。各時点で
「今のポリシーがぎりぎり届く速度」が指令されるのでキック報酬が払われ続ける。

**この構造は帯を触る限り常について回る。** 帯を再調整するときは、必ず
「継承元の実蹴り速度で採点したときにキック報酬が残るか」を先に検算すること。

学習手順 (loop_pass_360 の checkpoint から fine-tune)::

    ./scripts/rsl_rl/train_walk_long_pass.sh
    # または直接:
    _labpython2 scripts/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-K1-Walk-Long-Pass-v0 \
        --headless --num_envs 4096 \
        --load_pretrained logs/rsl_rl/k1_walk_loop_pass_360/<run>/model_<N>.pt \
        --reset_noise_std 0.3

--resume ではなく --load_pretrained を使うこと（理由は walk_pass_env_cfg の
docstring 参照）。--reset_noise_std は必須に近い: 収束済みポリシーは action std が
潰れていて、4-5 m/s は探索したことのない速度域なので、std を戻さないと慣れた
2-3 m/s の蹴り方に貼り付いたまま抜け出せない。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import _KICK_STATE_PARAMS
from ..walk_loop_pass.walk_loop_pass_env_cfg import K1WalkLoopPass360EnvCfg

# --------------------------------------------------------------------------- #
# ロングパス用の目標ボール速度レンジ [m/s]（3D ノルム基準）
#
# 下限 3.2 = 要件下限 5 m 相当 (a=1.0)。それ未満は loop_pass_360 の担当なので重ねない。
# 上限 5.0 = 10 m + α。ハード壁 (関節速度上限から出る足先速度 ≈ 6.5 m/s、loop_shoot
# 実測) に対して余裕があるので、帯全域が到達可能 = 威力指令が全域で意味を持つ。
#
# NOTE: a=1.0 は 1 点計測。実機で --kick_speed 3.0 を測って a を確定させたら、
#       この帯を締め直すこと (a=0.7 なら 10 m に 3.74 m/s で足りる)。
# NOTE: これは **カリキュラムの終点**。開始時は _LONG_PASS_SPEED_RANGE_START。
#       一段で張り替えると「蹴らない」に収束する (モジュール docstring の失敗記録参照)。
# --------------------------------------------------------------------------- #
_LONG_PASS_SPEED_RANGE = (3.2, 5.0)

# --------------------------------------------------------------------------- #
# 帯カリキュラムの開始点と区間
#
# 開始点は継承元 (loop_pass_360) の帯そのもの。そこでは実蹴り 2.4 m/s に対して
# kick_vel_ratio 0.866 / kick_rate 0.997 が出ているので、fine-tune の 1 iteration 目から
# キック報酬が満額で払われ、「蹴る」行動が維持される。
#
# 区間 500 → 3000 iteration:
#   * 500 まで待つのは、キック報酬 weight 自体のランプ (0 → 500) と重ねないため。
#     weight が立ち上がりきる前に帯まで動かすと、どちらが原因で報酬が動いたのか
#     読めなくなる。
#   * 2500 iteration かけるのは、帯の上限が 3.0 → 5.0 と 1.7 倍に伸びるため。
#     1 iteration あたり 0.0008 m/s ずつで、ポリシーが追従する余裕を取る。
#   * 終点 3000 の後に仕上げの余地を残す想定なので、ITER は 5000 以上で回すこと。
# --------------------------------------------------------------------------- #
_LONG_PASS_SPEED_RANGE_START = (2.0, 3.0)
_SPEED_RAMP_START_ITER = 500
_SPEED_RAMP_END_ITER = 3000

# --------------------------------------------------------------------------- #
# 値 latch のトリガー速度 [m/s]
#
# 継承値 0.8 は帯 (3.2-5.0) に対して低すぎる。接近中に足がかすっただけ (v_ball ≈ 1)
# で latch が不可逆に成立し、2 秒後に kick_finished がエピソードを終わらせるため、
# 「ちゃんと蹴った経験」がサンプリングされなくなる (walk_pass が 3000 iteration
# 潰した機序と同じ)。1.5 ならかすり当てでは latch せず、蹴り直しのチャンスが残る。
#
# 2.5 まで上げないのは fine-tune の出発点への配慮。開始時のポリシー
# (loop_pass_360) の実蹴りは 2.15-2.6 m/s (指令 2.0-3.0 × ratio 0.86) なので、
# それを上回る閾値にすると初期の正当な蹴りが latch せず、報酬信号が消えて
# fine-tune 自体が立ち上がらない。1.5 < 2.15 で初期の蹴りは全て拾える。
#
# NOTE: kick_state はステップ単位でキャッシュされ、最初に呼んだ項の v_thresh で
#       その step の状態が確定する。全ての項に同じ値を配る必要があるため、
#       :func:`_apply_v_thresh` でまとめて上書きしている (walk_pass と同じ手順)。
# --------------------------------------------------------------------------- #
_LONG_PASS_V_THRESH = 1.5

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# 継承値 0.7 は帯幅 1.0 (2.0-3.0) 用。帯幅 1.8 に広げるぶん少し緩めて、帯の端の
# 指令でも勾配が届くようにする。0.9 で帯の両端 (3.2 vs 5.0) は 2σ 離れるので
# 指令の識別は保たれる。
# --------------------------------------------------------------------------- #
_LONG_PASS_SIGMA_VELOCITY = 0.9

# --------------------------------------------------------------------------- #
# 項1 (kick_direction) の片側速度ゲート
#
# 項1 は重み最大 (6.0) なのに素の定義では速度非依存なので、v_thresh (1.5) を
# ぎりぎり超えるだけの弱い蹴りでも方向さえ合っていれば満点が出てしまう。
# g(v) = sigmoid((v_ball − 0.6·v_target) / 0.3) を掛けて弱すぎる蹴りを削る。
# 片側ゲートなので蹴りすぎは削らない (それは項2 の仕事)。
#
# 0.6 は walk_pass の 0.8 より緩い。fine-tune 開始時の実蹴り 2.2-2.6 m/s が
# v_target=3.2 のゲート (1.92) を十分通過できるようにするため
# (0.8 だと 2.56 で、初期の蹴りの大半が半減以下になり立ち上がりが遅れる)。
# sigma_gate 0.3 も同じ理由で walk_pass の 0.05 より緩い (速度スケールが 6-10 倍)。
# --------------------------------------------------------------------------- #
_LONG_PASS_V_GATE_FRAC = 0.6
_LONG_PASS_SIGMA_GATE = 0.3

# kick_state を参照する報酬項（v_thresh を配る対象）。
# この cfg で None の項 (kick_velocity_strong / approach_penalty) も名前だけ挙げて
# おき、getattr の None ガードで飛ばす (親の構成が変わっても取りこぼさないように)。
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


def _apply_v_thresh(cfg: "K1WalkLongPassEnvCfg", v_thresh: float) -> None:
    """kick_state を共有する全ての項に同じ v_thresh を配る。

    1 つでも取りこぼすと、その step で最初に評価された項の値で latch 状態が確定し、
    結果が報酬項の評価順に依存してしまう (walk_pass の同名ヘルパーと同じ)。
    報酬項を追加し終えた後に呼ぶこと。
    """
    cfg.commands.base_velocity.v_thresh = v_thresh
    cfg.terminations.kick_finished.params["v_thresh"] = v_thresh
    for _name in _KICK_STATE_REWARD_TERMS:
        _term = getattr(cfg.rewards, _name, None)
        if _term is not None:
            _term.params["v_thresh"] = v_thresh


@configclass
class K1WalkLongPassEnvCfg(K1WalkLoopPass360EnvCfg):
    """ロングパス専用。Walk-Loop-Pass-360 と観測・行動空間・蹴り方の報酬は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 目標ボール速度をロングパスの帯へ（このタスクの本体）
        #
        # cfg の初期値は **カリキュラムの開始点** (= 継承元の帯) にしておく。
        # 終点 _LONG_PASS_SPEED_RANGE へはカリキュラム (下の -- 6) が動かす。
        # ここに終点を直接書くと、カリキュラムが最初に走るまでの数ステップだけ
        # 帯が飛んでしまう上、意図が読めなくなる。
        self.commands.kick_direction.target_speed_range = _LONG_PASS_SPEED_RANGE_START

        # -- 2. 速度シェイピングを帯幅に合わせて緩める
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _LONG_PASS_SIGMA_VELOCITY

        # -- 3. 項1 に片側速度ゲートを掛けて、弱い蹴りで満点が出る穴を塞ぐ
        self.rewards.kick_direction.params["v_gate_frac"] = _LONG_PASS_V_GATE_FRAC
        self.rewards.kick_direction.params["sigma_gate"] = _LONG_PASS_SIGMA_GATE

        # -- 4. 項8: 2 回目以降のボール接触を罰する
        #
        # v_thresh を 1.5 に上げた副作用で「まず軽く触って、それから蹴る」が
        # 無コストになる (1.5 m/s 未満の接触では latch もエピソード終了も起きない)。
        # walk_pass と同じ対策。1 回目の接触は無料なので、罰を避ける唯一の道は
        # 最初の接触で蹴り切ること。loop_pass_360 の実測 ball_touch_count ≈ 1.29 が
        # 1.0 に寄るかで効果を見る。
        self.rewards.extra_ball_touch = RewTerm(
            func=mdp.extra_ball_touch,
            weight=0.0,
            params={**_KICK_STATE_PARAMS},
        )
        # ランプは帯カリキュラムの **後** (3000 → 3500 iteration)。
        # 初版は 0 → 500 で他のキック報酬と同時に立ち上げていたが、「触ると罰される」
        # 圧力と「蹴っても報酬が出ない」状況が重なると、ポリシーはボールを避ける方向へ
        # まとめて逃げる (実測 ball_touch_count 1.30 → 0.91)。帯が目標まで動いて
        # キックが安定してから、最後に多重接触だけを削るのが正しい順序。
        self.curriculum.extra_ball_touch_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "extra_ball_touch",
                "start_weight": 0.0,
                # イベント項なので value = weight * dt。-50.0 で 1 回あたり -1.0
                # (キック 1 回の収益 ≈ +5 の 2 割)。kick_rate が落ちるようなら
                # 弱めること (ボールに触ること自体を避け始めるサイン)。
                "end_weight": -50.0,
                "start_step": _SPEED_RAMP_END_ITER,
                "end_step": _SPEED_RAMP_END_ITER + 500,
                "steps_per_iteration": 24,
            },
        )

        # -- 5. latch のトリガー速度を帯に合わせて上げる
        # (全項に一括配布。extra_ball_touch を追加した後に呼ぶこと)
        _apply_v_thresh(self, _LONG_PASS_V_THRESH)

        # -- 6. 目標ボール速度の帯を (2.0,3.0) → (3.2,5.0) へ滑らかに動かす
        #
        # このタスクの成否を決めるカリキュラム。一段で張り替えると
        # 「蹴らずに time_out まで歩く」に収束する (モジュール docstring の失敗記録)。
        # 進捗は Curriculum/kick_speed_range/speed_{min,max} で追える。
        self.curriculum.kick_speed_range = CurrTerm(
            func=mdp.linear_command_speed_range,
            params={
                "command_name": "kick_direction",
                "start_range": _LONG_PASS_SPEED_RANGE_START,
                "end_range": _LONG_PASS_SPEED_RANGE,
                "start_step": _SPEED_RAMP_START_ITER,
                "end_step": _SPEED_RAMP_END_ITER,
                "steps_per_iteration": 24,
            },
        )


@configclass
class K1WalkLongPassEnvCfg_PLAY(K1WalkLongPassEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
