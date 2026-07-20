# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークパス環境（弱いキック専用ポリシー）。

:class:`K1WalkKickEnvCfg` を継承し、低速の目標ボール速度 (0.5-1.5 m/s) だけを
扱うように再設定したもの。強キック用の Walk-Kick と 2 本立てで運用し、実践では
状況に応じて使い分ける前提。

観測・行動空間・シーンは Walk-Kick と完全に同一 (policy 55 次元) なので、
walk phase の checkpoint (``k1_walk_kick_walk_phase``) をそのまま引き継げる。
パス専用の walk phase は不要（歩行の学習内容は蹴りの強弱に依存しないため）::

    # stage 1: 歩行のみ（Walk-Kick と共用。すでに学習済みならスキップ）
    _labpython2 scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0 \
        --headless --num_envs 4096
    # stage 2: パス（リポジトリルートで実行すること。train.py は logs/ を CWD 基準で作る）
    _labpython2 scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-K1-Walk-Pass-v0 \
        --headless --num_envs 4096 \
        --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase/<run>/model_<N>.pt

--resume ではなく --load_pretrained を使うこと。--resume は現在の experiment_name
(k1_walk_pass) 配下から checkpoint を探すので walk phase の run を拾えず、さらに
common_step_counter を stage 1 の到達 iteration に同期させるため、キック報酬の
カリキュラム (0 → 500 iteration でフェードイン) が「もう終わった」と判定されて
一切ランプしなくなる。--load_pretrained は形の合うテンソルだけをロードする。

Walk-Kick からの変更点は 5 つ。いずれも「弱いキックを学習可能にする」ための再設定。

報酬関数側 (walk_kick.mdp) への追加は :func:`~..walk_kick.mdp.rewards.kick_direction` の
``v_gate_frac`` (項1 の片側速度ゲート) のみで、デフォルト 0.0 = 無効なので
Walk-Kick / Walk-Loop の挙動は変わらない。これは v_thresh を下げただけでは
「接近中に足がかすっただけで latch が成立し、そのままエピソードが終わる」状態に
収束してしまったため（詳細は _PASS_V_THRESH のコメント参照）。
"""

from isaaclab.utils import configclass

from ..walk_kick.walk_kick_env_cfg import _KICK_W_SCALE, K1WalkKickEnvCfg

# --------------------------------------------------------------------------- #
# パス用の目標ボール速度レンジ [m/s]
#
# Walk-Kick の (1.0, 4.0) と意図的に重ねていない。重ねると 2 方策に分けた意味
# （それぞれが狭い速度域に集中できる）が薄れるため。
# --------------------------------------------------------------------------- #
_PASS_SPEED_RANGE = (0.5, 0.5)

# --------------------------------------------------------------------------- #
# 値 latch のトリガー速度 [m/s]
#
# Walk-Kick は 0.8。これは _PASS_SPEED_RANGE の下限 (0.5) を上回っているので、
# そのままだと 0.5 m/s のパスでは kick_state の latch が永久に発火せず、
# 項1-3 の報酬が全て 0 のまま・kick_finished も発火せず time_out まで走る。
#
# 当初この値は 0.25 (レンジ下限の半分) だったが、それが原因で 3000 iteration 回しても
# 「足でちょこちょこ触れるだけでキックにならない」状態に収束した。機序:
#
#   1. approach_penalty が足裏をボールに近づける圧力をかけるので、接近中の接触は不可避
#   2. P_kick はボール後方 (kick_dir 上) なので、ロボットは必ず真後ろから寄る
#      → かすり当ては「ほぼ狙った方向」に転がり、方向系のゲートでは弾けない
#   3. v_ball が 0.25 を超えた瞬間に latch が成立し、値が不可逆に凍結される
#   4. その 100 ステップ後に kick_finished がエピソードを終了させる
#
#   → 振り足を出す前にエピソードが終わるため、「ちゃんと蹴った経験」が一度も
#     サンプリングされない。単なる局所最適ではなく探索そのものが塞がれていた。
#
# 対策として v_target の 80% まで引き上げる。これ未満の接触では latch せず、
# エピソードも終わらないので、ボールが転がったまま蹴り直すチャンスが残る。
#
# NOTE: 上げるぶん「latch が全く発火せず報酬が 0 のまま」に振れるリスクは残る。
#       Metrics/kick_direction/kick_rate が 0 付近に張り付くようなら下げること。
#       ただし下げるときは kick_direction の v_gate_frac (項1 の速度ゲート) を
#       セットで残すこと。片方だけ戻すと上記のかすり当て収束が再発する。
# NOTE: kick_state はステップ単位でキャッシュされ、最初に呼んだ項の v_thresh で
#       その step の状態が確定する。全ての項に同じ値を配る必要があるため、
#       :func:`_apply_v_thresh` でまとめて上書きしている。
# --------------------------------------------------------------------------- #
_PASS_V_THRESH = 0.40

# --------------------------------------------------------------------------- #
# 項1 (kick_direction) の速度ゲート
#
# 項1 は重み最大 (6.0) なのに素の定義では速度非依存なので、かすり当てでも方向さえ
# 合っていれば満点が出てしまう。これがかすり当て収束の最大の報酬源だった。
# g(v) = sigmoid((v_ball − 0.8·v_target) / 0.05) を掛けて、弱すぎる蹴りを削る。
# v_target=0.5 のとき v_ball=0.5 → 0.88 / 0.4 → 0.50 / 0.3 → 0.12。
# 片側ゲートなので蹴りすぎは削らない（それは項2 の仕事）。
# --------------------------------------------------------------------------- #
_PASS_V_GATE_FRAC = 0.8
_PASS_SIGMA_GATE = 0.05

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# Walk-Kick は 1.0。これは 1.0-4.0 の帯には合っているが、パスの帯では粗すぎる。
# 例えば v_target=0.5 のときに 1.5 m/s (3 倍の速さ) で蹴っても
# exp(-((1.5-0.5)/1.0)^2) = 0.37 を貰えてしまい、弱いキックを識別できない。
# 0.3 なら同じケースが exp(-(1.0/0.3)^2) ≈ 1e-5 になり、指令追従が意味を持つ。
#
# 0.3 → 0.15 に絞った。0.3 だとかすり当て (v_ball≈0.4) でも
# exp(-((0.4-0.5)/0.3)^2) = 0.89 と、ほぼ満点が出てしまっていたため。
# 0.15 なら同じ 0.4 が 0.64、0.3 なら 0.17 まで落ちる。
# --------------------------------------------------------------------------- #
_PASS_SIGMA_VELOCITY = 0.15

# kick_state を参照する報酬項（v_thresh を配る対象）
_KICK_STATE_REWARD_TERMS = (
    "kick_direction",
    "kick_velocity_scaled",
    "kick_velocity_strong",
    "walk_speed",
    "approach_penalty",
    "kick_pose_overshoot",
)


def _apply_v_thresh(cfg: "K1WalkPassEnvCfg", v_thresh: float) -> None:
    """kick_state を共有する全ての項に同じ v_thresh を配る。

    1 つでも取りこぼすと、その step で最初に評価された項の値で latch 状態が確定し、
    結果が報酬項の評価順に依存してしまう。
    """
    cfg.commands.base_velocity.v_thresh = v_thresh
    cfg.terminations.kick_finished.params["v_thresh"] = v_thresh
    for _name in _KICK_STATE_REWARD_TERMS:
        _term = getattr(cfg.rewards, _name, None)
        if _term is not None:
            _term.params["v_thresh"] = v_thresh


@configclass
class K1WalkPassEnvCfg(K1WalkKickEnvCfg):
    """弱いキック（パス）専用。Walk-Kick と観測・行動空間は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 目標ボール速度を低速帯に張り替える
        self.commands.kick_direction.target_speed_range = _PASS_SPEED_RANGE

        # -- 2. latch のトリガー速度をレンジ下限より下に下げる（これが無いと報酬が出ない）
        _apply_v_thresh(self, _PASS_V_THRESH)

        # -- 3. 速度シェイピングを低速帯の分解能に合わせる
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _PASS_SIGMA_VELOCITY

        # -- 3b. 項1 に片側速度ゲートを掛けて、かすり当てで満点が出る穴を塞ぐ
        self.rewards.kick_direction.params["v_gate_frac"] = _PASS_V_GATE_FRAC
        self.rewards.kick_direction.params["sigma_gate"] = _PASS_SIGMA_GATE

        # -- 4. kick_velocity_strong を外す
        #
        # r_dir * v_ball（上限なし・速いほど得）は、低速の指令追従と真正面から
        # 衝突する。Walk-Kick では kick_velocity_scaled と 3:4 で綱引きさせて
        # いるが、v_target=0.5 の帯では strong 側が勝ってしまい、必ず蹴りすぎる。
        # 2 方策に分けた最大の利点がここで、パス側は素直に項ごと落とせる。
        self.rewards.kick_velocity_strong = None
        self.curriculum.kick_velocity_strong_weight = None

        # strong を落としたぶん、速度追従の取り分を厚くする。
        # NOTE: 4.0 -> 7.0 は「方向 (6.0) と同等以上に速度を合わせさせる」という
        #       意図の初期値。学習を見て調整すること。
        self.curriculum.kick_velocity_scaled_weight.params["end_weight"] = 7.0 * _KICK_W_SCALE


@configclass
class K1WalkPassEnvCfg_PLAY(K1WalkPassEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
