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

Walk-Kick からの変更点は 4 つ。いずれも「弱いキックを学習可能にする」ための最小限の
再設定で、報酬関数そのもの (walk_kick.mdp) には手を入れていない。
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
# レンジ下限の半分まで下げて、狙い通りの弱いキックを必ず捕捉できるようにする。
#
# NOTE: 下げるぶん「接近中に足が軽く触れただけで latch する」誤検出は増える。
#       latch は τ_direction / v_ball / p_style を同時凍結する一発勝負なので、
#       誤検出が多いようなら値を上げるか、latch 条件に接触判定を足すこと。
# NOTE: kick_state はステップ単位でキャッシュされ、最初に呼んだ項の v_thresh で
#       その step の状態が確定する。全ての項に同じ値を配る必要があるため、
#       :func:`_apply_v_thresh` でまとめて上書きしている。
# --------------------------------------------------------------------------- #
_PASS_V_THRESH = 0.25

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# Walk-Kick は 1.0。これは 1.0-4.0 の帯には合っているが、パスの帯では粗すぎる。
# 例えば v_target=0.5 のときに 1.5 m/s (3 倍の速さ) で蹴っても
# exp(-((1.5-0.5)/1.0)^2) = 0.37 を貰えてしまい、弱いキックを識別できない。
# 0.3 なら同じケースが exp(-(1.0/0.3)^2) ≈ 1e-5 になり、指令追従が意味を持つ。
# --------------------------------------------------------------------------- #
_PASS_SIGMA_VELOCITY = 0.3

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
