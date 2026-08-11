# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ロングパス + キック検出フラグ環境。

:class:`~..walk_long_pass.walk_long_pass_env_cfg.K1WalkLongPassEnvCfg` の
**継続学習用** バリアント。蹴り方の報酬・コマンド分布・カリキュラムは一切変えず、
以下の 3 つだけを足す:

1. **行動空間 +1 次元** … 「このエピソードで既にボールを蹴ったか」を 0/1 で出力する
   ダミー行動 (:class:`~.mdp.actions.KickFlagAction`)。物理には作用しない。
2. **policy 観測 +1 次元** … 前ステップのフラグ出力 (``prev_kick_flag``)。
3. **critic 観測 +8 次元** … 前ステップのフラグ + latch 状態 (2) + latch 時の凍結値 (5)。
   critic 専用の特権情報。

行動空間 55/12 → 56/13、critic 観測 61 → 69。

なぜフラグを「補助ヘッド」ではなく行動空間に置くのか
----------------------------------------------------
``kick_done`` は latch (0 → 1 の単調フラグ) なので判定に **記憶が要る**。ポリシーは
feedforward MLP なので、latch から 2 秒経った時点の観測 (ボールが遠い・遅い) だけでは
「蹴った後」と「一度も蹴っていない」が区別できず、原理的に表現できない。

行動空間に入れると前ステップの出力が観測として戻ってくるので、この自己ループが
latch を保持する 1 bit のメモリになる。GRU に切り替えるより遥かに安い。
詳細は :mod:`.mdp.actions` の docstring 参照。

なぜ critic に特権情報を足すのか
--------------------------------
post-latch の報酬は latch 時の凍結値だけの関数で、猶予窓 (2 秒 = 100 step) のあいだ
1 ステップも変化しない。ところが critic は latch の有無も凍結値も見えていなかったので、
その 100 ステップぶんの収入を予測できず大きな TD 誤差を出し続けていた。本当に学習させたい
転倒罰 (-100) がその予測誤差ノイズに埋もれる = 猶予窓を 2 秒取った設計
(walk_kick_env_cfg の ``_KICK_WINDOW_S``) が効きにくい。凍結値を渡すと残リターンが
ほぼ解析的に当たるようになり、「転ぶ/転ばない」の差分だけがアドバンテージに残る。
詳細は :func:`.mdp.observations.kick_frozen_values` の docstring 参照。

既存 checkpoint からの引き継ぎ (重要)
--------------------------------------
**``--load_pretrained`` に生の long_pass checkpoint を渡してはいけない。**
[train.py] の実装は「形の合わないテンソルを捨てる」ので、``actor.0.weight`` (入力層) と
``actor.6.weight`` (出力層) が両方ランダム初期化され、隠れ層だけ残った死んだポリシーになる。

代わりに :mod:`scripts.rsl_rl.expand_checkpoint_kick_flag` で checkpoint を
**ゼロパディング** してから渡す::

    python scripts/rsl_rl/expand_checkpoint_kick_flag.py \\
        logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \\
        -o /tmp/long_pass_flag_init.pt

新しい列/行が 0 なので、拡張直後のポリシーは **元と挙動が完全に一致する**
(新しい観測は無視され、フラグ出力は常に bias=0)。そこから fine-tune する。

このタスクは既存 55 次元の並びを 1 bit も動かさない
--------------------------------------------------
パディングで引き継ぐ前提なので、追加する次元は必ず **末尾** に置くこと:

* 行動: ``joint_pos`` (0-11) の後ろに ``kick_flag`` (12)
* policy 観測: 既存 55 次元の後ろに ``prev_kick_flag`` (55)
* critic 観測: 既存 61 次元の後ろに 8 次元

``prev_joint_request`` を ``last_action`` (行動全体) のままにすると 13 次元になって
観測の **途中** (index 47) に挿入され、後続 8 次元がずれる。そのため項ごとのスライスに
切り替えている (:func:`.mdp.observations.prev_action_of`)。

学習手順::

    ./scripts/rsl_rl/train_walk_long_pass_flag.sh

カリキュラムは :func:`~..walk_long_pass.walk_long_pass_env_cfg._freeze_curricula_at_final`
で全部終値に固定してあるので、``common_step_counter`` が 0 でも iter 0 から親タスクの
収束状態で始まる。``--reset_noise_std`` は **付けないこと** (蹴り方を壊す)。

見るべきもの
------------
* ``Metrics/kick_direction/flag_accuracy`` … **まずこれ**。0.5 で閾値を切った単純な
  正解率。1.0 が満点、0.5 付近なら何も学習できていない。
* ``Metrics/kick_direction/kick_rate`` … 0.99 付近を維持するはず。落ちたらフラグ報酬が
  歩行/キックを邪魔している。``kick_flag`` の weight を下げる。
* ``Metrics/kick_direction/flag_pred_final`` … kick_rate と同じ値に近づけば成功。
* ``Metrics/kick_direction/flag_pre_latch_pred`` … 誤検出。0 に近いほど良い。
* ``Metrics/kick_direction/flag_err_mean`` … 確率の絶対誤差 (正解率とは別物で、
  当たっていても確率が鈍いと大きくなる)。定数出力の下限 (≈ 正例の割合) に
  張り付いたら「常に 0」に潰れているサイン → ``pos_weight`` を上げる。
* ``Policy/mean_noise_std`` / ``Loss/learning_rate`` … フラグ次元が Gaussian に
  混ざるので、std が暴れないか・adaptive LR が荒れないかを見る。
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..locomotion.velocity_env_cfg import ActionsCfg
from ..walk_kick.walk_kick_env_cfg import (
    _KICK_STATE_PARAMS,
    K1WalkKickCriticCfg,
    K1WalkKickObservationsCfg,
    K1WalkKickPolicyCfg,
)
from ..walk_long_pass.walk_long_pass_env_cfg import (
    _LONG_PASS_SPEED_RANGE,
    K1WalkLongPassEnvCfg,
    _freeze_curricula_at_final,
)
from . import mdp

# --------------------------------------------------------------------------- #
# フラグ報酬のクラス不均衡補正
#
# 正例 (post-latch) は猶予窓の 100 step だけ、負例 (pre-latch) はそれ以前の全部。
# 出発点となる long_pass run (2026-08-09_11-03-31) の実測から:
#
#   Train/mean_episode_length         223.58 step
#   Episode_Termination/kick_finished 0.9935   ← ほぼ全エピソードが窓を使い切る
#   → 正例 100 step / 負例 123.58 step / 正例の割合 p = 0.447
#
# 無情報なポリシーが落ち着く定数出力は p* = w·n_pos / (w·n_pos + n_neg)。
# **これを 0.5 に合わせる** のが狙い。0.5 は sigmoid の勾配が最大で、正例側にも
# 負例側にも同じだけ動ける点なので、そこを出発点にすると学習が最も速い。
#
#   w = n_neg / n_pos = 123.58 / 100 = 1.24  →  p* = 0.501
#
# w=1.0 だと p*=0.447 で負例側に、w=2.0 だと p*=0.618 で正例側 (偽陽性) に偏る。
#
# NOTE: この値は **エピソード長に依存する**。速度帯・episode_length_s・ボール距離を
#       変えて mean_episode_length が動いたら w = (L - 100) / 100 で計算し直すこと
#       (L = Train/mean_episode_length [step]、100 = kick_finished の delay_steps)。
# --------------------------------------------------------------------------- #
_FLAG_POS_WEIGHT = 1.25

# --------------------------------------------------------------------------- #
# フラグ報酬の重み
#
# RewardManager は value = func * weight * dt (dt=0.02) で払う。「常に 0.5 を出す
# (無情報)」状態のエピソード損失は上の実測値を使って
#
#   0.25 * (n_neg + w·n_pos) * dt * weight = 0.25 * (123.58 + 125) * 0.02 * 1.0 = 1.24
#
# つまり「フラグを完璧に当てられるようになる価値 ≈ +1.24 / episode」。同じ run の
# 実測と比べると:
#
#   Train/mean_reward                     +11.91   → 約 10%
#   キック 3 項の合計 (dir+elev+vel)       + 6.97   → 約 18%
#   termination_penalty (-100 * dt)       - 2.00 の一撃
#
# 学習信号としては十分あり、収束済みの蹴り方を押しのけるほどではない。
# kick_rate が落ちたら真っ先にここを下げること (0.5 なら影響は半分)。
# --------------------------------------------------------------------------- #
_FLAG_REWARD_WEIGHT = 1.0


# --------------------------------------------------------------------------- #
# Actions: 既存 12 関節の **後ろ** にフラグを 1 次元足す
#
# ActionManager は cfg.__dict__ の順に項を並べる。dataclass の継承では親のフィールドが
# 先に来るので、joint_pos が 0-11、kick_flag が 12 になる。この並びが崩れると
# checkpoint のパディングが合わなくなるので、ここに項を足すときは必ず末尾に置くこと。
# --------------------------------------------------------------------------- #
@configclass
class K1KickFlagActionsCfg(ActionsCfg):
    kick_flag = mdp.KickFlagActionCfg(asset_name="robot")


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFlagPolicyCfg(K1WalkKickPolicyCfg):
    """既存 55 次元 + 前ステップのフラグ = 56 次元。

    ``prev_joint_request`` (index 35-46) の中身は :meth:`K1WalkLongPassFlagEnvCfg.__post_init__`
    で ``prev_action_of("joint_pos")`` に差し替える。12 次元のまま位置も変わらない。
    """

    # 12. Previous Kick Flag (1) — 自分の前ステップの出力。latch を保持する記憶になる。
    prev_kick_flag = ObsTerm(func=mdp.prev_action_of, params={"action_name": "kick_flag"})


@configclass
class K1WalkLongPassFlagCriticCfg(K1WalkKickCriticCfg):
    """既存 61 次元 + フラグ (1) + latch 状態 (2) + 凍結値 (5) = 69 次元。

    ``v_thresh`` / ``delay_steps`` / ``v_scale`` は親タスクが決めた値を
    :meth:`K1WalkLongPassFlagEnvCfg.__post_init__` が配り直す (long_pass は v_thresh を
    1.5 に上げ、速度帯をカリキュラムで動かしている)。ここに書いてあるのは既定値。
    """

    prev_kick_flag = ObsTerm(func=mdp.prev_action_of, params={"action_name": "kick_flag"})
    kick_latch = ObsTerm(
        func=mdp.kick_latch_state,
        params={**_KICK_STATE_PARAMS, "delay_steps": 100},
    )
    kick_frozen = ObsTerm(
        func=mdp.kick_frozen_values,
        params={**_KICK_STATE_PARAMS, "v_scale": _LONG_PASS_SPEED_RANGE[1]},
    )


@configclass
class K1WalkLongPassFlagObservationsCfg(K1WalkKickObservationsCfg):
    policy: K1WalkLongPassFlagPolicyCfg = K1WalkLongPassFlagPolicyCfg()
    critic: K1WalkLongPassFlagCriticCfg = K1WalkLongPassFlagCriticCfg()


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFlagEnvCfg(K1WalkLongPassEnvCfg):
    """ロングパス + キック検出フラグ。蹴り方そのものは long_pass と同一。"""

    actions: K1KickFlagActionsCfg = K1KickFlagActionsCfg()
    observations: K1WalkLongPassFlagObservationsCfg = K1WalkLongPassFlagObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. カリキュラムを終値で固定する（継続学習タスクなので必須）
        #
        # --load_pretrained は common_step_counter を 0 のままにするので、放置すると
        # キック報酬の weight も速度帯も親のランプをやり直してしまう。理由の詳細は
        # _freeze_curricula_at_final の docstring 参照。
        _freeze_curricula_at_final(self)

        # -- 2. prev_joint_request を「joint_pos 項だけのスライス」に差し替える
        #
        # last_action (引数なし) は行動全体を返すので、行動が 13 次元になると
        # 観測の index 47 に 1 次元挿入されて後続がずれる。既存 55 次元の並びを
        # 保つために項ごとのスライスへ切り替える。値そのものは変わらない。
        for _group in (self.observations.policy, self.observations.critic):
            _group.prev_joint_request.func = mdp.prev_action_of
            _group.prev_joint_request.params = {"action_name": "joint_pos"}

        # -- 3. kick_state のパラメータを critic の新しい観測項にも配る
        #
        # long_pass は v_thresh を 0.8 → 1.5 に上げている (_apply_v_thresh)。あちらは
        # 報酬・termination・command しか見ないので、観測項にはここで配り直す。
        #
        # 実際には TerminationManager が ObservationManager より先に走るため、観測が
        # kick_state の最初の呼び出し元になることは無い (= キャッシュを読むだけで
        # v_thresh は使われない)。それでも揃えるのは「kick_state を参照する全ての項に
        # 同じ値を渡す」という walk_kick の規約を破らないため。
        _v_thresh = self.terminations.kick_finished.params["v_thresh"]
        _delay_steps = self.terminations.kick_finished.params["delay_steps"]
        self.observations.critic.kick_latch.params["v_thresh"] = _v_thresh
        self.observations.critic.kick_latch.params["delay_steps"] = _delay_steps
        self.observations.critic.kick_frozen.params["v_thresh"] = _v_thresh

        # -- 3b. 凍結ボール速度の正規化スケールを「目標速度帯の上限」から取る
        #
        # 帯はカリキュラムが動かすので、cfg 時点の target_speed_range は **開始点**
        # (2.0, 3.0) でしかない。実際に到達する上限はカリキュラムの end_range 側なので
        # そちらを優先して読む (上の _freeze_curricula_at_final で start=end に潰した後
        # なので、どちらを読んでも同じ値になる)。帯を張り替えてもここが自動で追従する。
        _speed_curr = getattr(self.curriculum, "kick_speed_range", None)
        if _speed_curr is not None:
            _v_scale = _speed_curr.params["end_range"][1]
        else:
            _v_scale = self.commands.kick_direction.target_speed_range[1]
        self.observations.critic.kick_frozen.params["v_scale"] = _v_scale

        # -- 4. フラグのメトリクスを足す (class_type の差し替えだけ)
        self.commands.kick_direction.class_type = mdp.KickFlagDirectionCommand

        # -- 5. フラグ報酬
        #
        # カリキュラムは付けない。_freeze_curricula_at_final が start=end に潰すので
        # ランプが成立しないうえ、拡張直後の出力は sigmoid(0)=0.5 の一定値で、
        # 学習前でも報酬は -0.25 * weight * dt に留まる (立ち上げを段階化する必要が無い)。
        self.rewards.kick_flag = RewTerm(
            func=mdp.kick_flag,
            weight=_FLAG_REWARD_WEIGHT,
            params={
                **_KICK_STATE_PARAMS,
                "v_thresh": _v_thresh,
                "action_name": "kick_flag",
                "pos_weight": _FLAG_POS_WEIGHT,
            },
        )


@configclass
class K1WalkLongPassFlagEnvCfg_PLAY(K1WalkLongPassFlagEnvCfg):
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
        self.curriculum.kick_speed_range = None
        self.commands.kick_direction.target_speed_range = _LONG_PASS_SPEED_RANGE
