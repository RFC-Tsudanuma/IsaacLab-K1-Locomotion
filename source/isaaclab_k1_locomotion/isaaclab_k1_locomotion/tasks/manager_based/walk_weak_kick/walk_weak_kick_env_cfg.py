# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウィークキック環境（弱いキックを指令どおりに出せるようにする / walk_kick の作り直し）。

実機で walk_kick_360 のポリシーが **指令 0.5 m/s でも大きく飛びすぎる**問題への対策。
「もっと弱く蹴れ」という追加学習ではなく、**弱いキックが構造的に学習できなかった原因を
3 つ潰したうえで stage 2 からやり直す**タスク群。

なぜ fine-tune ではなく作り直しなのか
------------------------------------
walk_kick_360 のポリシーは「指令を無視して全力で蹴る」に収束しており、それが下の
3 つの理由で **報酬構造上ほんとうに最適**だった。報酬を直しても、収束済みポリシーは
探索 std が潰れているので弱いキックを再発見できない (walk_long_pass が 2 回踏んだ機序と
同じ)。歩行だけは共通なので、**stage 1 (walk phase) の checkpoint は再利用**して
stage 2 からやり直す。

原因 1: 値 latch が弱いキックで発火しない
    トリガーは ``v_ball > v_thresh`` で v_thresh = 0.8 m/s 固定なのに、指令帯は
    (0.25, 2.0)。**指令の下半分は「指令どおり正しく蹴ると latch しない」**。latch しなければ
    項1-3 は kick_done ゲートで全て 0、``kick_finished`` も発火しない。つまり弱いキックは
    どれだけ正確でも報酬 0 で、「強めに蹴る」が唯一の得点手段になっていた。

原因 2: ``kick_velocity_strong`` が指令を無視して強さを押す
    r_dir × v_ball (生の球速に比例・上限なし、実効 weight 0.9)。指令 0.5 のとき::

        正しく蹴る (v=0.5): scaled 1.2×1.00 + strong 0.9×0.5 ≈ 1.65
        蹴りすぎ  (v=1.5): scaled 1.2×e^-1 + strong 0.9×1.5 ≈ 1.79   ← こちらが得

    オーバーシュートの方が黒字なので、勾配は常に「もっと強く」を向く。

原因 3: ``kick_velocity_scaled`` の σ が太すぎる
    σ=1.0 は指令帯の幅 1.75 に対して太く、指令 0.5 に v=1.5 を出しても f_vel = 0.37 と
    そこそこ点が付く。指令間の報酬差が出ないので「指令を読む」動機が弱い。

対策 (:func:`_apply_weak_kick_recipe` が 3 つまとめて適用する)
-------------------------------------------------------------
a. latch 閾値を指令追従に (``v_thresh_eff = clamp(0.6·v_target, 0.2, 0.8)``)
b. ``kick_velocity_strong`` を折れ線カリキュラムで「立ち上げてから落とす」
c. ``sigma_velocity`` を 1.0 → 0.35 にアニール + 非対称な
   :func:`~..walk_kick.mdp.rewards.kick_velocity_overshoot` をフェードイン

さらに層2として、既存のボール物性 DR (walk_loop_shoot 由来、walk_long_pass_dr と同一範囲)
をそのまま適用する。実機で飛びすぎるもう一つの経路が「sim の接触モデルへの overfit」
なので、速度の再現性を上げるには物性の振れに強くしておく必要がある。

学習手順 (3 段。stage 1 はリポジトリ同梱の checkpoint を再利用)::

    ./scripts/rsl_rl/train_walk_kick_weak.sh

段ごとに手で回す場合は :class:`K1WalkKickWeakEnvCfg` の docstring を参照。
``--reset_noise_std`` は **使わないこと** (理由は walk_mid_kick の docstring と同じ)。
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _apply_walk_state_init,
    _disable_ball_obs_jitter,
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
    K1WalkKick360EnvCfg,
    K1WalkKickEnvCfg,
)

# ボール物性 DR の範囲は walk_loop_shoot の定義をそのまま使う (二重管理を避ける)。
# 範囲の根拠 (MuJoCo と IsaacLab の両方を内包する) はあちらのコメント参照。
# walk_long_pass_dr が使っているものと同一で、**新しい DR は足していない**。
from ..walk_loop_shoot.walk_loop_shoot_env_cfg import (
    _BALL_DYNAMIC_FRICTION_RANGE,
    _BALL_MASS_SCALE_RANGE,
    _BALL_RESTITUTION_RANGE,
    _BALL_STATIC_FRICTION_RANGE,
)

# steps_per_iteration = num_steps_per_env (PPO config)。基底の _spi と同じ値だが、
# あちらは __post_init__ のローカル変数なので参照できずリテラルで持つ。
_SPI = 24

# --------------------------------------------------------------------------- #
# (a) latch 閾値の指令追従
#
#   v_thresh_eff = clamp(0.6 * v_target, min=0.2, max=v_thresh(=0.8))
#
# 指令帯 (0.25, 2.0) に対する実効閾値と、指令どおり蹴ったときの余裕:
#
#   v_target   0.25    0.5     1.0     1.34 以上
#   閾値       0.20    0.30    0.60    0.80 (据え置き)
#   余裕      +0.05   +0.20   +0.40   +0.54 以上
#
# * frac 0.6: 「指令の 6 割出れば蹴ったと認める」。``kick_direction`` の片側速度ゲート
#   (``v_gate_frac``) の既定と同じ考え方で、指令の 6 割に届かないへなちょこ接触を
#   latch させない足切りとして働く。
# * **比例であること (切片つきの一次式にしないこと)**。0.5·v+0.2 の形にすると帯の下端で
#   閾値が指令を追い越し (0.325 > 0.25)、いちばん救いたい最弱の指令だけが救われない。
# * floor 0.2: 実効閾値の下限。接触カウントの閾値 (_TOUCH_DV_THRESH = 0.15 m/s) より
#   **上**なので「触ったが latch しない」領域が残り、歩行中のかすりで誤 latch しない。
#   同時に帯の下端 0.25 より **下**でなければならない (余裕は 0.05 しかない)。
#   指令帯の下端を 0.25 より下げるときは、この床も必ず一緒に下げること。
# * 上限は既存の v_thresh (0.8) のまま。指令 1.34 以上の挙動は一切変わらない。
# --------------------------------------------------------------------------- #
_V_THRESH_TARGET_FRAC = 0.6
_V_THRESH_FLOOR = 0.2

# --------------------------------------------------------------------------- #
# (b) kick_velocity_strong の折れ線スケジュール [(iteration, weight), ...]
#
# 0 から学習し直すので、この項を最初から 0 にすると **そもそもキックが発見されない**
# (弱い接触では latch も報酬も無く、勾配が立たない)。まず立ち上げて「強く蹴る」を
# 獲得させ、獲得できてから 0 へ落として「指令どおりの強さで蹴る」へ移す。
#
#   0 -> 500    : 他のキック報酬と同じランプで立ち上げ (基底の _phase2 と同じ窓)
#   500 -> 1500 : 満額で維持 = キック動作の獲得期間
#   1500 -> 3000: 0 へフェードアウト = 指令追従へ移行する期間
#
# 終値は必ず 0。少しでも残すと「速いほど得」が残り、指令 0.25-0.5 の帯では
# その分だけオーバーシュートが黒字になる。
# --------------------------------------------------------------------------- #
_STRONG_W = 3.0 * _KICK_W_SCALE  # 基底 walk_kick のフェードイン終値と同じ (= 0.9)
_STRONG_KNOTS = [(0, 0.0), (500, _STRONG_W), (1500, _STRONG_W), (3000, 0.0)]

# 移行期間 (strong を落としつつ σ を絞り overshoot 罰を入れる) の窓。
_SHIFT_START_ITER = 1500
_SHIFT_END_ITER = 3000

# --------------------------------------------------------------------------- #
# (c) kick_velocity_scaled の σ アニールと、非対称オーバーシュート罰
#
# σ 1.0 -> 0.35: 指令帯の幅 1.75 に対して 0.35 なら、指令 0.5 と 1.0 が 1.4σ 離れて
# 識別できる。指令 0.5 に v=1.5 を出すと f_vel = exp(−(1.0/0.35)²) ≈ 0.0003 とほぼ 0 に
# なるので、「太い σ に隠れて蹴りすぎる」が効かなくなる。
# 最初から絞らないのは、学習初期の下手なキックが軒並み 0 点になって勾配が死ぬため。
# 窓は strong のフェードアウトと同じ (500 -> 3000) にして、両者を一体の移行として動かす。
#
# overshoot 罰: 超過分だけを直接罰する非対称項。margin 0.2 は latch の量子化と接触の
# ばらつきぶんの遊び。weight -2.0 (×_KICK_W_SCALE) は他のキック項と同じ土俵の値で、
# 飽和 1.0 m/s と併せて「転んで損切り」が得にならない範囲に収めてある
# (:func:`~..walk_kick.mdp.rewards.kick_velocity_overshoot` の NOTE 参照)。
# フェードインは strong のフェードアウトと同期させる。先に入れるとキック獲得を妨げる。
# --------------------------------------------------------------------------- #
_SIGMA_VELOCITY_START = 1.0
_SIGMA_VELOCITY_END = 0.35
_SIGMA_ANNEAL_START_ITER = 500

_OVERSHOOT_MARGIN = 0.2
_OVERSHOOT_SAT = 1.0
_OVERSHOOT_W = -2.0 * _KICK_W_SCALE


def _apply_weak_kick_recipe(cfg: "K1WalkKickEnvCfg") -> None:
    """弱いキックを学習可能にする 3 点セット + ボール物性 DR を適用する。

    stage 2 (:class:`K1WalkKickWeakEnvCfg`) と stage 3
    (:class:`K1WalkKick360WeakEnvCfg`) で共通の処理をここに集約し、二重定義を避ける。
    観測・コマンド・地形・ボール配置には一切触らないので、観測 55 次元と並びは不変。
    ``__post_init__`` の最後 (基底クラスの設定が全部済んだ後) に呼ぶこと。
    """
    # -- (a) latch 閾値を指令追従に -------------------------------------- #
    #
    # kick_state はステップ単位でキャッシュされ、最初に呼んだ項の値でその step の状態が
    # 確定する。ただし閾値判定を行うのは kick_state 自身だけで、報酬項は結果
    # (kick_done と凍結値) を読むだけなので、報酬項の全てに配る必要はない。
    # 毎ステップ最初に走る kick_finished と、同じ状態を読む base_velocity の 2 か所へ。
    cfg.terminations.kick_finished.params["v_thresh_target_frac"] = _V_THRESH_TARGET_FRAC
    cfg.terminations.kick_finished.params["v_thresh_floor"] = _V_THRESH_FLOOR
    cfg.commands.base_velocity.v_thresh_target_frac = _V_THRESH_TARGET_FRAC
    cfg.commands.base_velocity.v_thresh_floor = _V_THRESH_FLOOR

    # -- (b) kick_velocity_strong を「立ち上げてから落とす」 -------------- #
    #
    # 基底の単調ランプ (linear_reward_weight) を折れ線に差し替える。報酬項自体は残す
    # (weight を折れ線で動かすだけ)。項ごと消してはいけない: 消すとキックが発見されない。
    cfg.curriculum.kick_velocity_strong_weight = CurrTerm(
        func=mdp.piecewise_reward_weight,
        params={
            "term_name": "kick_velocity_strong",
            "knots": _STRONG_KNOTS,
            "steps_per_iteration": _SPI,
        },
    )

    # -- (c1) kick_velocity_scaled の σ を絞る ---------------------------- #
    cfg.curriculum.kick_velocity_scaled_sigma = CurrTerm(
        func=mdp.linear_reward_param,
        params={
            "term_name": "kick_velocity_scaled",
            "param_name": "sigma_velocity",
            "start_value": _SIGMA_VELOCITY_START,
            "end_value": _SIGMA_VELOCITY_END,
            "start_step": _SIGMA_ANNEAL_START_ITER,
            "end_step": _SHIFT_END_ITER,
            "steps_per_iteration": _SPI,
        },
    )

    # -- (c2) 非対称オーバーシュート罰 ------------------------------------ #
    cfg.rewards.kick_velocity_overshoot = RewTerm(
        func=mdp.kick_velocity_overshoot,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "margin": _OVERSHOOT_MARGIN,
            "overshoot_sat": _OVERSHOOT_SAT,
        },
    )
    cfg.curriculum.kick_velocity_overshoot_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_velocity_overshoot",
            "start_weight": 0.0,
            "end_weight": _OVERSHOOT_W,
            "start_step": _SHIFT_START_ITER,
            "end_step": _SHIFT_END_ITER,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 層2. ボール物性のドメインランダマイゼーション --------------------- #
    #
    # walk_long_pass_dr / walk_loop_shoot とまったく同じ範囲。新しい DR は発明していない。
    # 実機で飛距離がばらつくもう一つの経路が「sim の接触モデルへの overfit」なので、
    # 指令どおりの速度を実機で再現したいこのタスクでは必須。
    # mode="startup" なので env ごとに固定値が 1 回だけ割り当てられる (4096 env あれば
    # 分布としては十分)。半径だけは spawn 後に env ごとへ変えられないので 0.11 m 固定。
    cfg.events.ball_physics_material = EventTerm(
        func=loco_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("soccer_ball"),
            "static_friction_range": _BALL_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": _BALL_DYNAMIC_FRICTION_RANGE,
            "restitution_range": _BALL_RESTITUTION_RANGE,
            "num_buckets": 64,
        },
    )
    cfg.events.ball_mass = EventTerm(
        func=loco_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("soccer_ball"),
            "mass_distribution_params": _BALL_MASS_SCALE_RANGE,
            "operation": "scale",
        },
    )


@configclass
class K1WalkKickWeakEnvCfg(K1WalkKickEnvCfg):
    """Stage 2 (weak): 限定レンジで「指令どおりの強さのキック」を獲得する。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickEnvCfg` との差は
    :func:`_apply_weak_kick_recipe` の 3 点セット + ボール物性 DR だけ。観測・コマンド・
    ボール配置・地形は同一なので、**stage 1 (walk phase) の checkpoint をそのまま使える**::

        # stage 2 (リポジトリ同梱の walk phase checkpoint から)
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-Weak-v0 \
            --headless --num_envs 4096 --max_iterations 5000 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt
        # stage 3 (stage 2 の checkpoint から)
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-v0 \
            --headless --num_envs 4096 --max_iterations 5000 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_weak/<run>/model_<N>.pt

    walk phase の checkpoint に 2026-08-03 の run を使うのは、こちらが
    ``knee_close_penalty`` を入れた後の学習で、現在のタスクの報酬構成と整合するため
    (:file:`.gitignore` のコメント参照)。

    **``--max_iterations`` は 3000 以上で回すこと。** カリキュラムが 3000 iteration で
    ようやく終点 (strong=0 / σ=0.35 / overshoot 満額) に着くので、それより短いと
    「まだ強く蹴った方が得」な途中状態で終わる。

    NOTE: ``--reset_noise_std`` は使わないこと。stage 1 からの引き継ぎなので歩行の
          スイングは既に精密で、std を戻すとそれを壊す (walk_mid_kick の NOTE と同じ)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_weak_kick_recipe(self)


@configclass
class K1WalkKickWeakEnvCfg_PLAY(K1WalkKickWeakEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKick360WeakEnvCfg(K1WalkKick360EnvCfg):
    """Stage 3 (weak): 全方位版。stage 2 (weak) の checkpoint から続ける。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKick360EnvCfg` との差は
    :func:`_apply_weak_kick_recipe` だけ。360 版固有の設定 (ボール配置の全方位化・
    ``approach_penalty`` → ``ball_avoidance`` の差し替え・``episode_length_s=15.0``) は
    基底の ``__post_init__`` で済んでおり、レシピはそれらに触れないのでそのまま残る。

    コマンド例は :class:`K1WalkKickWeakEnvCfg` の docstring を参照。

    NOTE: このタスクのカリキュラムは stage 2 と同じ窓 (0/500/1500/3000 iteration) を
          **もう一度 0 から**回す。``--load_pretrained`` は common_step_counter を
          0 のままにするので、strong が再び立ち上がって落ちる。stage 2 で獲得済みの
          キックに対しては「強く蹴る期間」が余計だが、回り込みという新しい行動を
          獲得する段でもあるので、ここでキック報酬を厚くしておく方が安全と判断した
          (360 が walk_kick から fine-tune するときにランプをやり直すのと同じ流儀)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_weak_kick_recipe(self)


@configclass
class K1WalkKick360WeakEnvCfg_PLAY(K1WalkKick360WeakEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKick360WeakNoisyBallEnvCfg(K1WalkKick360WeakEnvCfg):
    """Stage 4 (weak): 知覚ノイズ+遅延つき。stage 3 (weak, 360) の checkpoint から続ける。

    実機で蹴り損ねが出る原因のうち、報酬構造ではなく **観測の質** の側を潰す段。
    :class:`K1WalkKick360WeakEnvCfg` との差は policy のボール位置観測だけで、
    :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` が
    「エピソードごとランダム遅延 0-6 ステップ (0-120ms) + 30Hz サンプル&ホールド +
    フレーム同期ガウスジッタ σ=6.7cm・クリップ ±20cm」に差し替える (詳細はあちらの docstring)。

    weak のレシピ (latch 閾値の指令追従・strong の折れ線・σ アニール・overshoot 罰・
    ボール物性 DR) は基底の ``__post_init__`` で全て済んでおり、観測差し替えは
    報酬にもコマンドにも触れないのでそのまま残る::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Noisy-Ball-v0 \
            --headless --num_envs 4096 --max_iterations 3000 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_360_weak/<run>/model_<N>.pt

    NOTE: 基底のカリキュラム (0/500/1500/3000 iteration) は ``--load_pretrained`` では
          **もう一度 0 から**回る (stage 3 が stage 2 から続けるときと同じ)。獲得済みの
          「指令どおりの強さ」に対しては strong の再ランプが余計なので、この段は
          ノイズ耐性の獲得だけが目的であることを踏まえて短めに回すのでもよい。
          ただし途中で止めると strong が満額のまま終わる窓 (500-1500) があるので、
          止めるなら 500 以前か 3000 以降にすること。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_noisy_ball_obs(self)


@configclass
class K1WalkKick360WeakNoisyBallEnvCfg_PLAY(K1WalkKick360WeakNoisyBallEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)


@configclass
class K1WalkKick360WeakNoisyBallWalkInitEnvCfg(K1WalkKick360WeakNoisyBallEnvCfg):
    """Stage 5 (weak, 知覚ノイズ): 歩行ポリシーの歩行状態から reset して再学習する。

    :class:`K1WalkKick360WeakNoisyBallEnvCfg` との差は **リセットの初期状態だけ**。
    報酬・コマンド・カリキュラム・ボール観測のノイズはすべて基底のまま触らない。

    狙いは実機の walk (model_34995.pt) → walk_kick 切り替えで姿勢が飛ぶのを消すこと。
    立ち姿勢ではなく walk ポリシーが実際に作る歩行状態 (高さ・roll/pitch・base 速度・
    脚関節・歩行位相・前ステップの関節指令) から始めるので、切り替え直後の状態が
    学習分布の中に入る。詳細は :func:`~..walk_kick.walk_kick_env_cfg._apply_walk_state_init`。

    状態プール (model_34995.pt を学習したブランチで作る)::

        _labpython2 scripts/rsl_rl/record_walk_states.py \
            --checkpoint checkpoint/model_34995.pt --headless \
            --num_envs 128 --record_time 60 --out walk_states.npz

    再学習 (k1_walk_kick_360_weak_noisy_ball の checkpoint から続ける)::

        K1_WALK_STATES_NPZ=/abs/path/walk_states.npz \
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Noisy-Ball-Walk-Init-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_360_weak_noisy_ball/<run>/model_<N>.pt

    観測は 55 次元・並びとも同一なので checkpoint はそのまま載る。

    NOTE: 基底のカリキュラム (0/500/1500/3000 iteration) は ``--load_pretrained`` では
          また 0 から回る。ここでの目的は初期状態への適応だけなので、止めどころの
          注意は :class:`K1WalkKick360WeakNoisyBallEnvCfg` の NOTE と同じ。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_walk_state_init(self)


@configclass
class K1WalkKick360WeakNoisyBallWalkInitEnvCfg_PLAY(K1WalkKick360WeakNoisyBallWalkInitEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)
