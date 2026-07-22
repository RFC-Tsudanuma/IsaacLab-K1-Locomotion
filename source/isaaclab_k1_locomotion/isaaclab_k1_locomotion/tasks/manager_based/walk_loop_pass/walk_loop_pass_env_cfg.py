# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークループパス環境（軽く浮かせるパス専用ポリシー）。

:class:`K1WalkKickEnvCfg` を継承し、「狙った方位へ、指定の速さで、一定の仰角をつけて
飛ばす」タスクに再設定したもの。

もともとは「ループシュート」を狙って作った設定だが、実機で試したところ **強く飛ばす
シュートではなく、軽く浮かせるパスとして機能した**（実機では射出仰角が sim より大きく
目減りし、浮きが小さくなるため）。その挙動自体は有用なのでこの設定はそのまま残し、
本命のループシュートは :mod:`..walk_loop_shoot` で別タスクとして扱う。

したがって現在の 4 本立ては:

* Walk-Kick       : 強い地面蹴り
* Walk-Pass       : 弱い地面パス
* Walk-Loop-Pass  : 軽く浮かせるパス ← **これ**
* Walk-Loop-Shoot : はっきり浮かせるシュート

観測・行動空間・シーンは Walk-Kick と完全に同一 (policy 55 次元) なので、
walk phase の checkpoint (``k1_walk_kick_walk_phase``) をそのまま引き継げる::

    # stage 1: 歩行のみ（Walk-Kick と共用。すでに学習済みならスキップ）
    _labpython2 scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0 \
        --headless --num_envs 4096
    # stage 2: ループシュート（リポジトリルートで実行すること）
    _labpython2 scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0 \
        --headless --num_envs 4096 \
        --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase/<run>/model_<N>.pt

--resume ではなく --load_pretrained を使うこと（理由は walk_pass_env_cfg の docstring 参照）。

ボールの仰角は **コマンド化していない**。φ_target は固定で、専用ポリシーが常にループ
シュートを打つ。policy が飛翔中のボールの vz を観測しても、それが分かるのは蹴った後
（kick_finished の 2 秒窓）だけでスイング生成には使えないため、観測を 3D 化する価値が
無い。観測 55 次元を保つことで walk phase の checkpoint 再利用が効く方を取っている。

報酬関数そのもの (walk_kick.mdp) への追加は :func:`~...walk_kick.mdp.rewards.kick_elevation`
の 1 項のみ。Walk-Kick / Walk-Pass の挙動は一切変えていない。

NOTE: **この設定は実機で動作確認済みの「効いている」構成なので、安易に触らないこと。**
      仰角を上げる方向の変更は :mod:`..walk_loop_shoot` 側で行う。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
    _SIGMA_DIRECTION,
    K1WalkKickEnvCfg,
)

# --------------------------------------------------------------------------- #
# ループシュートの目標ボール速度 [m/s]（3D ノルム基準）
#
# Walk-Kick の (1.0, 4.0) / Walk-Pass の (0.5, 0.5) と帯を分ける。ループは「浮かせる」
# ことが目的なので、地面蹴りほどの最高速は要らない一方、仰角 30° をつけて前に飛ばすには
# それなりの初速が要る（v=2.0, φ=30° で頂点 ≈ 0.05m + 転がり出し、v=3.0 で ≈ 0.11m）。
# --------------------------------------------------------------------------- #
_LOOP_SPEED_RANGE = (2.0, 3.0)

# --------------------------------------------------------------------------- #
# 目標仰角 φ_target [rad] と、そのシェイピング係数 σ_φ [rad]
#
# 0.52 rad ≈ 30°。σ_φ = 0.25 rad ≈ 14° なので、±14° で報酬が 1/e、±28° でほぼ 0 になる。
# 30° を選んだのは飛距離が最大になる角度（45°）よりも低めに倒しておきたいため。実機の
# 足で 45° を出すのは物理的にかなり厳しく、届かない目標を置くと f(φ) がずっと 0 付近で
# 勾配が死ぬ。まず 30° で浮かせられることを確認してから上げること。
# --------------------------------------------------------------------------- #
_LOOP_PHI_TARGET = 0.52
_LOOP_SIGMA_PHI = 0.25

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# Walk-Kick は 1.0 (帯 1.0-4.0)。ループの帯は 2.0-3.0 と狭いので少し絞る。
# --------------------------------------------------------------------------- #
_LOOP_SIGMA_VELOCITY = 0.7


@configclass
class K1WalkLoopPassEnvCfg(K1WalkKickEnvCfg):
    """ループシュート専用。Walk-Kick と観測・行動空間は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 目標ボール速度をループ用の帯に張り替える
        self.commands.kick_direction.target_speed_range = _LOOP_SPEED_RANGE

        # -- 2. 速度の測り方を 3D ノルムに切り替える
        #
        # 仰角 30° で蹴ると水平成分は 3D ノルムの cos30° = 0.87 倍しかない。水平で測った
        # ままだと「指令速度に届いていない」と誤判定され、kick_elevation（浮かせろ）と
        # kick_velocity_scaled（もっと強く）が恒常的に綱引きしてしまう。
        self.rewards.kick_velocity_scaled.params["use_3d_speed"] = True
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _LOOP_SIGMA_VELOCITY

        # -- 3. kick_velocity_strong を外す
        #
        # r_dir * v_ball（水平速度、上限なし・速いほど得）は仰角をつける動機と真正面から
        # 衝突する。同じ運動エネルギーなら水平に蹴った方が必ず得になるので、この項が
        # 残っているとループシュートは学習されない。walk_pass と同じく項ごと落とす。
        self.rewards.kick_velocity_strong = None
        self.curriculum.kick_velocity_strong_weight = None

        # -- 4. 項7: ループシュート報酬を追加
        self.rewards.kick_elevation = RewTerm(
            func=mdp.kick_elevation,
            weight=0.0,
            params={
                **_KICK_STATE_PARAMS,
                "sigma_direction": _SIGMA_DIRECTION,
                "phi_target": _LOOP_PHI_TARGET,
                "sigma_phi": _LOOP_SIGMA_PHI,
            },
        )

        # 他のキック報酬と同じランプ (0 → 500 iteration) に載せる。
        # end_weight 5.0 は「方向 (6.0) に次ぐ重み、速度追従 (4.0) より上」という意図の
        # 初期値。strong を落として空いた 3.0 分がここに回っている形。学習を見て調整すること。
        _phase2 = {"start_step": 0, "end_step": 500, "steps_per_iteration": 24}
        self.curriculum.kick_elevation_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "kick_elevation",
                "start_weight": 0.0,
                "end_weight": 5.0 * _KICK_W_SCALE,
                **_phase2,
            },
        )

        # -- 5. ボールの物理マテリアルを実ボール寄りにする
        #
        # Walk-Kick 系は static_friction のみ指定で restitution は既定の 0.0。地面蹴りでは
        # 問題にならないが、ループでは着地後の跳ね方が全く変わるうえ、足がボールの下に
        # 潜り込む瞬間の接触も反発ゼロだと「押すだけ」になり浮きにくい。5 号球相当の値を入れる。
        # NOTE: この上書きは walk_loop だけに閉じている（Walk-Kick / Walk-Pass の
        #       ボールは従来どおり restitution=0.0 のまま）。
        self.scene.soccer_ball.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.6,
        )


@configclass
class K1WalkLoopPassEnvCfg_PLAY(K1WalkLoopPassEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkLoopPass360EnvCfg(K1WalkLoopPassEnvCfg):
    """全方位版ループパス。ボール出現・蹴り方向とも 360°、距離 0.5-1.5 m。

    :class:`K1WalkLoopPassEnvCfg` (±60°/±45°, 0.5-0.8 m) を全方位に広げたもの。
    質的に新しいのは「ボールの正面側から半周回り込んで蹴る」エピソードで、
    段階カリキュラムは使わず **loop_pass の checkpoint からの fine-tune を実質
    カリキュラムとする** (蹴り方は既知・回り込みだけ新規)::

        # 既存の loop_pass checkpoint から (推奨):
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Loop-Pass-360-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_loop_pass/<run>/model_<N>.pt
        # ゼロから通しで (walk phase → loop_pass → 360 の 3 段):
        ./scripts/rsl_rl/train_walk_loop_pass_360.sh

    回り込みは誘導 (G の迂回経路) ではなく **報酬の抑止で成立させる** (B-Human 方式)。
    approach_penalty (遠いほど罰 = 接近圧) を ball_avoidance (姿勢ができていないのに
    近いほど罰) に差し替えることで、ボール正面からまっすぐ突っ込む行動が罰され、
    距離を保って回り込む解に寄る。差し替え理由の詳細は
    :func:`~..walk_kick.mdp.rewards.ball_avoidance` の docstring を参照。

    init_side の確定遅延 (kick_state._INIT_SIDE_COMMIT_DIST) は共有コード側の修正で、
    ボール正面スタート (s≈0) の回り込みがコイントスで overshoot 罰されるのを防ぐ。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. サンプリングを全方位・遠距離へ
        self.events.reset_ball.params["half_angle"] = math.pi
        self.events.reset_ball.params["dist_range"] = (0.5, 1.5)
        self.commands.kick_direction.ranges.heading = (-math.pi, math.pi)

        # -- 2. approach_penalty → ball_avoidance
        #
        # approach_penalty は「姿勢が悪くても足をボールに近づけるだけで罰が減る」ので、
        # 正面からの突っ込み (→ 衝突で latch が悪い値で凍結 → エピソード終了) を
        # むしろ後押ししてしまう。ball_avoidance は同じ weight で向きだけ逆。
        self.rewards.approach_penalty = None
        self.curriculum.approach_penalty_weight = None
        self.rewards.ball_avoidance = RewTerm(
            func=mdp.ball_avoidance,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_sole": 0.35, "sigma_pose": 0.3},
        )
        self.curriculum.ball_avoidance_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "ball_avoidance",
                "start_weight": 0.0,
                "end_weight": -3.0,
                "start_step": 0,
                "end_step": 500,
                "steps_per_iteration": 24,
            },
        )

        # -- 3. 移動距離が伸びるぶんエピソードを延長
        #
        # 1.5 m + 半周回り込みで移動 2.5-3 m。学習初期の遅い歩きだと 10 秒では
        # 蹴る前に time_out するエピソードが増える。
        self.episode_length_s = 15.0


@configclass
class K1WalkLoopPass360EnvCfg_PLAY(K1WalkLoopPass360EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
