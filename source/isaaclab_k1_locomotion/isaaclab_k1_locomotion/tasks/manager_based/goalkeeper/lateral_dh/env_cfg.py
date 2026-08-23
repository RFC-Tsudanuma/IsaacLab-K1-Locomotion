# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""履歴 + CNN 版の横移動下位ポリシー (``Isaac-GKLateralDH-K1-v0``)。

``goalkeeper_lateral_env_cfg.py`` の :class:`K1GKLateralEnvCfg` を土台に、次を変える:

1. 観測を **3 グループ構成** に再編する (``direct`` / ``policy`` / ``critic``)。
   履歴は IsaacLab 標準の ``ObservationGroupCfg.history_length`` で取る。
2. 位相周波数を **指令ベース** (``cmd_gain`` 方式) に変える。
3. **関節ゼロ点オフセットの DR** を追加する。
4. 位相の DR を広げる。

報酬・指令分布・カリキュラムは非 DH 版と同一なので、差分がそのまま構造の効果になる。

☠ **観測構造が変わる = 既存 ckpt からの ``--resume`` は不可能。** from scratch 前提。

---

**なぜ自前のリングバッファをやめたか (2026-08-23)**

最初は履歴を自前で持つ実装にしていたが、``feat/inoue_walk_double_encoder`` を読んで
IsaacLab 標準の ``history_length`` に置き換えた。理由:

* 容量管理・リセット・ステップ二重進行の番人が **全部不要** になる (自前実装のバグ源)。
* ノイズは ObservationManager が **履歴へ push する前に 1 度だけ** 掛けるので、
  「各ステップのノイズがそのまま履歴に固定される」= 実機と同じ性質が自動で得られる。
  自前実装ではここを手で書いていた。

---

**なぜ位相を指令ベースにしたか (2026-08-23)**

☠ 実機の引きずりの原因は **位相観測の学習/推論不一致** だった (学習 3.2〜3.9Hz の
適応位相 / 推論 ``rl_policy_slide_walk_node.cpp`` は固定 1.6Hz)。シム実測:

    位相 1.6Hz固定 : 左横の引きずり率 10.3% / クリアランス下位10% 7.5mm
    指令ベース      : 3.1% / 46.7mm
    実速度ベース    : 2.0% / 46.5mm

``use_actual_speed=True`` は推論側に **真の base_lin_vel** を要求するが、実機の
``LowStateData`` は motors(q,dq) と imu(rpy,gyro,acc) だけで線速度を持たない。つまり
「学習と推論で別の式」になることが構造的に避けられず、事実その不一致で事故が起きた。

``cmd_gain`` 方式なら **学習と推論が同じ式**になり、速度推定も要らない。素の指令
(gain=1.0) だと届かない要求と戦って歩幅が縮む問題があったが (横 1.444→0.941 m/s)、
実測の定常追従率 0.92 を掛ければ実速度基準との誤差は **2〜4%** に収まる。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ...locomotion.rough_env_cfg import K1CriticCfg, K1PolicyCfg
from ..goalkeeper_lateral_env_cfg import K1GKLateralEnvCfg
from ..mdp.observations import zeros_obs
from .history_layout import HISTORY_LENGTH

@configclass
class K1GKLateralDHDirectCfg(ObsGroup):
    """履歴を持たない直接入力。最新の速度指令 + GK タスクスロット (Stage1 では全てゼロ)。

    ☠ 項の並びは :data:`~.history_layout.DIRECT_TERM_SPECS` と一致させること。
    ★ 速度指令は履歴グループにも入っている。ここは「最新値を CNN を通さず直接見る」ため
      の冗長な経路で、指令変化への即応を担う (あちらのブランチと同じ設計)。
    """

    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    ball_pos_rel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_vel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_active = ObsTerm(func=zeros_obs, params={"dim": 1})
    target_y = ObsTerm(func=zeros_obs, params={"dim": 1})
    self_state = ObsTerm(func=zeros_obs, params={"dim": 4})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1GKLateralDHPolicyCfg(K1PolicyCfg):
    """actor 用: 歩行観測 49 次元を HISTORY_LENGTH ステップ分バッファする。

    ☠ GK タスクスロット (ゼロ 10 次元) は **入れない**。定数ゼロを 100 フレーム分
      CNN に流しても情報は増えず、観測正規化に縮退列 (std=0) を大量に作るだけ。
      あれは ``direct`` グループへ移した。
    """

    def __post_init__(self):
        super().__post_init__()
        self.history_length = HISTORY_LENGTH
        self.flatten_history_dim = True


@configclass
class K1GKLateralDHCriticCfg(K1CriticCfg):
    """critic 用: 特権情報込み 54 次元の履歴 (ノイズなし)。"""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = HISTORY_LENGTH
        self.flatten_history_dim = True


@configclass
class K1GKLateralDHObservationsCfg:
    direct: K1GKLateralDHDirectCfg = K1GKLateralDHDirectCfg()
    policy: K1GKLateralDHPolicyCfg = K1GKLateralDHPolicyCfg()
    critic: K1GKLateralDHCriticCfg = K1GKLateralDHCriticCfg()


@configclass
class K1GKLateralDHEnvCfg(K1GKLateralEnvCfg):
    observations: K1GKLateralDHObservationsCfg = K1GKLateralDHObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # ☠ 位相の指令ベース化・位相 DR の拡大・関節ゼロ点 DR・l_back・body_jitter の
        #   停止時ゲートは **すべて土台の K1GKLateralEnvCfg 側** に入っている。
        #   ここで重複させないこと。**非DH版との差を「履歴 + CNN」だけに保つ**のが
        #   この cfg の存在理由で、他の差分を混ぜると A/B 比較が成立しなくなる。
        #
        #   ただし観測グループを 3 つに再編した都合で、土台が
        #   ``self.observations.policy.gait_phase`` などへ配った位相パラメータは
        #   **こちらの新しいグループにも配り直す必要がある** (土台の __post_init__ は
        #   古い policy/critic グループを触っているが、クラス属性としては別物)。
        from ..goalkeeper_lateral_env_cfg import _ADAPTIVE_PHASE_PARAMS

        self.observations.policy.gait_phase.params.update(_ADAPTIVE_PHASE_PARAMS)
        self.observations.critic.gait_phase.params.update(_ADAPTIVE_PHASE_PARAMS)


@configclass
class K1GKLateralDHEnvCfg_PLAY(K1GKLateralDHEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # ★ 評価では較正のずれと位相 DR を切る (方策の素の実力を見るため)。
        #   個体差への頑健性を測りたいときはここを戻して比較する。
        self.events.randomize_joint_offset = None
        self.events.randomize_phase_freq.params["offset_range"] = (0.0, 0.0)
