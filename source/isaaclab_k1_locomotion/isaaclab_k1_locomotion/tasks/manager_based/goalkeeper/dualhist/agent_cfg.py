# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー版ゴールキーパーの PPO 設定。

既存の階層版 v2 の設定 (:class:`~..agents.rsl_rl_ppo_cfg.K1GKHierPPORunnerCfg`) を継承し、

    * ``policy.class_name`` を :class:`~.networks.ActorCriticDualHistory` に差し替え
    * 履歴の長さ・CNN 構成を追加フィールドで渡す
    * 対称変換を履歴ブロック対応版に差し替え

の 3 点だけ変える。凍結下位 (``low_level_policy``) と PPO のハイパラは既存のまま。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlSymmetryCfg

from ..agents.rsl_rl_ppo_cfg import K1GKHierPPORunnerCfg
from .env_cfg import HIST_LONG_FRAMES, HIST_SHORT_FRAMES
from .observations import GK_HIST_FRAME_DIM
from .symmetry import compute_symmetric_states_dualhist

# import しておくことで rsl-rl の on_policy_runner 名前空間へクラスが注入される
# (``eval(class_name)`` から見えるようにするため)。:func:`.networks.register_with_rsl_rl`
from . import networks  # noqa: F401


@configclass
class RslRlDualHistoryActorCriticCfg(RslRlPpoActorCriticCfg):
    """actor をデュアルヒストリー構造にするための追加設定。

    ``hist_*`` はそのまま ``ActorCriticDualHistory.__init__`` の引数へ渡る。
    **``hist_short_frames`` / ``hist_long_frames`` は env 側 (dualhist/env_cfg.py) の
    観測項と必ず一致させること。** 食い違うと観測の切り出し位置がずれる。
    """

    class_name: str = "ActorCriticDualHistory"

    hist_frame_dim: int = GK_HIST_FRAME_DIM
    hist_short_frames: int = HIST_SHORT_FRAMES
    hist_long_frames: int = HIST_LONG_FRAMES

    # 長期履歴を圧縮する 1D CNN。既定は論文 (arXiv:2401.16889) と同じ構成。
    #   50 frame → k6/s3 で 15 → k4/s2 で 6 → 16ch × 6 = latent 96
    hist_long_channels: list = [32, 16]
    hist_long_kernels: list = [6, 4]
    hist_long_strides: list = [3, 2]


@configclass
class K1GKHierDHPPORunnerCfg(K1GKHierPPORunnerCfg):
    """デュアルヒストリー版の上位ポリシー設定。

    ``actor_hidden_dims`` は既存の階層版と同じ [256, 256, 128] のまま。CNN の latent (96)
    が入力に増えるだけで、比較対象 (既存階層版) と MLP 側の容量を揃えておきたいため。
    """

    policy: RslRlDualHistoryActorCriticCfg = RslRlDualHistoryActorCriticCfg(
        init_noise_std=0.7,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        super().__post_init__()
        # 親は 59 次元固定の反転関数を入れるので、履歴ブロック対応版へ差し替える。
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states_dualhist,
            mirror_loss_coeff=2.0,
        )


@configclass
class K1GKHierDHStage1PPORunnerCfg(K1GKHierDHPPORunnerCfg):
    """Stage 1: ボールなし。既存階層版 Stage1 と同じ反復数で比較できるようにする。"""

    max_iterations = 5000
    experiment_name = "k1_gk_hier_dh_stage1"


@configclass
class K1GKHierDHStage2PPORunnerCfg(K1GKHierDHPPORunnerCfg):
    """Stage 2: ゴール + ボール + 適応カリキュラム。Stage1 ckpt から --resume。

    ★ 既存階層版 (20000) より長い 60000 にしてある (ユーザー指示 2026-08-15)。
      適応カリキュラムはセーブ成功率 EMA が上がるたびに難易度を上げる作りなので、
      反復を増やしたぶんだけ「狙い先の広さ → ボール初速」の段が先へ進む。
      比較対象の直接制御版は 76k iter で success_ema 0.796 / ball_speed_hi 3.10 m/s。
    """

    max_iterations = 60000
    experiment_name = "k1_gk_hier_dh_stage2"
