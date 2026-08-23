# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""履歴 + CNN 版 横移動下位ポリシーの PPO 設定。

既存の :class:`~..agents.rsl_rl_ppo_cfg.K1GKLateralPPORunnerCfg` を継承し、

    * ``policy.class_name`` を :class:`~.networks.LateralHistoryActorCritic` に差し替え
    * 観測グループの組を ``{"policy": ["direct", "policy"], "critic": ["direct", "critic"]}`` に
    * 対称変換を **履歴対応版** に差し替え

の 3 点を変える。PPO のハイパラは非 DH 版と同一。

ネットワークの形 (既定値のとき)::

    direct 13 + policy 履歴 49×100 = 4913 次元
    履歴 100 frame → Conv1d(49→32, k8, s4) → 24 → Conv1d(32→16, k4, s2) → 11
                   → latent 16×11 = 176
    MLP 入力 = 13 (direct) + 49×4 (直近ステップ) + 176 (CNN) = 385
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ..agents.rsl_rl_ppo_cfg import K1GKLateralPPORunnerCfg
from .networks import RslRlLateralHistoryActorCriticCfg
from .symmetry import compute_symmetric_states_lateral_history

# import しておくことで rsl-rl の on_policy_runner 名前空間へクラスが注入される
# (``eval(class_name)`` から見えるようにするため)。networks.py の末尾を参照。
from . import networks  # noqa: F401


@configclass
class K1GKLateralDHPPORunnerCfg(K1GKLateralPPORunnerCfg):
    """横移動 DH 版。experiment_name を分けて非 DH 版と並べて比較できるようにする。"""

    experiment_name = "k1_gk_lateral_dh"

    # ★ MLP の幅は非 DH 版と同一 (actor [256,128,128] / critic [256,256,128])。
    #   CNN の latent が入力に増えるだけにして、容量差ではなく構造差を見る。
    policy: RslRlLateralHistoryActorCriticCfg = RslRlLateralHistoryActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        super().__post_init__()

        # ☠ 履歴グループは **必ず末尾** に置くこと。LateralHistoryActorCritic は
        #   「最後のグループが履歴、それより前は全部 direct」で入力を切り分ける。
        self.obs_groups = {
            "policy": ["direct", "policy"],
            "critic": ["direct", "critic"],
        }

        # ☠ 親が入れた対称変換 (59 次元の履歴なし版) を履歴対応版に差し替える。
        #   忘れると次元チェックで落ちる (静かに壊れるより良い)。
        #   ★★ 2026-08-23: use_data_augmentation を **False → True に戻した**。
        #     当初は「履歴グループ全体をミニバッチごと 2 倍に複製するとメモリが厳しい」
        #     という理由で切ったが、これは **非 DH 版 (aug=True) との差を履歴+CNN 以外にも
        #     作ってしまう手抜き**だった。実測でも A/B の DH 側だけ左右差が
        #     **1.5 で −7.8% / 1.8 で −7.5%** と一貫して残っている (右が速い)。
        #     mirror loss だけでは「出力の対称性」しか縛れず、critic の価値推定・
        #     advantage・観測正規化の統計は左右非対称なまま残るため。
        #   ☠ メモリ実測: H=50 なら 4096 env で 8.97GB / 16GB。aug で増えるのは
        #     ミニバッチ 12288 サンプル × 2 × (4900+5400) 次元 × 4B ≒ **+1.0GB** なので収まる。
        #     H を 100 に戻すときは再計算すること (履歴が倍なので +2GB 級になる)。
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states_lateral_history,
            mirror_loss_coeff=2.0,
        )

        # 履歴観測でロールアウトストレージが数 GB になるため、更新時のピークを抑える。
        # データ総量・イテレーション数は不変 (KL 適応 LR が勾配ステップ数の変化を吸収)。
        self.algorithm.num_mini_batches = 8
