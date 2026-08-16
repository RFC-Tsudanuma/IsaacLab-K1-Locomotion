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

        # ★ 2026-08-15: entropy_coef 0.008 → 0.02。
        #   Stage2 の実測で mean_noise_std が初期 0.7 から **0.053** まで潰れ、3500 iter
        #   のあいだ一度も回復しなかった (0.0534 → 0.0528)。方策がほぼ決定論的で、
        #   PPO が新しい挙動を試せない状態。
        #
        #   これが停滞の主因と見ている根拠: 同時点で success_ema は 0.648 で頭打ちだが、
        #   「物理的に取れない球」は発射時に判定されて **成功率の集計から除外済み**
        #   (events._mark_unreachable + curriculums._update_success_ema)。つまり 0.648 は
        #   *取れるはずの球* に対する成功率で、届く球の 35% を落としている = 伸びしろは
        #   まだある。下位の速度が天井なら EMA はもっと高い値で張り付くはずで、
        #   実際そうなっていない。
        #
        #   既存階層版も同じ症状で 0.004 → 0.008 に上げた経緯があるが、それでも足りて
        #   いなかった。DH は観測が 444 次元と広く、探索が潰れると届かない領域が増える。
        #
        # ★ 2026-08-15 (同日中に修正): 0.02 は **上げすぎだった**。σ が 0.239 まで上がり、
        #   `--high_action_deadband 0.1` と衝突した。上位 action は 3 次元なので指令
        #   ノルムの期待値は 1.6σ ≈ 0.38 となり、**平均指令が 0 でもノルムが 0.1 を
        #   超える確率が約 99%**。デッドバンドは「下位の停止規約に入れるための仕組み」
        #   なので、超え続けるとキーパーが物理的に止まれなくなる。実測:
        #       hold_at_target        0.523 → 0.069  (1/7.6)
        #       target_reach_velocity 1.669 → 0.466  (1/3.6)
        #   停止できる確率を残すには σ ≲ 0.06 が上限 (0.1/0.06 = 1.67 → 約 30%)。
        #   0.01 は元の 0.008 よりわずかに探索を残しつつ、その制約の内側に収まる狙い。
        #
        # ★ 2026-08-16: **0.01 でも足りなかった。** 実測で σ = 0.13 にしかならず
        #   (0.008 → 0.01 の 25% 増で σ が 2.5 倍)、止まれる確率は約 10%。15,000 iter の
        #   あいだ全指標が右肩下がりになった:
        #       success_ema 0.653 → 0.604 / mean_reward -17% / hold_at_target 0.16 → 0.13
        #       target_reach_velocity -14% / goal_conceded 0.62 → 0.70
        #   目標に着いても止まれず行き過ぎるので、到達も接触も一緒に悪化する。
        #   「探索を増やす」という方向自体が deadband=0.1 の制約下では成立しない、
        #   というのが結論。元の 0.008 (σ ≈ 0.053、止まれる確率 約70%) に戻す。
        #   ★ これを上げたくなったら、先に high_action_deadband との整合を確認すること。
        #     目安は σ ≲ 0.06 (= deadband / 1.6)。
        self.algorithm.entropy_coef = 0.008

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
