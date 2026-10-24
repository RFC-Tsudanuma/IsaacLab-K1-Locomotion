# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行 (K1FlatPPORunnerCfg) の学習設定をそのまま使う。

歩行の重みと観測の正規化の統計をそのまま引き継ぐので、PPO の設定も歩行に揃える
(履歴入力の HistoryActorCritic、観測の正規化あり、mirror loss あり)。変えるのは:

* 最初の 100 update は actor を固定して critic を慣らす (PPOActorFreeze)。
* env 数が 1024 (履歴観測が大きく GPU 8GB に載せるため)。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from ...locomotion.agents.rsl_rl_ppo_cfg import K1FlatPPORunnerCfg


@configclass
class ActorFreezePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOActorFreeze"
    actor_freeze_iterations: int = 100
    std_range: tuple | None = None


@configclass
class K1BHumanWalkKickPPORunnerCfg(K1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_bhuman_walk_kick"
        self.max_iterations = 10000
        self.save_interval = 100
        a = self.algorithm
        self.algorithm = ActorFreezePpoAlgorithmCfg(
            value_loss_coef=a.value_loss_coef,
            use_clipped_value_loss=a.use_clipped_value_loss,
            clip_param=a.clip_param,
            entropy_coef=a.entropy_coef,
            num_learning_epochs=a.num_learning_epochs,
            num_mini_batches=a.num_mini_batches,
            learning_rate=a.learning_rate,
            schedule=a.schedule,
            gamma=a.gamma,
            lam=a.lam,
            desired_kl=a.desired_kl,
            max_grad_norm=a.max_grad_norm,
            symmetry_cfg=a.symmetry_cfg,
        )
