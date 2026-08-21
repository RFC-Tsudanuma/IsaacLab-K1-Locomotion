# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the isolated WalkKick likelihood task."""

import math

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg


@configclass
class DirectKickingActorCriticCfg(RslRlPpoActorCriticCfg):
    """DirectKicking dimensions with the global-target v3 schema metadata."""

    class_name: str = "DirectKickingActorCritic"
    init_noise_std: float = math.exp(-2.0)
    noise_std_type: str = "log"
    state_dependent_std: bool = False
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [256, 128, 128]
    critic_hidden_dims: list[int] = [256, 256, 128]
    activation: str = "elu"

    prediction_horizons_s: list[float] = [
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
        2.50,
        3.00,
    ]
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    num_privileged_observations: int = 20

    mirror_consistency_enabled: bool = True
    nominal_strike_point_m: list[float] = [0.185, 0.096]
    ambiguity_cost_gap_m: list[float] = [0.02, 0.10]
    ball_position_observation_scale: float = 1.0


@configclass
class DirectKickingPPOCfg(RslRlPpoAlgorithmCfg):
    """Current WalkKick PPO settings plus the source mirror coefficient."""

    class_name: str = "DirectKickingPPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005399484409787433
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 0.00012551115172973836
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    symmetric_coef: float = 10.0


@configclass
class K1WalkKickLikelihoodPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Use the local 132D model/weighted-mirror PPO without changing WalkKick."""

    class_name: str = "DirectKickingOnPolicyRunner"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }
    policy: DirectKickingActorCriticCfg = DirectKickingActorCriticCfg()
    algorithm: DirectKickingPPOCfg = DirectKickingPPOCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "k1_walk_kick_likelihood_global_target"


@configclass
class K1WalkKickLikelihoodWalkPhasePPORunnerCfg(K1WalkKickLikelihoodPPORunnerCfg):
    """Stage 1 runner with the same 132D policy/checkpoint contract."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "k1_walk_kick_likelihood_global_target_walk_phase"


@configclass
class K1WalkKickLikelihoodStationaryPPORunnerCfg(K1WalkKickLikelihoodPPORunnerCfg):
    """Stage 2 runner for stationary-ball global-target training."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "k1_walk_kick_likelihood_global_target_stationary"
