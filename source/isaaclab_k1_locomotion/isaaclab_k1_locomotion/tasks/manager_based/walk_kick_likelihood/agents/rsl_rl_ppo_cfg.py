# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the isolated WalkKick likelihood task."""

import math

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg
from ...walk_long_pass_history.agents.rsl_rl_ppo_cfg import (
    K1WalkLongPassHistoryPPORunnerCfg,
)
from .symmetry import compute_inside_cvkf_symmetric_states


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


@configclass
class InsideCVKFActorCriticCfg(RslRlPpoActorCriticCfg):
    """Inside 223D MLP contract with the appended 83D CVKF horizon encoder."""

    class_name: str = "InsideCVKFActorCritic"
    init_noise_std: float = 0.7207805082202461
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
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


@configclass
class K1WalkKickInsideCVKFStationaryPPORunnerCfg(
    K1WalkLongPassHistoryPPORunnerCfg
):
    """Phase 1 runner for stationary inside kicks with the 306D actor schema."""

    class_name: str = "DirectKickingOnPolicyRunner"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy: InsideCVKFActorCriticCfg = InsideCVKFActorCriticCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "k1_walk_kick_inside_cvkf_stationary"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_inside_cvkf_symmetric_states,
            mirror_loss_coeff=0.5,
        )


@configclass
class K1WalkKickInsideCVKFMovingPPORunnerCfg(
    K1WalkKickInsideCVKFStationaryPPORunnerCfg
):
    """Phase 2 runner for the coupled incoming-ball curriculum."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "k1_walk_kick_inside_cvkf_moving"
