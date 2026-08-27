"""RSL-RL actor-critic for CVKF future-horizon observations.

The LSTMs encode the configured forecast horizon axis.  They intentionally do
not carry hidden state across environment steps, so RSL-RL must treat this as a
feed-forward policy (``is_recurrent = False``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from .symmetry import (
    BELIEF_STATUS_SIZE,
    BELIEF_VELOCITY_SIZE,
    DIRECT_KICKING_ACTION_SIZE,
    HORIZON_TOKEN_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    NON_FORECAST_OBSERVATION_SIZE,
    expected_direct_kicking_observation_size,
    kick_feasibility_ambiguity_weight,
    mirror_direct_kicking_observation,
    weighted_mirror_consistency_loss,
)

DEFAULT_PREDICTION_HORIZONS_S = (
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
)

_ACTOR_HIDDEN_DIMS = (256, 128, 128)
_CRITIC_HIDDEN_DIMS = (256, 256, 128)

INSIDE_OBSERVATION_SIZE = 223
INSIDE_PRIVILEGED_OBSERVATION_SIZE = 61
INSIDE_CVKF_OBSERVATION_SIZE = INSIDE_OBSERVATION_SIZE + 83


def _concatenate_groups(
    observations: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
) -> torch.Tensor:
    values = [observations[name] for name in group_names]
    if not values:
        raise ValueError("observation group list must not be empty")
    leading_shape = values[0].shape[:-1]
    if any(value.shape[:-1] != leading_shape for value in values[1:]):
        raise ValueError("observation groups must have equal leading dimensions")
    return torch.cat(values, dim=-1)


class DirectKickingObservationEncoder(nn.Module):
    """Encode the forecast horizon and preserve every non-forecast feature."""

    def __init__(
        self,
        num_observations: int,
        horizon_count: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_observations = int(num_observations)
        self.horizon_count = int(horizon_count)
        self.hidden_size = int(hidden_size)
        self.horizon_token_size = int(HORIZON_TOKEN_SIZE)
        self.forecast_start = LOCOMOTION_OBSERVATION_SIZE
        self.forecast_end = self.forecast_start + self.horizon_count * self.horizon_token_size
        self.output_size = NON_FORECAST_OBSERVATION_SIZE + self.hidden_size
        self.lstm = nn.LSTM(
            input_size=self.horizon_token_size,
            hidden_size=self.hidden_size,
            num_layers=int(num_layers),
            batch_first=True,
        )

    def forward(self, flat_observation: torch.Tensor) -> torch.Tensor:
        if flat_observation.dim() != 2:
            raise ValueError("DirectKicking encoder expects a two-dimensional batch")
        if flat_observation.shape[-1] != self.num_observations:
            raise ValueError("DirectKicking observation size does not match the model")

        forecast = flat_observation[:, self.forecast_start : self.forecast_end]
        forecast = forecast.reshape(
            flat_observation.shape[0],
            self.horizon_count,
            self.horizon_token_size,
        )
        # Omitting h_0/c_0 intentionally creates fresh zero state on every call.
        _, (hidden, _) = self.lstm(forecast)
        non_forecast = torch.cat(
            (
                flat_observation[:, : self.forecast_start],
                flat_observation[:, self.forecast_end :],
            ),
            dim=-1,
        )
        return torch.cat((non_forecast, hidden[-1]), dim=-1)


class InsideCVKFObservationEncoder(nn.Module):
    """Encode only the appended CVKF horizon and preserve the 223D inside input."""

    def __init__(
        self,
        horizon_count: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.horizon_count = int(horizon_count)
        self.hidden_size = int(hidden_size)
        self.forecast_start = INSIDE_OBSERVATION_SIZE
        self.forecast_end = self.forecast_start + self.horizon_count * HORIZON_TOKEN_SIZE
        self.output_size = (
            INSIDE_OBSERVATION_SIZE
            + BELIEF_VELOCITY_SIZE
            + BELIEF_STATUS_SIZE
            + self.hidden_size
        )
        self.lstm = nn.LSTM(
            input_size=HORIZON_TOKEN_SIZE,
            hidden_size=self.hidden_size,
            num_layers=int(num_layers),
            batch_first=True,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2:
            raise ValueError("Inside+CVKF encoder expects a two-dimensional batch")
        if observation.shape[-1] != INSIDE_CVKF_OBSERVATION_SIZE:
            raise ValueError(
                f"Inside+CVKF observation must have {INSIDE_CVKF_OBSERVATION_SIZE} values"
            )
        forecast = observation[:, self.forecast_start : self.forecast_end]
        forecast = forecast.reshape(
            observation.shape[0],
            self.horizon_count,
            HORIZON_TOKEN_SIZE,
        )
        _, (hidden, _) = self.lstm(forecast)
        non_forecast = torch.cat(
            (
                observation[:, : self.forecast_start],
                observation[:, self.forecast_end :],
            ),
            dim=-1,
        )
        return torch.cat((non_forecast, hidden[-1]), dim=-1)


class InsideCVKFActor(nn.Module):
    """Inside-kick MLP fed by raw inside features and the CVKF LSTM latent."""

    def __init__(
        self,
        num_actions: int,
        horizon_count: int,
        hidden_size: int,
        num_layers: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.encoder = InsideCVKFObservationEncoder(
            horizon_count,
            hidden_size,
            num_layers,
        )
        dims = (self.encoder.output_size, *tuple(int(value) for value in hidden_dims))
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dims, dims[1:]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
        layers.append(nn.Linear(dims[-1], int(num_actions)))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoder(observation))


class InsideCVKFActorCritic(ActorCritic):
    """Inside actor/critic contract with an appended stateless CVKF horizon LSTM."""

    is_recurrent = False
    observation_schema = "walk_long_pass_history_inside_plus_cvkf_v1"

    def __init__(
        self,
        obs: Mapping[str, torch.Tensor],
        obs_groups: Mapping[str, Sequence[str]],
        num_actions: int,
        *,
        prediction_horizons_s: Sequence[float] = DEFAULT_PREDICTION_HORIZONS_S,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        horizons = tuple(float(value) for value in prediction_horizons_s)
        if len(horizons) * HORIZON_TOKEN_SIZE + INSIDE_OBSERVATION_SIZE + 5 != (
            INSIDE_CVKF_OBSERVATION_SIZE
        ):
            raise ValueError("Inside+CVKF horizon count does not match the 306D schema")
        actor_example = _concatenate_groups(obs, obs_groups["policy"])
        critic_example = _concatenate_groups(obs, obs_groups["critic"])
        if actor_example.shape[-1] != INSIDE_CVKF_OBSERVATION_SIZE:
            raise ValueError(
                f"Inside+CVKF actor expected 306 observations, got {actor_example.shape[-1]}"
            )
        if critic_example.shape[-1] != INSIDE_PRIVILEGED_OBSERVATION_SIZE:
            raise ValueError(
                "Inside+CVKF critic must preserve the 61D inside privileged contract"
            )
        if activation.lower() != "elu":
            raise ValueError("Inside+CVKF policy requires ELU activation")

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            **kwargs,
        )
        self.prediction_horizons_s = horizons
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.actor = InsideCVKFActor(
            num_actions,
            len(horizons),
            self.lstm_hidden_size,
            self.lstm_num_layers,
            actor_hidden_dims,
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "observation_schema": self.observation_schema,
            "policy_observation_size": INSIDE_CVKF_OBSERVATION_SIZE,
            "critic_observation_size": INSIDE_PRIVILEGED_OBSERVATION_SIZE,
            "prediction_horizons_s": list(self.prediction_horizons_s),
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_num_layers": self.lstm_num_layers,
        }


class DirectKickingActor(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_observations: int,
        horizon_count: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.encoder = DirectKickingObservationEncoder(
            num_observations,
            horizon_count,
            hidden_size,
            num_layers,
        )
        self.network = nn.Sequential(
            nn.Linear(self.encoder.output_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, flat_observation: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoder(flat_observation))


class DirectKickingCritic(nn.Module):
    def __init__(
        self,
        num_privileged_observations: int,
        num_observations: int,
        horizon_count: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_privileged_observations = int(num_privileged_observations)
        self.encoder = DirectKickingObservationEncoder(
            num_observations,
            horizon_count,
            hidden_size,
            num_layers,
        )
        self.network = nn.Sequential(
            nn.Linear(self.encoder.output_size + self.num_privileged_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        flat_observation: torch.Tensor,
        flat_privileged_observation: torch.Tensor,
    ) -> torch.Tensor:
        if flat_privileged_observation.dim() != 2:
            raise ValueError("DirectKicking critic expects a two-dimensional privileged batch")
        if flat_privileged_observation.shape[-1] != self.num_privileged_observations:
            raise ValueError("DirectKicking privileged observation size mismatch")
        encoded = self.encoder(flat_observation)
        return self.network(torch.cat((encoded, flat_privileged_observation), dim=-1))


class DirectKickingActorCritic(nn.Module):
    """RSL-RL 3.0.1 policy with stateless future-horizon LSTM encoders."""

    is_recurrent = False
    observation_schema = "direct_kicking_horizon_lstm_global_target_direction_v3"

    def __init__(
        self,
        obs: Mapping[str, torch.Tensor],
        obs_groups: Mapping[str, Sequence[str]],
        num_actions: int,
        *,
        prediction_horizons_s: Sequence[float] = DEFAULT_PREDICTION_HORIZONS_S,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        num_privileged_observations: int = 20,
        mirror_consistency_enabled: bool = True,
        nominal_strike_point_m: Sequence[float] = (0.185, 0.096),
        ambiguity_cost_gap_m: Sequence[float] = (0.02, 0.10),
        ball_position_observation_scale: float = 1.0,
        init_noise_std: float = math.exp(-2.0),
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] | None = None,
        critic_hidden_dims: Sequence[int] | None = None,
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DirectKickingActorCritic arguments: {unknown}")
        if noise_std_type != "log":
            raise ValueError("DirectKickingActorCritic requires noise_std_type='log'")
        if state_dependent_std:
            raise ValueError("DirectKickingActorCritic does not support state-dependent standard deviation")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("DirectKicking checkpoint contract does not include observation normalizers")
        if actor_hidden_dims is not None and tuple(actor_hidden_dims) != _ACTOR_HIDDEN_DIMS:
            raise ValueError(f"actor_hidden_dims must be {_ACTOR_HIDDEN_DIMS}")
        if critic_hidden_dims is not None and tuple(critic_hidden_dims) != _CRITIC_HIDDEN_DIMS:
            raise ValueError(f"critic_hidden_dims must be {_CRITIC_HIDDEN_DIMS}")
        if activation.lower() != "elu":
            raise ValueError("DirectKickingActorCritic requires ELU activations")
        if init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive")

        horizons = tuple(float(value) for value in prediction_horizons_s)
        if not horizons or horizons[0] != 0.0:
            raise ValueError("prediction_horizons_s must start at zero")
        if any(next_value <= value for value, next_value in zip(horizons, horizons[1:])):
            raise ValueError("prediction_horizons_s must be strictly increasing")
        if lstm_hidden_size <= 0 or lstm_num_layers <= 0:
            raise ValueError("LSTM hidden size and layer count must be positive")

        self.obs_groups = {name: list(groups) for name, groups in obs_groups.items()}
        if "policy" not in self.obs_groups or "critic" not in self.obs_groups:
            raise ValueError("obs_groups must define policy and critic observation sets")

        actor_example = _concatenate_groups(obs, self.obs_groups["policy"])
        critic_example = _concatenate_groups(obs, self.obs_groups["critic"])
        self.num_observations = int(actor_example.shape[-1])
        expected_observations = expected_direct_kicking_observation_size(len(horizons))
        if self.num_observations != expected_observations:
            raise ValueError(
                f"DirectKicking LSTM expected {expected_observations} observations, "
                f"got {self.num_observations}"
            )

        self.num_actions = int(num_actions)
        self.num_privileged_observations = int(num_privileged_observations)
        if self.num_actions != DIRECT_KICKING_ACTION_SIZE:
            raise ValueError("DirectKicking mirror contract requires 12 actions")
        valid_critic_sizes = {
            self.num_privileged_observations,
            self.num_observations + self.num_privileged_observations,
        }
        if int(critic_example.shape[-1]) not in valid_critic_sizes:
            raise ValueError(
                "critic observation set must contain either 20 privileged values or "
                "the 132 policy values followed by 20 privileged values"
            )

        self.prediction_horizons_s = horizons
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.mirror_consistency_enabled = bool(mirror_consistency_enabled)
        self.nominal_strike_point_m = tuple(float(value) for value in nominal_strike_point_m)
        self.ambiguity_cost_gap_m = tuple(float(value) for value in ambiguity_cost_gap_m)
        self.ball_position_observation_scale = float(ball_position_observation_scale)
        if len(self.nominal_strike_point_m) != 2:
            raise ValueError("nominal_strike_point_m must contain two values")
        if len(self.ambiguity_cost_gap_m) != 2:
            raise ValueError("ambiguity_cost_gap_m must contain two values")
        if self.nominal_strike_point_m[0] <= 0.0 or self.nominal_strike_point_m[1] <= 0.0:
            raise ValueError("nominal strike coordinates must be positive")
        if self.ambiguity_cost_gap_m[0] < 0.0 or (
            self.ambiguity_cost_gap_m[1] <= self.ambiguity_cost_gap_m[0]
        ):
            raise ValueError("ambiguity cost gap must satisfy 0 <= low < high")
        if self.ball_position_observation_scale <= 0.0:
            raise ValueError("ball position observation scale must be positive")

        self.actor = DirectKickingActor(
            self.num_actions,
            self.num_observations,
            len(horizons),
            self.lstm_hidden_size,
            self.lstm_num_layers,
        )
        self.critic = DirectKickingCritic(
            self.num_privileged_observations,
            self.num_observations,
            len(horizons),
            self.lstm_hidden_size,
            self.lstm_num_layers,
        )
        self.logstd = nn.Parameter(
            torch.full((1, self.num_actions), math.log(float(init_noise_std))),
            requires_grad=True,
        )
        # Preserve the source checkpoint key (``logstd``) while exposing the
        # RSL-RL spelling used by training utilities.  A property does not add
        # a duplicate entry to state_dict().
        self.noise_std_type = "log"
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _actor_observation(self, obs: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(obs):
            actor_obs = obs
        else:
            actor_obs = _concatenate_groups(obs, self.obs_groups["policy"])
        if actor_obs.shape[-1] != self.num_observations:
            raise ValueError("DirectKicking actor observation shape mismatch")
        return actor_obs

    def _privileged_observation(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        critic_obs = _concatenate_groups(obs, self.obs_groups["critic"])
        if critic_obs.shape[-1] == self.num_privileged_observations:
            return critic_obs
        return critic_obs[..., -self.num_privileged_observations :]

    def get_actor_obs(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._actor_observation(obs)

    def get_critic_obs(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((self._actor_observation(obs), self._privileged_observation(obs)), dim=-1)

    def update_distribution(self, actor_observation: torch.Tensor) -> None:
        leading_shape = actor_observation.shape[:-1]
        action_mean = self.actor(actor_observation.reshape(-1, self.num_observations))
        action_mean = action_mean.reshape(leading_shape + (self.num_actions,))
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        self.distribution = Normal(action_mean, action_std)

    def act(self, obs: Mapping[str, torch.Tensor], **_: Any) -> torch.Tensor:
        self.update_distribution(self._actor_observation(obs))
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        actor_observation = self._actor_observation(obs)
        leading_shape = actor_observation.shape[:-1]
        action_mean = self.actor(actor_observation.reshape(-1, self.num_observations))
        return action_mean.reshape(leading_shape + (self.num_actions,))

    def evaluate(self, obs: Mapping[str, torch.Tensor], **_: Any) -> torch.Tensor:
        actor_observation = self._actor_observation(obs)
        privileged_observation = self._privileged_observation(obs)
        if actor_observation.shape[:-1] != privileged_observation.shape[:-1]:
            raise ValueError("actor and privileged observation leading shapes differ")
        leading_shape = actor_observation.shape[:-1]
        values = self.critic(
            actor_observation.reshape(-1, self.num_observations),
            privileged_observation.reshape(-1, self.num_privileged_observations),
        )
        return values.reshape(leading_shape + (1,))

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.stddev

    @property
    def log_std(self) -> nn.Parameter:
        """RSL-RL-compatible alias for the source ``logstd`` parameter."""
        return self.logstd

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: Mapping[str, torch.Tensor]) -> None:
        del obs

    def compute_symmetry_loss(
        self,
        observation: Mapping[str, torch.Tensor] | torch.Tensor,
        action_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted mirror loss and mean active-sample weight."""
        actor_observation = self._actor_observation(observation)
        expected_action_shape = actor_observation.shape[:-1] + (self.num_actions,)
        if action_mean.shape != expected_action_shape:
            raise ValueError("DirectKicking symmetry action mean shape mismatch")

        flat_observation = actor_observation.reshape(-1, self.num_observations)
        flat_action_mean = action_mean.reshape(-1, self.num_actions)
        if not self.mirror_consistency_enabled:
            zero = flat_action_mean.sum() * 0.0
            return zero, zero.detach()

        mirrored_observation = mirror_direct_kicking_observation(
            flat_observation,
            len(self.prediction_horizons_s),
        )
        mirrored_observation_action_mean = self.actor(mirrored_observation)
        weight = kick_feasibility_ambiguity_weight(
            flat_observation,
            len(self.prediction_horizons_s),
            self.nominal_strike_point_m,
            self.ambiguity_cost_gap_m,
            self.ball_position_observation_scale,
        )
        loss = weighted_mirror_consistency_loss(
            flat_action_mean,
            mirrored_observation_action_mean,
            weight,
        )
        return loss, weight.mean()

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_class": self.__class__.__name__,
            "observation_schema": self.observation_schema,
            "num_actions": self.num_actions,
            "num_observations": self.num_observations,
            "num_privileged_observations": self.num_privileged_observations,
            "prediction_horizons_s": list(self.prediction_horizons_s),
            "horizon_token_size": HORIZON_TOKEN_SIZE,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_num_layers": self.lstm_num_layers,
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

    def forward(self, *_: Any, **__: Any) -> torch.Tensor:
        raise NotImplementedError("Use act(), act_inference(), or evaluate()")


__all__ = [
    "DEFAULT_PREDICTION_HORIZONS_S",
    "DirectKickingActor",
    "DirectKickingActorCritic",
    "DirectKickingCritic",
    "DirectKickingObservationEncoder",
]
