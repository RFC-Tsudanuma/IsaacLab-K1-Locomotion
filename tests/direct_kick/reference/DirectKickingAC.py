"""DirectKicking actor-critic with stateless future-horizon LSTM encoders."""

from typing import Dict, Sequence, Tuple

import torch

from utils.direct_kicking_observation import (
    HORIZON_TOKEN_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    NON_FORECAST_OBSERVATION_SIZE,
    expected_direct_kicking_observation_size,
)
from utils.direct_kicking_symmetry import (
    DIRECT_KICKING_ACTION_SIZE,
    kick_feasibility_ambiguity_weight,
    mirror_direct_kicking_observation,
    weighted_mirror_consistency_loss,
)


class DirectKickingObservationEncoder(torch.nn.Module):
    """Encode only the forecast-horizon axis and keep policy time stateless."""

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
        self.horizon_token_size = int(HORIZON_TOKEN_SIZE)
        self.hidden_size = int(hidden_size)
        self.encoded_output_size = int(
            NON_FORECAST_OBSERVATION_SIZE + self.hidden_size
        )
        self.forecast_start = LOCOMOTION_OBSERVATION_SIZE
        self.forecast_end = (
            self.forecast_start + self.horizon_count * HORIZON_TOKEN_SIZE
        )
        self.lstm = torch.nn.LSTM(
            input_size=HORIZON_TOKEN_SIZE,
            hidden_size=self.hidden_size,
            num_layers=int(num_layers),
            batch_first=True,
        )

    @property
    def output_size(self) -> int:
        return self.encoded_output_size

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


class DirectKickingActor(torch.nn.Module):
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
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.output_size, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_actions),
        )
        self.post_kick_phase_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.output_size, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        torch.nn.init.zeros_(self.post_kick_phase_head[-1].weight)
        torch.nn.init.constant_(self.post_kick_phase_head[-1].bias, -4.0)

    def action_mean(self, flat_observation: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoder(flat_observation))

    def post_kick_phase_logit(
        self,
        flat_observation: torch.Tensor,
    ) -> torch.Tensor:
        return self.post_kick_phase_head(
            self.encoder(flat_observation)
        ).squeeze(-1)

    def forward(self, flat_observation: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(flat_observation)
        action_mean = self.network(encoded)
        post_kick_phase = torch.sigmoid(
            self.post_kick_phase_head(encoded)
        )
        return torch.cat((action_mean, post_kick_phase), dim=-1)


class DirectKickingCritic(torch.nn.Module):
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
        self.network = torch.nn.Sequential(
            torch.nn.Linear(
                self.encoder.output_size + self.num_privileged_observations,
                256,
            ),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )

    def forward(
        self,
        flat_observation: torch.Tensor,
        flat_privileged_observation: torch.Tensor,
    ) -> torch.Tensor:
        if flat_privileged_observation.dim() != 2:
            raise ValueError("DirectKicking critic expects a two-dimensional batch")
        if (
            flat_privileged_observation.shape[-1]
            != self.num_privileged_observations
        ):
            raise ValueError("DirectKicking privileged observation size mismatch")
        encoded = self.encoder(flat_observation)
        return self.network(
            torch.cat((encoded, flat_privileged_observation), dim=-1)
        ).squeeze(-1)


class DirectKickingActorCritic(torch.nn.Module):
    """Actor-critic whose LSTMs run across forecast horizons, never rollout time."""

    observation_schema = "direct_kicking_horizon_lstm_direction_only_v2"
    policy_output_schema = "joint_action_12_post_kick_phase_probability_v1"

    def __init__(
        self,
        num_act: int,
        num_obs: int,
        num_privileged_obs: int,
        prediction_horizons_s: Sequence[float],
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        mirror_consistency_enabled: bool = True,
        nominal_strike_point_m=(0.185, 0.096),
        ambiguity_cost_gap_m=(0.02, 0.10),
        ball_position_observation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        horizons = tuple(float(value) for value in prediction_horizons_s)
        if not horizons or horizons[0] != 0.0:
            raise ValueError("prediction_horizons_s must start at zero")
        if any(next_value <= value for value, next_value in zip(horizons, horizons[1:])):
            raise ValueError("prediction_horizons_s must be strictly increasing")
        if lstm_hidden_size <= 0 or lstm_num_layers <= 0:
            raise ValueError("LSTM hidden size and layer count must be positive")

        expected_observations = expected_direct_kicking_observation_size(len(horizons))
        if int(num_obs) != expected_observations:
            raise ValueError(
                "DirectKicking LSTM expected {} observations, got {}".format(
                    expected_observations,
                    num_obs,
                )
            )

        self.num_actions = int(num_act)
        self.num_observations = int(num_obs)
        self.num_privileged_observations = int(num_privileged_obs)
        if self.num_actions != DIRECT_KICKING_ACTION_SIZE:
            raise ValueError("DirectKicking mirror contract requires 12 actions")
        self.prediction_horizons_s = horizons
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.mirror_consistency_enabled = bool(mirror_consistency_enabled)
        self.nominal_strike_point_m = tuple(
            float(value) for value in nominal_strike_point_m
        )
        self.ambiguity_cost_gap_m = tuple(
            float(value) for value in ambiguity_cost_gap_m
        )
        self.ball_position_observation_scale = float(
            ball_position_observation_scale
        )
        if len(self.nominal_strike_point_m) != 2:
            raise ValueError("nominal_strike_point_m must contain two values")
        if len(self.ambiguity_cost_gap_m) != 2:
            raise ValueError("ambiguity_cost_gap_m must contain two values")
        if (
            self.nominal_strike_point_m[0] <= 0.0
            or self.nominal_strike_point_m[1] <= 0.0
        ):
            raise ValueError("nominal strike coordinates must be positive")
        if (
            self.ambiguity_cost_gap_m[0] < 0.0
            or self.ambiguity_cost_gap_m[1] <= self.ambiguity_cost_gap_m[0]
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
        self.logstd = torch.nn.parameter.Parameter(
            torch.full((1, self.num_actions), fill_value=-2.0),
            requires_grad=True,
        )

    @classmethod
    def from_config(
        cls,
        num_act: int,
        num_obs: int,
        num_privileged_obs: int,
        cfg: Dict,
    ) -> "DirectKickingActorCritic":
        observation_cfg = cfg["direct_kicking"]["observation"]
        mirror_cfg = cfg["direct_kicking"].get("mirror_consistency", {})
        return cls(
            num_act,
            num_obs,
            num_privileged_obs,
            prediction_horizons_s=observation_cfg["prediction_horizons_s"],
            lstm_hidden_size=int(observation_cfg.get("lstm_hidden_size", 64)),
            lstm_num_layers=int(observation_cfg.get("lstm_num_layers", 1)),
            mirror_consistency_enabled=bool(mirror_cfg.get("enabled", True)),
            nominal_strike_point_m=mirror_cfg.get(
                "nominal_strike_point_m",
                (0.185, 0.096),
            ),
            ambiguity_cost_gap_m=mirror_cfg.get(
                "ambiguity_cost_gap_m",
                (0.02, 0.10),
            ),
            ball_position_observation_scale=float(
                cfg.get("normalization", {}).get("ball_pos", 1.0)
            ),
        )

    def checkpoint_metadata(self) -> Dict:
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
            "policy_output_schema": self.policy_output_schema,
            "num_policy_outputs": self.num_actions + 1,
            "post_kick_phase_output_index": self.num_actions,
        }

    def initialize_from_checkpoint(self, checkpoint: Dict) -> None:
        """Load a current checkpoint or migrate the pre-phase policy once."""
        actual_metadata = checkpoint.get("model_metadata")
        expected_metadata = self.checkpoint_metadata()
        if actual_metadata == expected_metadata:
            self.load_state_dict(checkpoint["model"], strict=True)
            return

        legacy_metadata = dict(expected_metadata)
        legacy_metadata.pop("policy_output_schema")
        legacy_metadata.pop("num_policy_outputs")
        legacy_metadata.pop("post_kick_phase_output_index")
        if actual_metadata != legacy_metadata:
            raise ValueError(
                "Checkpoint model metadata is not compatible with "
                "DirectKicking post-kick-phase initialization"
            )

        state_dict = checkpoint["model"]
        expected_keys = set(self.state_dict())
        actual_keys = set(state_dict)
        missing_keys = expected_keys - actual_keys
        unexpected_keys = actual_keys - expected_keys
        allowed_missing_keys = {
            key
            for key in expected_keys
            if key.startswith("actor.post_kick_phase_head.")
        }
        if missing_keys != allowed_missing_keys or unexpected_keys:
            raise ValueError(
                "Legacy DirectKicking checkpoint differs outside the new "
                "post-kick-phase head"
            )
        self.load_state_dict(state_dict, strict=False)

    def act(self, obs: torch.Tensor) -> torch.distributions.Normal:
        if obs.dim() < 2 or obs.shape[-1] != self.num_observations:
            raise ValueError("DirectKicking actor observation shape mismatch")
        leading_shape = obs.shape[:-1]
        action_mean = self.actor.action_mean(
            obs.reshape(-1, self.num_observations)
        )
        action_mean = action_mean.reshape(leading_shape + (self.num_actions,))
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def post_kick_phase_logit(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() < 2 or obs.shape[-1] != self.num_observations:
            raise ValueError("DirectKicking phase observation shape mismatch")
        logits = self.actor.post_kick_phase_logit(
            obs.reshape(-1, self.num_observations)
        )
        return logits.reshape(obs.shape[:-1])

    def compute_symmetry_loss(
        self,
        observation: torch.Tensor,
        action_mean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return weighted mirror loss and mean active-sample weight."""
        if observation.dim() < 2 or observation.shape[-1] != self.num_observations:
            raise ValueError("DirectKicking symmetry observation shape mismatch")
        expected_action_shape = observation.shape[:-1] + (self.num_actions,)
        if action_mean.shape != expected_action_shape:
            raise ValueError("DirectKicking symmetry action mean shape mismatch")

        flat_observation = observation.reshape(-1, self.num_observations)
        flat_action_mean = action_mean.reshape(-1, self.num_actions)
        if not self.mirror_consistency_enabled:
            zero = flat_action_mean.sum() * 0.0
            return zero, zero.detach()

        mirrored_observation = mirror_direct_kicking_observation(
            flat_observation,
            len(self.prediction_horizons_s),
        )
        mirrored_observation_action_mean = self.actor.action_mean(
            mirrored_observation
        )
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

    def est_value(
        self,
        obs: torch.Tensor,
        privileged_obs: torch.Tensor,
    ) -> torch.Tensor:
        if obs.shape[:-1] != privileged_obs.shape[:-1]:
            raise ValueError("Actor and privileged observation leading shapes differ")
        if obs.shape[-1] != self.num_observations:
            raise ValueError("DirectKicking critic observation shape mismatch")
        values = self.critic(
            obs.reshape(-1, self.num_observations),
            privileged_obs.reshape(-1, self.num_privileged_observations),
        )
        return values.reshape(obs.shape[:-1])
