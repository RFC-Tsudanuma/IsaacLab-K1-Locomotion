"""Joga-sized PPO networks with the existing two-layer forecast representation."""
import copy

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP

from .config import FRAME_SIZE, PROPRIO_SIZE, FORECAST_SIZE, HISTORY_LENGTH, POLICY_SIZE, PRIVILEGED_SIZE


class ForecastHistoryNetwork(torch.nn.Module):
    def __init__(self, output_size, privileged_size, cfg):
        super().__init__()
        self.frame_size = FRAME_SIZE
        self.history_length = HISTORY_LENGTH
        self.proprio_size = PROPRIO_SIZE
        self.forecast_size = FORECAST_SIZE
        self.history_size = POLICY_SIZE
        model = cfg["model"]
        self.forecast = torch.nn.Sequential(
            torch.nn.Linear(FORECAST_SIZE, model["forecast_hidden_size"]),
            torch.nn.ELU(),
            torch.nn.Linear(model["forecast_hidden_size"], model["forecast_output_size"]),
        )
        encoded_size = (PROPRIO_SIZE + 3 + model["forecast_output_size"]) * HISTORY_LENGTH
        hidden = model["actor_hidden_dims"] if privileged_size == 0 else model["critic_hidden_dims"]
        self.network = MLP(encoded_size + privileged_size, output_size, hidden, "elu")

    def forward(self, observation):
        frames = observation[:, :self.history_size].reshape(-1, self.history_length, self.frame_size)
        forecast_end = self.proprio_size + self.forecast_size
        encoded = torch.cat((frames[..., :self.proprio_size],
                             self.forecast(frames[..., self.proprio_size:forecast_end]),
                             frames[..., forecast_end:]), dim=-1)
        combined = torch.cat((encoded.flatten(1), observation[:, self.history_size:]), dim=-1)
        return self.network(combined)


class ReceivingActorCritic(ActorCritic):
    def __init__(self, obs, cfg):
        super().__init__(
            obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
            actor_obs_normalization=True, critic_obs_normalization=True,
            actor_hidden_dims=cfg["model"]["actor_hidden_dims"],
            critic_hidden_dims=cfg["model"]["critic_hidden_dims"], activation="elu",
            init_noise_std=1., noise_std_type="log",
        )
        self.actor = ForecastHistoryNetwork(12, 0, cfg)
        self.critic = ForecastHistoryNetwork(1, PRIVILEGED_SIZE, cfg)

    def export_actor(self):
        """Self-contained CPU policy: normalized raw history -> 12 joint actions."""
        return torch.nn.Sequential(copy.deepcopy(self.actor_obs_normalizer),
                                   copy.deepcopy(self.actor)).cpu().eval()
