"""Display-only episode statistics for the IsaacLab-style training terminal."""
from collections import deque
from statistics import mean

import torch


class EpisodeStatistics:
    """Keep in-flight returns separate from the last 100 completed episodes.

    Reward terms use IsaacLab's episode-sum / maximum-duration normalization.
    Their iteration mean weights each completed episode equally.
    """

    def __init__(self, num_envs, device, episode_length_s):
        self.episode_length_s = episode_length_s
        self._returns = torch.zeros(num_envs, device=device)
        self._lengths = torch.zeros(num_envs, device=device)
        self._term_sums = {}
        self._completed_terms = {}
        self._completed_count = 0
        self._recent_returns = deque(maxlen=100)
        self._recent_lengths = deque(maxlen=100)

    def reset(self):
        self._returns.zero_()
        self._lengths.zero_()
        self._term_sums.clear()
        self._completed_terms.clear()
        self._completed_count = 0
        self._recent_returns.clear()
        self._recent_lengths.clear()

    @torch.no_grad()
    def record(self, rewards, dones, reward_terms):
        self._returns += rewards.to(self._returns.device)
        self._lengths += 1
        for name, values in reward_terms.items():
            if name not in self._term_sums:
                self._term_sums[name] = torch.zeros_like(self._returns)
                self._completed_terms[name] = self._returns.new_zeros(())
            self._term_sums[name] += values.to(self._returns.device)

        finished = dones.to(self._returns.device).nonzero(as_tuple=False).flatten()
        if finished.numel() == 0:
            return
        self._recent_returns.extend(self._returns[finished].cpu().tolist())
        self._recent_lengths.extend(self._lengths[finished].cpu().tolist())
        self._completed_count += finished.numel()
        for name, values in self._term_sums.items():
            self._completed_terms[name] += values[finished].sum()
            values[finished] = 0
        self._returns[finished] = 0
        self._lengths[finished] = 0

    def finish_iteration(self):
        terms = {}
        if self._completed_count:
            for name, value in self._completed_terms.items():
                terms[name] = value.item() / self._completed_count / self.episode_length_s
        result = {
            'reward': mean(self._recent_returns) if self._recent_returns else None,
            'episode_length': mean(self._recent_lengths) if self._recent_lengths else None,
            'reward_terms': terms,
        }
        for value in self._completed_terms.values():
            value.zero_()
        self._completed_count = 0
        return result


def format_duration(seconds):
    """Use an unbounded hour field, including for multi-day training runs."""
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
