"""Bounded, diagnostic-only counts of completed DirectKick episodes."""
import csv
from pathlib import Path

import torch


GROUPS = (
    'all', 'stationary', 'moving',
    'speed_0_1_mps', 'speed_1_2_mps', 'speed_2_3_mps',
    'speed_3_4_mps', 'speed_4_5_mps', 'speed_5_6_mps',
    'offset_0_0p25_m', 'offset_0p25_0p50_m', 'offset_0p50_0p75_m',
)
COUNTS = ('episodes', 'kicks', 'falls', 'timeouts', 'post_kick_completions', 'other_terminations')
RATES = ('kick_rate', 'fall_rate', 'timeout_rate', 'post_kick_completion_rate', 'other_termination_rate')


class KickEpisodeMetrics:
    """Keep one group mask per live environment, plus fixed-size counters.

    A kick uses the existing valid_kick event; it does not imply a correct pass
    direction or subsequent balance. All rates use completed episodes only.
    """
    def __init__(self, num_envs, device):
        self.membership = torch.zeros(num_envs, len(GROUPS), dtype=torch.bool, device=device)
        self.counts = torch.zeros(len(GROUPS), len(COUNTS), dtype=torch.long, device=device)

    def start(self, env_ids, speed, stationary, closest_offset):
        moving = ~stationary
        offset = closest_offset.abs()
        groups = [torch.ones_like(stationary), stationary, moving]
        for lower in range(6):
            upper = speed <= 6.0 if lower == 5 else speed < lower + 1.0
            groups.append(moving & (speed >= lower) & upper)
        groups.extend((
            moving & (offset < 0.25),
            moving & (offset >= 0.25) & (offset < 0.50),
            moving & (offset >= 0.50) & (offset <= 0.75),
        ))
        self.membership[env_ids] = torch.stack(groups, dim=1)

    def finish(self, env_ids, kicked, fell, timed_out, post_kick_done):
        membership = self.membership[env_ids]
        # Termination reasons are exclusive, with falls taking priority.
        timeout = timed_out & ~fell
        post_kick = post_kick_done & ~fell & ~timeout
        other = ~(fell | timeout | post_kick)
        outcomes = torch.stack((torch.ones_like(kicked), kicked, fell, timeout, post_kick, other), dim=1)
        self.counts += (membership[:, :, None] & outcomes[:, None, :]).sum(dim=0)
        completed_kicks = (membership[:, 0] & kicked).sum()
        # Consume membership so the same episode cannot count twice.
        self.membership[env_ids] = False
        return completed_kicks

    def summary(self, reset=False):
        rows = self.counts.cpu().tolist()
        result = {}
        for group, counts in zip(GROUPS, rows):
            row = dict(zip(COUNTS, counts))
            row.update({key: value / counts[0] if counts[0] else None
                        for key, value in zip(RATES, counts[1:])})
            result[group] = row
        if reset:
            self.counts.zero_()
        return result


def write_episode_metrics(path, summary, iteration=None, append=True):
    """Write group counts and rates; an empty rate means no completed samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = append and path.exists() and path.stat().st_size > 0
    with path.open('a' if append else 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('iteration', 'group', *COUNTS, *RATES))
        if not has_header:
            writer.writeheader()
        for group, row in summary.items():
            writer.writerow({'iteration': iteration, 'group': group, **row})


def episode_metric_scalars(summary):
    return {f'episodes/{group}/{key}': value
            for group, row in summary.items() for key, value in row.items() if value is not None}
