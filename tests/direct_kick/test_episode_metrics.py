"""Completed-episode denominators, immutable spawn categories and CSV output."""
import csv
from pathlib import Path
import tempfile
import unittest

import torch

from test_source_parity import load_port


metrics_module = load_port('episode_metrics')
KickEpisodeMetrics = metrics_module.KickEpisodeMetrics


class EpisodeMetricsTest(unittest.TestCase):
    def test_completed_only_categories_reasons_and_drain_with_live_episode(self):
        metrics = KickEpisodeMetrics(5, 'cpu')
        ids = torch.arange(5)
        speed = torch.tensor([0., 0.5, 1., 3., 6.])
        stationary = torch.tensor([True, False, False, False, False])
        offset = torch.tensor([0.75, -0.249, -0.25, 0.5, 0.75])
        metrics.start(ids, speed, stationary, offset)
        # Spawn categories must not follow later speed/position mutations.
        speed.zero_()
        offset.zero_()
        finished = torch.arange(4)
        kicked = torch.tensor([True, False, True, True])
        fell = torch.tensor([False, False, True, False])
        timeout = torch.tensor([False, True, True, False])
        post_kick = torch.tensor([True, False, True, True])
        self.assertEqual(metrics.finish(finished, kicked, fell, timeout, post_kick).item(), 3)
        # Repeated bookkeeping must not duplicate those episodes.
        self.assertEqual(metrics.finish(finished, kicked, fell, timeout, post_kick).item(), 0)
        summary = metrics.summary(reset=True)
        self.assertEqual(summary['all']['episodes'], 4)
        self.assertEqual(summary['all']['kick_rate'], 0.75)
        self.assertEqual(summary['all']['falls'], 1)
        self.assertEqual(summary['all']['timeouts'], 1)
        self.assertEqual(summary['all']['post_kick_completions'], 2)
        self.assertEqual(summary['all']['other_terminations'], 0)
        self.assertEqual(summary['stationary']['episodes'], 1)
        self.assertEqual(summary['stationary']['kick_rate'], 1.0)
        self.assertEqual(summary['moving']['episodes'], 3)
        self.assertEqual(summary['moving']['kick_rate'], 2 / 3)
        self.assertEqual(summary['speed_0_1_mps']['kick_rate'], 0.0)
        self.assertEqual(summary['speed_1_2_mps']['kick_rate'], 1.0)
        self.assertEqual(summary['speed_3_4_mps']['kick_rate'], 1.0)
        self.assertEqual(summary['offset_0_0p25_m']['kick_rate'], 0.0)
        self.assertEqual(summary['offset_0p25_0p50_m']['kick_rate'], 1.0)
        self.assertEqual(summary['offset_0p50_0p75_m']['kick_rate'], 1.0)
        self.assertIsNone(summary['speed_5_6_mps']['kick_rate'])
        self.assertIsNone(metrics.summary()['all']['kick_rate'])
        no = torch.tensor([False])
        metrics.finish(torch.tensor([4]), no, no, no, no)
        next_summary = metrics.summary()
        self.assertEqual(next_summary['all']['episodes'], 1)
        self.assertEqual(next_summary['all']['other_terminations'], 1)
        self.assertEqual(next_summary['speed_5_6_mps']['episodes'], 1)
        self.assertEqual(next_summary['offset_0p50_0p75_m']['episodes'], 1)

    def test_initial_reset_and_empty_rates_in_csv_and_wandb(self):
        metrics = KickEpisodeMetrics(1, 'cpu')
        yes = torch.tensor([True])
        no = torch.tensor([False])
        # No episode has been spawned yet; initialization is not a failure.
        metrics.finish(torch.tensor([0]), yes, no, no, yes)
        empty = metrics.summary()
        self.assertEqual(empty['all']['episodes'], 0)
        self.assertIsNone(empty['all']['kick_rate'])
        scalars = metrics_module.episode_metric_scalars(empty)
        self.assertNotIn('episodes/all/kick_rate', scalars)
        self.assertEqual(scalars['episodes/all/episodes'], 0)
        with tempfile.TemporaryDirectory(prefix='direct_kick_metrics_') as directory:
            path = Path(directory) / 'episode_metrics.csv'
            metrics_module.write_episode_metrics(path, empty, iteration=1)
            metrics.start(torch.tensor([0]), torch.tensor([3.]), no, torch.tensor([0.4]))
            metrics.finish(torch.tensor([0]), yes, no, no, yes)
            metrics_module.write_episode_metrics(path, metrics.summary(), iteration=2)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2 * len(metrics_module.GROUPS))
            overall = [r for r in rows if r['group'] == 'all']
            self.assertEqual(overall[0]['kick_rate'], '')
            self.assertEqual(overall[1]['kick_rate'], '1.0')
            # A new play evaluation replaces its prior evaluation summary.
            metrics_module.write_episode_metrics(path, empty, append=False)
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(metrics_module.GROUPS))


if __name__ == '__main__':
    unittest.main()
