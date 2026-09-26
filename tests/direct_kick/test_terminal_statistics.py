"""Completed-episode console statistics without changing training tensors."""
import unittest

import torch

from test_source_parity import load_port


EpisodeStatistics = load_port('terminal').EpisodeStatistics


class TerminalStatisticsTest(unittest.TestCase):
    def test_staggered_completions_keep_live_episodes_and_drain_term_totals(self):
        statistics = EpisodeStatistics(3, 'cpu', episode_length_s=17.0)
        statistics.record(
            torch.tensor([2., 3., 100.]), torch.tensor([False, False, False]),
            {'approach': torch.tensor([1., 2., 1000.]),
             'penalty': torch.tensor([-0.5, -1., -20.])},
        )
        statistics.record(
            torch.tensor([4., 5., 100.]), torch.tensor([True, False, False]),
            {'approach': torch.tensor([3., 4., 1000.]),
             'penalty': torch.tensor([-1., -2., -20.])},
        )
        first = statistics.finish_iteration()
        self.assertEqual(first['reward'], 6.)
        self.assertEqual(first['episode_length'], 2.)
        self.assertAlmostEqual(first['reward_terms']['approach'], 4. / 17.)
        self.assertAlmostEqual(first['reward_terms']['penalty'], -1.5 / 17.)

        statistics.record(
            torch.tensor([7., 11., 100.]), torch.tensor([False, True, False]),
            {'approach': torch.tensor([5., 6., 1000.]),
             'penalty': torch.tensor([-3., -4., -20.])},
        )
        statistics.record(
            torch.tensor([13., 17., 100.]), torch.tensor([True, True, False]),
            {'approach': torch.tensor([7., 8., 1000.]),
             'penalty': torch.tensor([-5., -6., -20.])},
        )
        second = statistics.finish_iteration()
        # Four completed episodes: returns 6, 19, 20, 17; lengths 2, 3, 2, 1.
        # The large third environment has never completed and is excluded.
        self.assertEqual(second['reward'], 15.5)
        self.assertEqual(second['episode_length'], 2.)
        # Term means use this iteration's three episodes, including their
        # pre-boundary rewards, and IsaacLab's configured 17 second divisor.
        self.assertAlmostEqual(second['reward_terms']['approach'], 32. / 3. / 17.)
        self.assertAlmostEqual(second['reward_terms']['penalty'], -21. / 3. / 17.)
        drained = statistics.finish_iteration()
        self.assertEqual(drained['reward'], 15.5)
        self.assertEqual(drained['episode_length'], 2.)
        self.assertEqual(drained['reward_terms'], {})

    def test_no_completion_is_distinct_from_a_completed_zero_return(self):
        statistics = EpisodeStatistics(1, 'cpu', episode_length_s=17.0)
        empty = statistics.finish_iteration()
        self.assertIsNone(empty['reward'])
        self.assertIsNone(empty['episode_length'])
        self.assertEqual(empty['reward_terms'], {})
        statistics.record(
            torch.tensor([0.]), torch.tensor([False]), {'kick': torch.tensor([0.])},
        )
        unfinished = statistics.finish_iteration()
        self.assertIsNone(unfinished['reward'])
        self.assertIsNone(unfinished['episode_length'])
        self.assertEqual(unfinished['reward_terms'], {})
        statistics.record(
            torch.tensor([0.]), torch.tensor([True]), {'kick': torch.tensor([0.])},
        )
        completed = statistics.finish_iteration()
        self.assertEqual(completed['reward'], 0.)
        self.assertEqual(completed['episode_length'], 2.)
        self.assertEqual(completed['reward_terms'], {'kick': 0.})

    def test_return_window_keeps_only_the_last_100_completed_episodes(self):
        statistics = EpisodeStatistics(1, 'cpu', episode_length_s=17.0)
        for reward in range(1, 106):
            statistics.record(torch.tensor([float(reward)]), torch.tensor([True]), {})
        summary = statistics.finish_iteration()
        self.assertEqual(summary['reward'], sum(range(6, 106)) / 100.)
        self.assertEqual(summary['episode_length'], 1.)
        self.assertEqual(summary['reward_terms'], {})

    def test_reset_discards_completed_history_and_unfinished_episode(self):
        statistics = EpisodeStatistics(2, 'cpu', episode_length_s=17.0)
        statistics.record(
            torch.tensor([9., 100.]), torch.tensor([True, False]),
            {'approach': torch.tensor([2., 1000.])},
        )
        statistics.reset()
        cleared = statistics.finish_iteration()
        self.assertIsNone(cleared['reward'])
        self.assertIsNone(cleared['episode_length'])
        self.assertEqual(cleared['reward_terms'], {})
        statistics.record(
            torch.tensor([3., 4.]), torch.tensor([False, True]),
            {'approach': torch.tensor([6., 8.])},
        )
        fresh = statistics.finish_iteration()
        self.assertEqual(fresh['reward'], 4.)
        self.assertEqual(fresh['episode_length'], 1.)
        self.assertAlmostEqual(fresh['reward_terms']['approach'], 8. / 17.)

    def test_record_leaves_inputs_unchanged_and_does_not_alias_reused_buffers(self):
        statistics = EpisodeStatistics(2, 'cpu', episode_length_s=17.0)
        rewards = torch.tensor([1., 2.])
        dones = torch.tensor([False, True])
        terms = {'kick': torch.tensor([10., 20.])}
        originals = [rewards.clone(), dones.clone(), terms['kick'].clone()]
        statistics.record(rewards, dones, terms)
        for actual, original in zip([rewards, dones, terms['kick']], originals):
            self.assertTrue(torch.equal(actual, original))
        rewards.copy_(torch.tensor([3., 4.]))
        dones.copy_(torch.tensor([True, False]))
        terms['kick'].copy_(torch.tensor([30., 40.]))
        statistics.record(rewards, dones, terms)
        rewards.fill_(999.)
        dones.fill_(False)
        terms['kick'].fill_(999.)
        summary = statistics.finish_iteration()
        self.assertEqual(summary['reward'], 3.)
        self.assertEqual(summary['episode_length'], 1.5)
        self.assertAlmostEqual(summary['reward_terms']['kick'], 60. / 2. / 17.)


if __name__ == '__main__':
    unittest.main()
