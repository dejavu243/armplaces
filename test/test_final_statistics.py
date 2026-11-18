import unittest

from simulation import Config, simulate
from simulation.final_statistics import final_record, summarize_finals
from test.test_simulation import FixedRandom


class FinalStatisticsTests(unittest.TestCase):
    def test_branch_orientation_and_superfinal(self):
        config = Config(participants=2)
        for draw, expected_upper in (([0, 1], 0), ([1, 0], 1)):
            result = simulate(config, 'elo', [1000., 1100.], draw,
                              rng=FixedRandom([0., .999, 0.]))
            row = final_record(result)
            self.assertEqual(row['upper'], expected_upper)
            self.assertEqual(row['lower'], 1-expected_upper)
            self.assertEqual(row['gap'], -100 if expected_upper == 0 else 100)
            self.assertEqual(row['reset_gap'], row['gap'])
            self.assertTrue(row['reset'])
            self.assertEqual(row['abs_gap'], 100)
            self.assertEqual(row['champion'], result['champion'])

    def test_no_superfinal_and_singleton(self):
        result = simulate(Config(participants=2), 'elo', [1000., 1000.], [0, 1],
                          rng=FixedRandom([0., 0.]))
        row = final_record(result)
        self.assertEqual(row['gap'], 0)
        self.assertEqual(row['lower_win_probability'], .5)
        self.assertFalse(row['reset'])
        self.assertIsNone(row['reset_gap'])
        single = simulate(Config(participants=1), 'elo', [1000.], [0])
        self.assertIsNone(final_record(single))

    def test_statistics_keep_signed_gaps_and_conditional_denominators(self):
        rows = [{'gap': -100, 'reset': True, 'reset_gap': -50},
                {'gap': 200, 'reset': False, 'reset_gap': None},
                {'gap': -300, 'reset': False, 'reset_gap': None}]
        summary = summarize_finals(rows)
        self.assertEqual(summary['reset_count'], 1)
        self.assertEqual(summary['reset_rate'], 1/3)
        self.assertEqual(summary['final']['samples'], 3)
        self.assertAlmostEqual(summary['final']['mean_gap'], -200/3)
        self.assertEqual(summary['final']['mean_abs_gap'], 200)
        self.assertEqual(summary['final']['median_abs_gap'], 200)
        self.assertEqual(summary['final']['p95_abs_gap'], 290)
        self.assertEqual(summary['superfinal']['samples'], 1)
        self.assertEqual(summary['superfinal']['mean_abs_gap'], 50)
        self.assertIsNone(summarize_finals([])['reset_rate'])
        self.assertIsNone(summarize_finals([])['final']['mean_abs_gap'])
