import unittest

import numpy as np

from simulation import Config, build_bracket, elo_probability, generate_field, simulate, validate_result


class FixedRandom:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


class SimulationTests(unittest.TestCase):
    def test_all_sizes_and_models(self):
        for n in range(2, 129):
            config = Config(participants=n)
            ratings, draw = generate_field(config, 0)
            for model in ('strong-win', 'elo', 'elo-stamina'):
                result = simulate(config, model, ratings, draw)
                validate_result(result)
                if model == 'strong-win':
                    self.assertEqual(result['champion'], int(np.argmax(ratings)))
                bracket = build_bracket(n)
                for i, match in enumerate(bracket.matches):
                    for destination in (match.winner_to, match.loser_to):
                        if destination is not None:
                            self.assertGreater(destination, 2*i+1)

    def test_singleton(self):
        result = simulate(Config(participants=1), 'elo', [10.], [0], grin_tour=True)
        self.assertEqual(result['places'], {0: 1})
        self.assertEqual(result['bouts'], [])
        self.assertEqual(result['new_ratings'], [10.])
        self.assertEqual(result['grin_tour']['places'], {'0': 1})

    def test_finals_and_delayed_updates(self):
        config = Config(participants=2)
        result = simulate(config, 'elo', [100., 100.], [0, 1], rng=FixedRandom([.1, .1]))
        self.assertFalse(result['reset'])
        self.assertEqual(result['new_ratings'], [260., -60.])
        self.assertEqual([b['p_a'] for b in result['bouts']], [.5, .5])
        reset = simulate(config, 'elo', [100., 100.], [0, 1], rng=FixedRandom([.1, .9, .1]))
        self.assertTrue(reset['reset'])
        self.assertEqual(reset['champion'], 1)
        self.assertEqual(reset['new_ratings'], [20., 180.])

    def test_probabilities(self):
        self.assertEqual(elo_probability(100, 100), .5)
        self.assertEqual(elo_probability(1e300, 1e300, 1e-300), .5)
        self.assertEqual(elo_probability(1e300, 1e299, 1e-300), 1.)
        self.assertAlmostEqual(elo_probability(300, 100), 10/11)
        for a, b in ((100, 700), (1e300, 1), (1, 1e300)):
            p = elo_probability(a, b)
            self.assertTrue(0 <= p <= 1)
            self.assertAlmostEqual(p + elo_probability(b, a), 1)

    def test_stamina_zero_and_reproducibility(self):
        config = Config(participants=17, fatigue_rate=0)
        ratings, draw = generate_field(config, 3)
        a = simulate(config, 'elo', ratings, draw, repeat=3, grin_tour=True)
        b = simulate(config, 'elo-stamina', ratings, draw, repeat=3, grin_tour=True)
        self.assertEqual(a['bouts'], b['bouts'])
        self.assertEqual(a['places'], b['places'])
        self.assertEqual(a, simulate(config, 'elo', ratings, draw, repeat=3, grin_tour=True))
        self.assertEqual((ratings, draw), generate_field(config, 3))

    def test_byes_and_stamina(self):
        config = Config(participants=5)
        ratings, draw = generate_field(config, 0)
        result = simulate(config, 'elo-stamina', ratings, draw)
        self.assertTrue(any(b['technical'] for b in result['bouts']))
        played = [0]*5
        for b in result['bouts']:
            for side in ('a', 'b'):
                who = b[side]
                if who is not None:
                    self.assertEqual(b['previous_bouts_'+side], played[who])
                    if not b['technical']:
                        self.assertAlmostEqual(b['effective_'+side], ratings[who]*.95**played[who])
                        played[who] += 1
        self.assertEqual(played, result['played'])

    def test_shared_places(self):
        result = simulate(Config(participants=8), 'strong-win', list(range(1,9)), list(range(8)))
        self.assertEqual(sorted(result['places'].values()), [1,2,3,4,5,5,7,7])

    def test_invalid_inputs_and_corrupt_journal(self):
        for config in (Config(participants=129), Config(seed=-1), Config(alpha=float('nan')),
                       Config(repeats=0), Config(fatigue_rate=1), Config(k=-1)):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            simulate(Config(participants=2), 'elo', [1., 2.], [0,0])
        result = simulate(Config(participants=2), 'strong-win', [1., 2.], [0,1])
        result['bouts'][0]['winner'] = result['bouts'][0]['loser']
        with self.assertRaises(RuntimeError):
            validate_result(result)
