"""Tabulated simulation must replay the actual fixed DE_OLD columns."""
import copy
import unittest

from armplaces.tables.DE_OLD_Winner_fix import DE_OLD_winner_fix as winners
from armplaces.tables.DE_OLD_Loser_fix import DE_OLD_loser_fix as losers
from simulation import Config, build_bracket, generate_field, simulate, validate_result


class TableSimulationTests(unittest.TestCase):
    def test_routes_equal_fixed_tables(self):
        for n in range(2, 33):
            bracket = build_bracket(n)
            self.assertEqual(len(bracket.matches), 2*n-1)
            for i, match in enumerate(bracket.matches):
                self.assertEqual(match.winner_to+1, winners[i][n-2])
                self.assertEqual(match.loser_to+1, losers[i][n-2])

    def test_replay_all_tables_and_both_final_outcomes(self):
        resets = set()
        for n in range(2, 33):
            for repeat in range(30):
                ratings, draw = generate_field(Config(participants=n), repeat)
                result = simulate(Config(participants=n), 'elo', ratings, draw, repeat=repeat, grin_tour=True)
                resets.add(result['reset'])
                gs = [None]*(5*n)
                gs[:n] = draw
                for i, bout in enumerate(result['bouts']):
                    self.assertEqual([bout['a'], bout['b']], gs[2*i:2*i+2])
                    for who, table in ((bout['winner'], winners), (bout['loser'], losers)):
                        gs[table[i][n-2]-1] = who
                self.assertEqual(result['sequence'], gs[:4*n])
                self.assertEqual(result['eliminated_slots'],
                                 {i+1: who for i, who in enumerate(gs) if i >= 4*n and who is not None})
                self.assertEqual(result['grin_tour']['status'], 'ok')
        self.assertEqual(resets, {True, False})

    def test_nine_initial_cells_feed_pair_five_without_a_bye(self):
        result = simulate(Config(participants=9), 'strong-win', [9-i for i in range(9)], list(range(9)))
        pairs = [(b['a'], b['b']) for b in result['bouts']]
        self.assertEqual(pairs[:5], [(0,1), (2,3), (4,5), (6,7), (8,0)])
        self.assertEqual(len(result['sequence']), 36)

    def test_missing_tables_and_corrupted_sequence(self):
        for n in (33, 64, 128):
            with self.assertRaisesRegex(ValueError, 'no tables'):
                Config(participants=n).validate()
            with self.assertRaisesRegex(ValueError, 'no tables'):
                build_bracket(n)
        original = simulate(Config(participants=9), 'strong-win', [1.]*9, list(range(9)))
        for field in ('sequence', 'eliminated_slots', 'bouts'):
            result = copy.deepcopy(original)
            if field == 'sequence':
                result[field][-1] = 123
            elif field == 'eliminated_slots':
                result[field].clear()
            else:
                result[field][0]['a'] = 123
            with self.assertRaises(RuntimeError):
                validate_result(result)
