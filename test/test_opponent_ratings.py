"""Regression coverage for rating comparisons after graph failure."""
import unittest
from armplaces.topological_sort import rank_by_opponents, rank_tournament
from simulation import Config, generate_field, simulate


class OpponentRatingTests(unittest.TestCase):
    def test_loss_priority_then_best_win_then_draw(self):
        names = dict(enumerate('ABCDEF'))
        ratings = dict(zip('ABCDEF', [100, 80, 60, 40, 30, 20]))
        pairs = [('D','A'), ('D','B'), ('E','A'), ('E','C'), ('F','A'), ('F','B'),
                 ('C','D'), ('B','F')]
        places = rank_by_opponents(names, pairs, ratings, {'A':1,'B':2,'C':3})
        self.assertEqual(places, {'A':1,'B':2,'C':3,'F':4,'D':5,'E':6})
        tied = [('D','A'), ('E','A'), ('F','A')]
        self.assertEqual(rank_by_opponents(names, tied, ratings, {'A':1,'B':2,'C':3})['D'], 4)

    def test_both_graph_failures_resolved_without_changing_podium_or_journal(self):
        reasons = set()
        for repeat in range(100):
            config = Config(participants=16)
            ratings, draw = generate_field(config, repeat)
            result = simulate(config, 'elo', ratings, draw, repeat=repeat, grin_tour=True)
            pairs = [(str(b['loser']), str(b['winner'])) for b in result['bouts'] if not b['technical']]
            names = {i: str(i) for i in draw}
            old = rank_tournament(names, pairs)
            new = result['grin_tour']
            self.assertEqual(new['status'], 'ok')
            self.assertEqual(sorted(new['places'].values()), list(range(1,17)))
            if old['status'] == 'partial':
                reasons.add(old['reason'])
                self.assertEqual(new['method'], 'opponent_ratings')
                self.assertEqual(new['graph_reason'], old['reason'])
                self.assertTrue(all(new['places'][p] == place for p, place in old['places'].items()))
            else:
                self.assertEqual(old, new)
            self.assertEqual(rank_tournament(names, pairs), old)
        self.assertEqual(len(reasons), 2)

    def test_invalid_ratings(self):
        for ratings in ({}, {'A':float('nan')}, {'A':0}, {'A':1,'B':2}):
            with self.assertRaises(ValueError):
                rank_tournament({0:'A'}, [], ratings=ratings)
