"""Проверка правил призовых мест по финалу, суперфиналу и полуфиналу."""
import unittest

from armplaces.podium import infer_podium
from simulation import Config, generate_field, simulate


class PodiumTests(unittest.TestCase):
    def test_final_without_reset(self):
        names = {1:'A', 2:'B', 3:'C'}
        pairs = [('B','A'), ('C','A'), ('C','B'), ('B','A')]
        result = infer_podium(names, pairs)
        self.assertEqual(result['places'], {'A':1, 'B':2, 'C':3})
        self.assertEqual(result['final_kind'], 'final')

    def test_both_superfinal_outcomes(self):
        names = {1:'A', 2:'B', 3:'C'}
        prefix = [('B','A'), ('C','A'), ('C','B'), ('A','B')]
        for final, expected in ((('A','B'), {'B':1,'A':2,'C':3}),
                                (('B','A'), {'A':1,'B':2,'C':3})):
            result = infer_podium(names, prefix+[final])
            self.assertEqual(result['places'], expected)
            self.assertEqual(result['final_kind'], 'superfinal')

    def test_small_categories_and_passes(self):
        self.assertEqual(infer_podium({1:'A'}, [])['places'], {'A':1})
        self.assertEqual(infer_podium({1:'A',2:'B'}, [('', 'A'), ('B','A'), ('B','A')])['places'],
                         {'A':1,'B':2})
        self.assertEqual(infer_podium({1:'A',2:'B'}, [('B','A'), ('A','B'), ('A','B')])['places'],
                         {'B':1,'A':2})

    def test_incomplete_or_invalid_tournaments_do_not_lock_places(self):
        names = {1:'A',2:'B',3:'C'}
        for pairs in ([], [('B','A')], [('B','A'),('B','A'),('B','C')],
                      [('B','A'),('C','A'),('C','B'),('A','B')]):
            result = infer_podium(names, pairs)
            self.assertEqual(result['status'], 'undefined')
            self.assertEqual(result['places'], {})
        with self.assertRaises(ValueError):
            infer_podium(names, [('X','A')])

    def test_classification_bout_does_not_replace_final_or_semifinal(self):
        names = dict(enumerate('ABCDEF'))
        pairs = [('E','A'),('F','B'),('C','A'),('D','B'),('E','C'),('F','D'),
                 ('B','A'),('D','C'),('C','B'),('B','A')]
        expected = {'A':1,'B':2,'C':3}
        self.assertEqual(infer_podium(names, pairs+[('E','F')])['places'], expected)
        self.assertEqual(infer_podium(names, pairs[:-1]+[('E','F')]+pairs[-1:])['places'], expected)
        self.assertEqual(pairs[-1], ('B','A'))

    def test_all_simulated_sizes_agree_with_bracket_podium(self):
        final_kinds = set()
        for n in range(2, 65):
            for repeat in range(5):
                config = Config(participants=n)
                ratings, draw = generate_field(config, repeat)
                result = simulate(config, 'elo', ratings, draw, repeat=repeat)
                names = {i:str(i) for i in draw}
                pairs = [(str(b['loser']),str(b['winner'])) for b in result['bouts'] if not b['technical']]
                podium = infer_podium(names, pairs)
                self.assertEqual(podium['places'], {str(i):p for i,p in result['places'].items() if p <= 3}, (n,repeat))
                final_kinds.add(podium['final_kind'])
        self.assertEqual(final_kinds, {'final','superfinal'})
