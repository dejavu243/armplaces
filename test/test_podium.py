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

    def test_fixed_podium_breaks_its_cycles(self):
        from armplaces.topological_sort import rank_tournament
        # B and C each beat the other, but C is the semifinal loser.
        names = {1:'A',2:'B',3:'C'}
        pairs = [('B','C'),('C','A'),('C','B'),('B','A')]
        self.assertEqual(rank_tournament(names,pairs),
                         {'status':'ok','reason':'','places':{'A':1,'B':2,'C':3}})
        self.assertEqual(pairs[0], ('B','C'))

    def test_remaining_cycle_preserves_the_podium(self):
        from armplaces.topological_sort import rank_tournament
        from armplaces.grinev_algorithm import TournamentGraphConstructor, RankingUndefined
        names = dict(enumerate('ABCDEF'))
        # A/B/C are proven separately; a cycle among D/E/F must not be invented away.
        pairs = [('D','E'), ('E','F'), ('F','D')]
        with self.assertRaises(RankingUndefined):
            TournamentGraphConstructor(names, pairs, fixed_places={'A':1,'B':2,'C':3})
        config = Config(participants=16)
        seen_reasons = set()
        for repeat in range(100):
            ratings, draw = generate_field(config, repeat)
            result = simulate(config, 'elo', ratings, draw, repeat=repeat)
            bout_pairs = [(str(b['loser']),str(b['winner'])) for b in result['bouts'] if not b['technical']]
            ranking = rank_tournament({i:str(i) for i in draw}, bout_pairs)
            if ranking['status'] == 'partial':
                self.assertEqual(ranking['places'], {str(i):p for i,p in result['places'].items() if p<=3})
                self.assertTrue(ranking['reason'])
                import json
                import tempfile
                from pathlib import Path
                from armplaces.__main__ import save_graph
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    (output/'edgelist.txt').write_text('stale graph')
                    saved = save_graph({i:str(i) for i in draw}, bout_pairs, output)
                    self.assertEqual(saved, ranking)
                    self.assertEqual(len((output/'places.txt').read_text().splitlines()), 3)
                    self.assertEqual(json.loads((output/'places.json').read_text()), ranking)
                    if 'Cycle' in ranking['reason']:
                        self.assertFalse((output/'edgelist.txt').exists())
                    else:
                        self.assertNotEqual((output/'edgelist.txt').read_text(), 'stale graph')
                seen_reasons.add(ranking['reason'])
                if len(seen_reasons) == 2:
                    break
        else:
            self.fail('Expected both cyclic and disconnected partial-ranking examples')
