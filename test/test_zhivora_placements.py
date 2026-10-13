import json
import unittest
from pathlib import Path

from armplaces.elimination import rank_tournament_far
from armplaces.zhivora import rank_tournament_zhivora
from simulation import Config, generate_field, simulate


class ZhivoraPlacementTests(unittest.TestCase):
    def test_hand_checked_eight_and_sixteen_person_histories(self):
        cases = json.loads(Path('test/fixtures/zhivora_tournaments.json').read_text())
        for case in cases:
            names = {who: str(who) for who in case['draw']}
            for fixed_top, field in ((4, 'zhivora_four_order'), (6, 'zhivora_six_order')):
                with self.subTest(n=case['participants'], fixed_top=fixed_top):
                    ranked = rank_tournament_zhivora(names, case['pairs'], fixed_top=fixed_top)
                    expected = {name: place for place, name in enumerate(case[field], 1)}
                    self.assertEqual(ranked['places'], expected)
            far = rank_tournament_far(names, case['pairs'])
            self.assertEqual(far['places'], {name: place for place, name in enumerate(case['far_order'], 1)})
        # At N=16 a two-win athlete who lost to second place beats another
        # two-win athlete who lost to fourth place in the optional top-four mode.
        case = cases[1]
        names = {who: str(who) for who in case['draw']}
        four = rank_tournament_zhivora(names, case['pairs'], fixed_top=4)
        self.assertEqual((four['places']['5'], four['places']['10']), (6, 7))
        self.assertNotEqual(case['zhivora_four_order'], case['zhivora_six_order'])
        default = rank_tournament_zhivora(names, case['pairs'])
        self.assertEqual(default['places'], {name: place for place, name in enumerate(case['zhivora_six_order'], 1)})
        self.assertEqual(len(default['diagnostics']['fixed_places']), 6)
        self.assertEqual(min(default['places'][who] for group in default['diagnostics']['groups']
                             for who in group['order']), 7)

    def test_equal_win_and_opponent_places_use_draw_order(self):
        names = {i: name for i, name in enumerate(['a', 'b', 'h', 'g', 'f', 'c', 'e', 'd'])}
        pairs = [(name, 'a') for name in 'bcdefgh']+[(name, 'a') for name in 'cdefghb']
        ranked = rank_tournament_zhivora(names, pairs, fixed_top=4)
        self.assertEqual(ranked['places'], {'a': 1, 'b': 2, 'h': 3, 'g': 4,
                                          'f': 5, 'c': 6, 'e': 7, 'd': 8})
        self.assertEqual(ranked['diagnostics']['groups'][0]['order'], ['f', 'c', 'e', 'd'])

    def test_all_sizes_with_description_and_normal_pascal_sizes(self):
        for n in range(1, 33):
            for repeat in range(3):
                config = Config(participants=n)
                ratings, draw = generate_field(config, repeat)
                original = simulate(config, 'elo-tire', ratings, draw, repeat,
                                    grin_tour=True, grin_cycle_score=True)
                ranked = simulate(config, 'elo-tire', ratings, draw, repeat,
                                  grin_tour=True, grin_cycle_score=True,
                                  far_tour=True, zhivora_tour=True, zhivora_fixed_top=4)
                self.assertEqual(original['bouts'], ranked['bouts'])
                self.assertEqual(original['new_ratings'], ranked['new_ratings'])
                self.assertEqual(original['grin_tour'], ranked['grin_tour'])
                self.assertEqual(original['grin_cycle_score'], ranked['grin_cycle_score'])
                for method in ('far_tour', 'zhivora_tour'):
                    self.assertEqual(sorted(ranked[method]['places'].values()), list(range(1, n+1)))
                    for who, place in original['places'].items():
                        if place <= 3:
                            self.assertEqual(ranked[method]['places'][str(who)], place)
                if n in (1, 2, 3, 4, 8, 16, 32):
                    pascal = simulate(config, 'elo-tire', ratings, draw, repeat,
                                      far_tour=True, zhivora_tour=True)
                    self.assertEqual(pascal['zhivora_tour']['diagnostics']['fixed_top'], 6)

    def test_reset_and_invalid_or_incomplete_input(self):
        self.assertEqual(rank_tournament_zhivora({0: 'a', 1: 'b'},
                                                [('b', 'a'), ('a', 'b'), ('b', 'a')])['places'], {'a': 1, 'b': 2})
        self.assertEqual(rank_tournament_zhivora({0: 'a', 1: 'b'}, [('a', 'b')])['status'], 'undefined')
        for value in (0, 5, True):
            with self.assertRaises(ValueError):
                rank_tournament_zhivora({0: 'a'}, [], fixed_top=value)
        with self.assertRaises(ValueError):
            rank_tournament_zhivora({0: 'a', 1: 'a'}, [])
        with self.assertRaises(ValueError):
            rank_tournament_zhivora({0: 'a', 1: 'b'}, [('a', 'unknown')])


if __name__ == '__main__':
    unittest.main()
