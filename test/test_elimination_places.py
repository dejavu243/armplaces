import unittest

from armplaces.elimination import elimination_history, rank_tournament_far


class EliminationPlacementTests(unittest.TestCase):
    def test_first_eliminated_gets_last_and_podium_is_preserved(self):
        names = {i: str(i) for i in range(4)}
        pairs = [('1', '0'), ('3', '2'), ('3', '1'), ('2', '0'), ('2', '1'), ('1', '0')]
        result = rank_tournament_far(names, pairs)
        self.assertEqual(result['places'], {'0': 1, '1': 2, '2': 3, '3': 4})
        self.assertEqual([entry['participant'] for entry in result['diagnostics']['eliminated']], ['3', '2', '1'])
        history = elimination_history(names, pairs)
        self.assertEqual(history['wins'], {'0': 3, '1': 2, '2': 1, '3': 0})
        self.assertEqual(history['last_winner']['2'], '1')

    def test_reset_and_singleton(self):
        names = {0: 'a', 1: 'b'}
        pairs = [('b', 'a'), ('a', 'b'), ('b', 'a')]
        self.assertEqual(rank_tournament_far(names, pairs)['places'], {'a': 1, 'b': 2})
        self.assertEqual(rank_tournament_far({0: 'only'}, [])['places'], {'only': 1})

    def test_technical_and_classification_bouts_do_not_change_places(self):
        names = {i: str(i) for i in range(4)}
        pairs = [('1', '0'), ('3', '2'), ('3', '1'), ('2', '0'), ('2', '1'), ('1', '0')]
        expected = rank_tournament_far(names, pairs)['places']
        self.assertEqual(rank_tournament_far(names, [('', '2')]+pairs+[('3', '2')])['places'], expected)

    def test_incomplete_or_contradictory_history_has_no_full_ranking(self):
        names = {0: 'a', 1: 'b', 2: 'c'}
        for pairs in ([('b', 'a')], [('b', 'a'), ('b', 'c'), ('a', 'b')]):
            self.assertEqual(rank_tournament_far(names, pairs)['status'], 'undefined')
        with self.assertRaises(ValueError):
            rank_tournament_far(names, [('unknown', 'a')])


if __name__ == '__main__':
    unittest.main()
