"""Regression checks for historical recovery and the GrinTour heuristic."""
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import networkx as nx

from data.DE_OLD_Loser_fix import DE_OLD_loser_fix as DE_OLD_loser
from data.DE_OLD_Winner_fix import DE_OLD_winner_fix as DE_OLD_winner
from grinev_algorithm import TournamentGraphConstructor
from read_tournament import (RESULT_FILE_SUFFIX, drop_simple_cycles, read_names,
                             read_tournament_files, tournament_recovery)
from topological_sort import (get_places, get_target_points_sorted, get_tournament_dict,
                              rank_tournament)
from utils import drop_duplicates, get_max_chains


class HistoricalTests(unittest.TestCase):
    def test_hyphenated_names_and_technical_results(self):
        self.assertEqual(read_names(['Иванов-Петров', 'Б', '><'])[0][1], 'Иванов-Петров')
        names = {1: 'A', 2: 'B'}
        self.assertEqual(tournament_recovery({RESULT_FILE_SUFFIX: (names, ['>', '+'])}),
                         [('B', 'A')])
        self.assertEqual(tournament_recovery({RESULT_FILE_SUFFIX: (names, ['<', '+'])}),
                         [('A', 'B')])

    def test_raw_repeated_bouts_preserved(self):
        pairs = tournament_recovery({RESULT_FILE_SUFFIX: ({1: 'A', 2: 'B'}, ['+', '+'])})
        self.assertEqual(pairs, [('B', 'A'), ('B', 'A')])
        self.assertEqual(drop_simple_cycles(pairs), [('B', 'A')])
        self.assertEqual(drop_simple_cycles([('A', 'B'), ('B', 'A')]),
                         [('A', 'B'), ('B', 'A')])

    def test_paths_and_sample(self):
        for directory in (Path('data/left_hand_75kg'), 'data/left_hand_75kg'):
            files = read_tournament_files(directory)
            pairs = tournament_recovery(files)
            self.assertEqual(len(pairs), 40)
            ranking = rank_tournament(files[RESULT_FILE_SUFFIX][0], pairs)
            self.assertEqual(ranking['status'], 'ok')
            self.assertEqual(ranking['places']['Болдырев Эдуард'], 1)
            self.assertEqual(len(ranking['places']), 20)

    def test_bad_files(self):
        for lines in ([], ['A'], ['A', 'A', '+'], ['A', 'B', '+x']):
            with self.assertRaises(ValueError):
                read_names(lines)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                read_tournament_files(directory)
            Path(directory, 'a[DE].txt').write_text('A\nB\n++\n')
            Path(directory, 'b[DE].txt').write_text('A\nB\n++\n')
            with self.assertRaises(ValueError):
                read_tournament_files(directory)

    def test_legacy_tables_full_tournaments(self):
        for n in range(2, 33):
            for seed in range(50):
                rng = random.Random(seed)
                sequence = list(range(1, n + 1)) + [0] * (6 * n)
                losses = Counter()
                for i in range(2 * n - 1):
                    a, b = sequence[2*i:2*i+2]
                    self.assertTrue(a and b and a != b, (n, i))
                    self.assertLess(losses[a], 2, (n, i))
                    self.assertLess(losses[b], 2, (n, i))
                    winner, loser = (a, b) if rng.random() < .5 else (b, a)
                    losses[loser] += 1
                    if sum(losses[x] < 2 for x in range(1, n+1)) == 1:
                        break
                    for who, table in ((winner, DE_OLD_winner), (loser, DE_OLD_loser)):
                        dest = table[i][n-2]
                        if dest:
                            self.assertGreater(dest, 2*i+2)
                            self.assertEqual(sequence[dest-1], 0)
                            sequence[dest-1] = who
                self.assertEqual(sum(losses[x] < 2 for x in range(1, n+1)), 1)

    def test_original_tables_remain_unchanged(self):
        from data.DE_OLD_Loser import DE_OLD_loser as original
        self.assertEqual(original[17][11], 47)
        self.assertEqual(original[22][14], 59)
        self.assertEqual(original[44][26], 105)
        self.assertEqual(original[45][26], 131)
        self.assertEqual(DE_OLD_loser[17][11], 45)
        self.assertIsNot(original, DE_OLD_loser)
        from data.DE_OLD_Winner import DE_OLD_winner as original_winner
        self.assertEqual(original_winner[31][21], 82)
        self.assertEqual(DE_OLD_winner[31][21], 74)
        self.assertEqual(DE_OLD_winner[40][21], 84)



class GrinTourTests(unittest.TestCase):
    def test_names_are_not_substrings(self):
        constructor = TournamentGraphConstructor({1: 'Ann', 2: 'Anna', 3: 'B'},
                                                [('Ann', 'Anna'), ('B', 'Anna')])
        self.assertEqual(constructor.find_total_losers(), ['Ann', 'B'])

    def test_short_and_long_chains(self):
        self.assertEqual(rank_tournament({1: 'A'}, [])['places'], {'A': 1})
        self.assertEqual(rank_tournament({1: 'A', 2: 'B'}, [('B', 'A')])['places'],
                         {'A': 1, 'B': 2})
        names = dict(enumerate(map(str, range(160))))
        pairs = [(str(i+1), str(i)) for i in range(159)]
        result = rank_tournament(names, pairs)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['places']['159'], 160)
        self.assertEqual(drop_duplicates([['A'], ['A']]), [['A']])

    def test_cycles_and_disconnected(self):
        names = {1: 'A', 2: 'B', 3: 'C'}
        for pairs in ([('A', 'B'), ('B', 'C'), ('C', 'A')],
                      [('A', 'B'), ('B', 'A')], []):
            result = rank_tournament(names, pairs)
            self.assertEqual(result['status'], 'undefined')
            self.assertTrue(result['reason'])
            self.assertEqual(result['places'], {})

    def test_dynamic_weights_match_enumeration(self):
        for seed in range(20):
            rng = random.Random(seed)
            names = {i: str(i) for i in range(8)}
            pairs = [(str(i), str(j)) for i in range(8) for j in range(i+1, 8)
                     if rng.random() < .4]
            alg = TournamentGraphConstructor(names, pairs)
            expected = Counter()
            for chains in alg.make_all_chains().values():
                for chain in get_max_chains(chains):
                    expected.update((b, a) for a, b in zip(chain, chain[1:]))
            actual = {(a, b): d['weight'] for a, b, d in alg.make_graph().edges(data=True)}
            self.assertEqual(actual, dict(expected))

    def test_places_do_not_mutate_inputs_and_ties_are_stable(self):
        graph = nx.DiGraph([('A', 'B'), ('B', 'C'), ('B', 'D')])
        tour = get_tournament_dict(graph)
        result = get_places(tour, graph)
        self.assertEqual({name: item['place'] for name, item in result.items()},
                         {'A': 1, 'B': 2, 'C': 4, 'D': 3})
        self.assertEqual(len(graph), 4)
        self.assertTrue(all('place' not in item for item in tour.values()))
        self.assertEqual(get_target_points_sorted({'A': {'x': 2}, 'B': {'x': 1}}, 'x')[0][0], 'B')
        self.assertEqual(get_target_points_sorted({'A': {'x': 2}, 'B': {'x': 1}}, ['x'], True)[0][0], 'A')
