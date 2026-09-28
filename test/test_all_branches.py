"""Short and joining branches must survive graph construction."""
import json
from pathlib import Path
import unittest
import networkx as nx
from armplaces.grinev_algorithm import TournamentGraphConstructor
from armplaces.topological_sort import rank_tournament


class AllBranchesTests(unittest.TestCase):
    def test_shortcut_and_shared_descendant_keep_every_edge(self):
        pairs = [('a','b'),('b','c'),('c','d'),('d','e'),('a','x'),('x','e')]
        constructor = TournamentGraphConstructor(dict(enumerate('abcdex')), pairs)
        graph = constructor.make_graph()
        self.assertEqual(set(graph.edges()), {(b,a) for a,b in pairs})
        self.assertEqual(len(graph), 6)
        self.assertEqual(set(graph.successors('e')), {'d','x'})
        self.assertEqual(graph.in_degree('a'), 2)
        self.assertTrue(all(data['weight'] > 0 for _,_,data in graph.edges(data=True)))

    def test_documented_isolates_rank_without_initial_ratings(self):
        root = Path(__file__).resolve().parents[1]/'test/fixtures'
        for name in ('03-isolated-participant', '04-isolated-group'):
            case = json.loads((root/(name+'.json')).read_text())
            names = dict(enumerate(case['draw']))
            podium = {n:p for n,p in case['graph_only']['places'].items() if p <= 3}
            constructor = TournamentGraphConstructor(names, case['pairs'], fixed_places=podium)
            graph = constructor.make_graph()
            self.assertEqual(set(graph.edges()), {(b,a) for a,b in constructor.up.edges()})
            self.assertEqual(list(nx.isolates(graph)), [])
            result = rank_tournament(names, case['pairs'])
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(sorted(result['places'].values()), list(range(1,17)))
            self.assertTrue(all(result['places'][n] == p for n,p in podium.items()))
            self.assertEqual(result, rank_tournament(names, case['pairs'], ratings=case['ratings']))
