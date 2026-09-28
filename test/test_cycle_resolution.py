"""Residual-cycle scoring and its strict standard-pass dispatch."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from armplaces.cycle_resolution import (normalize_weights, ranked_scores,
                                        rank_tournament_cycle_score, score_component)
from armplaces.grinev_algorithm import RankingUndefined, TournamentGraphConstructor
from armplaces.topological_sort import rank_tournament

ROOT = Path(__file__).resolve().parents[1]


def case(path):
    return json.loads((ROOT/path).read_text())


class CycleResolutionTests(unittest.TestCase):
    def test_only_residual_cycles_invoke_scoring(self):
        complete = case('docs/grintour-examples/03-isolated-participant.json')
        names = dict(enumerate(complete['draw']))
        with patch('armplaces.cycle_resolution.score_component', side_effect=AssertionError('called')):
            result = rank_tournament_cycle_score(names, complete['pairs'])
        self.assertFalse(result['diagnostics']['activated'])
        self.assertEqual(result['places'], rank_tournament(names, complete['pairs'])['places'])
        with patch('armplaces.cycle_resolution.score_component', side_effect=AssertionError('called')):
            disconnected = rank_tournament_cycle_score(dict(enumerate('ABC')), [])
        self.assertEqual(disconnected['status'], 'undefined')
        self.assertFalse(disconnected['diagnostics']['activated'])
        self.assertNotEqual(disconnected['diagnostics']['standard']['reason_code'], 'remaining_cycle')
        cyclic = case('docs/grintour-examples/01-mutual-cycle.json')
        names = dict(enumerate(cyclic['draw']))
        import armplaces.cycle_resolution as module
        with patch.object(module, 'score_component', wraps=module.score_component) as score:
            result = rank_tournament_cycle_score(names, cyclic['pairs'])
        self.assertEqual(score.call_count, 1)
        self.assertTrue(result['diagnostics']['activated'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['diagnostics']['standard']['reason_code'], 'remaining_cycle')

    def test_average_ranks_and_weights(self):
        self.assertEqual([ranked_scores({'a':3,'b':2,'c':1}, higher_is_better=True)[x]['score']
                          for x in 'abc'], [1., .5, 0.])
        tied = ranked_scores({'a':3,'b':3,'c':1}, higher_is_better=True)
        self.assertEqual([tied[x]['score'] for x in 'abc'], [.75,.75,0.])
        self.assertEqual([x['score'] for x in ranked_scores(dict.fromkeys('abc', 1),
                         higher_is_better=True).values()], [.5]*3)
        self.assertEqual(normalize_weights((.33,.33,.33)), normalize_weights((1,1,1)))
        for values in ((0,0,0),(-1,1,1),(float('nan'),1,1),(float('inf'),1,1),(1,2)):
            with self.assertRaises(ValueError):
                normalize_weights(values)

    def test_chronology_internal_losses_and_draw_tie(self):
        members={'A','B','C'}
        places={'A':4,'B':5,'C':6,'X':12}
        pairs=[('A','B'),('B','C'),('C','A'),('B','A'),('A','X'),('B','X')]
        order, scored=score_component(members,pairs,places,dict(zip('ABC',range(3))),
                                      normalize_weights((1,0,0)))
        self.assertEqual(order, ['A','C','B'])
        self.assertEqual(scored['A']['values']['chronology'], 4)
        self.assertEqual(scored['B']['values']['loss_place'], 12)
        self.assertEqual(scored['C']['values']['loss_place'], 4)
        self.assertEqual(scored['A']['values']['graph_place'], 4)
        for weights in ((0,1,0),(0,0,1)):
            resolved,details=score_component(members,pairs,places,dict(zip('ABC',range(3))),
                                              normalize_weights(weights))
            self.assertEqual(len(resolved),3)
            self.assertEqual(len(details),3)

    def test_examples_preserve_podium_external_edges_and_input(self):
        for path in ('01-mutual-cycle','02-three-person-cycle','four_person_cycle'):
            item = case(('test/fixtures/' if path=='four_person_cycle' else
                         'docs/grintour-examples/')+path+'.json')
            draw=item['draw'] if path=='four_person_cycle' else item['draw']
            names={i:str(i) for i in draw} if path=='four_person_cycle' else dict(enumerate(draw))
            pairs=[tuple(x) for x in item['pairs']]
            original=list(pairs)
            standard=rank_tournament(names,pairs)
            result=rank_tournament_cycle_score(names,pairs)
            self.assertEqual(pairs,original)
            self.assertEqual(result['status'],'ok')
            self.assertEqual(sorted(result['places'].values()),list(range(1,len(names)+1)))
            self.assertTrue(all(result['places'][name]==place for name,place in
                                standard['places'].items() if place<=3))
            components=result['diagnostics']['components']
            self.assertEqual(max(map(lambda x:len(x['members']),components)),
                             4 if path=='four_person_cycle' else 3 if 'three' in path else 2)
            original_graph=TournamentGraphConstructor(names,pairs,fixed_places=standard['places'],
                                                        allow_cycles=True).up
            rewritten=original_graph.copy()
            rewritten.remove_edges_from(map(tuple,result['diagnostics']['removed_edges']))
            rewritten.add_edges_from(map(tuple,result['diagnostics']['added_edges']))
            self.assertTrue(nx.is_directed_acyclic_graph(rewritten))
            for a,b in original_graph.edges():
                if not any(a in c['members'] and b in c['members'] for c in components):
                    self.assertTrue(rewritten.has_edge(a,b))

    def test_two_components_share_one_preliminary_graph(self):
        names=dict(enumerate('ABCDEFG'))
        pairs=[('D','E'),('E','D'),('F','G'),('G','F'),
               ('D','C'),('E','C'),('F','C'),('G','C')]
        podium={'A':1,'B':2,'C':3}
        graph=TournamentGraphConstructor(names,pairs,fixed_places=podium,allow_cycles=True).up
        reason='Cycle remains after resolving head-to-head majorities'
        failure=RankingUndefined(reason,reason_code='remaining_cycle')
        standard={'status':'partial','reason':reason,'places':podium}
        with patch('armplaces.cycle_resolution.standard_graph_pass',
                   return_value=(standard,failure,graph)):
            result=rank_tournament_cycle_score(names,pairs)
        self.assertEqual(result['status'],'ok')
        self.assertEqual([component['members'] for component in result['diagnostics']['components']],
                         [['D','E'],['F','G']])
        preliminary=result['diagnostics']['preliminary_places']
        self.assertEqual(set(preliminary),set(names.values()))
        self.assertEqual(sorted(result['places'].values()),list(range(1,8)))
