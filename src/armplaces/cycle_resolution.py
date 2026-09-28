"""Resolve residual GrinTour cycles using chronological and graph-place evidence.

The standard graph pass always runs first. Initial athlete ratings are never read.
Detailed rules and examples are documented in README.md.
"""
from math import fsum, isfinite

import networkx as nx

from armplaces.grinev_algorithm import RankingUndefined, TournamentGraphConstructor
from armplaces.topological_sort import (get_places, get_tournament_dict,
                                        standard_graph_pass)

DEFAULT_WEIGHTS = (0.33, 0.33, 0.33)
CRITERIA = ('chronology', 'loss_place', 'graph_place')


def normalize_weights(weights) -> tuple[float, float, float]:
    """Validate finite nonnegative weights and normalize their sum to one."""
    try:
        values = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as exc:
        raise ValueError('cycle weights must be three finite nonnegative numbers') from exc
    if len(values) != 3 or any(not isfinite(value) or value < 0 for value in values):
        raise ValueError('cycle weights must be three finite nonnegative numbers')
    total = fsum(values)
    if not isfinite(total) or total <= 0:
        raise ValueError('cycle weights must have a finite positive sum')
    if values[0] == values[1] == values[2]:
        return (1/3, 1/3, 1/3)
    return tuple(value/total for value in values)


def ranked_scores(values: dict, *, higher_is_better: bool) -> dict:
    """Assign average-rank scores in [0, 1] to a cyclic component."""
    size = len(values)
    if size < 2:
        raise ValueError('Expected at least two cyclic participants')
    ordered = sorted(values, key=values.get, reverse=higher_is_better)
    scores = {}
    start = 0
    while start < size:
        end = start + 1
        while end < size and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for name in ordered[start:end]:
            scores[name] = {'rank': rank, 'score': (size-rank)/(size-1)}
        start = end
    return scores


def score_component(component, pairs, preliminary_places, draw_order, weights):
    """Return best-to-worst order and independently inspectable score details."""
    members = sorted(component, key=draw_order.get)
    names = set(members)
    raw = {name: {'chronology': None, 'loss_place': None,
                  'graph_place': preliminary_places[name]} for name in members}
    losses = {name: [] for name in members}
    for index, (loser, winner) in enumerate(pairs, start=1):
        if not loser:
            continue
        if winner in names and loser in names:
            raw[winner]['chronology'] = index
        if loser in names:
            losses[loser].append(preliminary_places[winner])
    for name in members:
        if raw[name]['chronology'] is None or not losses[name]:
            raise RankingUndefined('Cyclic participant lacks an internal win or a loss',
                                   reason_code='inconsistent_cycle')
        raw[name]['loss_place'] = max(losses[name])
    ranks = {criterion: ranked_scores({name: raw[name][criterion] for name in members},
                                     higher_is_better=(criterion == 'chronology'))
             for criterion in CRITERIA}
    scores = {}
    for name in members:
        weighted = fsum(weights[i]*ranks[criterion][name]['score']
                        for i, criterion in enumerate(CRITERIA))
        scores[name] = {'values': raw[name],
                        'ranks': {criterion: ranks[criterion][name]['rank'] for criterion in CRITERIA},
                        'scores': {criterion: ranks[criterion][name]['score'] for criterion in CRITERIA},
                        'total': weighted}
    order = sorted(members, key=lambda name: (-round(scores[name]['total'], 12),
                                              draw_order[name]))
    return order, scores


def rank_tournament_cycle_score(names, pairs, *, weights=DEFAULT_WEIGHTS) -> dict:
    """Run standard GrinTour first; score SCCs only after its residual-cycle failure."""
    normalized = normalize_weights(weights)
    standard, failure, projection = standard_graph_pass(names, pairs)
    diagnostics = {'standard': {'status': standard['status'], 'reason': standard['reason'],
                                'reason_code': failure.reason_code if failure else None},
                   'activated': False, 'weights': list(normalized), 'components': [],
                   'preliminary_places': {}, 'removed_edges': [], 'added_edges': []}
    if failure is None or failure.reason_code != 'remaining_cycle':
        return {**standard, 'diagnostics': diagnostics}

    diagnostics['activated'] = True
    draw_order = {name: index for index, name in enumerate(names.values())}
    components = sorted((set(nodes) for nodes in nx.strongly_connected_components(projection)
                         if len(nodes) > 1), key=lambda nodes: min(draw_order[node] for node in nodes))
    preliminary = projection.copy()
    for members in components:
        preliminary.remove_edges_from((a, b) for a, b in list(preliminary.edges())
                                      if a in members and b in members)
    fixed_places = standard['places']
    try:
        constructor = TournamentGraphConstructor(names, pairs, fixed_places=fixed_places,
                                                   allow_cycles=True)
        constructor.up = preliminary
        graph = constructor.make_graph()
        preliminary_ranking = get_places(get_tournament_dict(graph), graph,
                                         fixed_places=fixed_places)
        preliminary_places = {name: values['place'] for name, values in preliminary_ranking.items()}
        diagnostics['preliminary_places'] = preliminary_places

        resolved = projection.copy()
        for members in components:
            order, scores = score_component(members, pairs, preliminary_places,
                                            draw_order, normalized)
            priority = {name: index for index, name in enumerate(order)}
            removed = sorted(([a, b] for a, b in resolved.edges()
                              if a in members and b in members and priority[a] < priority[b]),
                             key=lambda edge: (draw_order[edge[0]], draw_order[edge[1]]))
            resolved.remove_edges_from(removed)
            added = []
            for higher, lower in zip(order, order[1:]):
                if not resolved.has_edge(lower, higher):
                    resolved.add_edge(lower, higher)
                    added.append([lower, higher])
            diagnostics['components'].append({'members': sorted(members, key=draw_order.get),
                                               'order': order, 'scores': scores,
                                               'removed_edges': removed, 'added_edges': added})
            diagnostics['removed_edges'].extend(removed)
            diagnostics['added_edges'].extend(added)
        if not nx.is_directed_acyclic_graph(resolved):
            raise RankingUndefined('Cycle remains after scoring components',
                                   reason_code='remaining_cycle')
        constructor.up = resolved
        graph = constructor.make_graph()
        final = get_places(get_tournament_dict(graph), graph, fixed_places=fixed_places)
    except RankingUndefined as exc:
        diagnostics['failure_code'] = exc.reason_code
        return {'status': 'partial' if fixed_places else 'undefined', 'reason': str(exc),
                'places': dict(fixed_places), 'diagnostics': diagnostics}
    return {'status': 'ok', 'reason': '',
            'places': {name: values['place'] for name, values in final.items()},
            'diagnostics': diagnostics}
