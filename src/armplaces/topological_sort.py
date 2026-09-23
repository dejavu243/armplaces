"""GrinTour: места из ациклического графа побед.

Единственный источник и его неразветвлённое продолжение получают верхние места.
Остальные вершины удаляются снизу: сравниваются длина пути, уровни победителей,
число их побед и порядок жеребьёвки. Степени графа не равны реальным счётчикам боёв.
Призёры сначала фиксируются по финалу/суперфиналу/полуфиналу. Возвращается полная
расстановка либо partial с сохранёнными призёрами, undefined — без известных мест.
Входы не изменяются.
Формулы и порядок сравнений подробно описаны в README.md.
"""
import json
from pathlib import Path

import networkx as nx

from armplaces.grinev_algorithm import RankingUndefined, TournamentGraphConstructor
from armplaces.podium import infer_podium


def get_tournament_dict(graph) -> dict:
    return {node: {"level": graph.out_degree(node) - graph.in_degree(node) + 2,
                   "losses": graph.in_degree(node), "wins": graph.out_degree(node)}
            for node in graph}


def get_target_points_sorted(target_points, keys=None, reverse=None):
    """Sort ascending by default; True means descending for each selected key."""
    keys = keys if isinstance(keys, list) else [keys]
    directions = reverse if isinstance(reverse, list) else [bool(reverse)] * len(keys)
    if len(directions) != len(keys):
        raise ValueError("keys and reverse must have matching lengths")
    return sorted(target_points.items(), key=lambda item: tuple(
        (-1 if descending else 1) * item[1][key]
        for key, descending in zip(keys, directions)))


def get_places(tournament: dict, graph, fixed_places: dict | None = None) -> dict:
    if not graph or not nx.is_directed_acyclic_graph(graph):
        raise RankingUndefined("Expected a nonempty acyclic graph")
    sources = [node for node in graph if graph.in_degree(node) == 0]
    if len(sources) != 1 or set(tournament) != set(graph):
        raise RankingUndefined("Graph must have exactly one source and include all participants")
    result = {node: dict(values) for node, values in tournament.items()}
    order = {node: i for i, node in enumerate(graph)}
    source = sources[0]
    depth = {source: 1}
    for node in nx.topological_sort(graph):
        for child in graph.successors(node):
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
    assigned = set()
    if fixed_places:
        for node, place in fixed_places.items():
            result[node]['place'] = place
            assigned.add(node)
    else:
        node, position = source, 1
        while True:
            result[node]["place"] = position
            assigned.add(node)
            children = list(graph.successors(node))
            if len(children) != 1:
                break
            node, position = children[0], position + 1
    remaining = graph.copy()
    next_place = len(graph)
    while len(assigned) < len(graph):
        targets = [node for node in remaining if remaining.out_degree(node) == 0
                   and node not in assigned]
        if not targets:
            raise RankingUndefined("No progress while assigning remaining places")

        def priority(target):
            winners = list(remaining.predecessors(target))
            if not winners:
                raise RankingUndefined("Participant has no ranked path to the winner")
            levels = [result[winner]["level"] for winner in winners]
            # Longest path first, then weakest opposition first; places descend.
            return (-depth[target], min(levels), sum(levels) / len(levels),
                    max(result[winner]["wins"] for winner in winners), order[target])

        for target in sorted(targets, key=priority):
            result[target]["place"] = next_place
            assigned.add(target)
            next_place -= 1
        remaining.remove_nodes_from(targets)
    if sorted(item["place"] for item in result.values()) != list(range(1, len(graph) + 1)):
        raise RankingUndefined("Incomplete or duplicate placement")
    return result


def rank_tournament(names: dict, pairs: list) -> dict:
    """Lock the decisive-bout podium, then rank the rest; preserve it on graph failure."""
    podium = infer_podium(names, pairs)
    fixed_places = podium['places']
    try:
        graph = TournamentGraphConstructor(names, pairs, fixed_places=fixed_places).make_graph()
        tournament = get_places(get_tournament_dict(graph), graph, fixed_places=fixed_places)
    except RankingUndefined as exc:
        return {'status': 'partial' if fixed_places else 'undefined',
                'reason': str(exc), 'places': dict(fixed_places)}
    return {'status': 'ok', 'reason': '',
            'places': {name: values['place'] for name, values in tournament.items()}}


def calc_and_save_places(names: dict, pairs: list, filename: Path = Path('places.txt')):
    ranking = rank_tournament(names, pairs)
    filename = Path(filename)
    filename.write_text(''.join(f'{place} {name}\n' for name, place in
                               sorted(ranking['places'].items(), key=lambda item: item[1])),
                        encoding='utf-8')
    filename.with_suffix('.json').write_text(json.dumps(ranking, ensure_ascii=False, indent=2)+'\n',
                                            encoding='utf-8')
    return ranking
