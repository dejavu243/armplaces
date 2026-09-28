"""GrinTour: граф мест из пар (проигравший, победитель).

Повторы сворачиваются, встречные победы разрешаются большинством. Доказанные
по решающим боям призовые места устраняют противоречащие им рёбра; оставшийся
цикл даёт RankingUndefined. Сохраняются все полные пути, включая короткие ветви.
Частоты рёбер всех путей вычисляются динамическим программированием, затем
рёбра разворачиваются от победителя к проигравшему. Рейтинги не используются.
Точный алгоритм, пример и ограничения: README.md, раздел «Полный алгоритм Гринёва».
"""
from pathlib import Path

import networkx as nx

from armplaces.read_tournament import drop_simple_cycles


class RankingUndefined(ValueError):
    """The heuristic cannot provide a complete, unambiguous graph ranking."""

    def __init__(self, message: str, reason_code: str = 'graph_failure'):
        super().__init__(message)
        self.reason_code = reason_code


def build_projection(participants, pairs, fixed_places) -> nx.DiGraph:
    """Return the podium-anchored loser-to-winner projection, even if cyclic."""
    graph = nx.DiGraph()
    graph.add_nodes_from(participants)
    graph.add_edges_from(drop_simple_cycles(pairs))
    for loser, winner in list(graph.edges()):
        if loser in fixed_places and (winner not in fixed_places or
                fixed_places[loser] < fixed_places[winner]):
            graph.remove_edge(loser, winner)
    podium = sorted(fixed_places, key=fixed_places.get)
    if podium:
        graph.add_edges_from((lower, higher) for higher, lower in zip(podium, podium[1:]))
        graph.add_edges_from((name, podium[-1]) for name in participants if name not in fixed_places)
    return graph


class TournamentGraphConstructor:
    def __init__(self, names: dict, pairs: list, fixed_places: dict | None = None,
                 *, allow_cycles: bool = False):
        self.names = dict(names)
        self.pairs = [tuple(pair) for pair in pairs]
        participants = list(self.names.values())
        if not participants or len(set(participants)) != len(participants) or "" in participants:
            raise ValueError("Expected nonempty, unique participant names")
        for pair in self.pairs:
            if len(pair) != 2 or pair[1] not in participants or pair[0] == pair[1]:
                raise ValueError(f"Invalid bout: {pair}")
            if pair[0] != "" and pair[0] not in participants:
                raise ValueError(f"Unknown participant: {pair[0]}")
        self.fixed_places = dict(fixed_places or {})
        if (not set(self.fixed_places) <= set(participants)
                or sorted(self.fixed_places.values()) != list(range(1, len(self.fixed_places)+1))
                or len(self.fixed_places) > 3):
            raise ValueError('Fixed places must be a unique podium prefix')
        self.up = build_projection(participants, self.pairs, self.fixed_places)
        if not allow_cycles and not nx.is_directed_acyclic_graph(self.up):
            raise RankingUndefined("Cycle remains after resolving head-to-head majorities",
                                   reason_code='remaining_cycle')

    def find_winner(self, sportsman: str) -> list:
        return list(self.up.successors(sportsman))

    def find_total_losers(self) -> list:
        return [name for name in self.names.values() if self.up.in_degree(name) == 0]

    def make_chains(self, first_sportsman: str) -> list:
        """Enumerate complete paths for inspection; graph building uses dynamic programming."""
        chains, stack = [], [[first_sportsman]]
        while stack:
            chain = stack.pop()
            winners = self.find_winner(chain[-1])
            if winners:
                stack.extend(chain + [winner] for winner in reversed(winners))
            else:
                chains.append(chain)
        return chains

    def make_all_chains(self) -> dict:
        return {name: self.make_chains(name) for name in self.find_total_losers()}

    def make_graph(self) -> nx.DiGraph:
        # Every edge of a DAG lies on a complete root-to-terminal path.
        # Count all such paths through an edge without pruning short branches.
        order = list(nx.topological_sort(self.up))
        prefix = {node: int(self.up.in_degree(node) == 0) for node in order}
        suffix = {node: int(self.up.out_degree(node) == 0) for node in order}
        for node in order:
            for successor in self.up.successors(node):
                prefix[successor] += prefix[node]
        for node in reversed(order):
            for successor in self.up.successors(node):
                suffix[node] += suffix[successor]
        graph = nx.DiGraph()
        graph.add_nodes_from(self.names.values())
        graph.add_weighted_edges_from((winner, loser, prefix[loser]*suffix[winner])
                                     for loser, winner in self.up.edges())
        return graph

    def save_edgelist(self, filename: Path | str = "edgelist.txt"):
        graph = self.make_graph()
        with Path(filename).open("w", encoding="utf-8") as stream:
            stream.write(f"{graph.number_of_edges()}\n")
            for a, b, data in graph.edges(data=True):
                stream.write(f"{a} {b} {data['weight']}\n")
