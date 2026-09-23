"""GrinTour: граф мест из пар (проигравший, победитель).

Повторы сворачиваются, встречные победы разрешаются большинством. Доказанные
по решающим боям призовые места устраняют противоречащие им рёбра; оставшийся
цикл даёт RankingUndefined. От участников без побед выбираются пути длины L и L−1.
Частоты рёбер этих путей вычисляются динамическим программированием, затем
рёбра разворачиваются от победителя к проигравшему. Рейтинги не используются.
Точный алгоритм, пример и ограничения: README.md, раздел «Полный алгоритм Гринёва».
"""
from collections import Counter
from pathlib import Path

import networkx as nx

from armplaces.read_tournament import drop_simple_cycles


class RankingUndefined(ValueError):
    """The heuristic cannot provide a complete, unambiguous graph ranking."""


class TournamentGraphConstructor:
    def __init__(self, names: dict, pairs: list, fixed_places: dict | None = None):
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
        self.up = nx.DiGraph()
        self.up.add_nodes_from(participants)
        self.up.add_edges_from(drop_simple_cycles(self.pairs))
        # Only the graph projection is corrected; the chronological journal is untouched.
        for loser, winner in list(self.up.edges()):
            if loser in self.fixed_places and (winner not in self.fixed_places or
                    self.fixed_places[loser] < self.fixed_places[winner]):
                self.up.remove_edge(loser, winner)
        podium = sorted(self.fixed_places, key=self.fixed_places.get)
        if podium:
            self.up.add_edges_from((lower, higher) for higher, lower in zip(podium, podium[1:]))
            self.up.add_edges_from((name, podium[-1]) for name in participants if name not in self.fixed_places)
        if not nx.is_directed_acyclic_graph(self.up):
            raise RankingUndefined("Cycle remains after resolving head-to-head majorities")

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
        # Count lengths of suffix paths once. Length is measured in edges here.
        order = list(nx.topological_sort(self.up))
        suffix = {node: Counter() for node in order}
        for node in reversed(order):
            if self.up.out_degree(node) == 0:
                suffix[node][0] = 1
            for successor in self.up.successors(node):
                for length, count in suffix[successor].items():
                    suffix[node][length + 1] += count
        weights = Counter()
        for root in self.find_total_losers():
            longest = max(suffix[root])
            prefix = {node: Counter() for node in order}
            prefix[root][0] = 1
            for node in order:
                for successor in self.up.successors(node):
                    for length, count in prefix[node].items():
                        prefix[successor][length + 1] += count
                        for target_length in (longest, longest - 1):
                            weights[successor, node] += count * suffix[successor].get(
                                target_length - length - 1, 0)
        graph = nx.DiGraph()
        graph.add_nodes_from(self.names.values())
        graph.add_weighted_edges_from((a, b, count) for (a, b), count in weights.items() if count)
        return graph

    def save_edgelist(self, filename: Path | str = "edgelist.txt"):
        graph = self.make_graph()
        with Path(filename).open("w", encoding="utf-8") as stream:
            stream.write(f"{graph.number_of_edges()}\n")
            for a, b, data in graph.edges(data=True):
                stream.write(f"{a} {b} {data['weight']}\n")
