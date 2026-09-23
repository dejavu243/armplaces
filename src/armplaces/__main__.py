"""CLI исторических протоколов: python -m armplaces.

Читает файлы выбранной категории, восстанавливает журнал, сохраняет граф и места
по Гринёву в --output. Визуализация доступна отдельной функцией run_alg.
Новые эксперименты запускаются через python -m simulation. Команды — в README.md.
"""
import argparse
from pathlib import Path

from armplaces.grinev_algorithm import RankingUndefined, TournamentGraphConstructor
from armplaces.podium import infer_podium
from armplaces.read_tournament import RESULT_FILE_SUFFIX, read_tournament_files, tournament_recovery
from armplaces.topological_sort import calc_and_save_places


def run_alg(names: dict, pairs: list):
    import matplotlib.pyplot as plt
    import networkx as nx
    graph = TournamentGraphConstructor(names, pairs, fixed_places=infer_podium(names, pairs)['places']).make_graph()
    nx.draw_networkx(graph, pos=nx.circular_layout(graph), with_labels=True)
    plt.show()


def save_graph(names: dict, pairs: list, output: Path = Path(".")):
    output.mkdir(parents=True, exist_ok=True)
    ranking = calc_and_save_places(names, pairs, output / "places.txt")
    try:
        TournamentGraphConstructor(names, pairs, fixed_places=infer_podium(names, pairs)['places']).save_edgelist(output / 'edgelist.txt')
    except RankingUndefined:
        # Do not leave a previous run's full graph next to a partial new result.
        (output / 'edgelist.txt').unlink(missing_ok=True)
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/left_hand_75kg"))
    parser.add_argument("--output", type=Path, default=Path("results/historical"))
    args = parser.parse_args()
    try:
        files = read_tournament_files(args.input)
        ranking = save_graph(files[RESULT_FILE_SUFFIX][0], tournament_recovery(files), args.output)
        print(f"GrinTour: {ranking['status']}; known places: {len(ranking['places'])}")
        if ranking['reason']:
            print(ranking['reason'])
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
