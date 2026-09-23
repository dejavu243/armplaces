"""Historical protocol entry point; simulation is available via python -m simulation."""
import argparse
from pathlib import Path

from grinev_algorithm import TournamentGraphConstructor
from read_tournament import RESULT_FILE_SUFFIX, read_tournament_files, tournament_recovery
from topological_sort import calc_and_save_places


def run_alg(names: dict, pairs: list):
    import matplotlib.pyplot as plt
    import networkx as nx
    graph = TournamentGraphConstructor(names, pairs).make_graph()
    nx.draw_networkx(graph, pos=nx.circular_layout(graph), with_labels=True)
    plt.show()


def save_graph(names: dict, pairs: list, output: Path = Path(".")):
    output.mkdir(parents=True, exist_ok=True)
    TournamentGraphConstructor(names, pairs).save_edgelist(output / "edgelist.txt")
    return calc_and_save_places(names, pairs, output / "places.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/left_hand_75kg"))
    parser.add_argument("--output", type=Path, default=Path("results/historical"))
    args = parser.parse_args()
    try:
        files = read_tournament_files(args.input)
        save_graph(files[RESULT_FILE_SUFFIX][0], tournament_recovery(files), args.output)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
