"""Проверка непобедимого №1: 1000 турниров для каждого N=2..64.

Участники имеют имена 1..N; жеребьёвка случайная. №1 всегда выигрывает,
остальные пары получают честный случайный исход. Проверяются чемпион сетки
и отдельно результат GrinTour. Полный прогон запускается явно, вне discover:
.venv/bin/python -m test.forced_champion
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from armplaces.topological_sort import rank_tournament
from simulation import Config, simulate
from simulation.experiments import environment, write_json


def run(output: Path, seed: int = 42, repeats: int = 1000):
    if repeats < 1 or seed < 0:
        raise ValueError('repeats must be positive; seed must be nonnegative')
    output.mkdir(parents=True, exist_ok=True)
    rows, reasons, examples = [], Counter(), {}
    for n in range(2, 65):
        counts = Counter()
        for repeat in range(repeats):
            draw_rng = np.random.default_rng(np.random.SeedSequence([seed, n, repeat, 0]))
            outcome_rng = np.random.default_rng(np.random.SeedSequence([seed, n, repeat, 1]))
            draw = [int(who) for who in draw_rng.permutation(n)]
            # Exact requested rule using the existing model: only №1 has a greater
            # rating; all other ratings are equal, hence their outcomes have p=1/2.
            # Internal ID 0 corresponds to the external name "1".
            result = simulate(Config(participants=n, seed=seed), 'strong-win',
                              [2.] + [1.]*(n-1), draw, repeat=repeat, rng=outcome_rng)
            counts['tournaments'] += 1
            if result['champion'] == 0 and result['places'][0] == 1 and result['losses'][0] == 0:
                counts['bracket_correct'] += 1
            else:
                counts['bracket_wrong'] += 1
            pairs = []
            for bout in result['bouts']:
                if bout['technical']:
                    continue
                a, b = bout['a'], bout['b']
                expected_probability = 1. if a == 0 else (0. if b == 0 else .5)
                if bout['p_a'] != expected_probability or (0 in (a, b) and bout['winner'] != 0):
                    counts['outcome_rule_errors'] += 1
                pairs.append((str(bout['loser']+1), str(bout['winner']+1)))
            names = {who+1: str(who+1) for who in draw}
            ranking = rank_tournament(names, pairs)
            if ranking['status'] == 'ok':
                counts['grin_correct' if ranking['places']['1'] == 1 else 'grin_wrong'] += 1
            else:
                counts['grin_undefined'] += 1
                reasons[ranking['reason']] += 1
                if ranking['reason'] not in examples:
                    examples[ranking['reason']] = {'participants': n, 'repeat': repeat,
                        'draw': [who+1 for who in draw], 'pairs': pairs, 'ranking': ranking}
            if counts['bracket_wrong'] or counts['outcome_rule_errors'] or counts['grin_wrong']:
                write_json(output/'failure.json', {'participants': n, 'repeat': repeat,
                           'result': result, 'ranking_with_names_1_to_n': ranking})
                raise AssertionError(f'Incorrect champion or bout rule: N={n}, repeat={repeat}')
        row = {'participants': n, **{key: counts[key] for key in (
            'tournaments', 'bracket_correct', 'bracket_wrong', 'outcome_rule_errors',
            'grin_correct', 'grin_wrong', 'grin_undefined')}}
        rows.append(row)
        print(f"N={n:2}: champion #1 {counts['bracket_correct']}/{repeats}; "
              f"GrinTour #1={counts['grin_correct']}, undefined={counts['grin_undefined']}", flush=True)
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key != 'participants'}
    with (output/'by_size.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'seed': seed, 'repeats_per_size': repeats, 'min_n': 2, 'max_n': 64,
               'environment': environment(), 'totals': totals, 'grin_undefined_reasons': dict(reasons)}
    write_json(output/'summary.json', summary)
    write_json(output/'undefined_examples.json', examples)
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('results/forced-champion'))
    args = parser.parse_args()
    run(args.output, args.seed, args.repeats)


if __name__ == '__main__':
    main()
