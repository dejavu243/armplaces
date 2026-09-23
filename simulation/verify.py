"""Validate exported experiment artifacts: python -m simulation.verify DIRECTORY."""
import argparse
import csv
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

from .core import Config, MODELS
from .experiments import require_finite


def verify_artifacts(directory: Path | str):
    directory = Path(directory)
    for filename in ('config.json', 'summary.json', 'participants.csv', 'bouts.csv',
                     'tournaments.jsonl', 'report.md', 'run.log'):
        if not (directory/filename).is_file() or not (directory/filename).stat().st_size:
            raise ValueError(f'Missing or empty artifact: {filename}')
    config = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    require_finite(config)
    require_finite(summary)
    settings = Config(**config['config'])
    settings.validate()
    models = config['models']
    n, repeats = settings.participants, settings.repeats
    if config['status'] != 'complete' or not models or set(summary) != set(models) or not set(models) <= set(MODELS):
        raise ValueError('Incomplete experiment manifest or summary')
    expected = {(repeat, model) for repeat in range(repeats) for model in models}
    tournament_records = {}
    with (directory/'tournaments.jsonl').open() as stream:
        for line in stream:
            item = json.loads(line)
            require_finite(item)
            key = item['repeat'], item['model']
            if key not in expected or key in tournament_records or sorted(item['draw']) != list(range(n)):
                raise ValueError('Invalid or duplicate tournament record')
            tournament_records[key] = item
    if set(tournament_records) != expected:
        raise ValueError('Missing tournament records')
    seen = set()
    undefined = Counter()
    with (directory/'participants.csv').open(newline='') as stream:
        for key, rows in groupby(csv.DictReader(stream), lambda row: (int(row['repeat']), row['model'])):
            rows = list(rows)
            if key not in expected or key in seen or sorted(int(row['participant']) for row in rows) != list(range(n)):
                raise ValueError('Missing or duplicate participant records')
            seen.add(key)
            record = tournament_records[key]
            places = []
            for row in rows:
                require_finite([float(row[field]) for field in ('rating', 'new_rating', 'place', 'wins', 'losses', 'played')])
                losses, wins = int(row['losses']), int(row['wins'])
                champion = int(row['participant']) == record['champion']
                if losses != (losses if champion and losses in (0,1) else 2) or int(row['played']) != losses+wins:
                    raise ValueError('Invalid participant counters')
                if champion and (int(row['place']) != 1 or losses >= 2):
                    raise ValueError('Invalid champion')
                if row['grin_status'] != record['grin_tour']['status']:
                    raise ValueError('Inconsistent GrinTour status')
                if row['grin_status'] == 'ok':
                    if int(row['grin_place']) != record['grin_tour']['places'][row['participant']]:
                        raise ValueError('Inconsistent GrinTour place')
                places.append(int(row['place']))
            cursor = 1
            for place, count in sorted(Counter(places).items()):
                if place != cursor:
                    raise ValueError('Invalid shared-place numbering')
                cursor += count
            undefined[key[1]] += record['grin_tour']['status'] == 'undefined'
    if seen != expected:
        raise ValueError('Missing participant groups')
    bout_counts = Counter()
    with (directory/'bouts.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = int(row['repeat']), row['model']
            if key not in expected:
                raise ValueError('Unexpected bout')
            if row['technical'] == 'False':
                for field in ('p_a', 'p_start_a'):
                    value = float(row[field])
                    if not 0 <= value <= 1:
                        raise ValueError('Invalid probability')
                bout_counts[key] += 1
    for key, item in tournament_records.items():
        if bout_counts[key] != (0 if n == 1 else 2*n-2+int(item['reset'])):
            raise ValueError('Incomplete bout journal')
    for model, item in summary.items():
        if item['completed'] != repeats or item['bracket']['samples'] != repeats:
            raise ValueError('Incomplete summary')
        if config['grin_tour'] and (item['grin_tour']['samples'] != repeats-undefined[model] or
                                   item['grin_undefined_rate'] != undefined[model]/repeats):
            raise ValueError('Incorrect GrinTour sample counts')
        if not 0 <= item['strongest_win_rate'] <= 1:
            raise ValueError('Invalid strongest-win rate')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        verify_artifacts(args.directory)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print('Artifacts verified')


if __name__ == '__main__':
    main()
