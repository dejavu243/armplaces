"""Проверка полноты артефактов: python -m simulation.verify DIRECTORY.

Манифест и записи всех участников сопоставляются с журналом боёв, который
независимо проигрывается повторно для проверки счётчиков, нагрузки и мест.
Метрики пересчитываются по CSV и сверяются со сводкой. Повреждённый экспорт
завершает CLI ошибкой. Форматы файлов и ограничения описаны в README.md.
"""
import argparse
import csv
import json
from collections import Counter
from itertools import groupby
from math import isclose
from pathlib import Path

from .core import Config, MODELS, elo_probability, validate_result
from .experiments import rank_metrics, require_finite


def verify_artifacts(directory: Path | str):
    directory = Path(directory)
    for filename in ('config.json', 'summary.json', 'participants.csv', 'bouts.csv',
                     'tournaments.jsonl', 'report.md', 'run.log'):
        if not (directory/filename).is_file() or not (directory/filename).stat().st_size:
            raise ValueError(f'Missing or empty artifact: {filename}')
    manifest = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('Expected artifact schema 2; verify older artifacts with their recorded code revision')
    require_finite(manifest)
    require_finite(summary)
    config = Config(**manifest['config'])
    config.validate()
    models = manifest['models']
    n, repeats = config.participants, config.repeats
    if manifest['status'] != 'complete' or not models or len(set(models)) != len(models) or set(summary) != set(models) or not set(models) <= set(MODELS):
        raise ValueError('Incomplete experiment manifest or summary')
    expected = {(repeat, model) for repeat in range(repeats) for model in models}
    records = {}
    with (directory/'tournaments.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            require_finite(item)
            key = item['repeat'], item['model']
            if key not in expected or key in records or sorted(item['draw']) != list(range(n)):
                raise ValueError('Invalid or duplicate tournament record')
            records[key] = item
    if set(records) != expected:
        raise ValueError('Missing tournament records')
    seen = set()
    totals = {model: {'strongest': 0, 'undefined': Counter(), 'partial': Counter(), 'bracket': [], 'grin_tour': []}
              for model in models}
    def group_key(row):
        return int(row['repeat']), row['model']
    with (directory/'participants.csv').open(newline='', encoding='utf-8') as participant_stream, \
            (directory/'bouts.csv').open(newline='', encoding='utf-8') as bout_stream:
        bout_groups = iter(groupby(csv.DictReader(bout_stream), group_key))
        for key, rows in groupby(csv.DictReader(participant_stream), group_key):
            rows = sorted(rows, key=lambda row: int(row['participant']))
            if key not in expected or key in seen or [int(row['participant']) for row in rows] != list(range(n)):
                raise ValueError('Missing or duplicate participant records')
            seen.add(key)
            record = records[key]
            result = {**record, 'ratings': [float(row['rating']) for row in rows],
                      'new_ratings': [float(row['new_rating']) for row in rows],
                      'wins': [int(row['wins']) for row in rows],
                      'losses': [int(row['losses']) for row in rows],
                      'played': [int(row['played']) for row in rows],
                      'places': {int(row['participant']): int(row['place']) for row in rows},
                      'bouts': []}
            require_finite(result)
            ranking = result['grin_tour']
            if (not manifest['grin_tour']) != (ranking['status'] == 'disabled'):
                raise ValueError('Unexpected GrinTour status')
            for i, row in enumerate(rows):
                if result['ratings'][i] <= 0 or int(row['draw_position']) != record['draw'].index(i)+1:
                    raise ValueError('Invalid rating or draw position')
                if row['grin_status'] != ranking['status'] or row['grin_reason'] != ranking['reason']:
                    raise ValueError('Inconsistent GrinTour status')
                if row['participant'] in ranking['places']:
                    if not row['grin_place'] or int(row['grin_place']) != ranking['places'][row['participant']]:
                        raise ValueError('Inconsistent GrinTour place')
                elif row['grin_place']:
                    raise ValueError('Unexpected GrinTour place')
            if n > 1:
                bout_key, bout_rows = next(bout_groups, (None, ()))
                if bout_key != key:
                    raise ValueError('Missing or out-of-order bout journal')
                for row in bout_rows:
                    if row['technical'] not in ('True', 'False'):
                        raise ValueError('Invalid technical-pass flag')
                    bout = {'stage': row['stage'], 'technical': row['technical'] == 'True'}
                    for field in ('match', 'a', 'b', 'winner', 'loser', 'previous_bouts_a', 'previous_bouts_b'):
                        bout[field] = int(row[field]) if row[field] else None
                    for field in ('p_a', 'p_start_a', 'effective_a', 'effective_b'):
                        bout[field] = float(row[field]) if row[field] else None
                    result['bouts'].append(bout)
            validate_result(result)
            changes = [0.]*n
            for bout in result['bouts']:
                if not bout['technical']:
                    a, b = bout['a'], bout['b']
                    baseline = elo_probability(result['ratings'][a], result['ratings'][b], config.dr)
                    if not isclose(baseline, bout['p_start_a'], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError('Incorrect baseline probability')
                    change = config.k * (int(bout['winner'] == a)-baseline)
                    changes[a] += change
                    changes[b] -= change
            for initial, updated, change in zip(result['ratings'], result['new_ratings'], changes):
                if not isclose(initial+change, updated, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError('Incorrect post-tournament rating')
            total = totals[key[1]]
            total['strongest'] += result['ratings'][result['champion']] == max(result['ratings'])
            total['bracket'].append(rank_metrics(result['ratings'], result['places']))
            if ranking['status'] == 'ok':
                total['grin_tour'].append(rank_metrics(result['ratings'], {int(k): v for k,v in ranking['places'].items()}))
            elif ranking['status'] in ('undefined', 'partial'):
                total[ranking['status']][ranking['reason']] += 1
        if next(bout_groups, None) is not None:
            raise ValueError('Unexpected extra bout group')
    if seen != expected:
        raise ValueError('Missing participant groups')
    for model, item in summary.items():
        total = totals[model]
        if item['completed'] != repeats or not isclose(item['strongest_win_rate'], total['strongest']/repeats):
            raise ValueError('Incorrect tournament count or strongest-win rate')
        for status in ('undefined', 'partial'):
            if item[f'grin_{status}_reasons'] != dict(total[status]):
                raise ValueError(f'Incorrect GrinTour {status} reasons')
            expected_rate = sum(total[status].values())/repeats if manifest['grin_tour'] else None
            if item[f'grin_{status}_rate'] != expected_rate:
                raise ValueError(f'Incorrect GrinTour {status} rate')
        for method in ('bracket', 'grin_tour'):
            samples = total[method]
            correlations = [x['spearman'] for x in samples if x['spearman'] is not None]
            if item[method]['samples'] != len(samples) or item[method]['correlation_samples'] != len(correlations):
                raise ValueError('Incorrect metric sample count')
            for metric, values in (('mae', [x['mae'] for x in samples]), ('spearman', correlations)):
                actual = item[method][metric]
                if not values:
                    if actual is not None:
                        raise ValueError('Metric without samples must be null')
                elif actual is None or not isclose(actual, sum(values)/len(values), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Incorrect aggregate metric')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        verify_artifacts(args.directory)
    except (ValueError, RuntimeError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print('Artifacts verified')


if __name__ == '__main__':
    main()
