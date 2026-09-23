"""Stream experiment artifacts and aggregate rank metrics."""
import csv
import json
import logging
import platform
import subprocess
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import distributions
from math import isfinite
from pathlib import Path

import numpy as np

from .core import MODELS, Config, generate_field, simulate


def average_ranks(values, descending=False):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    ranks = [0.]*len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranks[i] = (start + 1 + end)/2
        start = end
    return np.array(ranks)


def rank_metrics(ratings, places):
    expected = average_ranks(ratings, descending=True)
    actual = average_ranks([places[i] for i in range(len(ratings))])
    mae = float(np.mean(np.abs(actual-expected)))
    a, b = actual-actual.mean(), expected-expected.mean()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    correlation = float(np.clip(np.dot(a, b)/denominator, -1, 1)) if denominator else None
    return {'mae': mae, 'spearman': correlation}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def environment():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'python': platform.python_version(), 'commit': commit, 'dirty': dirty,
            'dependencies': dict(sorted((dist.metadata['Name'], dist.version) for dist in distributions()))}


def report_markdown(config, summary):
    lines = ['# ArmPlaces: результаты эксперимента', '',
             f'N={config.participants}; повторов на модель: {config.repeats}; seed={config.seed}.', '',
             '| Модель | Завершено | Сильнейший победил | MAE сетки | Spearman сетки | Гринёв: успешных | Гринёв: неопределённых | MAE Гринёва | Spearman Гринёва |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{value:.6f}'
    for model, item in summary.items():
        lines.append(f"| {model} | {item['completed']} | {fmt(item['strongest_win_rate'])} | "
                     f"{fmt(item['bracket']['mae'])} | {fmt(item['bracket']['spearman'])} | "
                     f"{item['grin_tour']['samples']} | {fmt(item['grin_undefined_rate'])} | "
                     f"{fmt(item['grin_tour']['mae'])} | {fmt(item['grin_tour']['spearman'])} |")
    lines.extend(['', 'Неопределённые результаты Гринёва исключены из его MAE и Spearman.',
                  'Число определённых корреляций указано в summary.json; для N=1 корреляция отсутствует.',
                  'Рейтинг внутри турнира фиксирован, обновление выполняется после окончания.', ''])
    return '\n'.join(lines)


def run_experiments(config: Config, models, grin_tour: bool, output: Path | str) -> dict:
    config.validate()
    if not models or any(model not in MODELS for model in models):
        raise ValueError('Select at least one known experiment')
    models = [model for model in MODELS if model in models]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'config': asdict(config), 'models': models,
                'grin_tour': grin_tour, 'environment': environment(), 'status': 'running'}
    write_json(output/'config.json', manifest)
    logger = logging.getLogger('armplaces.experiments')
    handler = logging.FileHandler(output/'run.log', mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    accumulators = {model: {'completed': 0, 'strongest_wins': 0, 'undefined': Counter(),
                           'bracket': [], 'grin_tour': []} for model in models}
    participants_fields = ['repeat', 'model', 'participant', 'draw_position', 'rating', 'new_rating',
                           'wins', 'losses', 'played', 'place', 'grin_place', 'grin_status', 'grin_reason']
    bouts_fields = ['repeat', 'model', 'match', 'stage', 'a', 'b', 'technical', 'winner', 'loser',
                    'p_a', 'p_start_a', 'effective_a', 'effective_b', 'previous_bouts_a', 'previous_bouts_b']
    try:
        with ExitStack() as stack:
            participants_file = stack.enter_context((output/'participants.csv').open('w', newline='', encoding='utf-8'))
            bouts_file = stack.enter_context((output/'bouts.csv').open('w', newline='', encoding='utf-8'))
            tournaments = stack.enter_context((output/'tournaments.jsonl').open('w', encoding='utf-8'))
            participants_writer = csv.DictWriter(participants_file, participants_fields)
            bouts_writer = csv.DictWriter(bouts_file, bouts_fields)
            participants_writer.writeheader()
            bouts_writer.writeheader()
            for repeat in range(config.repeats):
                ratings, draw = generate_field(config, repeat)
                for model in models:
                    result = simulate(config, model, ratings, draw, repeat=repeat, grin_tour=grin_tour)
                    ranking = result['grin_tour']
                    accumulator = accumulators[model]
                    accumulator['completed'] += 1
                    accumulator['strongest_wins'] += ratings[result['champion']] == max(ratings)
                    accumulator['bracket'].append(rank_metrics(ratings, result['places']))
                    if ranking['status'] == 'ok':
                        accumulator['grin_tour'].append(rank_metrics(ratings, {int(k):v for k,v in ranking['places'].items()}))
                    elif ranking['status'] == 'undefined':
                        accumulator['undefined'][ranking['reason']] += 1
                    draw_positions = {who: i+1 for i, who in enumerate(draw)}
                    for who in range(config.participants):
                        participants_writer.writerow({'repeat': repeat, 'model': model, 'participant': who,
                            'draw_position': draw_positions[who], 'rating': ratings[who],
                            'new_rating': result['new_ratings'][who], 'wins': result['wins'][who],
                            'losses': result['losses'][who], 'played': result['played'][who],
                            'place': result['places'][who], 'grin_place': ranking['places'].get(str(who)),
                            'grin_status': ranking['status'], 'grin_reason': ranking['reason']})
                    for bout in result['bouts']:
                        bouts_writer.writerow({'repeat': repeat, 'model': model, **bout})
                    tournaments.write(json.dumps({key: result[key] for key in
                        ('repeat', 'model', 'draw', 'champion', 'reset', 'sequence', 'grin_tour')},
                        ensure_ascii=False, allow_nan=False)+'\n')
                if (repeat+1) % 100 == 0 or repeat+1 == config.repeats:
                    logger.info('Completed %d/%d repetitions for %s', repeat+1, config.repeats, ', '.join(models))
        summary = {}
        for model, accumulator in accumulators.items():
            item = {'completed': accumulator['completed'],
                    'strongest_win_rate': accumulator['strongest_wins']/config.repeats,
                    'grin_undefined_rate': sum(accumulator['undefined'].values())/config.repeats if grin_tour else None,
                    'grin_undefined_reasons': dict(accumulator['undefined'])}
            for method in ('bracket', 'grin_tour'):
                samples = accumulator[method]
                correlations = [sample['spearman'] for sample in samples if sample['spearman'] is not None]
                item[method] = {'samples': len(samples),
                                'mae': float(np.mean([sample['mae'] for sample in samples])) if samples else None,
                                'spearman': float(np.mean(correlations)) if correlations else None,
                                'correlation_samples': len(correlations)}
            summary[model] = item
        write_json(output/'summary.json', summary)
        (output/'report.md').write_text(report_markdown(config, summary), encoding='utf-8')
        manifest['status'] = 'complete'
        write_json(output/'config.json', manifest)
        from .verify import verify_artifacts
        verify_artifacts(output)
        logger.info('Artifact verification passed')
        return summary
    except Exception:
        manifest['status'] = 'failed'
        write_json(output/'config.json', manifest)
        logger.exception('Experiment failed')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)
    elif isinstance(value, (int, float)) and not isfinite(value):
        raise ValueError('Nonfinite value in artifacts')
