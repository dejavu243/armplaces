"""Независимые повторения экспериментов, метрики и потоковый экспорт.

Внутри повтора модели получают одинаковые рейтинги и жеребьёвку. Результаты
пишутся в CSV/JSONL, затем агрегируются MAE, Spearman и частота победы сильнейшего.
Неопределённые результаты Гринёва учитываются отдельно. В манифесте сохраняются
параметры, seed и окружение. Форматы и воспроизводимость описаны в README.md.
"""
import csv
import hashlib
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
from .place_statistics import (RANKING_NAMES, ALL_RANKING_NAMES, place_errors,
                               summarize_places, write_place_artifacts)
from .comparison_statistics import individual_metrics, metric_details, paired_comparisons
from armplaces.cycle_resolution import DEFAULT_WEIGHTS, normalize_weights


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
    source_root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for package in ('armplaces', 'simulation'):
        for path in sorted((source_root/package).rglob('*.py')):
            digest.update(str(path.relative_to(source_root)).encode())
            digest.update(b'\0')
            digest.update(path.read_bytes())
            digest.update(b'\0')
    return {'python': platform.python_version(), 'commit': commit, 'dirty': dirty,
            'source_sha256': digest.hexdigest(),
            'dependencies': dict(sorted((dist.metadata['Name'], dist.version) for dist in distributions()))}


def report_markdown(config, summary, *, zhivora_fixed_top=None):
    lines = ['# ArmPlaces: результаты эксперимента', '',
             f'N={config.participants}; повторов на модель: {config.repeats}; seed={config.seed}.',
             f'Распределение: `{config.distribution}`; границы: {config.rating_bounds() or "без усечения"}.', '',
             '`grin-cycle` — основной метод без исходных рейтингов; `grin-elo` использует рейтинговый резерв.', '',
             '| Модель | Завершено | Сильнейший победил | MAE сетки | Spearman сетки | grin-elo: полных | grin-elo: только призёры | grin-elo: без мест | MAE grin-elo | Spearman grin-elo |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{value:.6f}'
    for model, item in summary.items():
        lines.append(f"| {model} | {item['completed']} | {fmt(item['strongest_win_rate'])} | "
                     f"{fmt(item['bracket']['mae'])} | {fmt(item['bracket']['spearman'])} | "
                     f"{item['grin_tour']['samples']} | {fmt(item['grin_partial_rate'])} | {fmt(item['grin_undefined_rate'])} | "
                     f"{fmt(item['grin_tour']['mae'])} | {fmt(item['grin_tour']['spearman'])} |")
    lines.extend(['', 'MAE и Spearman графовых методов рассчитаны только по полным расстановкам; partial сохраняет призёров.',
                  'Число определённых корреляций указано в summary.json; для N=1 корреляция отсутствует.',
                  'Исходный рейтинг фиксирован; эффективная сила зависит от модели. Обновление рейтинга — после турнира.', ''])
    lines.extend(['| Модель | Граф без цикла | grin-cycle: полных | Доля полных | Этап циклов запущен | Доля запусков | Этап завершён | MAE grin-cycle | Spearman grin-cycle |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for model, item in summary.items():
        cycle = item['grin_cycle_score']
        lines.append(f"| {model} | {item['cycle_standard_success']} | {cycle['samples']} | "
                     f"{fmt(cycle['samples']/config.repeats)} | {item['cycle_activated']} | "
                     f"{fmt(item['cycle_activated']/config.repeats)} | {item['cycle_success']} | "
                     f"{fmt(cycle['mae'])} | {fmt(cycle['spearman'])} |")
    lines.append('')
    if any('far_tour' in item for item in summary.values()):
        lines += ['## Сравнение индивидуальных мест', '',
                  f'Zhivora-Tour: фиксированы места 1–{zhivora_fixed_top}.',
                  'FAR-tour назначает индивидуальные места по вторым поражениям; места сетки выше допускают общие места.',
                  'Эталонные рейтинги отсортированы по убыванию с разрешением равенств по ID.', '',
                  '| Модель | Метод | Полных | Пропущено | MAE | 95% интервал MAE | Spearman | 95-й процентиль MAE | Максимальная MAE |',
                  '|---|---|---:|---:|---:|---|---:|---:|---:|']
        for model, item in summary.items():
            for internal, name in ALL_RANKING_NAMES.items():
                if name not in item['place_deviations']:
                    continue
                metric = item[internal]
                interval = metric['mae_ci95']
                interval_text = '—' if interval is None else f'[{interval[0]:.6f}, {interval[1]:.6f}]'
                lines.append(f'| {model} | {name} | {metric["samples"]} | {config.repeats-metric["samples"]} | '
                             f'{fmt(metric["mae"])} | {interval_text} | {fmt(metric["spearman"])} | '
                             f'{fmt(metric["mae_p95"])} | {fmt(metric["mae_max"])} |')
        lines += ['', 'Интервал: mean ± 1.96·std(ddof=1)/sqrt(T), приближённая оценка неопределённости среднего.',
                  'Все повторы, включая редкие исходы, учтены. Перцентили описывают хвост, без удаления турниров.', '',
                  '| Модель | Сравнение (левая − правая MAE) | Общих повторов | Средняя разность | 95% интервал |',
                  '|---|---|---:|---:|---|']
        for model, item in summary.items():
            for pair in item['paired_comparisons'].values():
                interval = pair['ci95']
                interval_text = '—' if interval is None else f'[{interval[0]:.6f}, {interval[1]:.6f}]'
                lines.append(f'| {model} | {pair["left"]} − {pair["right"]} | {pair["samples"]} | '
                             f'{fmt(pair["mean_difference"])} | {interval_text} |')
        lines += ['', 'Парные сравнения используют только общие полные повторы; отрицательная разность означает меньшую MAE слева.',
                  '[Метрики каждого турнира](ranking-metrics.csv).', '']
    lines.extend(['| Модель | Доля полных grin-elo | Рейтинговый резерв запущен | Доля запусков резерва |',
                  '|---|---:|---:|---:|'])
    for model, item in summary.items():
        lines.append(f"| {model} | {fmt(item['grin_tour']['samples']/config.repeats)} | "
                     f"{item['rating_fallback']} | {fmt(item['rating_fallback']/config.repeats)} |")
    lines.append('')
    return '\n'.join(lines)


def run_experiments(config: Config, models, grin_tour: bool, output: Path | str, *,
                    grin_cycle_score: bool = False, cycle_weights=DEFAULT_WEIGHTS,
                    far_tour: bool = False, zhivora_tour: bool = False,
                    zhivora_fixed_top: int = 6) -> dict:
    config.validate()
    normalized_weights = normalize_weights(cycle_weights)
    if not models or any(model not in MODELS for model in models):
        raise ValueError('Select at least one known experiment')
    models = [model for model in MODELS if model in models]
    extended = far_tour or zhivora_tour
    if zhivora_tour and (isinstance(zhivora_fixed_top, bool) or zhivora_fixed_top not in (4, 6)):
        raise ValueError('zhivora-fixed-top must be 4 or 6')
    if zhivora_tour and zhivora_fixed_top == 6 and config.participants not in (1, 2, 3, 4, 8, 16, 22, 32):
        raise ValueError('Pascal top-six mode requires N=8,16,22,32 or N<=4')
    ranking_names = ALL_RANKING_NAMES if extended else RANKING_NAMES
    enabled = {'grin_tour': grin_tour, 'grin_cycle_score': grin_cycle_score,
               'far_tour': far_tour, 'zhivora_tour': zhivora_tour}
    measure = individual_metrics if extended else rank_metrics
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 6 if extended else 5, 'config': asdict(config), 'models': models,
                'grin_tour': grin_tour, 'grin_cycle_score': grin_cycle_score,
                'ranking_names': ranking_names, 'rating_bounds': config.rating_bounds(),
                'cycle_weights': list(cycle_weights),
                'normalized_cycle_weights': list(normalized_weights),
                'environment': environment(), 'status': 'running'}
    if extended:
        manifest.update(far_tour=far_tour, zhivora_tour=zhivora_tour, zhivora_fixed_top=zhivora_fixed_top)
    write_json(output/'config.json', manifest)
    logger = logging.getLogger('armplaces.experiments')
    handler = logging.FileHandler(output/'run.log', mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    accumulators = {model: {'completed': 0, 'strongest_wins': 0, 'undefined': Counter(), 'partial': Counter(),
                           'bracket': [], 'grin_tour': [], 'grin_cycle_score': [],
                           'cycle_activated': 0, 'cycle_success': 0,
                           'cycle_standard_success': 0, 'cycle_failures': Counter(),
                           'cycle_subset_grin_tour': [], 'cycle_subset_grin_cycle_score': [],
                           'place_errors': {name: [] for name in ranking_names if enabled[name]},
                           'rating_fallback': 0}
                    for model in models}
    if extended:
        for accumulator in accumulators.values():
            accumulator.update(far_tour=[], zhivora_tour=[],
                               paired_mae={ranking_names[name]: [] for name in ranking_names if enabled[name]})
    participants_fields = ['repeat', 'model', 'participant', 'draw_position', 'rating', 'new_rating',
                           'wins', 'losses', 'played', 'place', 'grin_place', 'grin_status', 'grin_reason',
                           'cycle_place', 'cycle_status', 'cycle_reason']
    if extended:
        participants_fields += ['far_place', 'far_status', 'far_reason',
                                'zhivora_place', 'zhivora_status', 'zhivora_reason']
    bouts_fields = ['repeat', 'model', 'match', 'stage', 'a', 'b', 'technical', 'winner', 'loser',
                    'p_a', 'p_start_a', 'effective_a', 'effective_b', 'previous_bouts_a', 'previous_bouts_b',
                    'fatigue_before_a', 'fatigue_before_b', 'fatigue_after_a', 'fatigue_after_b']
    try:
        with ExitStack() as stack:
            participants_file = stack.enter_context((output/'participants.csv').open('w', newline='', encoding='utf-8'))
            bouts_file = stack.enter_context((output/'bouts.csv').open('w', newline='', encoding='utf-8'))
            tournaments = stack.enter_context((output/'tournaments.jsonl').open('w', encoding='utf-8'))
            participants_writer = csv.DictWriter(participants_file, participants_fields)
            bouts_writer = csv.DictWriter(bouts_file, bouts_fields)
            participants_writer.writeheader()
            bouts_writer.writeheader()
            metrics_writer = None
            if extended:
                metrics_file = stack.enter_context((output/'ranking-metrics.csv').open('w', newline='', encoding='utf-8'))
                metrics_writer = csv.DictWriter(metrics_file, ['repeat', 'model', 'method', 'status', 'mae', 'spearman'])
                metrics_writer.writeheader()
            for repeat in range(config.repeats):
                ratings, draw = generate_field(config, repeat)
                for model in models:
                    result = simulate(config, model, ratings, draw, repeat=repeat, grin_tour=grin_tour,
                                      grin_cycle_score=grin_cycle_score,
                                      cycle_weights=normalized_weights, far_tour=far_tour,
                                      zhivora_tour=zhivora_tour, zhivora_fixed_top=zhivora_fixed_top)
                    ranking = result['grin_tour']
                    cycle = result['grin_cycle_score']
                    accumulator = accumulators[model]
                    accumulator['completed'] += 1
                    accumulator['strongest_wins'] += ratings[result['champion']] == max(ratings)
                    accumulator['bracket'].append(rank_metrics(ratings, result['places']))
                    accumulator['rating_fallback'] += ranking.get('method') == 'opponent_ratings'
                    for method, errors in accumulator['place_errors'].items():
                        if result[method]['status'] == 'ok':
                            errors.append(place_errors(ratings, {int(k): v for k, v in result[method]['places'].items()}))
                    if extended:
                        for internal, name in ranking_names.items():
                            if not enabled[internal]:
                                continue
                            ranked = result[internal]
                            metric = (measure(ratings, {int(k): v for k, v in ranked['places'].items()})
                                      if ranked['status'] == 'ok' else {'mae': None, 'spearman': None})
                            accumulator['paired_mae'][name].append(metric['mae'])
                            metrics_writer.writerow({'repeat': repeat, 'model': model, 'method': name,
                                                     'status': ranked['status'], **metric})
                            if internal in ('far_tour', 'zhivora_tour') and ranked['status'] == 'ok':
                                accumulator[internal].append(metric)
                    if ranking['status'] == 'ok':
                        accumulator['grin_tour'].append(measure(ratings, {int(k):v for k,v in ranking['places'].items()}))
                    elif ranking['status'] in ('undefined', 'partial'):
                        accumulator[ranking['status']][ranking['reason']] += 1
                    if grin_cycle_score:
                        diagnostics = cycle['diagnostics']
                        accumulator['cycle_activated'] += diagnostics['activated']
                        accumulator['cycle_standard_success'] += not diagnostics['activated'] and cycle['status']=='ok'
                        if cycle['status'] == 'ok':
                            measurement = measure(ratings, {int(k):v for k,v in cycle['places'].items()})
                            accumulator['grin_cycle_score'].append(measurement)
                            if diagnostics['activated']:
                                accumulator['cycle_success'] += 1
                                accumulator['cycle_subset_grin_cycle_score'].append(measurement)
                        elif diagnostics['activated']:
                            accumulator['cycle_failures'][cycle['reason']] += 1
                        if diagnostics['activated'] and ranking['status']=='ok':
                            accumulator['cycle_subset_grin_tour'].append(
                                measure(ratings, {int(k):v for k,v in ranking['places'].items()}))
                    draw_positions = {who: i+1 for i, who in enumerate(draw)}
                    for who in range(config.participants):
                        row = {'repeat': repeat, 'model': model, 'participant': who,
                            'draw_position': draw_positions[who], 'rating': ratings[who],
                            'new_rating': result['new_ratings'][who], 'wins': result['wins'][who],
                            'losses': result['losses'][who], 'played': result['played'][who],
                            'place': result['places'][who], 'grin_place': ranking['places'].get(str(who)),
                            'grin_status': ranking['status'], 'grin_reason': ranking['reason'],
                            'cycle_place': cycle['places'].get(str(who)),
                            'cycle_status': cycle['status'], 'cycle_reason': cycle['reason']}
                        if extended:
                            for internal, prefix in (('far_tour', 'far'), ('zhivora_tour', 'zhivora')):
                                row.update({prefix+'_place': result[internal]['places'].get(str(who)),
                                            prefix+'_status': result[internal]['status'],
                                            prefix+'_reason': result[internal]['reason']})
                        participants_writer.writerow(row)
                    for bout in result['bouts']:
                        bouts_writer.writerow({'repeat': repeat, 'model': model, **bout})
                    record_fields = ['repeat', 'model', 'draw', 'champion', 'reset', 'sequence', 'grin_tour',
                                     'grin_cycle_score', 'eliminated_slots', 'bracket']
                    if extended:
                        record_fields += ['far_tour', 'zhivora_tour']
                    tournaments.write(json.dumps({key: result[key] for key in record_fields},
                        ensure_ascii=False, allow_nan=False)+'\n')
                if (repeat+1) % 100 == 0 or repeat+1 == config.repeats:
                    logger.info('Completed %d/%d repetitions for %s', repeat+1, config.repeats, ', '.join(models))
        summary = {}
        for model, accumulator in accumulators.items():
            item = {'completed': accumulator['completed'],
                    'strongest_win_rate': accumulator['strongest_wins']/config.repeats,
                    'grin_undefined_rate': sum(accumulator['undefined'].values())/config.repeats if grin_tour else None,
                    'grin_undefined_reasons': dict(accumulator['undefined']),
                    'grin_partial_rate': sum(accumulator['partial'].values())/config.repeats if grin_tour else None,
                    'grin_partial_reasons': dict(accumulator['partial']),
                    'cycle_standard_success': accumulator['cycle_standard_success'],
                    'cycle_activated': accumulator['cycle_activated'],
                    'cycle_success': accumulator['cycle_success'],
                    'cycle_failures': dict(accumulator['cycle_failures']),
                    'rating_fallback': accumulator['rating_fallback'],
                    'place_deviations': {ranking_names[method]: {
                        'samples': len(errors), 'skipped': config.repeats-len(errors),
                        'places': summarize_places(errors, config.participants)}
                        for method, errors in accumulator['place_errors'].items()}}
            methods = ['bracket', 'grin_tour', 'grin_cycle_score', 'cycle_subset_grin_tour', 'cycle_subset_grin_cycle_score']
            if extended:
                methods += ['far_tour', 'zhivora_tour']
                item['paired_comparisons'] = paired_comparisons(accumulator['paired_mae'])
            for method in methods:
                samples = accumulator[method]
                correlations = [sample['spearman'] for sample in samples if sample['spearman'] is not None]
                item[method] = {'samples': len(samples),
                                'mae': float(np.mean([sample['mae'] for sample in samples])) if samples else None,
                                'spearman': float(np.mean(correlations)) if correlations else None,
                                'correlation_samples': len(correlations)}
                if extended and method in ranking_names:
                    item[method].update(metric_details([sample['mae'] for sample in samples]))
            summary[model] = item
        write_json(output/'summary.json', summary)
        report = report_markdown(config, summary, zhivora_fixed_top=zhivora_fixed_top if extended else None)
        for model, item in summary.items():
            for method, statistics in item['place_deviations'].items():
                report += '\n' + write_place_artifacts(output, model, method, statistics, config)
        (output/'report.md').write_text(report, encoding='utf-8')
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
