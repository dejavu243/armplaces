"""Run and report the reproducible 36000-tournament series, or a 300-tournament CI series.

Usage: .venv/bin/python scripts/run_elo_tire_experiment.py [--smoke]
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

from simulation import Config
from simulation.core import DISTRIBUTIONS, MODELS
from simulation.experiments import run_experiments


def write_batch_report(root, report, sizes, models, repeats, smoke):
    report.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}
    hashes = set()
    for distribution in DISTRIBUTIONS:
        for n in sizes:
            folder = root/distribution/f'n{n}'
            manifest = json.loads((folder/'config.json').read_text())
            if manifest['status'] != 'complete' or manifest['config']['repeats'] != repeats:
                raise ValueError('Incomplete experiment category')
            hashes.add(manifest['environment']['source_sha256'])
            summaries[distribution, n] = json.loads((folder/'summary.json').read_text())
    if len(hashes) != 1:
        raise ValueError('Experiment categories used different source code')
    total = sum(item['completed'] for summary in summaries.values() for item in summary.values())
    lines = ['# Эксперимент elo-tire: точность мест', '',
             f'Всего **{total} турниров**; N={", ".join(map(str, sizes))}; '
             f'по {repeats} повторов на сочетание; seed=42.', '',
             'Случайная жеребьёвка, сетка DE_OLD_fix. Веса циклов: 0.33, 0.33, 0.33, нормированы до 1.',
             '`grin-cycle` — основной метод без исходных рейтингов. `grin-elo` использует рейтинги соперников при отказе графа.',
             'Призёры в обоих методах фиксируются по решающим боям.', '',
             'Распределения: beta-prime (2, 3, scale=1000), Weibull (2.2, scale=220, location=852),',
             'uniform [800,2400). Бета-прайм и Вейбулла без усечения.',
             'elo-tire: A=0.85, A′=0.90, C=400, N_f=3; dR=200; K=160.',
             'Широкий равномерный диапазон — параметр эксперимента, не универсальная граница рейтингов.',
             'Сравнение распределений включает различия их формы и масштаба.', '']
    complete_elo = sum(item['grin_tour']['samples'] for summary in summaries.values() for item in summary.values())
    complete_cycle = sum(item['grin_cycle_score']['samples'] for summary in summaries.values() for item in summary.values())
    activated = sum(item['cycle_activated'] for summary in summaries.values() for item in summary.values())
    resolved = sum(item['cycle_success'] for summary in summaries.values() for item in summary.values())
    fallback = sum(item['rating_fallback'] for summary in summaries.values() for item in summary.values())
    lines += [f'Полные расстановки: grin-cycle **{complete_cycle}/{total}**, grin-elo **{complete_elo}/{total}**.',
              f'Этап циклов запущен {activated} раз, завершён {resolved} раз; рейтинговый резерв — {fallback} раз.',
              'Полнота на этой серии не является доказательством разрешимости произвольного входа.', '']
    for distribution in DISTRIBUTIONS:
        lines += [f'## {distribution}', '',
                  '| N | Модель | Полных cycle / elo | Циклов | Резерв Эло | MAE cycle | MAE elo | Spearman cycle | Spearman elo | Сильнейший победил |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for n in sizes:
            for model in models:
                item = summaries[distribution, n][model]
                cycle, elo = item['grin_cycle_score'], item['grin_tour']
                def number(value):
                    return '—' if value is None else f'{value:.6f}'
                lines.append(f'| {n} | {model} | {cycle["samples"]} / {elo["samples"]} | '
                             f'{item["cycle_activated"]} | {item["rating_fallback"]} | '
                             f'{number(cycle["mae"])} | {number(elo["mae"])} | '
                             f'{number(cycle["spearman"])} | {number(elo["spearman"])} | '
                             f'{number(item["strongest_win_rate"])} |')
        lines.append('')
        if not smoke:
            item = summaries[distribution, 32]['elo-tire']
            for method in ('grin-cycle', 'grin-elo'):
                statistics = item['place_deviations'][method]
                lines += [f'### elo-tire, N=32, {method}: отклонения по местам', '',
                          f'Учтено {statistics["samples"]} полных расстановок; пропущено {statistics["skipped"]}.', '',
                          '| Место | Дисперсия δ | Среднее \|δ\| | Максимум вниз | Максимум вверх |',
                          '|---:|---:|---:|---:|---:|']
                for row in statistics['places']:
                    lines.append(f'| {row["place"]} | {number(row["variance"])} | '
                                 f'{number(row["mean_abs_deviation"])} | '
                                 f'{number(row["max_down"])} | {number(row["max_up"])} |')
                lines.append('')
            figures = report.parent/'figures'
            figures.mkdir(exist_ok=True)
            source = root/distribution/'n32'/'place-deviations-elo-tire-grin-cycle.png'
            destination = figures/f'elo-tire-{distribution}-n32.png'
            shutil.copyfile(source, destination)
            lines += [f'![Отклонения elo-tire: {distribution}, N=32](figures/{destination.name})', '']
    lines += ['## Артефакты и воспроизведение', '',
              'δ = итоговое место − место по исходному рейтингу. Дисперсия генеральная (ddof=0),',
              'среднее — абсолютное расстояние. Максимум вниз = max(0,max δ), вверх = max(0,−min δ).',
              'Равные исходные рейтинги упорядочиваются по ID. Неполные расстановки исключаются целиком;',
              'число учтённых и пропущенных повторов указано для каждого метода.', '',
              'Все категории прошли независимую проверку журналов, усталости, эффективных рейтингов,',
              'вероятностей, счётчиков, циклов, метрик и пяти столбцов по местам.',
              f'SHA256 исходного кода пакетов: `{next(iter(hashes))}`.',
              'Git SHA, признак незакоммиченных изменений и зависимости сохранены в каждом config.json.',
              'Сырые CSV/JSONL и графики всех сочетаний находятся в results и не коммитятся.', '',
              '```bash', '.venv/bin/python scripts/run_elo_tire_experiment.py' + (' --smoke' if smoke else ''),
              '.venv/bin/python -m unittest discover', '.venv/bin/python -m ruff check .', '```', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {report}: {total} independently verified tournaments', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='N=16, 50 repeats, elo and elo-tire: 300 tournaments')
    parser.add_argument('--output', type=Path, default=Path('results/elo-tire'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    sizes = (16,) if args.smoke else (8, 16, 32)
    models = ('elo', 'elo-tire') if args.smoke else MODELS
    repeats = 50 if args.smoke else 1000
    report = args.report or (args.output/'batch-report.md' if args.smoke else Path('docs/elo-tire-report.md'))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    for distribution in DISTRIBUTIONS:
        for n in sizes:
            config = Config(participants=n, repeats=repeats, distribution=distribution,
                            rating_min=800 if distribution == 'uniform' else None,
                            rating_max=2400 if distribution == 'uniform' else None)
            print(f'Running {distribution}, N={n}, {repeats} repeats, {models}', flush=True)
            run_experiments(config, models, True, args.output/distribution/f'n{n}', grin_cycle_score=True)
    write_batch_report(args.output, report, sizes, models, repeats, args.smoke)


if __name__ == '__main__':
    main()
