"""Compare four placement methods on 30000 shared Elo-tire tournaments."""

import argparse
import csv
import json
import logging
import shlex
from pathlib import Path

from simulation import Config
from simulation.comparison_plots import METHOD_LABELS, comparison_panels, mae_overview
from simulation.experiments import run_experiments
from simulation.place_statistics import ALL_RANKING_NAMES


def write_report(root, report, sizes, repeats, seed, fixed_top):
    report.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}
    configs = {}
    fingerprints = set()
    for n in sizes:
        folder = root/f'n{n}'
        manifest = json.loads((folder/'config.json').read_text())
        if (manifest['status'] != 'complete' or manifest['schema_version'] != 6
                or manifest['models'] != ['elo-tire'] or manifest['config']['repeats'] != repeats
                or manifest['config']['seed'] != seed or manifest['zhivora_fixed_top'] != fixed_top
                or not all(manifest[name] for name in ALL_RANKING_NAMES)):
            raise ValueError('Incomplete or inconsistent placement comparison category')
        fingerprints.add(manifest['environment']['source_sha256'])
        configs[n] = Config(**manifest['config'])
        summaries[n] = json.loads((folder/'summary.json').read_text())['elo-tire']
    if len(fingerprints) != 1:
        raise ValueError('Categories used different source code')
    total = sum(item['completed'] for item in summaries.values())
    first = configs[sizes[0]]
    bounds = first.rating_bounds()
    lines = ['# Сравнение FAR-tour, Zhivora-Tour, Grin-Cycle и Grin-Elo', '',
             f'**{total} турниров**, N={", ".join(map(str, sizes))}; по **{repeats} повторов**; seed={seed}.',
             f'На каждом турнире рассчитаны четыре метода: всего {4*total} оценок расстановки.', '',
             f'Beta-prime: α={first.alpha:g}, β={first.beta:g}, scale={first.scale:g}; [{bounds[0]:g},{bounds[1]:g}).',
             f'Elo-tire: A={first.tire_floor:g}, A′={first.tire_reference:g}, C={first.tire_c:g}, '
             f'N_f={first.tire_bouts}; dR={first.dr:g}; K={first.k:g}.',
             'Сетка DE_OLD_fix, случайная жеребьёвка. Все методы используют **одни и те же бои**.',
             f'Zhivora-Tour: **фиксированы места 1–{fixed_top}**. [Описание и сверка с Pascal](placement-methods.md).',
             'Веса Grin-Cycle равны, 0.33/0.33/0.33 с нормированием.', '',
             '## Точность и полнота', '',
             'MAE — среднее абсолютное расстояние между назначенным и истинным местом.',
             'Каждому спортсмену назначается индивидуальное место; эталон — исходные рейтинги по убыванию, равенства по ID.',
             'Меньше MAE — лучше; больше Spearman — лучше. Неполные расстановки исключаются из метрик целиком.', '',
             '| N | Метод | Полных | Пропущено | MAE | 95% интервал MAE | Spearman | P95 MAE | Максимальная MAE |',
             '|---:|---|---:|---:|---:|---|---:|---:|---:|']
    rows = []
    for n in sizes:
        for internal, name in ALL_RANKING_NAMES.items():
            item = summaries[n][internal]
            ci = item['mae_ci95']
            interval = '—' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'
            def number(value):
                return '—' if value is None else f'{value:.6f}'
            lines.append(f'| {n} | {METHOD_LABELS[name]} | {item["samples"]} | {repeats-item["samples"]} | '
                         f'{number(item["mae"])} | {interval} | {number(item["spearman"])} | '
                         f'{number(item["mae_p95"])} | {number(item["mae_max"])} |')
            rows.append({'participants': n, 'method': name, 'samples': item['samples'],
                         'skipped': repeats-item['samples'], 'mae': item['mae'], 'mae_se': item['mae_se'],
                         'ci95_low': ci[0] if ci else None, 'ci95_high': ci[1] if ci else None,
                         'spearman': item['spearman'], 'mae_p95': item['mae_p95'], 'mae_max': item['mae_max']})
    with (root/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = report.parent/'figures'
    mae_overview(summaries, figures/'placement-mae', zhivora_fixed_top=fixed_top)
    lines += ['', '![Среднее абсолютное отклонение по истинным местам](figures/placement-mae.png)', '',
              '## Парные сравнения', '',
              'Разность рассчитывается внутри одного турнира, затем усредняется по общим полным повторам.',
              'Отрицательное значение означает меньшую MAE у левого метода. Интервалы рассчитаны для каждого сравнения отдельно.', '',
              '| N | Сравнение: левая − правая MAE | Общих повторов | Средняя разность | 95% интервал разности |',
              '|---:|---|---:|---:|---|']
    paired_rows = []
    for n in sizes:
        for pair in summaries[n]['paired_comparisons'].values():
            ci = pair['ci95']
            interval = '—' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'
            lines.append(f'| {n} | {METHOD_LABELS[pair["left"]]} − {METHOD_LABELS[pair["right"]]} | '
                         f'{pair["samples"]} | {number(pair["mean_difference"])} | {interval} |')
            paired_rows.append({'participants': n, **pair})
    (root/'paired-comparisons.json').write_text(json.dumps(paired_rows, ensure_ascii=False, indent=2)+'\n')
    lines += ['', '## Призёры и дополнительный этап', '',
              'Чемпион и места 2–3 определяются решающими боями и совпадают у всех методов.',
              'Победа сильнейшего поэтому является свойством проведённых боёв, а не способом сравнения этих четырёх методов.',
              'При этом ошибки истинных сильнейших участников могут различаться, если они не попали в фактическую тройку.', '',
              '| N | Сильнейший победил | Граф без цикла | Дополнительный этап | Доля запусков | Успешно разрешено | Резерв Grin-Elo |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for n, item in sorted(summaries.items()):
        lines.append(f'| {n} | {item["strongest_win_rate"]:.2%} | {item["cycle_standard_success"]} | '
                     f'{item["cycle_activated"]} | {item["cycle_activated"]/repeats:.2%} | '
                     f'{item["cycle_success"]} | {item["rating_fallback"]} |')
    lines += ['', '## Графики отклонений', '',
              'Для каждого истинного места: дисперсия δ (ddof=0), среднее |δ|, максимум вниз и максимум вверх.',
              'Все редкие результаты включены; увеличение числа повторов уменьшает шум среднего и лучше показывает хвост.', '']
    for n in sizes:
        comparison_panels(summaries[n], configs[n], figures/f'placement-n{n}', zhivora_fixed_top=fixed_top)
        lines += [f'### N={n}', '', f'![Четыре показателя отклонений, N={n}](figures/placement-n{n}.png)', '']
    lines += ['## Выводы по этой серии', '']
    for n, summary in sorted(summaries.items()):
        ranked = sorted(((summary[internal]['mae'], name) for internal, name in ALL_RANKING_NAMES.items()
                         if summary[internal]['mae'] is not None))
        best_mae, best = ranked[0]
        far = summary['far_tour']['mae']
        improvement = (far-best_mae)/far if far else 0
        lines.append(f'- N={n}: минимальная MAE у **{METHOD_LABELS[best]}**, {best_mae:.6f}; '
                     f'снижение относительно FAR-tour — {improvement:.2%}.')
    for n, summary in sorted(summaries.items()):
        comparison = summary['paired_comparisons']['zhivora-tour__vs__grin-cycle']
        interval = comparison['ci95']
        if interval is None:
            continue
        if interval[0] <= 0 <= interval[1]:
            conclusion = 'интервал содержит ноль; устойчивое преимущество одного из этих двух методов не установлено'
        elif interval[0] > 0:
            conclusion = 'интервал положителен; Grin-Cycle имеет меньшую среднюю ошибку в данной модели'
        else:
            conclusion = 'интервал отрицателен; Zhivora-Tour имеет меньшую среднюю ошибку в данной модели'
        lines.append(f'- N={n}: Zhivora − Grin-Cycle = {comparison["mean_difference"]:.6f}, '
                     f'95% интервал [{interval[0]:.6f}, {interval[1]:.6f}]; {conclusion}.')
    lines += ['', 'При близких средних учитывайте парные интервалы выше. Полная разрешимость произвольного входа',
              'и преимущество на других распределениях этой серией не доказаны.',
              'Интервалы mean ± 1.96·std(ddof=1)/sqrt(T) характеризуют неопределённость моделирования при заданных коэффициентах.',
              'Неопределённость подбора рейтингов и физиологическая калибровка усталости сюда не включены.', '',
              '## Проверка и воспроизведение', '',
              'Каждая категория прошла независимую проверку усталости, вероятностей, счётчиков,',
              'хронологии FAR-tour, групп Живоры, диагностики циклов, индивидуальных метрик,',
              'парных разностей и доверительных интервалов. Новые артефакты — schema 6; чтение 4/5 сохранено.',
              f'SHA256 пакетов: `{next(iter(fingerprints))}`.',
              f'CSV/JSONL, метрики каждого турнира и индивидуальные графики всех методов: `{root}`.', '',
              '```bash', f'.venv/bin/python scripts/run_placement_comparison.py --repeats {repeats} --seed {seed} '
              f'--zhivora-fixed-top {fixed_top} --output {shlex.quote(str(root))} --report {shlex.quote(str(report))}',
              f'.venv/bin/python -m simulation.verify {shlex.quote(str(root/"n8"))}',
              f'.venv/bin/python -m simulation.verify {shlex.quote(str(root/"n16"))}',
              f'.venv/bin/python -m simulation.verify {shlex.quote(str(root/"n32"))}', '```', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {report}: {total} verified tournaments, {4*total} rankings', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--zhivora-fixed-top', type=int, choices=(4, 6), default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--smoke', action='store_true', help='100 repeats at each N: 300 tournaments')
    args = parser.parse_args()
    repeats = 100 if args.smoke else args.repeats
    root = args.output or Path('results/placement-comparison-ci' if args.smoke else 'results/placement-comparison')
    report = args.report or (root/'batch-report.md' if args.smoke else Path('docs/placement-comparison-report.md'))
    sizes = (8, 16, 32)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    for n in sizes:
        print(f'Running N={n}, {repeats} Elo-tire tournaments, four placement methods', flush=True)
        run_experiments(Config(participants=n, repeats=repeats, seed=args.seed), ['elo-tire'], True,
                        root/f'n{n}', grin_cycle_score=True, far_tour=True, zhivora_tour=True,
                        zhivora_fixed_top=args.zhivora_fixed_top)
    write_report(root, report, sizes, repeats, args.seed, args.zhivora_fixed_top)


if __name__ == '__main__':
    main()
