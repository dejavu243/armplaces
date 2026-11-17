"""Three N=22 examples and 1000 tournaments per size and rating distribution.

Run with .venv/bin/python scripts/run_formula_experiment.py.
Every category uses the normal export and independent artifact verification.
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import shlex
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import Config
from simulation.core import DISTRIBUTIONS
from simulation.experiments import run_experiments, write_json
from simulation.final_statistics import final_record, summarize_finals


def read_category(folder):
    """Join verified participant and bout CSVs to their tournament records."""
    manifest = json.loads((folder/'config.json').read_text())
    if manifest['status'] != 'complete' or manifest['models'] != ['elo-tire']:
        raise ValueError('Expected a complete elo-tire category')
    participants, bouts = defaultdict(list), defaultdict(list)
    with (folder/'participants.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            participants[int(row['repeat'])].append(row)
    with (folder/'bouts.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            bout = dict(row)
            for key in ('match', 'a', 'b', 'winner', 'loser'):
                bout[key] = int(row[key])
            for key in ('p_a', 'effective_a', 'effective_b', 'fatigue_before_a', 'fatigue_before_b',
                        'fatigue_after_a', 'fatigue_after_b'):
                bout[key] = float(row[key])
            bout['technical'] = row['technical'] == 'True'
            bouts[int(row['repeat'])].append(bout)
    results = []
    with (folder/'tournaments.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            result = json.loads(line)
            rows = sorted(participants[result['repeat']], key=lambda row: int(row['participant']))
            result.update(ratings=[float(row['rating']) for row in rows],
                          participants=rows, bouts=bouts[result['repeat']])
            results.append(result)
    if len(results) != manifest['config']['repeats']:
        raise ValueError('Incomplete category')
    finals = [final_record(result) for result in results]
    if any(row is None for row in finals):
        raise ValueError('The experiment requires at least two participants')
    with (folder/'finals.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(finals[0]))
        writer.writeheader()
        writer.writerows(finals)
    summary = summarize_finals(finals)
    write_json(folder/'finals-summary.json', summary)
    return manifest, results, finals, summary


def relative_link(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def write_example(result, final, distribution, path, folder):
    lines = [f'# Турнир с 22 участниками {distribution}', '',
             'Модель elo-tire с убыванием силы; seed=42, repeat=0. Случайная жеребьёвка, DE_OLD_fix.',
             'R(S)=R₀·0.85^(S/1002); L=max(0,334−|ΔR_eff|); dR=167.', '',
             'Номера участников начинаются с 1; CSV использует ID от 0.', '',
             'Жеребьёвка: '+', '.join(str(who+1) for who in result['draw'])+'.', '',
             '## Участники и места', '',
             '| Участник | R₀ | Место сетки | Grin-Cycle | Grin-Elo |',
             '|---:|---:|---:|---:|---:|']
    for row in result['participants']:
        lines.append(f'| {int(row["participant"])+1} | {float(row["rating"]):.3f} | '
                     f'{row["place"]} | {row["cycle_place"] or "—"} | {row["grin_place"] or "—"} |')
    lines += ['', '## Все бои', '',
              'Эффективные рейтинги и S указаны перед боем. Нагрузка L одновременно добавляется обоим.', '',
              '| Бой | Стадия | a | b | R_eff a | R_eff b | S a | S b | L | P(a), % | Победил |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for bout in result['bouts']:
        load = bout['fatigue_after_a']-bout['fatigue_before_a']
        lines.append(f'| {bout["match"]+1} | {bout["stage"]} | {bout["a"]+1} | {bout["b"]+1} | '
                     f'{bout["effective_a"]:.3f} | {bout["effective_b"]:.3f} | '
                     f'{bout["fatigue_before_a"]:.3f} | {bout["fatigue_before_b"]:.3f} | '
                     f'{load:.3f} | {100*bout["p_a"]:.3f} | {bout["winner"]+1} |')
    lines += ['', '## Финал', '',
              f'Верхняя ветка: №{final["upper"]+1}, R_eff={final["effective_upper"]:.3f}. '
              f'Нижняя ветка: №{final["lower"]+1}, R_eff={final["effective_lower"]:.3f}.',
              f'Разность верхняя − нижняя: {final["gap"]:.3f}; модуль: {final["abs_gap"]:.3f}.',
              f'Вероятность победы нижней ветки в первом финале: {final["lower_win_probability"]:.3%}.',
              f'Суперфинал: {"да" if final["reset"] else "нет"}. Чемпион: №{final["champion"]+1}.', '']
    if final['reset']:
        lines += [f'Перед суперфиналом разность верхняя − нижняя: {final["reset_gap"]:.3f}; '
                  f'модуль: {final["reset_abs_gap"]:.3f}.', '']
    lines += [f'[CSV и JSONL полного прогона]({relative_link(folder, path.parent)}).', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def plot_categories(categories, destination):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for distribution in DISTRIBUTIONS:
        items = sorted((item for item in categories if item['distribution'] == distribution),
                       key=lambda item: item['participants'])
        sizes = [item['participants'] for item in items]
        axes[0].plot(sizes, [item['statistics']['final']['mean_abs_gap'] for item in items],
                     marker='o', label=distribution)
        axes[1].plot(sizes, [100*item['statistics']['reset_rate'] for item in items],
                     marker='o', label=distribution)
    for axis in axes:
        axis.set_xlabel('Число участников')
        axis.grid(alpha=.3)
        axis.legend()
    axes[0].set_ylabel('Средняя |ΔR_eff| перед финалом')
    axes[1].set_ylabel('Суперфиналы, % турниров')
    axes[1].set_ylim(0, 100)
    figure.suptitle('Новые формулы: убывание силы, A′=0.85, C=334, N_f=3, dR=167')
    for extension in ('png', 'svg'):
        figure.savefig(destination.with_suffix('.'+extension), dpi=160)
    plt.close(figure)


def write_report(report, data, root, sizes, repeats):
    report.parent.mkdir(parents=True, exist_ok=True)
    data_path = report.parent/'data'/f'{report.stem}.json'
    data_path.parent.mkdir(exist_ok=True)
    write_json(data_path, data)
    figures = report.parent/'figures'
    figures.mkdir(exist_ok=True)
    figure = figures/f'{report.stem}-finals'
    plot_categories(data['categories'], figure)
    total = len(data['categories'])*repeats
    lines = ['# Финалы и суперфиналы по новым формулам', '',
             f'Проведено **{total:,} турниров**: по {repeats} повторов для N={", ".join(map(str, sizes))} '
             'на каждом из трёх распределений. Ещё три отдельных примера содержат по 22 участника.',
             'Seed=42; случайная жеребьёвка, сетка DE_OLD_fix, выбывание после двух поражений.', '',
             '## Формулы и исходные параметры', '',
             'Источник: предоставленный PDF «Система_проведения_соревнований_новая_формула.pdf», с. 3–5.',
             'Знак степени согласован с пользователем: **положительный**, сила уменьшается.',
             'В PDF напечатана отрицательная степень; при A′=0.85 она увеличивала бы силу.', '',
             '```text', 'R_eff(S) = R₀ · 0.85^(S/(3·334))',
             'S_a ← S_a + max(0,334−|R_eff,a−R_eff,b|)',
             'S_b ← S_b + max(0,334−|R_eff,a−R_eff,b|)',
             'P(a победит) = 1 / (1+10^((R_eff,b−R_eff,a)/167))', '```', '',
             'Нагрузка рассчитывается по эффективным рейтингам до боя и одинакова для обоих участников.',
             'При |ΔR_eff|≥334 нагрузка равна нулю, в том числе при разнице больше 400.',
             'После трёх равных боёв S=1002, R_eff=0.85·R₀. Параметр A=0.9 из PDF',
             'в новой формуле не используется. При преимуществе 167 пунктов вероятность победы 90.909%.', '',
             '| Распределение | Параметры |', '|---|---|',
             '| beta-prime | α=135.1, β=65.2, scale=530; усечение [600,1900) |',
             '| weibull | shape=2.2, scale=220, location=852; без усечения |',
             '| uniform | [600,1900) |', '',
             'Три варианта трактуются как три поддерживаемых распределения. Их форма и масштаб различаются.',
             'Рейтинги генерируются заново для каждого повтора. Коэффициенты усталости не подбирались.', '',
             '## Три примера с 22 участниками', '',
             '| Распределение | R_eff верхней ветки | R_eff нижней ветки | Модуль разности | Суперфинал | Чемпион |',
             '|---|---:|---:|---:|---|---:|']
    for item in data['examples']:
        row = item['final']
        link = f'{report.stem}-n22-{item["distribution"]}.md'
        lines.append(f'| [{item["distribution"]}]({link}) | {row["effective_upper"]:.3f} | '
                     f'{row["effective_lower"]:.3f} | {row["abs_gap"]:.3f} | '
                     f'{"да" if row["reset"] else "нет"} | {row["champion"]+1} |')
    lines += ['', 'В каждом примере приведены все участники, жеребьёвка, места и полный журнал боёв.', '',
              '## Статистика серий', '',
              'Разность измеряется **перед первым финалом**, до добавления нагрузки за него.',
              'Знак Δ означает верхняя ветка минус нижняя; модуль характеризует близость силы финалистов.',
              'Верхняя ветка определяется по нулю предыдущих поражений, независимо от порядка ячеек.', '',
              '| Распределение | N | Средняя Δ | Средняя ∣Δ∣ | Медиана ∣Δ∣ | P95 ∣Δ∣ | Макс. ∣Δ∣ | Суперфиналов | Доля |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in data['categories']:
        stat = item['statistics']
        gap = stat['final']
        lines.append(f'| {item["distribution"]} | {item["participants"]} | {gap["mean_gap"]:.3f} | '
                     f'{gap["mean_abs_gap"]:.3f} | {gap["median_abs_gap"]:.3f} | {gap["p95_abs_gap"]:.3f} | '
                     f'{gap["max_abs_gap"]:.3f} | {stat["reset_count"]}/{gap["samples"]} | {stat["reset_rate"]:.1%} |')
    lines += ['', f'![Разность рейтингов и доля суперфиналов](figures/{figure.name}.png)', '',
              '## Рейтинги перед суперфиналом', '',
              'Здесь учтены только состоявшиеся суперфиналы. Для их доли в предыдущей таблице',
              f'знаменатель — все турниры соответствующей категории ({repeats}).', '',
              '| Распределение | N | Суперфиналов | Средняя Δ | Средняя ∣Δ∣ | Медиана ∣Δ∣ | P95 ∣Δ∣ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for item in data['categories']:
        gap = item['statistics']['superfinal']
        values = ['—' if gap[key] is None else f'{gap[key]:.3f}'
                  for key in ('mean_gap', 'mean_abs_gap', 'median_abs_gap', 'p95_abs_gap')]
        lines.append(f'| {item["distribution"]} | {item["participants"]} | {gap["samples"]} | '+
                     ' | '.join(values)+' |')
    lines += ['', '## Проверка и воспроизведение', '',
              'Каждая категория прошла независимое воспроизведение таблиц переходов, усталости,',
              'эффективных рейтингов, вероятностей, обновления рейтинга и расстановок мест.',
              'Финальные показатели рассчитаны по проверенным CSV и JSONL. В каждой категории',
              'добавлены finals.csv и finals-summary.json; исходные результаты сохранены в каталоге ниже.',
              'Примеры — отдельные запуски с repeat=0; при наличии серии N=22 совпадают с её первым повтором.', '',
              f'Сводные числа и манифесты: [JSON]({relative_link(data_path, report.parent)}).',
              f'Каталог сырых артефактов: `{root}`.',
              f'SHA256 исходного кода пакетов: `{data["source_sha256"]}`.', '',
              '```bash', '.venv/bin/python scripts/run_formula_experiment.py '+
              f'--sizes {" ".join(map(str, sizes))} --repeats {repeats} '+
              f'--output {shlex.quote(str(root))} --report {shlex.quote(str(report))}',
              '.venv/bin/python -m unittest discover', '.venv/bin/python -m ruff check .', '```', '',
              'Результаты описывают заданную модель и случайную жеребьёвку; посев SDE не реализован.',
              f'При {repeats} повторах максимальная стандартная ошибка доли составляет около '
              f'{50/repeats**.5:.2f} процентного пункта.', '']
    report.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8, 16, 22, 32])
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('results/new-formulas'))
    parser.add_argument('--report', type=Path, default=Path('docs/new-formulas-report.md'))
    args = parser.parse_args()
    if len(set(args.sizes)) != len(args.sizes) or any(not 2 <= n <= 32 for n in args.sizes):
        parser.error('sizes must be distinct integers in 2..32')
    if args.repeats < 1:
        parser.error('repeats must be positive')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data = {'categories': [], 'examples': [], 'source_sha256': None,
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for distribution in DISTRIBUTIONS:
        example_folder = args.output/'examples'/distribution
        config = Config(participants=22, repeats=1, distribution=distribution)
        run_experiments(config, ['elo-tire'], True, example_folder, grin_cycle_score=True)
        manifest, results, finals, _ = read_category(example_folder)
        data['examples'].append({'distribution': distribution, 'final': finals[0], 'manifest': manifest})
        write_example(results[0], finals[0], distribution,
                      args.report.with_name(f'{args.report.stem}-n22-{distribution}.md'), example_folder)
        for n in args.sizes:
            print(f'Running {distribution}, N={n}, {args.repeats} tournaments', flush=True)
            folder = args.output/distribution/f'n{n}'
            config = Config(participants=n, repeats=args.repeats, distribution=distribution)
            run_experiments(config, ['elo-tire'], True, folder, grin_cycle_score=True)
            manifest, _, _, summary = read_category(folder)
            fingerprint = manifest['environment']['source_sha256']
            if data['source_sha256'] not in (None, fingerprint):
                raise ValueError('Categories used different source code')
            data['source_sha256'] = fingerprint
            data['categories'].append({'distribution': distribution, 'participants': n,
                                       'statistics': summary, 'manifest': manifest})
    write_report(args.report, data, args.output, args.sizes, args.repeats)
    write_json(args.output/'finals-summary.json', data)
    print(f'Wrote {args.report}: {len(data["categories"])*args.repeats} tournaments and 3 examples', flush=True)


if __name__ == '__main__':
    main()
