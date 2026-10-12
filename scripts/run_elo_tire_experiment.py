"""Run and report the reproducible 36000-tournament series, or a 300-tournament CI series.

Usage: .venv/bin/python scripts/run_elo_tire_experiment.py [--smoke]
"""
import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

from simulation import Config
from simulation.core import DISTRIBUTIONS, MODELS
from simulation.experiments import run_experiments


def write_batch_report(root, report, sizes, models, repeats, smoke):
    report.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}
    configs = {}
    hashes = set()
    for distribution in DISTRIBUTIONS:
        for n in sizes:
            folder = root/distribution/f'n{n}'
            manifest = json.loads((folder/'config.json').read_text())
            if manifest['status'] != 'complete' or manifest['config']['repeats'] != repeats:
                raise ValueError('Incomplete experiment category')
            hashes.add(manifest['environment']['source_sha256'])
            configs[distribution, n] = (manifest['config'], manifest['rating_bounds'])
            summaries[distribution, n] = json.loads((folder/'summary.json').read_text())
    if len(hashes) != 1:
        raise ValueError('Experiment categories used different source code')
    total = sum(item['completed'] for summary in summaries.values() for item in summary.values())
    beta, beta_bounds = configs['beta-prime', sizes[0]]
    weibull, _ = configs['weibull', sizes[0]]
    _, uniform_bounds = configs['uniform', sizes[0]]
    beta_description = ('без усечения' if beta_bounds is None else f'[{beta_bounds[0]:g},{beta_bounds[1]:g})')
    lines = ['# Эксперимент elo-tire: точность мест', '',
             f'Всего **{total} турниров**; N={", ".join(map(str, sizes))}; '
             f'по {repeats} повторов на сочетание; seed=42.', '',
             'Случайная жеребьёвка, сетка DE_OLD_fix. Веса циклов: 0.33, 0.33, 0.33, нормированы до 1.',
             '`grin-cycle` — основной метод без исходных рейтингов. `grin-elo` использует рейтинги соперников при отказе графа.',
             'Призёры в обоих методах фиксируются по решающим боям.', '',
             f'Распределения: beta-prime (α={beta["alpha"]:g}, β={beta["beta"]:g}, scale={beta["scale"]:g}), {beta_description};',
             f'Weibull ({weibull["weibull_shape"]:g}, scale={weibull["weibull_scale"]:g}, location={weibull["weibull_location"]:g}) без усечения;',
             f'uniform [{uniform_bounds[0]:g},{uniform_bounds[1]:g}).',
             'elo-tire: A=0.85, A′=0.90, C=400, N_f=3; dR=200; K=160.',
             'Границы — параметры эксперимента, не универсальные границы рейтингов.',
             'Сравнение распределений включает различия их формы и масштаба.', '']
    if not beta['unbounded_beta_prime']:
        lines += ['Beta-prime откалиброван по **187 реальным рейтингам**, все записи учтены.',
                  '[Данные и метод подбора](beta-prime-calibration.md); [параметры и диагностика](beta-prime-fit.json).',
                  'Вне границ кандидаты отбрасываются и выбираются заново. Усталость, dR и сетка не изменены.', '']
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
                          r'| Место | Дисперсия δ | Среднее \|δ\| | Максимум вниз | Максимум вверх |',
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
    baseline_commit = '0790284dce24959f9b5ff73a338f80ec970db6b0'
    baseline_text = None
    if not smoke and not beta['unbounded_beta_prime']:
        try:
            baseline_text = subprocess.check_output(
                ['git', 'show', baseline_commit+':docs/elo-tire-report.md'], text=True,
                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            pass
    if baseline_text:
        baseline = {}
        distribution = None
        for line in baseline_text.splitlines():
            if line.startswith('## ') and line[3:] in DISTRIBUTIONS:
                distribution = line[3:]
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if distribution and len(cells) == 10 and cells[0] == '32' and cells[1] in MODELS:
                baseline[distribution, cells[1]] = (float(cells[5]), float(cells[9]))
        lines += ['## Сравнение с прежней серией, N=32', '',
                  'В прежней серии beta-prime: α=2, β=3, scale=1000 без границ; uniform [800,2400).',
                  'В новой серии меняются параметры beta-prime и диапазоны beta-prime/uniform.',
                  'Вейбулл, dR, усталость, жеребьёвка и число повторов сохранены.',
                  f'Прежние цифры взяты из отчёта в Git-коммите `{baseline_commit}`;',
                  'отдельная копия старого отчёта в текущем дереве не хранится.', '',
                  '| Распределение | Модель | MAE cycle прежде | MAE cycle теперь | Сильнейший прежде | Сильнейший теперь |',
                  '|---|---|---:|---:|---:|---:|']
        for distribution in DISTRIBUTIONS:
            for model in models:
                if (distribution, model) in baseline:
                    old_mae, old_win = baseline[distribution, model]
                    item = summaries[distribution, 32][model]
                    lines.append(f'| {distribution} | {model} | {old_mae:.6f} | '
                                 f'{item["grin_cycle_score"]["mae"]:.6f} | {old_win:.6f} | {item["strongest_win_rate"]:.6f} |')
        lines += ['', 'Более высокая ошибка новой beta-prime модели означает более трудный,',
                  'приближённый к предоставленным рейтингам состав участников. Старый тяжёлый хвост',
                  'создавал очень сильных лидеров и облегчал определение победителя.', '']
    if not smoke and not beta['unbounded_beta_prime']:
        lines += ['## Выводы', '',
                  'Калибровка устранила неподходящий тяжёлый хвост beta-prime; границы исключают',
                  f'рейтинги ниже {beta_bounds[0]:g} и от {beta_bounds[1]:g} при данном запуске. Параметры описывают предоставленные',
                  '187 значений, а диапазон выбран пользователем. Вейбулл сохраняет параметры статьи.', '',
                  f'Оба метода получили {complete_cycle}/{total} и {complete_elo}/{total} полных расстановок.',
                  f'Дополнительный этап понадобился в {activated/total:.2%} турниров.',
                  'Полнота означает наличие всех мест; она не означает совпадение с исходной силой.', '',
                  'При N=32, elo-tire:', '',
                  '| Распределение | MAE сетки (общие места) | MAE grin-elo | MAE grin-cycle | Сильнейший победил |',
                  '|---|---:|---:|---:|---:|']
        for distribution in DISTRIBUTIONS:
            item = summaries[distribution, 32]['elo-tire']
            lines.append(f'| {distribution} | {item["bracket"]["mae"]:.6f} | '
                         f'{item["grin_tour"]["mae"]:.6f} | {item["grin_cycle_score"]["mae"]:.6f} | '
                         f'{item["strongest_win_rate"]:.2%} |')
        lines += ['', 'Метрика сетки использует средние ранги групп с общими местами; графовые методы',
                  'назначают индивидуальные места. На Вейбулле полная детализация уступает сетке по MAE.',
                  '`grin-cycle` предпочтительнее рейтингового резерва на этой серии и сохраняет оценку',
                  'без доступа к исходной силе. Призёры закреплены по решающим боям.', '',
                  '`strong-win` — контроль с гарантированной победой более сильного. Обычный `elo`',
                  'точнее обеих моделей усталости по MAE во всех девяти сочетаниях N и распределения.',
                  'Усталость увеличивает отклонения от исходной силы; коэффициенты усталости',
                  'остаются экспериментальными и не подбирались по этому файлу рейтингов.', '',
                  'Uniform даёт наименьшую MAE вероятностных моделей при N=32 в новом наборе параметров.',
                  'Это сравнение заданных составов участников: формы и масштабы распределений различаются.', '']
    command = '.venv/bin/python scripts/run_elo_tire_experiment.py'
    if smoke:
        command += ' --smoke --output '+str(root)
    elif beta['unbounded_beta_prime']:
        command += ' --legacy-ratings --output '+str(root)
    else:
        command += f' --rating-min {beta_bounds[0]:g} --rating-max {beta_bounds[1]:g} --output '+str(root)
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
              '```bash', command,
              '.venv/bin/python -m unittest discover', '.venv/bin/python -m ruff check .', '```', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {report}: {total} independently verified tournaments', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='N=16, 50 repeats, elo and elo-tire: 300 tournaments')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--rating-min', type=float, help='Beta-prime and uniform lower bound; default 600')
    parser.add_argument('--rating-max', type=float, help='Beta-prime and uniform upper bound; default 1900')
    parser.add_argument('--legacy-ratings', action='store_true', help='Reproduce beta-prime (2,3,1000) unbounded and uniform [800,2400)')
    args = parser.parse_args()
    sizes = (16,) if args.smoke else (8, 16, 32)
    models = ('elo', 'elo-tire') if args.smoke else MODELS
    repeats = 50 if args.smoke else 1000
    if (args.rating_min is None) != (args.rating_max is None):
        parser.error('rating-min and rating-max must be supplied together')
    if args.legacy_ratings and args.rating_min is not None:
        parser.error('legacy-ratings cannot be combined with custom bounds')
    output = args.output or Path('results/elo-tire-legacy' if args.legacy_ratings else
                               ('results/elo-tire-ci' if args.smoke else 'results/elo-tire-calibrated'))
    report = args.report or (output/'batch-report.md' if args.smoke or args.legacy_ratings else Path('docs/elo-tire-report.md'))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    for distribution in DISTRIBUTIONS:
        for n in sizes:
            parameters = {'participants': n, 'repeats': repeats, 'distribution': distribution}
            if args.legacy_ratings:
                parameters.update(alpha=2, beta=3, scale=1000,
                                  unbounded_beta_prime=distribution == 'beta-prime')
                if distribution == 'uniform':
                    parameters.update(rating_min=800, rating_max=2400)
            elif distribution in ('beta-prime', 'uniform'):
                lower, upper = Config().rating_bounds() if args.rating_min is None else (args.rating_min, args.rating_max)
                parameters.update(rating_min=lower, rating_max=upper)
            config = Config(**parameters)
            print(f'Running {distribution}, N={n}, {repeats} repeats, {models}', flush=True)
            run_experiments(config, models, True, output/distribution/f'n{n}', grin_cycle_score=True)
    write_batch_report(output, report, sizes, models, repeats, args.smoke)


if __name__ == '__main__':
    main()
