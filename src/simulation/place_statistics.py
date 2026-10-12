"""Five-column deviations by true place and standalone scientific plots."""
import csv

import numpy as np

RANKING_NAMES = {'grin_tour': 'grin-elo', 'grin_cycle_score': 'grin-cycle'}
PLACE_FIELDS = ('place', 'variance', 'mean_abs_deviation', 'max_down', 'max_up')


def place_errors(ratings, places):
    """Signed distance at every true place; rating ties use participant ID."""
    order = sorted(range(len(ratings)), key=lambda who: (-ratings[who], who))
    return [places[who] - rank for rank, who in enumerate(order, 1)]


def summarize_places(samples, participants):
    if not samples:
        return [{'place': rank, **dict.fromkeys(PLACE_FIELDS[1:])}
                for rank in range(1, participants+1)]
    errors = np.asarray(samples)
    return [{'place': index+1, 'variance': float(np.var(errors[:, index], ddof=0)),
             'mean_abs_deviation': float(np.mean(np.abs(errors[:, index]))),
             'max_down': max(0, int(np.max(errors[:, index]))),
             'max_up': max(0, int(-np.min(errors[:, index])))}
            for index in range(participants)]


def write_place_artifacts(output, model, method, statistics, config):
    # Import lazily so pure simulation and independent verification need no GUI.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stem = f'place-deviations-{model}-{method}'
    rows = statistics['places']
    with (output/f'{stem}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, PLACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    titles = ('Дисперсия отклонений', 'Среднее абсолютное отклонение',
              'Максимальное отклонение вниз', 'Максимальное отклонение вверх')
    for axis, field, title in zip(axes.flat, PLACE_FIELDS[1:], titles, strict=True):
        if statistics['samples']:
            axis.bar([row['place'] for row in rows], [row[field] for row in rows], color='#2878a5')
        else:
            axis.text(.5, .5, 'Нет полных расстановок', ha='center', transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel('Место по исходному рейтингу')
        axis.set_ylabel('Места²' if field == 'variance' else 'Места')
        axis.set_xticks(range(1, config.participants+1), minor=False)
        axis.tick_params(axis='x', labelsize=7)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle(f'{config.distribution}; N={config.participants}; {model}; {method}\n'
                    f'Полных: {statistics["samples"]}/{config.repeats}; seed={config.seed}')
    try:
        figure.savefig(output/f'{stem}.png', dpi=150)
        figure.savefig(output/f'{stem}.svg')
    finally:
        plt.close(figure)
    lines = [f'### {model}: {method}', '',
             f'Полных расстановок: {statistics["samples"]}; пропущено: {statistics["skipped"]}.', '',
             '| Место | Дисперсия δ | Среднее |δ| | Максимум вниз | Максимум вверх |'.replace('|δ|', '\\|δ\\|'),
             '|---:|---:|---:|---:|---:|']
    for row in rows:
        values = ['—' if row[field] is None else f'{row[field]:.6f}' for field in PLACE_FIELDS[1:]]
        lines.append(f'| {row["place"]} | ' + ' | '.join(values) + ' |')
    lines.extend(['', f'[CSV]({stem}.csv) · [SVG]({stem}.svg)', '', f'![Отклонения по местам]({stem}.png)', ''])
    return '\n'.join(lines)
