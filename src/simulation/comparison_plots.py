"""Comparable scientific plots of individual place errors for all four methods."""

from pathlib import Path

from .place_statistics import ALL_RANKING_NAMES, PLACE_FIELDS

METHOD_LABELS = {'far-tour': 'FAR-tour', 'zhivora-tour': 'Zhivora-Tour',
                 'grin-cycle': 'Grin-Cycle', 'grin-elo': 'Grin-Elo'}
METHOD_COLORS = {'far-tour': '#216eac', 'zhivora-tour': '#d56623',
                 'grin-cycle': '#22845c', 'grin-elo': '#a84c83'}


def save(figure, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        path = destination.with_suffix('.'+extension)
        figure.savefig(path, dpi=150)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def comparison_panels(summary, config, destination, *, zhivora_fixed_top=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    titles = ('Дисперсия отклонений', 'Среднее абсолютное отклонение',
              'Максимальное отклонение вниз', 'Максимальное отклонение вверх')
    for axis, field, title in zip(axes.flat, PLACE_FIELDS[1:], titles, strict=True):
        for method in ALL_RANKING_NAMES.values():
            if method not in summary['place_deviations']:
                continue
            statistics = summary['place_deviations'][method]
            if not statistics['samples']:
                continue
            rows = statistics['places']
            axis.plot([row['place'] for row in rows], [row[field] for row in rows],
                      marker='o', markersize=3, linewidth=1.5,
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set_title(title)
        axis.set_xlabel('Истинное место по исходному рейтингу')
        axis.set_ylabel('Места²' if field == 'variance' else 'Места')
        axis.set_xticks(range(1, config.participants+1))
        axis.tick_params(axis='x', labelsize=7)
        axis.grid(alpha=.2)
    axes[0, 1].legend(fontsize=9)
    figure.suptitle(f'Elo-tire · beta-prime · N={config.participants} · {config.repeats} повторов · seed={config.seed}\n'
                    f'Zhivora: фиксированы 1–{zhivora_fixed_top}; все методы используют один журнал турнира')
    save(figure, destination)
    plt.close(figure)


def mae_overview(categories, destination, *, zhivora_fixed_top=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(categories), figsize=(5*len(categories), 4.5),
                                constrained_layout=True, squeeze=False)
    for axis, (n, summary) in zip(axes.flat, sorted(categories.items()), strict=True):
        for method in ALL_RANKING_NAMES.values():
            statistics = summary['place_deviations'][method]
            if statistics['samples']:
                axis.plot([row['place'] for row in statistics['places']],
                          [row['mean_abs_deviation'] for row in statistics['places']],
                          marker='o', markersize=3, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set_title(f'N={n}')
        axis.set_xlabel('Истинное место')
        axis.set_ylabel('Среднее абсолютное отклонение, мест')
        axis.set_xticks(range(1, n+1), minor=False)
        axis.tick_params(axis='x', labelsize=7)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f'Точность индивидуальных мест · Elo-tire · beta-prime · Zhivora фиксирует 1–{zhivora_fixed_top}')
    save(figure, destination)
    plt.close(figure)
