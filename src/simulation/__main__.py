"""CLI независимых экспериментов: python -m simulation.

Флаги --strong-win, --elo, --elo-stamina добавляют модели к серии; --grin-tour
добавляет эвристическую расстановку мест к каждой выбранной модели.
Параметры проверяются до запуска; ошибки дают ненулевой код завершения.
Таблица флагов и примеры команд приведены в README.md.
"""
import argparse
import logging
from pathlib import Path

from .core import Config, MODELS
from .experiments import run_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strong-win', action='store_true', help='Higher initial rating wins; ties are random')
    parser.add_argument('--elo', action='store_true', help='Sample outcomes using Elo probabilities')
    parser.add_argument('--elo-stamina', action='store_true', help='Sample Elo using fatigue-adjusted ratings')
    parser.add_argument('--grin-tour', action='store_true', help='Also rank each tournament with GrinTour')
    defaults = Config()
    for name in ('participants', 'repeats', 'seed'):
        parser.add_argument('--'+name, type=int, default=getattr(defaults, name),
                            help=f'Default: {getattr(defaults, name)}')
    for name in ('alpha', 'beta', 'scale', 'dr', 'k', 'fatigue_rate'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=getattr(defaults, name),
                            help=f'Default: {getattr(defaults, name)}')
    parser.add_argument('--output', type=Path, default=Path('results/comparison'), help='Output directory (overwrites previous run files)')
    args = parser.parse_args(argv)
    models = [model for model in MODELS if getattr(args, model.replace('-', '_'))]
    if not models:
        parser.error('Select at least one experiment: --strong-win, --elo, --elo-stamina')
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        run_experiments(config, models, args.grin_tour, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Results and report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
