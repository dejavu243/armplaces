"""CLI независимых экспериментов: python -m simulation.

Флаги --strong-win, --elo, --elo-stamina, --elo-tire добавляют модели к серии.
--grin-elo и --grin-cycle добавляют два варианта расстановки мест.
Параметры проверяются до запуска; ошибки дают ненулевой код завершения.
Таблица флагов и примеры команд приведены в README.md.
"""
import argparse
import logging
from pathlib import Path

from .core import Config, MODELS, DISTRIBUTIONS
from .experiments import run_experiments
from armplaces.cycle_resolution import DEFAULT_WEIGHTS, normalize_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strong-win', action='store_true', help='Higher initial rating wins; ties are random')
    parser.add_argument('--elo', action='store_true', help='Sample outcomes using Elo probabilities')
    parser.add_argument('--elo-stamina', action='store_true', help='Sample Elo using fatigue-adjusted ratings')
    parser.add_argument('--elo-tire', action='store_true', help='Sample Elo using the article cumulative-fatigue curve')
    parser.add_argument('--grin-elo', '--grin-tour', dest='grin_tour', action='store_true',
                        help='GrinTour with initial opponent ratings as fallback')
    parser.add_argument('--grin-cycle', '--grin-cycle-score', dest='grin_cycle_score', action='store_true',
                        help='Resolve residual GrinTour cycles using three weighted criteria')
    parser.add_argument('--cycle-weights', nargs=3, type=float, metavar=('CHRONOLOGY','LOSSES','PLACE'),
                        help='Three nonnegative cycle scores; requires --grin-cycle')
    defaults = Config()
    for name in ('participants', 'repeats', 'seed', 'tire_bouts'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=getattr(defaults, name),
                            help=f'Default: {getattr(defaults, name)}')
    for name in ('alpha', 'beta', 'scale', 'dr', 'k', 'fatigue_rate',
                 'weibull_shape', 'weibull_scale', 'weibull_location',
                 'rating_min', 'rating_max', 'tire_floor', 'tire_reference', 'tire_c'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=getattr(defaults, name),
                            help=f'Default: {getattr(defaults, name)}')
    parser.add_argument('--distribution', choices=DISTRIBUTIONS, default=defaults.distribution,
                        help='Rating distribution; uniform defaults to [800, 2400)')
    parser.add_argument('--output', type=Path, default=Path('results/comparison'), help='Output directory (overwrites previous run files)')
    args = parser.parse_args(argv)
    models = [model for model in MODELS if getattr(args, model.replace('-', '_'))]
    if not models:
        parser.error('Select at least one experiment: --strong-win, --elo, --elo-stamina, --elo-tire')
    if args.cycle_weights is not None and not args.grin_cycle_score:
        parser.error('--cycle-weights requires --grin-cycle (alias --grin-cycle-score)')
    try:
        normalize_weights(args.cycle_weights or DEFAULT_WEIGHTS)
    except ValueError as exc:
        parser.error(str(exc))
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        run_experiments(config, models, args.grin_tour, args.output,
                        grin_cycle_score=args.grin_cycle_score,
                        cycle_weights=args.cycle_weights or DEFAULT_WEIGHTS)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Results and report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
