"""Пошаговый Markdown-протокол GS: python -m simulation.trace.

Выводит рейтинги, жеребьёвку и состояние GS после каждого табличного перехода.
Для чтения человеком ID отображаются как номера 1…N. Полный пайплайн — README.
"""
import argparse

from .core import Config, MODELS, build_bracket, generate_field, simulate


def trace_markdown(config: Config, model: str = 'strong-win', repeat: int = 0) -> str:
    ratings, draw = generate_field(config, repeat)
    result = simulate(config, model, ratings, draw, repeat=repeat, grin_tour=True)
    bracket = build_bracket(config.participants)
    gs = [None]*(4*config.participants)
    gs[:config.participants] = draw

    def cells():
        return '[' + ', '.join('·' if who is None else str(who+1) for who in gs) + ']'

    lines = ['# Пошаговый пример GS', '',
             f'N={config.participants}, seed={config.seed}, repeat={repeat}, модель `{model}`.', '',
             'Номера спортсменов и ячеек здесь начинаются с 1; `·` — пустая ячейка.',
             'В Python и CSV ID начинаются с 0: показанный номер = ID + 1.', '',
             '## 1. Рейтинги BetaPrime', '', '| Спортсмен | Исходный рейтинг |', '|---:|---:|']
    lines.extend(f'| {i+1} | {rating:.6f} |' for i, rating in enumerate(ratings))
    lines += ['', '## 2. Жеребьёвка', '', ', '.join(str(who+1) for who in draw), '',
              '## 3. GS до первого боя', '', f'Длина: {len(gs)} = 4N.', '', f'```text\n{cells()}\n```', '',
              '## 4. Проход по парам', '']
    for i, (bout, route) in enumerate(zip(result['bouts'], bracket.matches)):
        lines += [f'### Бой {i+1} ({bout["stage"]})', '',
                  f'Ячейки {2*i+1} и {2*i+2}: № {bout["a"]+1} — № {bout["b"]+1}.',
                  f'P(первый победит) = {bout["p_a"]:.8f}. '
                  f'Победил № {bout["winner"]+1}, проиграл № {bout["loser"]+1}.', '']
        for who, address, label in ((bout['winner'], route.winner_to+1, 'Победитель'),
                                    (bout['loser'], route.loser_to+1, 'Проигравший')):
            if address > len(gs):
                lines.append(f'{label} № {who+1} → адрес {address} вне GS: второе поражение, учёт выбывших.')
            else:
                gs[address-1] = who
                lines.append(f'{label} № {who+1} → GS[{address}].')
        lines += ['', f'```text\n{cells()}\n```', '']
    if gs != result['sequence']:
        raise RuntimeError('Trace does not match saved GS')
    lines += ['## 5. Завершение и места', '',
              'Суперфинал проведён.' if result['reset'] else 'Суперфинал не нужен.', '',
              '| Спортсмен | Место по сетке | Место GrinTour |', '|---:|---:|---:|']
    lines.extend(f'| {who+1} | {place} | {result["grin_tour"]["places"][str(who)]} |'
                 for who, place in sorted(result['places'].items(), key=lambda pair: pair[1]))
    lines += ['', 'После завершения турнира рассчитываются новые рейтинги; '
              'они не изменяют исходы уже проведённых боёв.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participants', type=int, default=9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--model', choices=MODELS, default='strong-win')
    args = parser.parse_args()
    if args.repeat < 0:
        parser.error('repeat must be nonnegative')
    try:
        print(trace_markdown(Config(participants=args.participants, seed=args.seed), args.model, args.repeat))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
