"""Генеральная последовательность double elimination и три модели исхода.

Сетка заранее компилируется в адреса переходов. Участник выбывает после двух
поражений; технические проходы не расходуют силы. strong-win сравнивает рейтинги,
elo выбирает исход случайно, elo-stamina уменьшает силу после реальных боёв.
Рейтинги обновляются только после турнира; журнал независимо проверяется.
Формулы, порядок раундов и инварианты подробно описаны в README.md.
"""
from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np

from armplaces.topological_sort import rank_tournament

MODELS = ('strong-win', 'elo', 'elo-stamina')


@dataclass(frozen=True)
class Config:
    participants: int = 32
    repeats: int = 1000
    seed: int = 42
    alpha: float = 2.0
    beta: float = 3.0
    scale: float = 1000.0
    dr: float = 200.0
    k: float = 160.0
    fatigue_rate: float = .05

    def validate(self):
        if isinstance(self.participants, bool) or not isinstance(self.participants, int) or not 1 <= self.participants <= 128:
            raise ValueError('participants must be an integer in 1..128')
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError('repeats must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if any(not isfinite(x) or x <= 0 for x in (self.alpha, self.beta, self.scale, self.dr)):
            raise ValueError('alpha, beta, scale and dr must be finite and positive')
        if not isfinite(self.k) or self.k < 0:
            raise ValueError('k must be finite and nonnegative')
        if not isfinite(self.fatigue_rate) or not 0 <= self.fatigue_rate < 1:
            raise ValueError('fatigue-rate must be finite in [0, 1)')


@dataclass(frozen=True)
class Match:
    stage: str
    winner_to: int | None
    loser_to: int | None


@dataclass(frozen=True)
class Bracket:
    size: int
    matches: tuple[Match, ...]


def build_bracket(n: int) -> Bracket:
    """Compile output references into zero-based addresses of the general sequence."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 128:
        raise ValueError('participants must be an integer in 1..128')
    size = 1 << (n - 1).bit_length()
    if n == 1:
        return Bracket(1, ())
    stages, inputs = [], []

    def add(stage, a, b):
        index = len(stages)
        stages.append(stage)
        inputs.append((a, b))
        return (index, 0), (index, 1)

    upper, dropped = [], []
    for i in range(size // 2):
        winner, loser = add('W1', None, None)
        upper.append(winner)
        dropped.append(loser)
    if size == 2:
        lower = dropped
    else:
        lower = [add('L1', dropped[i], dropped[i+1])[0] for i in range(0, len(dropped), 2)]
    lower_round = 1
    for upper_round in range(2, size.bit_length()):
        outcomes = [add(f'W{upper_round}', upper[i], upper[i+1])
                    for i in range(0, len(upper), 2)]
        upper = [outcome[0] for outcome in outcomes]
        dropped = [outcome[1] for outcome in outcomes]
        lower_round += 1
        lower = [add(f'L{lower_round}', survivor, newcomer)[0]
                 for survivor, newcomer in zip(lower, reversed(dropped), strict=True)]
        if len(upper) > 1:
            lower_round += 1
            lower = [add(f'L{lower_round}', lower[i], lower[i+1])[0]
                     for i in range(0, len(lower), 2)]
    winner, loser = add('final', upper[0], lower[0])
    add('reset', winner, loser)
    routes = [[None, None] for _ in stages]
    for index, pair in enumerate(inputs):
        for side, ref in enumerate(pair):
            if ref is not None:
                origin, outcome = ref
                if origin >= index or routes[origin][outcome] is not None:
                    raise RuntimeError('Invalid bracket transition')
                routes[origin][outcome] = 2 * index + side
    return Bracket(size, tuple(Match(stage, *route) for stage, route in zip(stages, routes)))


def elo_probability(a: float, b: float, dr: float = 200.) -> float:
    if not all(isfinite(x) for x in (a, b, dr)) or dr <= 0:
        raise ValueError('Elo inputs must be finite and dr positive')
    # Scale before subtracting to avoid overflow for large ratings.
    z = (b / dr - a / dr) * log(10.)
    if z >= 0:
        e = exp(-z)
        return e / (1 + e)
    e = exp(z)
    return 1 / (1 + e)


def generate_field(config: Config, repeat: int):
    config.validate()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, repeat, 0]))
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        ratings = config.scale * (rng.gamma(config.alpha, size=config.participants) /
                                  rng.gamma(config.beta, size=config.participants))
    if not np.all(np.isfinite(ratings)) or np.any(ratings <= 0):
        raise ValueError('Distribution produced nonfinite/zero ratings; choose less extreme parameters')
    draw_rng = np.random.default_rng(np.random.SeedSequence([config.seed, repeat, 1]))
    draw = [int(x) for x in draw_rng.permutation(config.participants)]
    return [float(x) for x in ratings], draw


def simulate(config: Config, model: str, ratings: list[float], draw: list[int],
             repeat: int = 0, grin_tour: bool = False, rng=None) -> dict:
    config.validate()
    n = config.participants
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    if len(ratings) != n or any(not isfinite(r) or r <= 0 for r in ratings):
        raise ValueError('Expected one finite positive initial rating per participant')
    if sorted(draw) != list(range(n)):
        raise ValueError('draw must be a permutation of participant IDs')
    bracket = build_bracket(n)
    sequence = [None] * (2 * len(bracket.matches))
    if n > 1:
        half = bracket.size // 2
        for i in range(half):
            sequence[2*i] = draw[i]
            if i + half < n:
                sequence[2*i+1] = draw[i + half]
    if rng is None:
        # Common random numbers for the two Elo variants; separate generator per run.
        stream = 2 if model in ('elo', 'elo-stamina') else 3
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, repeat, stream]))
    losses, wins, played = [0]*n, [0]*n, [0]*n
    deltas, bouts, eliminated = [0.]*n, [], {}
    real_pairs = []
    reset_needed = False
    for index, match in enumerate(bracket.matches):
        if match.stage == 'reset' and not reset_needed:
            break
        a, b = sequence[2*index:2*index+2]
        technical = a is None or b is None
        entry = {'match': index, 'stage': match.stage, 'a': a, 'b': b,
                 'technical': technical, 'p_a': None, 'p_start_a': None,
                 'effective_a': None, 'effective_b': None,
                 'previous_bouts_a': played[a] if a is not None else 0,
                 'previous_bouts_b': played[b] if b is not None else 0}
        if technical:
            winner, loser = (a if a is not None else b), None
        else:
            if a == b or losses[a] >= 2 or losses[b] >= 2:
                raise RuntimeError('Invalid or eliminated participant in a bout')
            effective_a, effective_b = ratings[a], ratings[b]
            if model == 'elo-stamina':
                effective_a *= (1-config.fatigue_rate)**played[a]
                effective_b *= (1-config.fatigue_rate)**played[b]
            baseline = elo_probability(ratings[a], ratings[b], config.dr)
            probability = (float(ratings[a] > ratings[b]) if ratings[a] != ratings[b] else .5) if model == 'strong-win' else elo_probability(effective_a, effective_b, config.dr)
            winner, loser = (a, b) if rng.random() < probability else (b, a)
            change = config.k * (float(winner == a) - baseline)
            deltas[a] += change
            deltas[b] -= change
            wins[winner] += 1
            losses[loser] += 1
            played[a] += 1
            played[b] += 1
            real_pairs.append((str(loser), str(winner)))
            if losses[loser] == 2:
                eliminated.setdefault(match.stage, []).append(loser)
            if match.stage == 'final':
                reset_needed = losses[loser] == 1
            entry.update(p_a=probability, p_start_a=baseline,
                         effective_a=effective_a, effective_b=effective_b)
        entry.update(winner=winner, loser=loser)
        bouts.append(entry)
        for who, destination in ((winner, match.winner_to), (loser, match.loser_to)):
            if destination is not None:
                if destination <= 2*index+1 or sequence[destination] is not None:
                    raise RuntimeError('Conflicting general-sequence transition')
                sequence[destination] = who
    survivors = [i for i in range(n) if losses[i] < 2]
    if len(survivors) != 1:
        raise RuntimeError('Tournament did not produce exactly one champion')
    champion = survivors[0]
    places = {champion: 1}
    cursor = 2
    for group in reversed(list(eliminated.values())):
        for participant in group:
            places[participant] = cursor
        cursor += len(group)
    ranking = (rank_tournament({i: str(i) for i in draw}, real_pairs) if grin_tour else
               {'status': 'disabled', 'reason': '', 'places': {}})
    result = {'model': model, 'repeat': repeat, 'ratings': list(ratings), 'draw': list(draw),
              'new_ratings': [r+d for r, d in zip(ratings, deltas)],
              'wins': wins, 'losses': losses, 'played': played, 'champion': champion,
              'places': places, 'grin_tour': ranking, 'bouts': bouts, 'sequence': sequence,
              'reset': reset_needed}
    validate_result(result)
    return result


def validate_result(result: dict):
    """Replay counters independently of simulation and check complete output."""
    n = len(result['ratings'])
    wins, losses = [0]*n, [0]*n
    real = 0
    for bout in result['bouts']:
        if bout['technical']:
            if bout['a'] is not None and bout['b'] is not None:
                raise RuntimeError('A technical pass contains two participants')
            continue
        a, b, winner, loser = (bout[key] for key in ('a', 'b', 'winner', 'loser'))
        if a == b or {a, b} != {winner, loser} or losses[a] >= 2 or losses[b] >= 2:
            raise RuntimeError('Invalid bout in saved journal')
        if any(not isfinite(bout[key]) or not 0 <= bout[key] <= 1 for key in ('p_a', 'p_start_a')):
            raise RuntimeError('Invalid bout probability')
        wins[winner] += 1
        losses[loser] += 1
        real += 1
    champion = result['champion']
    if losses != result['losses'] or wins != result['wins'] or result['played'] != [a+b for a,b in zip(wins, losses)]:
        raise RuntimeError('Journal and counters disagree')
    if any(loss != 2 for i, loss in enumerate(losses) if i != champion) or losses[champion] >= 2:
        raise RuntimeError('Invalid elimination counters')
    if real != (0 if n == 1 else 2*n-2+int(result['reset'])):
        raise RuntimeError('Unexpected bout count')
    places = result['places']
    if set(places) != set(range(n)) or places[champion] != 1:
        raise RuntimeError('Incomplete placements')
    cursor = 1
    for place in sorted(set(places.values())):
        if place != cursor:
            raise RuntimeError('Invalid shared-place numbering')
        cursor += list(places.values()).count(place)
    if not all(isfinite(value) for value in result['new_ratings']):
        raise RuntimeError('Nonfinite updated rating')
    ranking = result['grin_tour']
    if ranking['status'] == 'ok':
        if set(ranking['places']) != set(map(str, range(n))) or sorted(ranking['places'].values()) != list(range(1,n+1)):
            raise RuntimeError('Incomplete GrinTour result')
    elif ranking['status'] not in ('undefined', 'disabled') or (ranking['status'] == 'undefined' and not ranking['reason']):
        raise RuntimeError('Invalid GrinTour status')
