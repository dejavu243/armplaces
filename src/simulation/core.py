"""Генеральная последовательность double elimination и три модели исхода.

Переходы берутся из DE_OLD_fix для N=2–32; GS содержит ровно 4N ячеек. Участник выбывает после двух
поражений; технические проходы не расходуют силы. strong-win сравнивает рейтинги,
elo выбирает исход случайно, elo-stamina уменьшает силу после реальных боёв.
Рейтинги обновляются только после турнира; журнал независимо проверяется.
Формулы, порядок раундов и инварианты подробно описаны в README.md.
"""
from dataclasses import dataclass
from math import exp, isfinite, log
from functools import lru_cache

import numpy as np

from armplaces.topological_sort import rank_tournament
from armplaces.cycle_resolution import (DEFAULT_WEIGHTS, normalize_weights,
                                        rank_tournament_cycle_score)
from armplaces.tables.DE_OLD_Winner_fix import DE_OLD_winner_fix
from armplaces.tables.DE_OLD_Loser_fix import DE_OLD_loser_fix

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
        if isinstance(self.participants, bool) or not isinstance(self.participants, int) or not 1 <= self.participants <= 32:
            raise ValueError('participants must be an integer in 1..32; DE_OLD_fix has no tables for N > 32')
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


@lru_cache(maxsize=32)
def build_bracket(n: int) -> Bracket:
    """Read DE_OLD_fix column N-2; convert one-based addresses to Python indices."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 32:
        raise ValueError('participants must be an integer in 1..32; DE_OLD_fix has no tables for N > 32')
    if n == 1:
        return Bracket(1, ())
    routes = [(DE_OLD_winner_fix[i][n-2]-1, DE_OLD_loser_fix[i][n-2]-1)
              for i in range(2*n-1)]
    # Structural loss counts identify upper/lower matches without choosing outcomes.
    states = [None]*(4*n)
    states[:n] = [(0, 0)]*n
    stages = []
    for i, (winner_to, loser_to) in enumerate(routes):
        if any(destination <= 2*i+1 for destination in (winner_to, loser_to)):
            raise RuntimeError('Invalid DE_OLD_fix forward address')
        if i >= 2*n-3:
            stages.append('final' if i == 2*n-3 else 'reset')
            continue
        a, b = states[2*i:2*i+2]
        if a is None or b is None or a[0] != b[0] or a[0] not in (0, 1):
            raise RuntimeError('Invalid DE_OLD_fix pair')
        level = max(a[1], b[1])+1
        stages.append(f'W{level}' if a[0] == 0 else 'L')
        for destination, state in ((winner_to, (a[0], level)), (loser_to, (a[0]+1, 0))):
            if destination >= 4*n:
                if state[0] != 2:
                    raise RuntimeError('Only eliminated participants may leave GS')
            else:
                if states[destination] is not None:
                    raise RuntimeError('Conflicting DE_OLD_fix address')
                states[destination] = state
    # Equal remaining distances to the final define shared lower-bracket places.
    distance = {}
    for i in reversed(range(2*n-3)):
        if stages[i] == 'L':
            following = routes[i][0]//2
            distance[i] = 1 + distance.get(following, 0)
    longest = max(distance.values(), default=0)
    for i, depth in distance.items():
        stages[i] = f'L{longest-depth+1}'
    return Bracket(n, tuple(Match(stage, winner, loser)
                           for stage, (winner, loser) in zip(stages, routes, strict=True)))


def elo_probability(a: float, b: float, dr: float = 200.) -> float:
    if not all(isfinite(x) for x in (a, b, dr)) or dr <= 0:
        raise ValueError('Elo inputs must be finite and dr positive')
    # Subtract first: equal huge ratings with tiny dr must still give exactly 1/2.
    # An infinite signed difference/exponent safely saturates the logistic function.
    z = ((b - a) / dr) * log(10.)
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
             repeat: int = 0, grin_tour: bool = False, rng=None, *,
             grin_cycle_score: bool = False, cycle_weights=DEFAULT_WEIGHTS) -> dict:
    config.validate()
    normalized_weights = normalize_weights(cycle_weights)
    n = config.participants
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    if len(ratings) != n or any(not isfinite(r) or r <= 0 for r in ratings):
        raise ValueError('Expected one finite positive initial rating per participant')
    if sorted(draw) != list(range(n)):
        raise ValueError('draw must be a permutation of participant IDs')
    bracket = build_bracket(n)
    sequence = [None] * (4*n)
    eliminated_slots = {}
    sequence[:n] = draw
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
                if destination >= len(sequence):
                    if who is None or losses[who] != 2 or destination+1 in eliminated_slots:
                        raise RuntimeError('Invalid eliminated-participant address')
                    eliminated_slots[destination+1] = who
                    continue
                if destination <= 2*index+1 or sequence[destination] is not None:
                    raise RuntimeError('Conflicting general-sequence transition')
                sequence[destination] = who
    survivors = [i for i in range(n) if losses[i] < 2]
    if len(survivors) != 1:
        raise RuntimeError('Tournament did not produce exactly one champion')
    champion = survivors[0]
    places = {champion: 1}
    cursor = 2
    def stage_order(stage):
        return (1, 0) if stage == 'final' else ((2, 0) if stage == 'reset' else (0, int(stage[1:])))

    for stage in sorted(eliminated, key=stage_order, reverse=True):
        group = eliminated[stage]
        for participant in group:
            places[participant] = cursor
        cursor += len(group)
    ranking = (rank_tournament({i: str(i) for i in draw}, real_pairs,
                               ratings={str(i): rating for i, rating in enumerate(ratings)}) if grin_tour else
               {'status': 'disabled', 'reason': '', 'places': {}})
    cycle_ranking = (rank_tournament_cycle_score({i: str(i) for i in draw}, real_pairs,
                                                  weights=normalized_weights)
                     if grin_cycle_score else {'status': 'disabled', 'reason': '', 'places': {}})
    result = {'model': model, 'repeat': repeat, 'ratings': list(ratings), 'draw': list(draw),
              'new_ratings': [r+d for r, d in zip(ratings, deltas)],
              'wins': wins, 'losses': losses, 'played': played, 'champion': champion,
              'places': places, 'grin_tour': ranking, 'grin_cycle_score': cycle_ranking,
              'bouts': bouts, 'sequence': sequence,
              'reset': reset_needed, 'eliminated_slots': eliminated_slots, 'bracket': 'DE_OLD_fix'}
    validate_result(result)
    return result


def validate_result(result: dict):
    """Replay counters independently of simulation and check complete output."""
    n = len(result['ratings'])
    validate_sequence(result)
    wins, losses = [0]*n, [0]*n
    real = 0
    for bout in result['bouts']:
        for side in ('a', 'b'):
            who = bout[side]
            if who is not None:
                if not isinstance(who, int) or not 0 <= who < n or losses[who] >= 2:
                    raise RuntimeError('Invalid participant in journal')
                if bout['previous_bouts_'+side] != wins[who]+losses[who]:
                    raise RuntimeError('Incorrect prior bout count')
        if bout['technical']:
            if bout['a'] is not None and bout['b'] is not None:
                raise RuntimeError('A technical pass contains two participants')
            expected_winner = bout['a'] if bout['a'] is not None else bout['b']
            if bout['winner'] != expected_winner or bout['loser'] is not None:
                raise RuntimeError('Invalid technical pass result')
            continue
        a, b, winner, loser = (bout[key] for key in ('a', 'b', 'winner', 'loser'))
        if a == b or {a, b} != {winner, loser} or losses[a] >= 2 or losses[b] >= 2:
            raise RuntimeError('Invalid bout in saved journal')
        if any(not isfinite(bout[key]) or bout[key] < 0 for key in ('effective_a', 'effective_b')):
            raise RuntimeError('Invalid effective rating')
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
    elif ranking['status'] == 'partial':
        expected_podium = {str(i): place for i, place in result['places'].items() if place <= 3}
        if ranking['places'] != expected_podium or not ranking['reason']:
            raise RuntimeError('Partial GrinTour must preserve the decisive-bout podium')
    elif ranking['status'] not in ('undefined', 'disabled') or (ranking['status'] == 'undefined' and not ranking['reason']):
        raise RuntimeError('Invalid GrinTour status')
    if ranking['status'] == 'ok':
        expected_podium = {str(i): place for i, place in result['places'].items() if place <= 3}
        if any(ranking['places'][name] != place for name, place in expected_podium.items()):
            raise RuntimeError('GrinTour changed a decisive-bout podium place')
    cycle_ranking = result['grin_cycle_score']
    if cycle_ranking['status'] == 'ok':
        if (set(cycle_ranking['places']) != set(map(str, range(n)))
                or sorted(cycle_ranking['places'].values()) != list(range(1, n+1))):
            raise RuntimeError('Incomplete cycle-score ranking')
        expected_podium = {str(i): place for i, place in result['places'].items() if place <= 3}
        if any(cycle_ranking['places'][name] != place for name, place in expected_podium.items()):
            raise RuntimeError('Cycle-score ranking changed the decisive-bout podium')
    elif cycle_ranking['status'] == 'partial':
        expected_podium = {str(i): place for i, place in result['places'].items() if place <= 3}
        if cycle_ranking['places'] != expected_podium or not cycle_ranking['reason']:
            raise RuntimeError('Invalid partial cycle-score result')
    elif cycle_ranking['status'] == 'undefined':
        if cycle_ranking['places'] or not cycle_ranking['reason']:
            raise RuntimeError('Invalid undefined cycle-score result')
    elif cycle_ranking['status'] != 'disabled' or cycle_ranking['places']:
        raise RuntimeError('Invalid cycle-score status')


def validate_sequence(result: dict):
    """Replay raw table addresses independently, including eliminated storage slots."""
    n = len(result['ratings'])
    if not 1 <= n <= 32 or result.get('bracket') != 'DE_OLD_fix':
        raise RuntimeError('Expected DE_OLD_fix tournament for N=1..32')
    draw = result['draw']
    if sorted(draw) != list(range(n)):
        raise RuntimeError('Invalid draw')
    expected = [None]*(4*n)
    expected[:n] = draw
    external = {}
    count = 0 if n == 1 else 2*n-2+int(result['reset'])
    if len(result['bouts']) != count:
        raise RuntimeError('Incorrect number of table matches')
    stages = build_bracket(n).matches
    for i, bout in enumerate(result['bouts']):
        if (bout['match'] != i or bout['stage'] != stages[i].stage
                or bout['technical'] or expected[2*i:2*i+2] != [bout['a'], bout['b']]):
            raise RuntimeError('Bout does not match the DE_OLD_fix general sequence')
        for who, table in ((bout['winner'], DE_OLD_winner_fix), (bout['loser'], DE_OLD_loser_fix)):
            address = table[i][n-2]
            if address <= 2*i+2:
                raise RuntimeError('Invalid forward table address')
            if address > 4*n:
                if address in external:
                    raise RuntimeError('Conflicting elimination address')
                external[address] = who
            else:
                if expected[address-1] is not None:
                    raise RuntimeError('Conflicting GS address')
                expected[address-1] = who
    if expected != result['sequence']:
        raise RuntimeError('Saved GS does not match table transitions')
    if external != {int(k): v for k, v in result['eliminated_slots'].items()}:
        raise RuntimeError('Saved elimination slots do not match table transitions')
