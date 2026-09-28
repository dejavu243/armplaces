"""Призёры по хронологии реальных боёв double elimination.

Проигравший получает поражение; второе означает выбывание. Бой, оставивший
двух активных участников, определяет третье место. Финал с вторым поражением
проигравшего или суперфинал определяют места 1–2. Технические проходы и встречи
двух уже выбывших участников (например, за 5–6) не влияют на призёров.
Неполный или противоречивый журнал не используется для фиксации призёров.
"""


def infer_podium(names: dict, pairs: list) -> dict:
    """Return proven places or an empty undefined result; preserve the raw journal."""
    participants = list(names.values())
    if not participants or len(set(participants)) != len(participants) or '' in participants:
        raise ValueError('Expected nonempty, unique participant names')
    pairs = [tuple(pair) for pair in pairs]
    for pair in pairs:
        if len(pair) != 2 or pair[1] not in participants or pair[0] == pair[1]:
            raise ValueError(f'Invalid bout: {pair}')
        if pair[0] != '' and pair[0] not in participants:
            raise ValueError(f'Unknown participant: {pair[0]}')

    def undefined(reason):
        return {'status': 'undefined', 'reason': reason, 'places': {}, 'final_kind': None}

    if len(participants) == 1:
        return {'status': 'ok', 'reason': '', 'places': {participants[0]: 1}, 'final_kind': 'singleton'}
    losses = dict.fromkeys(participants, 0)
    active = len(participants)
    bronze = None
    finalist_pair = None
    places = {}
    final_kind = None
    for loser, winner in pairs:
        if loser == '':
            continue
        # Classification bouts may be interleaved with the final or appended to it.
        if losses[loser] == losses[winner] == 2:
            continue
        if losses[loser] >= 2 or losses[winner] >= 2 or places:
            return undefined('A main bout involves an already eliminated participant')
        before = active
        previous_losses = (losses[loser], losses[winner])
        if before == 2:
            if previous_losses == (0, 0):
                if len(participants) != 2:
                    return undefined('Two undefeated finalists after earlier eliminations')
            elif sorted(previous_losses) == [0, 1]:
                if finalist_pair is not None:
                    return undefined('Final repeated without a valid reset')
                finalist_pair = {loser, winner}
                final_kind = 'final'
            elif previous_losses == (1, 1):
                if finalist_pair != {loser, winner}:
                    return undefined('Superfinal without the preceding final')
                final_kind = 'superfinal'
            else:
                return undefined('Invalid losses before final')
        losses[loser] += 1
        if losses[loser] == 2:
            active -= 1
            if before == 3:
                bronze = loser
            if active == 1:
                places = {winner: 1, loser: 2}
                if len(participants) > 2:
                    if bronze is None:
                        return undefined('Missing semifinal elimination')
                    places[bronze] = 3
    if not places:
        return undefined('Main tournament is incomplete')
    return {'status': 'ok', 'reason': '', 'places': places, 'final_kind': final_kind}
