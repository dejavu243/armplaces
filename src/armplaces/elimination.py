"""Complete double-elimination history and individual FAR-tour places."""

from .podium import infer_podium


def elimination_history(names: dict, pairs: list) -> dict:
    """Replay real bouts; second losses determine the elimination chronology.

    Technical passes and classification bouts between two eliminated athletes
    are ignored, consistently with podium inference. Incomplete or contradictory
    main histories cannot produce a complete ranking.
    """
    pairs = [tuple(pair) for pair in pairs]
    podium = infer_podium(names, pairs)
    if podium['status'] != 'ok':
        return {'status': 'undefined', 'reason': podium['reason'], 'places': {}}
    participants = list(names.values())
    wins = dict.fromkeys(participants, 0)
    losses = dict.fromkeys(participants, 0)
    last_winner = dict.fromkeys(participants)
    eliminated = []
    for number, (loser, winner) in enumerate(pairs, 1):
        if loser == '' or losses[loser] == losses[winner] == 2:
            continue
        wins[winner] += 1
        losses[loser] += 1
        last_winner[loser] = winner
        if losses[loser] == 2:
            eliminated.append({'participant': loser, 'winner': winner, 'bout': number})
    places = {entry['participant']: len(participants)-index for index, entry in enumerate(eliminated)}
    champion = next(name for name, place in podium['places'].items() if place == 1)
    places[champion] = 1
    if set(places) != set(participants) or sorted(places.values()) != list(range(1, len(participants)+1)):
        raise RuntimeError('Complete history did not produce individual elimination places')
    return {'status': 'ok', 'reason': '', 'places': places, 'wins': wins,
            'losses': losses, 'last_winner': last_winner, 'eliminated': eliminated,
            'podium': podium['places']}


def rank_tournament_far(names: dict, pairs: list) -> dict:
    """First eliminated gets N, next N-1; the final survivor gets first place."""
    history = elimination_history(names, pairs)
    result = {key: history[key] for key in ('status', 'reason', 'places')}
    if history['status'] == 'ok':
        result['diagnostics'] = {'eliminated': history['eliminated']}
    return result
