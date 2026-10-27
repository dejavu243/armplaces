"""Zhivora-Tour: fixed leading places, win groups, last winner's place.

The default fixes places 1–6, reproducing the normal power-of-two path of the
supplied Unit11.pas (1–4, then 5–6 by their last winners). fixed_top=4 is the
alternative from the initial prose description.
The group pass follows lines 1530–1570, including stable input order and temporary
keys for an opponent whose place is not assigned yet. No initial ratings are used.
"""

from .elimination import elimination_history


def rank_tournament_zhivora(names: dict, pairs: list, *, fixed_top: int = 6) -> dict:
    if isinstance(fixed_top, bool) or fixed_top not in (4, 6):
        raise ValueError('Zhivora fixed_top must be 4 or 6')
    history = elimination_history(names, pairs)
    if history['status'] != 'ok':
        return {key: history[key] for key in ('status', 'reason', 'places')}
    participants = list(names.values())
    if fixed_top == 6 and len(participants) not in (1, 2, 3, 4, 8, 16, 32):
        raise ValueError('The Pascal fixed-top-six path requires N=8,16,32 or N<=4; irregular 5–6 rules need weight data')
    places = {name: place for name, place in history['places'].items() if place <= 4}
    if fixed_top == 6 and len(participants) > 4:
        candidates = [name for name in participants if history['places'][name] in (5, 6)]
        opponent_places = {places.get(history['last_winner'][name]) for name in candidates}
        if len(candidates) != 2 or None in opponent_places or len(opponent_places) != 2 or 4 not in opponent_places:
            raise ValueError('Cannot reproduce the normal Pascal 5–6 placement from this history')
        for place, name in enumerate(sorted(candidates, key=lambda name: places[history['last_winner'][name]]), 5):
            places[name] = place
    fixed_places = dict(places)
    groups = []
    remaining = [name for name in participants if name not in places]
    for win_count in sorted({history['wins'][name] for name in remaining}, reverse=True):
        group = [name for name in remaining if history['wins'][name] == win_count]
        temporary = len(places)+1
        entries = []
        for name in group:
            opponent = history['last_winner'][name]
            opponent_place = places.get(opponent)
            if opponent_place is None:
                sort_place = temporary
                temporary += 1
            else:
                sort_place = opponent_place
            entries.append({'participant': name, 'last_winner': opponent,
                            'opponent_place': opponent_place, 'sort_place': sort_place})
        ordered = sorted(entries, key=lambda entry: entry['sort_place'])
        for place, entry in enumerate(ordered, len(places)+1):
            places[entry['participant']] = place
        groups.append({'wins': win_count, 'entries': entries,
                       'order': [entry['participant'] for entry in ordered]})
    return {'status': 'ok', 'reason': '', 'places': places,
            'diagnostics': {'fixed_top': fixed_top, 'fixed_places': fixed_places,
                            'wins': history['wins'], 'last_winner': history['last_winner'], 'groups': groups}}
