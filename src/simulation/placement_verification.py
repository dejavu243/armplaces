"""Independent replay of FAR/Zhivora placements and comparison statistics."""

import csv
from itertools import combinations
from math import isclose, sqrt


def same_number(actual, expected):
    if expected is None:
        return actual is None
    return isinstance(actual, (int, float)) and isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def independent_metrics(ratings, places):
    order = sorted(range(len(ratings)), key=lambda who: (-ratings[who], who))
    n = len(order)
    distances = [places[who]-place for place, who in enumerate(order, 1)]
    mean = (n+1)/2
    numerator = sum((rank-mean)*(places[who]-mean) for rank, who in enumerate(order, 1))
    denominator = sqrt(sum((rank-mean)**2 for rank in range(1, n+1))
                       *sum((places[who]-mean)**2 for who in order))
    return {'mae': sum(abs(value) for value in distances)/n,
            'spearman': numerator/denominator if denominator else None}


def verify_new_rankings(result, manifest):
    names = [str(who) for who in result['draw']]
    wins = dict.fromkeys(names, 0)
    losses = dict.fromkeys(names, 0)
    last = dict.fromkeys(names)
    eliminations = []
    number = 0
    for bout in result['bouts']:
        if bout['technical']:
            continue
        number += 1
        loser, winner = str(bout['loser']), str(bout['winner'])
        wins[winner] += 1
        losses[loser] += 1
        last[loser] = winner
        if losses[loser] == 2:
            eliminations.append({'participant': loser, 'winner': winner, 'bout': number})
    far = {str(result['champion']): 1}
    for index, entry in enumerate(eliminations):
        far[entry['participant']] = len(names)-index
    for method in ('far_tour', 'zhivora_tour'):
        if not manifest[method]:
            if result[method] != {'status': 'disabled', 'reason': '', 'places': {}}:
                raise ValueError('Unexpected enabled '+method)
    if manifest['far_tour']:
        expected = {'status': 'ok', 'reason': '', 'places': far,
                    'diagnostics': {'eliminated': eliminations}}
        if result['far_tour'] != expected:
            raise ValueError('Incorrect independently replayed FAR-tour places')
    if not manifest['zhivora_tour']:
        return
    fixed_top = manifest['zhivora_fixed_top']
    places = {who: place for who, place in far.items() if place <= 4}
    if fixed_top == 6 and len(names) > 4:
        candidates = [who for who in names if far[who] in (5, 6)]
        for place, who in enumerate(sorted(candidates, key=lambda who: places[last[who]]), 5):
            places[who] = place
    fixed = dict(places)
    groups = []
    # This is the array/bubble-sort pass from Pascal, not the production sorted pass.
    maximum = max((wins[who] for who in names if who not in places), default=-1)
    for win_count in range(maximum, -1, -1):
        temporary = len(places)+1
        group = []
        entries = []
        for who in names:
            if who in places or wins[who] != win_count:
                continue
            assigned = places.get(last[who])
            score = assigned
            if assigned is None:
                score = temporary
                temporary += 1
            group.append([score, who])
            entries.append({'participant': who, 'last_winner': last[who],
                            'opponent_place': assigned, 'sort_place': score})
        if not group:
            continue
        for stop in range(len(group)-1, 0, -1):
            for index in range(stop):
                if group[index][0] > group[index+1][0]:
                    group[index], group[index+1] = group[index+1], group[index]
        offset = len(places)+1
        for index, (_, who) in enumerate(group):
            places[who] = offset+index
        groups.append({'wins': win_count, 'entries': entries, 'order': [entry[1] for entry in group]})
    expected = {'status': 'ok', 'reason': '', 'places': places,
                'diagnostics': {'fixed_top': fixed_top, 'fixed_places': fixed,
                                'wins': wins, 'last_winner': last, 'groups': groups}}
    if result['zhivora_tour'] != expected:
        raise ValueError('Incorrect independently replayed Zhivora-Tour places')


def estimate(values):
    count = len(values)
    mean = sum(values)/count if count else None
    std = sqrt(sum((value-mean)**2 for value in values)/(count-1)) if count > 1 else None
    se = std/sqrt(count) if std is not None else None
    return mean, std, se, [mean-1.96*se, mean+1.96*se] if se is not None else None


def verify_metric_details(item, values):
    _, std, se, ci = estimate(values)
    ordered = sorted(values)
    percentile = None
    if ordered:
        position = (len(ordered)-1)*.95
        lower = int(position)
        upper = min(lower+1, len(ordered)-1)
        percentile = ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)
    wanted = {'mae_std': std, 'mae_se': se, 'mae_p95': percentile,
              'mae_max': max(values) if values else None,
              'perfect_order_rate': sum(value == 0 for value in values)/len(values) if values else None}
    for key, value in wanted.items():
        if not same_number(item[key], value):
            raise ValueError('Incorrect MAE uncertainty/tail statistic: '+key)
    if ci is None:
        if item['mae_ci95'] is not None:
            raise ValueError('MAE confidence interval without enough samples')
    elif (not isinstance(item['mae_ci95'], list) or len(item['mae_ci95']) != 2
          or any(not same_number(a, b) for a, b in zip(item['mae_ci95'], ci))):
        raise ValueError('Incorrect MAE confidence interval')


def verify_paired_comparisons(saved, series):
    expected_keys = {a+'__vs__'+b for a, b in combinations(series, 2)}
    if set(saved) != expected_keys:
        raise ValueError('Incorrect paired comparison methods')
    for left, right in combinations(series, 2):
        differences = [a-b for a, b in zip(series[left], series[right], strict=True)
                       if a is not None and b is not None]
        mean, std, se, ci = estimate(differences)
        item = saved[left+'__vs__'+right]
        wanted = {'left': left, 'right': right, 'samples': len(differences),
                  'skipped': len(series[left])-len(differences),
                  'left_better': sum(value < 0 for value in differences),
                  'tied': sum(value == 0 for value in differences),
                  'right_better': sum(value > 0 for value in differences)}
        if any(item[key] != value for key, value in wanted.items()):
            raise ValueError('Incorrect paired comparison counts')
        if any(not same_number(item[key], value) for key, value in
               (('mean_difference', mean), ('std_difference', std), ('se_difference', se))):
            raise ValueError('Incorrect paired MAE difference')
        if ci is None:
            if item['ci95'] is not None:
                raise ValueError('Paired confidence interval without enough samples')
        elif (not isinstance(item['ci95'], list) or len(item['ci95']) != 2
              or any(not same_number(a, b) for a, b in zip(item['ci95'], ci))):
            raise ValueError('Incorrect paired confidence interval')


def verify_metrics_csv(directory, expected):
    with (directory/'ranking-metrics.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['repeat', 'model', 'method', 'status', 'mae', 'spearman']:
            raise ValueError('Invalid per-tournament metrics columns')
        seen = set()
        for row in reader:
            key = int(row['repeat']), row['model'], row['method']
            if key not in expected or key in seen:
                raise ValueError('Invalid or duplicate per-tournament metric row')
            seen.add(key)
            status, metric = expected[key]
            if row['status'] != status:
                raise ValueError('Incorrect per-tournament metric status')
            for field in ('mae', 'spearman'):
                value = float(row[field]) if row[field] else None
                if not same_number(value, metric[field]):
                    raise ValueError('Incorrect per-tournament '+field)
        if seen != set(expected):
            raise ValueError('Missing per-tournament metrics')
