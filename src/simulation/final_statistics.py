"""Final ratings before each deciding bout, oriented by the bracket branches."""
from collections import Counter

import numpy as np


def final_record(result):
    """One row per tournament; a singleton has no final and returns None.

    A signed gap is upper-branch rating minus lower-branch rating. Branches are
    identified from prior losses, independently of the order of the GS cells.
    """
    losses = Counter()
    row = None
    for bout in result['bouts']:
        if bout['technical']:
            continue
        if bout['stage'] == 'final':
            upper_side = next(side for side in ('a', 'b') if losses[bout[side]] == 0)
            lower_side = 'b' if upper_side == 'a' else 'a'
            upper, lower = bout[upper_side], bout[lower_side]
            gap = bout['effective_'+upper_side] - bout['effective_'+lower_side]
            row = {'repeat': result['repeat'], 'upper': upper, 'lower': lower,
                   'initial_upper': result['ratings'][upper],
                   'initial_lower': result['ratings'][lower],
                   'effective_upper': bout['effective_'+upper_side],
                   'effective_lower': bout['effective_'+lower_side],
                   'fatigue_upper': bout['fatigue_before_'+upper_side],
                   'fatigue_lower': bout['fatigue_before_'+lower_side],
                   'gap': gap, 'abs_gap': abs(gap),
                   'lower_win_probability': bout['p_a'] if lower_side == 'a' else 1-bout['p_a'],
                   'final_winner': bout['winner'], 'reset': result['reset'],
                   'reset_gap': None, 'reset_abs_gap': None,
                   'champion': result['champion']}
        elif bout['stage'] == 'reset':
            if row is None:
                raise ValueError('Superfinal without a final')
            upper_side = 'a' if bout['a'] == row['upper'] else 'b'
            lower_side = 'b' if upper_side == 'a' else 'a'
            gap = bout['effective_'+upper_side] - bout['effective_'+lower_side]
            row.update(reset_gap=gap, reset_abs_gap=abs(gap))
        losses[bout['loser']] += 1
    if row is not None and row['reset'] != (row['reset_gap'] is not None):
        raise ValueError('Superfinal flag disagrees with the bout journal')
    return row


def gap_statistics(gaps):
    values = np.asarray(gaps, dtype=float)
    if not len(values):
        return {'samples': 0, 'mean_gap': None, 'mean_abs_gap': None,
                'median_abs_gap': None, 'p95_abs_gap': None, 'max_abs_gap': None}
    absolute = np.abs(values)
    return {'samples': len(values), 'mean_gap': float(values.mean()),
            'mean_abs_gap': float(absolute.mean()),
            'median_abs_gap': float(np.median(absolute)),
            'p95_abs_gap': float(np.percentile(absolute, 95)),
            'max_abs_gap': float(absolute.max())}


def summarize_finals(rows):
    """Reset rate uses all first finals; reset gaps use only actual superfinals."""
    rows = list(rows)
    resets = [row for row in rows if row['reset']]
    return {'final': gap_statistics([row['gap'] for row in rows]),
            'superfinal': gap_statistics([row['reset_gap'] for row in resets]),
            'reset_count': len(resets),
            'reset_rate': len(resets)/len(rows) if rows else None}
