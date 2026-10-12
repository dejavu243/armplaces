"""Individual-place metrics and Monte Carlo uncertainty for paired tournaments."""

from itertools import combinations

import numpy as np

from .place_statistics import place_errors


def individual_metrics(ratings, places):
    errors = place_errors(ratings, places)
    n = len(errors)
    return {'mae': sum(abs(value) for value in errors)/n,
            'spearman': 1-6*sum(value*value for value in errors)/(n*(n*n-1)) if n > 1 else None}


def uncertainty(values):
    count = len(values)
    mean = float(np.mean(values)) if count else None
    std = float(np.std(values, ddof=1)) if count > 1 else None
    se = std/count**.5 if std is not None else None
    return {'mean': mean, 'std': std, 'se': se,
            'ci95': [mean-1.96*se, mean+1.96*se] if se is not None else None}


def metric_details(values):
    estimate = uncertainty(values)
    return {'mae_std': estimate['std'], 'mae_se': estimate['se'], 'mae_ci95': estimate['ci95'],
            'mae_p95': float(np.quantile(values, .95)) if values else None,
            'mae_max': max(values) if values else None,
            'perfect_order_rate': sum(value == 0 for value in values)/len(values) if values else None}


def paired_comparisons(series):
    result = {}
    for left, right in combinations(series, 2):
        pairs = [(a, b) for a, b in zip(series[left], series[right], strict=True)
                 if a is not None and b is not None]
        differences = [a-b for a, b in pairs]
        estimate = uncertainty(differences)
        result[left+'__vs__'+right] = {
            'left': left, 'right': right, 'samples': len(differences),
            'skipped': len(series[left])-len(differences),
            'mean_difference': estimate['mean'],
            'std_difference': estimate['std'],
            'se_difference': estimate['se'],
            'ci95': estimate['ci95'],
            'left_better': sum(value < 0 for value in differences),
            'tied': sum(value == 0 for value in differences),
            'right_better': sum(value > 0 for value in differences)}
    return result
