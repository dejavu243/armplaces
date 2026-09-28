"""Проверка полноты артефактов: python -m simulation.verify DIRECTORY.

Манифест и записи всех участников сопоставляются с журналом боёв, который
независимо проигрывается повторно для проверки счётчиков, нагрузки и мест.
Метрики пересчитываются по CSV и сверяются со сводкой. Повреждённый экспорт
завершает CLI ошибкой. Форматы файлов и ограничения описаны в README.md.
"""
import argparse
import csv
import json
from collections import Counter
from itertools import groupby
from math import isclose
from pathlib import Path

import networkx as nx

from armplaces.cycle_resolution import CRITERIA, normalize_weights
from armplaces.grinev_algorithm import TournamentGraphConstructor
from armplaces.topological_sort import (get_places, get_tournament_dict,
                                        standard_graph_pass)
from .core import Config, MODELS, elo_probability, validate_result
from .experiments import rank_metrics, require_finite


def verify_cycle_diagnostics(result, manifest):
    """Recompute chronology, tied ranks, graph transformations and final places."""
    cycle = result['grin_cycle_score']
    if not manifest['grin_cycle_score']:
        if cycle != {'status':'disabled','reason':'','places':{}}:
            raise ValueError('Unexpected enabled cycle-score ranking')
        return
    diagnostic = cycle['diagnostics']
    weights = normalize_weights(manifest['cycle_weights'])
    if (list(weights) != manifest['normalized_cycle_weights']
            or any(not isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
                   for a,b in zip(weights, diagnostic['weights']))):
        raise ValueError('Incorrect cycle-score weights')
    draw = result['draw']
    names = {i:str(i) for i in draw}
    pairs = [(str(b['loser']),str(b['winner'])) for b in result['bouts'] if not b['technical']]
    standard, failure, projection = standard_graph_pass(names,pairs)
    expected_standard = {'status':standard['status'],'reason':standard['reason'],
                         'reason_code':failure.reason_code if failure else None}
    if diagnostic['standard'] != expected_standard:
        raise ValueError('Incorrect standard-pass cycle diagnostics')
    activated = failure is not None and failure.reason_code == 'remaining_cycle'
    if diagnostic['activated'] is not activated:
        raise ValueError('Cycle resolver ran under the wrong condition')
    if not activated:
        if (cycle['status'] != standard['status'] or cycle['reason'] != standard['reason']
                or cycle['places'] != standard['places'] or diagnostic['components']
                or diagnostic['preliminary_places'] or diagnostic['removed_edges']
                or diagnostic['added_edges']):
            raise ValueError('Standard graph result changed without a residual cycle')
        return

    draw_order = {str(who):i for i,who in enumerate(draw)}
    groups = sorted((set(group) for group in nx.strongly_connected_components(projection)
                     if len(group)>1),key=lambda members:min(draw_order[name] for name in members))
    if [sorted(group,key=draw_order.get) for group in groups] != [c['members'] for c in diagnostic['components']]:
        raise ValueError('Incorrect cycle components')
    preliminary = projection.copy()
    for members in groups:
        preliminary.remove_edges_from((a,b) for a,b in list(preliminary.edges())
                                      if a in members and b in members)
    constructor = TournamentGraphConstructor(names,pairs,fixed_places=standard['places'],allow_cycles=True)
    constructor.up = preliminary
    preliminary_graph = constructor.make_graph()
    preliminary_places = {name:values['place'] for name,values in get_places(
        get_tournament_dict(preliminary_graph),preliminary_graph,
        fixed_places=standard['places']).items()}
    if preliminary_places != diagnostic['preliminary_places']:
        raise ValueError('Incorrect preliminary cycle places')
    resolved = projection.copy()
    all_removed,all_added = [],[]
    for members,component in zip(groups,diagnostic['components']):
        raw = {name:{'chronology':None,'loss_place':None,'graph_place':preliminary_places[name]}
               for name in members}
        losses = {name:[] for name in members}
        for bout_number,(loser,winner) in enumerate(pairs,1):
            if loser in members and winner in members:
                raw[winner]['chronology']=bout_number
            if loser in members:
                losses[loser].append(preliminary_places[winner])
        for name in members:
            if raw[name]['chronology'] is None or not losses[name]:
                raise ValueError('Invalid cyclic participant history')
            raw[name]['loss_place']=max(losses[name])
        ranks={}
        for criterion in CRITERIA:
            ordered=sorted(members,key=lambda name:raw[name][criterion],
                           reverse=(criterion=='chronology'))
            values={}
            for name in members:
                matching=[i+1 for i,who in enumerate(ordered)
                          if raw[who][criterion]==raw[name][criterion]]
                rank=sum(matching)/len(matching)
                values[name]=(rank,(len(members)-rank)/(len(members)-1))
            ranks[criterion]=values
        totals={}
        for name in members:
            saved=component['scores'][name]
            total=sum(weights[i]*ranks[criterion][name][1]
                      for i,criterion in enumerate(CRITERIA))
            if (saved['values']!=raw[name]
                    or any(not isclose(saved['ranks'][criterion],ranks[criterion][name][0])
                           or not isclose(saved['scores'][criterion],ranks[criterion][name][1])
                           for criterion in CRITERIA)
                    or not isclose(saved['total'],total,rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError('Incorrect independent cycle criterion or score')
            totals[name]=total
        order=sorted(members,key=lambda name:(-round(totals[name],12),draw_order[name]))
        if order!=component['order']:
            raise ValueError('Incorrect cycle score order')
        priority={name:i for i,name in enumerate(order)}
        removed=sorted(([a,b] for a,b in resolved.edges() if a in members and b in members
                        and priority[a]<priority[b]),key=lambda edge:(draw_order[edge[0]],draw_order[edge[1]]))
        resolved.remove_edges_from(removed)
        added=[]
        for higher,lower in zip(order,order[1:]):
            if not resolved.has_edge(lower,higher):
                resolved.add_edge(lower,higher)
                added.append([lower,higher])
        if component['removed_edges']!=removed or component['added_edges']!=added:
            raise ValueError('Incorrect cycle graph changes')
        all_removed.extend(removed)
        all_added.extend(added)
    if diagnostic['removed_edges']!=all_removed or diagnostic['added_edges']!=all_added:
        raise ValueError('Incorrect combined cycle graph changes')
    if not nx.is_directed_acyclic_graph(resolved):
        raise ValueError('Scored cycle graph still contains a cycle')
    constructor.up=resolved
    graph=constructor.make_graph()
    final={name:values['place'] for name,values in get_places(
        get_tournament_dict(graph),graph,fixed_places=standard['places']).items()}
    if cycle['status']!='ok' or cycle['places']!=final or cycle['reason']:
        raise ValueError('Incorrect final cycle-score placement')


def verify_artifacts(directory: Path | str):
    directory = Path(directory)
    for filename in ('config.json', 'summary.json', 'participants.csv', 'bouts.csv',
                     'tournaments.jsonl', 'report.md', 'run.log'):
        if not (directory/filename).is_file() or not (directory/filename).stat().st_size:
            raise ValueError(f'Missing or empty artifact: {filename}')
    manifest = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if manifest.get('schema_version') != 4:
        raise ValueError('Expected artifact schema 4; verify older artifacts with their recorded code revision')
    require_finite(manifest)
    require_finite(summary)
    config = Config(**manifest['config'])
    config.validate()
    models = manifest['models']
    n, repeats = config.participants, config.repeats
    if manifest['status'] != 'complete' or not models or len(set(models)) != len(models) or set(summary) != set(models) or not set(models) <= set(MODELS):
        raise ValueError('Incomplete experiment manifest or summary')
    expected = {(repeat, model) for repeat in range(repeats) for model in models}
    records = {}
    with (directory/'tournaments.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            require_finite(item)
            key = item['repeat'], item['model']
            if key not in expected or key in records or sorted(item['draw']) != list(range(n)):
                raise ValueError('Invalid or duplicate tournament record')
            records[key] = item
    if set(records) != expected:
        raise ValueError('Missing tournament records')
    seen = set()
    totals = {model: {'strongest': 0, 'undefined': Counter(), 'partial': Counter(),
                      'bracket': [], 'grin_tour': [], 'grin_cycle_score': [],
                      'cycle_activated':0,'cycle_success':0,'cycle_standard_success':0,
                      'cycle_failures':Counter(), 'cycle_subset_grin_tour':[],
                      'cycle_subset_grin_cycle_score':[]}
              for model in models}
    def group_key(row):
        return int(row['repeat']), row['model']
    with (directory/'participants.csv').open(newline='', encoding='utf-8') as participant_stream, \
            (directory/'bouts.csv').open(newline='', encoding='utf-8') as bout_stream:
        bout_groups = iter(groupby(csv.DictReader(bout_stream), group_key))
        for key, rows in groupby(csv.DictReader(participant_stream), group_key):
            rows = sorted(rows, key=lambda row: int(row['participant']))
            if key not in expected or key in seen or [int(row['participant']) for row in rows] != list(range(n)):
                raise ValueError('Missing or duplicate participant records')
            seen.add(key)
            record = records[key]
            result = {**record, 'ratings': [float(row['rating']) for row in rows],
                      'new_ratings': [float(row['new_rating']) for row in rows],
                      'wins': [int(row['wins']) for row in rows],
                      'losses': [int(row['losses']) for row in rows],
                      'played': [int(row['played']) for row in rows],
                      'places': {int(row['participant']): int(row['place']) for row in rows},
                      'bouts': []}
            require_finite(result)
            ranking = result['grin_tour']
            cycle = result['grin_cycle_score']
            if (not manifest['grin_tour']) != (ranking['status'] == 'disabled'):
                raise ValueError('Unexpected GrinTour status')
            for i, row in enumerate(rows):
                if result['ratings'][i] <= 0 or int(row['draw_position']) != record['draw'].index(i)+1:
                    raise ValueError('Invalid rating or draw position')
                if row['grin_status'] != ranking['status'] or row['grin_reason'] != ranking['reason']:
                    raise ValueError('Inconsistent GrinTour status')
                if row['participant'] in ranking['places']:
                    if not row['grin_place'] or int(row['grin_place']) != ranking['places'][row['participant']]:
                        raise ValueError('Inconsistent GrinTour place')
                elif row['grin_place']:
                    raise ValueError('Unexpected GrinTour place')
                if row['cycle_status'] != cycle['status'] or row['cycle_reason'] != cycle['reason']:
                    raise ValueError('Inconsistent cycle-score status')
                if row['participant'] in cycle['places']:
                    if not row['cycle_place'] or int(row['cycle_place']) != cycle['places'][row['participant']]:
                        raise ValueError('Inconsistent cycle-score place')
                elif row['cycle_place']:
                    raise ValueError('Unexpected cycle-score place')
            if n > 1:
                bout_key, bout_rows = next(bout_groups, (None, ()))
                if bout_key != key:
                    raise ValueError('Missing or out-of-order bout journal')
                for row in bout_rows:
                    if row['technical'] not in ('True', 'False'):
                        raise ValueError('Invalid technical-pass flag')
                    bout = {'stage': row['stage'], 'technical': row['technical'] == 'True'}
                    for field in ('match', 'a', 'b', 'winner', 'loser', 'previous_bouts_a', 'previous_bouts_b'):
                        bout[field] = int(row[field]) if row[field] else None
                    for field in ('p_a', 'p_start_a', 'effective_a', 'effective_b'):
                        bout[field] = float(row[field]) if row[field] else None
                    result['bouts'].append(bout)
            validate_result(result)
            verify_cycle_diagnostics(result, manifest)
            changes = [0.]*n
            for bout in result['bouts']:
                if not bout['technical']:
                    a, b = bout['a'], bout['b']
                    baseline = elo_probability(result['ratings'][a], result['ratings'][b], config.dr)
                    if not isclose(baseline, bout['p_start_a'], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError('Incorrect baseline probability')
                    change = config.k * (int(bout['winner'] == a)-baseline)
                    changes[a] += change
                    changes[b] -= change
            for initial, updated, change in zip(result['ratings'], result['new_ratings'], changes):
                if not isclose(initial+change, updated, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError('Incorrect post-tournament rating')
            total = totals[key[1]]
            total['strongest'] += result['ratings'][result['champion']] == max(result['ratings'])
            total['bracket'].append(rank_metrics(result['ratings'], result['places']))
            if ranking['status'] == 'ok':
                total['grin_tour'].append(rank_metrics(result['ratings'], {int(k): v for k,v in ranking['places'].items()}))
            elif ranking['status'] in ('undefined', 'partial'):
                total[ranking['status']][ranking['reason']] += 1
            if manifest['grin_cycle_score']:
                activated=cycle['diagnostics']['activated']
                total['cycle_activated']+=activated
                total['cycle_standard_success']+=not activated and cycle['status']=='ok'
                if cycle['status']=='ok':
                    metric=rank_metrics(result['ratings'],{int(k):v for k,v in cycle['places'].items()})
                    total['grin_cycle_score'].append(metric)
                    if activated:
                        total['cycle_success']+=1
                        total['cycle_subset_grin_cycle_score'].append(metric)
                elif activated:
                    total['cycle_failures'][cycle['reason']]+=1
                if activated and ranking['status']=='ok':
                    total['cycle_subset_grin_tour'].append(rank_metrics(
                        result['ratings'],{int(k):v for k,v in ranking['places'].items()}))
        if next(bout_groups, None) is not None:
            raise ValueError('Unexpected extra bout group')
    if seen != expected:
        raise ValueError('Missing participant groups')
    for model, item in summary.items():
        total = totals[model]
        if item['completed'] != repeats or not isclose(item['strongest_win_rate'], total['strongest']/repeats):
            raise ValueError('Incorrect tournament count or strongest-win rate')
        for status in ('undefined', 'partial'):
            if item[f'grin_{status}_reasons'] != dict(total[status]):
                raise ValueError(f'Incorrect GrinTour {status} reasons')
            expected_rate = sum(total[status].values())/repeats if manifest['grin_tour'] else None
            if item[f'grin_{status}_rate'] != expected_rate:
                raise ValueError(f'Incorrect GrinTour {status} rate')
        for field in ('cycle_activated','cycle_success','cycle_standard_success'):
            if item[field]!=total[field]:
                raise ValueError('Incorrect cycle-score activation counts')
        if item['cycle_failures']!=dict(total['cycle_failures']):
            raise ValueError('Incorrect cycle-score failure reasons')
        for method in ('bracket', 'grin_tour', 'grin_cycle_score',
                       'cycle_subset_grin_tour','cycle_subset_grin_cycle_score'):
            samples = total[method]
            correlations = [x['spearman'] for x in samples if x['spearman'] is not None]
            if item[method]['samples'] != len(samples) or item[method]['correlation_samples'] != len(correlations):
                raise ValueError('Incorrect metric sample count')
            for metric, values in (('mae', [x['mae'] for x in samples]), ('spearman', correlations)):
                actual = item[method][metric]
                if not values:
                    if actual is not None:
                        raise ValueError('Metric without samples must be null')
                elif actual is None or not isclose(actual, sum(values)/len(values), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Incorrect aggregate metric')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        verify_artifacts(args.directory)
    except (ValueError, RuntimeError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print('Artifacts verified')


if __name__ == '__main__':
    main()
