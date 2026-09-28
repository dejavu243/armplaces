"""Render reproducible SVG/PNG diagrams from the committed GrinTour examples.

Run with .venv/bin/python scripts/render_grintour_examples.py; requires Graphviz dot.
All diagrams use loser -> winner direction, including the reversed final graph.
"""
import json
from pathlib import Path
import shutil
import subprocess

import networkx as nx

from armplaces.grinev_algorithm import TournamentGraphConstructor
from armplaces.read_tournament import drop_simple_cycles
from armplaces.topological_sort import rank_tournament
from armplaces.cycle_resolution import rank_tournament_cycle_score

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'grintour-examples'


def quoted(value):
    return json.dumps(str(value), ensure_ascii=False)


def render(path, graph, podium, highlighted, marked_edges, title, subtitle, raw_edges,
           ratings=None, restored=False):
    lines = ['digraph G {',
             'graph [rankdir=BT, bgcolor="#ffffff", pad="0.35", nodesep="0.32", '
             'ranksep="0.55", fontname="DejaVu Sans", fontsize=18, labelloc=t, dpi=130,',
             f'label={quoted(title + chr(10) + subtitle)}];',
             'node [shape=ellipse, style=filled, fontname="DejaVu Sans", fontsize=13, '
             'color="#7891ad", fillcolor="#edf3fa", penwidth=1.4, margin="0.14,0.10"];',
             'edge [color="#76879a", arrowsize=0.65, penwidth=1.1];']
    for name in sorted(graph, key=int):
        label = '№ ' + name
        if name in podium:
            label += '\nместо ' + str(podium[name])
        if ratings:
            label += f'\nR = {ratings[name]:.1f}'
        attrs = [f'label={quoted(label)}']
        if name in podium:
            attrs += ['fillcolor="#fff0b3"', 'color="#a87d16"']
        if name in highlighted:
            attrs += ['fillcolor="#dcf5e8"', 'color="#18754a"', 'penwidth=2.5'] if restored else ['fillcolor="#ffe1de"', 'color="#bf3029"', 'penwidth=2.5']
            if graph.degree(name) == 0:
                attrs += ['style="filled,dashed"']
        lines.append(f'{quoted(name)} [{", ".join(attrs)}];')
    for a, b in sorted(graph.edges(), key=lambda e: (int(e[0]), int(e[1]))):
        attrs = []
        if (a, b) not in raw_edges:
            attrs += ['style=dashed']
        if (a, b) in marked_edges:
            attrs += ['color="#18754a"', 'penwidth=2.7'] if restored else ['color="#bf3029"', 'penwidth=2.7']
        lines.append(f'{quoted(a)} -> {quoted(b)} [{", ".join(attrs)}];')
    lines.append('}')
    source = '\n'.join(lines)
    for extension in ('svg', 'png'):
        subprocess.run(['dot', '-T'+extension, '-o', str(path.with_suffix('.'+extension))],
                       input=source, text=True, check=True)


def main():
    if not shutil.which('dot'):
        raise SystemExit('Graphviz dot is required; install Graphviz and rerun this command.')
    images = ROOT / 'images'
    images.mkdir(exist_ok=True)
    for path in sorted(ROOT.glob('*.json')):
        case = json.loads(path.read_text())
        names = dict(enumerate(case['draw']))
        assert rank_tournament(names, case['pairs']) == case['graph_only']
        podium = {name: place for name, place in case['graph_only']['places'].items() if place <= 3}
        raw = set(map(tuple, drop_simple_cycles(case['pairs'])))
        before = nx.DiGraph()
        before.add_nodes_from(names.values())
        before.add_edges_from(raw)
        for a, b in list(before.edges()):
            if a in podium and (b not in podium or podium[a] < podium[b]):
                before.remove_edge(a, b)
        ordered = sorted(podium, key=podium.get)
        before.add_edges_from((b, a) for a, b in zip(ordered, ordered[1:]))
        before.add_edges_from((name, ordered[-1]) for name in names.values() if name not in podium)
        if 'cycles' in case['details']:
            cycle = max(case['details']['cycles'], key=len)
            edges = set(zip(cycle, cycle[1:]+cycle[:1]))
            assert edges <= set(before.edges())
            render(images/(path.stem+'-context'), before, podium, set(cycle), edges,
                   'Цикл среди непризёров', 'Граф после фиксации тройки • проигравший → победитель', raw)
            detail = nx.DiGraph()
            detail.add_edges_from(edges)
            render(images/(path.stem+'-detail'), detail, {}, set(cycle), edges,
                   'Почему расстановка остановилась', 'Замкнутый цикл • проигравший → победитель', raw,
                   ratings=case['ratings'])
            scored = rank_tournament_cycle_score(names, case['pairs'])
            assert scored['status'] == 'ok' and scored['diagnostics']['activated']
            diagnostics = scored['diagnostics']
            preliminary = before.copy()
            for component in diagnostics['components']:
                members = set(component['members'])
                preliminary.remove_edges_from((a, b) for a, b in list(preliminary.edges())
                                              if a in members and b in members)
            render(images/(path.stem+'-preliminary'), preliminary, podium, set(cycle), set(),
                   'Временный граф без внутренних связей цикла',
                   'По этому графу рассчитываются предварительные места', raw)
            resolved = before.copy()
            resolved.remove_edges_from(map(tuple, diagnostics['removed_edges']))
            resolved.add_edges_from(map(tuple, diagnostics['added_edges']))
            render(images/(path.stem+'-resolved'), resolved, podium, set(cycle),
                   set(map(tuple, diagnostics['added_edges'])),
                   'Дополнительное разрешение цикла',
                   'Зелёным — добавленные связи порядка • проигравший → победитель',
                   raw, restored=True)
        else:
            isolated = set(case['details']['isolated'])
            after = nx.DiGraph()
            after.add_nodes_from(names.values())
            after.add_weighted_edges_from(case['details']['edges'])
            after = after.reverse()
            assert set(nx.isolates(after)) == isolated
            incident = {(a, b) for a, b in before.edges() if a in isolated or b in isolated}
            render(images/(path.stem+'-before'), before, podium, isolated, incident,
                   'До отбора цепочек', 'Связи выделенных участников ещё существуют • проигравший → победитель', raw)
            render(images/(path.stem+'-after'), after, podium, isolated, set(),
                   'После отбора цепочек L и L−1', 'Выделенные участники потеряли все связи • проигравший → победитель', raw)
            restored_graph = TournamentGraphConstructor(names, case['pairs'], fixed_places=podium).make_graph().reverse()
            assert not list(nx.isolates(restored_graph))
            assert set(restored_graph.edges()) == set(before.edges())
            render(images/(path.stem+'-restored'), restored_graph, podium, isolated, incident,
                   'Исправлено: сохранены все ветви',
                   'Зелёным — восстановленные связи • проигравший → победитель', raw, restored=True)
        print('Rendered', path.stem)


if __name__ == '__main__':
    main()
