"""Generate three fully verified Elo-tire examples and readable N=22 protocols."""

import argparse
import csv
import json
from collections import defaultdict
from html import escape
from pathlib import Path

from simulation import Config
from simulation.experiments import run_experiments
from simulation.verify import verify_artifacts

METHODS = (('far_tour', 'FAR-tour'), ('zhivora_tour', 'Zhivora-Tour'),
           ('grin_cycle_score', 'Grin-Cycle'), ('grin_tour', 'Grin-Elo'))


def table(headers, rows):
    markdown = ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']
    markdown += ['| '+' | '.join(map(str, row))+' |' for row in rows]
    html = '<div class="scroll"><table><thead><tr>'
    html += ''.join('<th>'+escape(str(value))+'</th>' for value in headers)
    html += '</tr></thead><tbody>'
    html += ''.join('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in row)+'</tr>' for row in rows)
    return '\n'.join(markdown), html+'</tbody></table></div>'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_examples(artifacts, output):
    verify_artifacts(artifacts)
    manifest = json.loads((artifacts/'config.json').read_text())
    config = Config(**manifest['config'])
    if (config.participants != 22 or config.repeats != 3 or manifest['models'] != ['elo-tire']
            or manifest['zhivora_fixed_top'] != 6 or not all(manifest[key] for key, _ in METHODS)):
        raise ValueError('Expected three N=22 Elo-tire examples with all four methods and Pascal top six')
    records = [json.loads(line) for line in (artifacts/'tournaments.jsonl').read_text().splitlines()]
    groups = {}
    for name in ('participants', 'bouts', 'ranking-metrics'):
        grouped = defaultdict(list)
        with (artifacts/(name+'.csv')).open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                grouped[int(row['repeat'])].append(row)
        groups[name] = grouped
    output.mkdir(parents=True, exist_ok=True)
    intro = (f'Три независимых турнира Elo-tire: N=22, seed={config.seed}, повторы 0, 1, 2. '
             f'Рейтинги beta-prime: α={config.alpha:g}, β={config.beta:g}, scale={config.scale:g}, '
             f'диапазон [600,1900). A={config.tire_floor:g}, A′={config.tire_reference:g}, '
             f'C={config.tire_c:g}, N_f={config.tire_bouts}, dR={config.dr:g}. '
             'Каждый пример имеет собственные рейтинги, жеребьёвку и исходы. '
             'Четыре метода распределяют места по одному и тому же журналу боёв.')
    notes = ('Спортсмены обозначены номерами 1–22 во всех таблицах и CSV этого отчёта; '
             'в исходных артефактах ID начинаются с 0. Истинное место — порядок исходных рейтингов '
             'по убыванию, равенства по ID. Для исхода боя используются эффективные рейтинги с усталостью, '
             'без округления; показанные числа округлены. P(A) — вероятность победы первого в паре. '
             'W — верхняя сетка, L — нижняя. Технический проход не считается боем.')
    zhivora = ('Zhivora-Tour: фиксированы места 1–6, групповой проход начинается с 7-го. '
               'Для N=22 сохранена проверка обычной ветви: два претендента на 5–6 должны проиграть '
               'разным участникам из первой четвёрки, один — будущему четвёртому. '
               'Во всех трёх примерах условие выполнено; вес спортсменов не используется.')
    html_sections = []
    overview = []
    rating_rows, bout_rows, place_rows = [], [], []
    for record in records:
        repeat = record['repeat']
        number = repeat+1
        participants = sorted(groups['participants'][repeat], key=lambda row: int(row['participant']))
        bouts = groups['bouts'][repeat]
        true_order = sorted(participants, key=lambda row: (-float(row['rating']), int(row['participant'])))
        true_place = {int(row['participant']): place for place, row in enumerate(true_order, 1)}
        real = [bout for bout in bouts if bout['technical'] == 'False']
        champion = record['champion']+1
        metrics = {row['method']: row for row in groups['ranking-metrics'][repeat]}
        summary = (f'Повтор {repeat}; реальных боёв: {len(real)}; технических проходов: {len(bouts)-len(real)}. '
                   f'Суперфинал: {"да" if record["reset"] else "нет"}. '
                   f'Чемпион — №{champion}, его истинное место: {true_place[champion-1]}.')
        markdown = [f'# Пример {number}: 22 участника', '', intro, '', notes, '', zhivora, '', summary, '']
        html = [f'<section id="example-{number}"><h2>Пример {number}</h2>', '<p>'+escape(summary)+'</p>']

        def add_section(title, headers, rows):
            md_table, html_table = table(headers, rows)
            markdown.extend(['## '+title, '', md_table, ''])
            html.extend(['<h3>'+escape(title)+'</h3>', html_table])

        add_section('Исходные рейтинги (по силе)', ['Истинное место', 'Спортсмен', 'Рейтинг', 'Позиция жеребьёвки'],
                    [(true_place[int(row['participant'])], int(row['participant'])+1,
                      f'{float(row["rating"]):.6f}', int(row['draw_position'])) for row in true_order])
        draw = ', '.join(str(who+1) for who in record['draw'])
        markdown += ['## Жеребьёвка', '', 'Позиции 1–22: **'+draw+'**.', '']
        html += ['<h3>Жеребьёвка</h3><p>Позиции 1–22: '+escape(draw)+'</p>']
        losses = defaultdict(int)
        journal = []
        for bout_number, bout in enumerate(real, 1):
            loser = int(bout['loser'])+1
            losses[loser] += 1
            stage = {'final': 'Финал', 'reset': 'Суперфинал'}.get(bout['stage'], bout['stage'])
            journal.append((bout_number, stage, f'{int(bout["a"])+1} — {int(bout["b"])+1}',
                            f'{float(bout["effective_a"]):.2f}', f'{float(bout["effective_b"]):.2f}',
                            f'{float(bout["p_a"]):.2%}', int(bout['winner'])+1,
                            f'{loser}: {losses[loser]}'+(' (выбыл)' if losses[loser] == 2 else '')))
        add_section('Пары в порядке проведения',
                    ['Бой', 'Стадия', 'Пара A — B', 'R эфф. A', 'R эфф. B', 'P(A)',
                     'Победитель', 'Проигравший: поражений'], journal)
        orders = [{place: int(who)+1 for who, place in record[key]['places'].items()} for key, _ in METHODS]
        placements = [(place, int(true_order[place-1]['participant'])+1,
                       *(order[place] for order in orders)) for place in range(1, 23)]
        add_section('Итоговая расстановка', ['Место', 'Эталон по рейтингу', *(label for _, label in METHODS)], placements)
        markdown += ['В ячейках этой таблицы — **номера спортсменов**, а не их рейтинги.', '']
        html += ['<p>В ячейках итоговой таблицы — <strong>номера спортсменов</strong>.</p>']
        method_names = ('far-tour', 'zhivora-tour', 'grin-cycle', 'grin-elo')
        errors = [(label, f'{float(metrics[name]["mae"]):.6f}', f'{float(metrics[name]["spearman"]):.6f}')
                  for (_, label), name in zip(METHODS, method_names, strict=True)]
        add_section('Точность мест', ['Метод', 'Среднее абсолютное отклонение', 'Spearman'], errors)
        html.append('</section>')
        html_sections.append('\n'.join(html))
        (output/f'example-{number}.md').write_text('\n'.join(markdown), encoding='utf-8')
        overview.append((number, len(real), 'да' if record['reset'] else 'нет', champion,
                         true_place[champion-1], *(f'{float(metrics[name]["mae"]):.6f}' for name in method_names)))
        for row in participants:
            who = int(row['participant'])
            rating_rows.append({'example': number, 'number': who+1, 'rating': row['rating'],
                                'true_place': true_place[who], 'draw_position': int(row['draw_position']),
                                'wins': row['wins'], 'losses': row['losses'],
                                **{key+'_place': record[key]['places'][str(who)] for key, _ in METHODS}})
        for bout in bouts:
            converted = {'example': number, **{key: value for key, value in bout.items() if key != 'repeat'}}
            for field in ('match', 'a', 'b', 'winner', 'loser'):
                converted[field] = int(bout[field])+1 if bout[field] else ''
            bout_rows.append(converted)
        for place, true_number, *numbers in placements:
            place_rows.append({'example': number, 'place': place, 'true_number': true_number,
                               **{key: who for (key, _), who in zip(METHODS, numbers, strict=True)}})
    headers = ['Пример', 'Боёв', 'Суперфинал', 'Чемпион', 'Истинное место чемпиона',
               *(label+': MAE' for _, label in METHODS)]
    md_overview, html_overview = table(headers, overview)
    reproduction = ('.venv/bin/python scripts/render_n22_examples.py\n'
                    '.venv/bin/python -m simulation.verify results/n22-examples')
    index = ['# Три турнира для 22 участников', '', intro, '', notes, '', zhivora, '', md_overview, '',
             '[Открыть все примеры в одном HTML-файле](all-examples.html)', '',
             *[f'- [Пример {i}: рейтинги, жеребьёвка, бои и четыре расстановки](example-{i}.md)' for i in range(1, 4)], '',
             'CSV с номерами спортсменов 1–22: [рейтинги и места](ratings.csv), '
             '[журнал боёв с усталостью](bouts.csv), [порядки мест](places.csv).', '',
             'Артефакты schema 6 и независимая проверка: `'+str(artifacts)+'`.',
             'SHA256 пакетов: `'+manifest['environment']['source_sha256']+'`.', '',
             '## Воспроизведение', '', '```bash', reproduction, '```', '']
    (output/'index.md').write_text('\n'.join(index), encoding='utf-8')
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>Три турнира: N=22</title><style>'
            'body{font:16px/1.5 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1250px;color:#18212b}'
            'table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:1rem 0}'
            'th,td{border:1px solid #cdd5dd;padding:.35rem .6rem;text-align:right;white-space:nowrap}'
            'th{background:#eaf0f6}tr:nth-child(even){background:#f6f8fa}.scroll{overflow-x:auto}'
            'h2{border-top:3px solid #376a97;padding-top:1rem;margin-top:3rem}nav a{margin-right:2rem}'
            '@media print{body{font-size:10px}.scroll{overflow:visible}section{break-before:page}thead{display:table-header-group}}'
            '</style><body><h1>Три турнира для 22 участников</h1><p>'+escape(intro)+'</p><p>'+escape(notes)
            +'</p><p>'+escape(zhivora)+'</p>'+html_overview+'<nav>'
            +''.join(f'<a href="#example-{i}">Пример {i}</a>' for i in range(1, 4))+'</nav>'
            +'\n'.join(html_sections)+'</body></html>\n')
    (output/'all-examples.html').write_text(html, encoding='utf-8')
    for name, rows in (('ratings', rating_rows), ('bouts', bout_rows), ('places', place_rows)):
        write_csv(output/(name+'.csv'), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('results/n22-examples'))
    parser.add_argument('--output', type=Path, default=Path('docs/examples/n22'))
    args = parser.parse_args()
    run_experiments(Config(participants=22, repeats=3, seed=42), ['elo-tire'], True, args.artifacts,
                    grin_cycle_score=True, far_tour=True, zhivora_tour=True, zhivora_fixed_top=6)
    write_examples(args.artifacts, args.output)
    print(f'Wrote three verified tournament protocols: {args.output}/index.md')


if __name__ == '__main__':
    main()
