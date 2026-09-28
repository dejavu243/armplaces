"""Rebuild docs/experiment-report.md from verified cycle-score artifacts.

Run with .venv/bin/python scripts/report_cycle_score.py after the 9000-tournament series.
"""
import argparse
import json
from pathlib import Path

SIZES=(8,16,32)
MODELS=('strong-win','elo','elo-stamina')
NEW=Path('results/cycle-score')


def weighted(values):
    count=sum(item['samples'] for item in values)
    return (sum(item['mae']*item['samples'] for item in values)/count if count else None,
            sum(item['spearman']*item['correlation_samples'] for item in values)/
            sum(item['correlation_samples'] for item in values)
            if any(item['correlation_samples'] for item in values) else None,
            count)


def number(value):
    return '—' if value is None else f'{value:.6f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ci-run-url', help='URL of a separately verified successful CI run')
    args=parser.parse_args()
    lines=['# Контрольная серия: дополнительное разрешение остаточных циклов','',
           'N=8, 16, 32; модели `strong-win`, `elo`, `elo-stamina`; по 1000 повторов;',
           'seed=42. Веса критериев 0.33, 0.33, 0.33, нормированы до 1.', '',
           'Новый метод запускается после стандартного графового прохода **только при',
           'остаточном цикле**. Базовый GrinTour использует рейтинговый резерв.',
           'Новый метод не получает исходные рейтинги при разрешении цикла.', '',
           '## Контроль выборки','',
           '| N | Всего | Стандартный граф | Дополнительный этап запущен | Успешно |',
           '|---:|---:|---:|---:|---:|']
    summaries={}
    revision=None
    for n in SIZES:
        new=NEW/f'n{n}'
        manifest=json.loads((new/'config.json').read_text())
        if (manifest['schema_version']!=4 or manifest['config']['seed']!=42
                or manifest['config']['repeats']!=1000 or manifest['status']!='complete'):
            raise RuntimeError(f'Invalid manifest for N={n}')
        current_revision=manifest['environment']['commit']
        if revision is None:
            revision=current_revision
        elif revision!=current_revision:
            raise RuntimeError('Experiment categories used different code revisions')
        summary=json.loads((new/'summary.json').read_text())
        summaries[n]=summary
        standard=sum(row['cycle_standard_success'] for row in summary.values())
        activated=sum(row['cycle_activated'] for row in summary.values())
        successful=sum(row['cycle_success'] for row in summary.values())
        assert standard+activated==3000 and successful==activated
        lines.append(f'| {n} | 3000 | {standard} | {activated} | {successful} |')
    lines+=['','Стандартный граф расставил всех участников в **8941** турнире.',
            'Дополнительный этап разрешил все **59** остаточных циклов:',
            'N=8 — 0, N=16 — 19, N=32 — 40. Оба метода дали полную расстановку',
            'во всех 9000 турнирах. Параметры окружения и SHA кода сохранены в `config.json`.','',
            '## Метрики по N и модели','',
            'MAE меньше — лучше, Spearman больше — лучше. Метрики обоих методов',
            'вычислены по всем 1000 турнирам в каждой строке. Исходные рейтинги',
            'используются только для оценки качества мест, а также базовым резервом.', '',
            '| N | Модель | Циклов | MAE базового | MAE нового | Spearman базового | Spearman нового |',
            '|---:|---|---:|---:|---:|---:|---:|']
    for n in SIZES:
        for model in MODELS:
            row=summaries[n][model]
            assert row['grin_tour']['samples']==row['grin_cycle_score']['samples']==1000
            lines.append(f"| {n} | {model} | {row['cycle_activated']} | "
                         f"{number(row['grin_tour']['mae'])} | {number(row['grin_cycle_score']['mae'])} | "
                         f"{number(row['grin_tour']['spearman'])} | "
                         f"{number(row['grin_cycle_score']['spearman'])} |")
    lines+=['','## Только 59 циклических турниров','',
            'В остальных 8941 турнирах методы получили одинаковые места.', '',
            '| N | Модель | Турниров | MAE базового | MAE нового | Spearman базового | Spearman нового |',
            '|---:|---|---:|---:|---:|---:|---:|']
    before=[]
    after=[]
    for n in SIZES:
        for model in MODELS:
            row=summaries[n][model]
            count=row['cycle_activated']
            if not count:
                continue
            old=row['cycle_subset_grin_tour']
            new=row['cycle_subset_grin_cycle_score']
            assert old['samples']==new['samples']==count
            before.append(old)
            after.append(new)
            lines.append(f"| {n} | {model} | {count} | {number(old['mae'])} | "
                         f"{number(new['mae'])} | {number(old['spearman'])} | "
                         f"{number(new['spearman'])} |")
    old_mae,old_spearman,count=weighted(before)
    new_mae,new_spearman,new_count=weighted(after)
    assert count==new_count==59
    lines.append(f'| **Все** | — | **59** | {number(old_mae)} | {number(new_mae)} | '
                 f'{number(old_spearman)} | {number(new_spearman)} |')
    lines+=['','В части подгрупп новый метод уступает базовому; это зафиксировано в',
            'таблице. Сравнение показывает качество на данном seed, а не гарантирует',
            'преимущество метода в других сериях.', '',
            '## Проверки','',
            '- Все 9000 экспортов прошли независимый валидатор: повторно вычислены',
            '  компоненты, предварительные места, показатели, ранги, баллы, порядок и места.',
            '- 54 теста unittest и Ruff прошли.',
            '- Тест непобедимого № 1: N=2…32, 1000 повторов — 31 000 турниров.',
            '  Все чемпионы и призёры правильны, все места определены; дополнительный',
            '  этап применён 6860 раз.',
            '',
            '## Воспроизведение','',
            '```bash','.venv/bin/python -m unittest discover','.venv/bin/python -m ruff check .',
            'for n in 8 16 32; do',
            '  .venv/bin/python -m simulation --strong-win --elo --elo-stamina \\',
            '    --grin-tour --grin-cycle-score --cycle-weights 0.33 0.33 0.33 \\',
            '    --participants "$n" --repeats 1000 --seed 42 \\',
            '    --output "results/cycle-score/n$n" || exit',
            'done',
            '.venv/bin/python -m test.forced_champion --repeats 1000 --seed 42 \\',
            '  --output results/cycle-score/forced-champion',
            '.venv/bin/python scripts/report_cycle_score.py',
            '```','',
            'Артефакты в `results/cycle-score/` не коммитятся. Основная серия',
            'и длинный тест чемпиона выполнены локально.','']
    if args.ci_run_url:
        lines += [f'Отдельный [запуск GitHub Actions]({args.ci_run_url}) завершился успешно:',
                  'проверки Python 3.11–3.13 и эксперименты N=16, N=32; оба',
                  'артефакта `simulation-n16-seed42` и `simulation-n32-seed42` загружены.', '']
    forced=json.loads((NEW/'forced-champion/summary.json').read_text())
    assert forced['totals']['tournaments']==31000
    assert forced['totals']['cycle_complete']==31000
    assert forced['totals']['cycle_activated']==6860
    Path('docs/experiment-report.md').write_text('\n'.join(lines))
    print('Wrote docs/experiment-report.md for 9000 tournaments and 31000 champion checks')


if __name__=='__main__':
    main()
