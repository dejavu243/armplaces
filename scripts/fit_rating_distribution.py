"""Fit truncated beta-prime to the supplied real ratings (SciPy is a dev dependency)."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import betaprime


def read_ratings(path):
    values = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = float(line.strip())
        except ValueError as exc:
            raise ValueError(f'Non-numeric rating on line {line_number}') from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Expected a finite positive rating on line {line_number}')
        values.append(value)
    if len(values) < 20:
        raise ValueError('Need at least 20 observed ratings for calibration')
    return np.array(values)


def negative_log_likelihood(parameters, ratings, bounds):
    alpha, beta, scale = parameters
    if not all(math.isfinite(value) and value > 0 for value in parameters):
        return math.inf
    low, high = betaprime.cdf(bounds, alpha, beta, scale=scale)
    mass = high-low
    if not math.isfinite(mass) or mass <= 0:
        return math.inf
    value = -np.sum(betaprime.logpdf(ratings, alpha, beta, scale=scale))+len(ratings)*np.log(mass)
    return float(value) if np.isfinite(value) else math.inf


def fit_beta_prime(ratings, bounds=(600., 1900.)):
    ratings = np.asarray(ratings, dtype=float)
    if (ratings.ndim != 1 or len(ratings) < 20 or not np.all(np.isfinite(ratings))
            or not all(math.isfinite(value) for value in bounds) or not 0 < bounds[0] < bounds[1]
            or np.any(ratings < bounds[0]) or np.any(ratings >= bounds[1])):
        raise ValueError('Expected at least 20 finite ratings inside finite positive bounds [min,max)')
    attempts = []
    def objective(log_parameters):
        with np.errstate(over='ignore', invalid='ignore'):
            return negative_log_likelihood(np.exp(log_parameters), ratings, bounds)
    # Multiple starts reduce sensitivity to a local optimum; location is fixed at zero.
    for start in ((80., 73., 1000.), (500., 47., 100.), (40., 100., 2800.)):
        result = minimize(objective, np.log(start), method='Nelder-Mead',
                          options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 4000})
        attempts.append({'start': list(start), 'success': bool(result.success),
                         'parameters': np.exp(result.x).tolist(),
                         'negative_log_likelihood': float(result.fun), 'iterations': int(result.nit)})
    successful = [item for item in attempts if item['success'] and math.isfinite(item['negative_log_likelihood'])]
    if not successful:
        raise ValueError('Beta-prime calibration did not converge')
    best = min(successful, key=lambda item: item['negative_log_likelihood'])
    rounded = [float(round(value, 1)) for value in best['parameters']]
    if not all(value > 0 for value in rounded):
        raise ValueError('Rounded fit is nonpositive; this dataset requires finer parameter precision')
    return {'mle_parameters': dict(zip(('alpha', 'beta', 'scale'), best['parameters'], strict=True)),
            'parameters': dict(zip(('alpha', 'beta', 'scale'), rounded, strict=True)),
            'attempts': attempts,
            'negative_log_likelihood': best['negative_log_likelihood'],
            'rounded_negative_log_likelihood': negative_log_likelihood(rounded, ratings, bounds)}


def diagnostics(ratings, parameters, bounds):
    alpha, beta, scale = (parameters[name] for name in ('alpha', 'beta', 'scale'))
    low, high = betaprime.cdf(bounds, alpha, beta, scale=scale)
    mass = high-low
    sorted_ratings = np.sort(ratings)
    cdf = (betaprime.cdf(sorted_ratings, alpha, beta, scale=scale)-low)/mass
    n = len(ratings)
    distance = max(np.max(np.arange(1, n+1)/n-cdf), np.max(cdf-np.arange(n)/n))
    probabilities = np.array([.05, .25, .5, .75, .95])
    quantiles = betaprime.ppf(low+probabilities*mass, alpha, beta, scale=scale)
    mean = quad(lambda x: x*betaprime.pdf(x, alpha, beta, scale=scale)/mass, *bounds)[0]
    variance = quad(lambda x: (x-mean)**2*betaprime.pdf(x, alpha, beta, scale=scale)/mass, *bounds)[0]
    return {'ks_distance': float(distance), 'acceptance_probability': float(mass),
            'mean': float(mean), 'std_ddof0': math.sqrt(variance),
            'quantiles': quantiles.tolist()}


def write_outputs(source, output, report, figures, ratings, bounds, fit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    old = {'alpha': 2., 'beta': 3., 'scale': 1000.}
    fit.update({'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'observations': len(ratings), 'bounds': list(bounds), 'location': 0,
                'method': 'truncated maximum likelihood; 3 deterministic Nelder-Mead starts; rounding to 0.1',
                'scipy_version': scipy.__version__, 'numpy_version': np.__version__,
                'observed': {'minimum': float(ratings.min()), 'maximum': float(ratings.max()),
                             'mean': float(ratings.mean()), 'std_ddof0': float(ratings.std(ddof=0)),
                             'quantiles': np.quantile(ratings, [.05, .25, .5, .75, .95]).tolist()},
                'diagnostics': diagnostics(ratings, fit['parameters'], bounds),
                'legacy_truncated_diagnostics': diagnostics(ratings, old, bounds)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    alpha, beta, scale = (fit['parameters'][name] for name in ('alpha', 'beta', 'scale'))
    low, high = betaprime.cdf(bounds, alpha, beta, scale=scale)
    x = np.linspace(*bounds, 600)
    new_pdf = betaprime.pdf(x, alpha, beta, scale=scale)/(high-low)
    new_cdf = (betaprime.cdf(x, alpha, beta, scale=scale)-low)/(high-low)
    old_mass = betaprime.cdf(bounds[1], 2, 3, scale=1000)-betaprime.cdf(bounds[0], 2, 3, scale=1000)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(ratings, bins='fd', density=True, alpha=.3, color='#216eac', label=f'{len(ratings)} реальных рейтингов')
    axes[0].plot(x, new_pdf, color='#216eac', label='Новый beta-prime с усечением')
    axes[0].plot(x, betaprime.pdf(x, 2, 3, scale=1000), color='#c74c32', linestyle='--', label='Прежний без границ')
    axes[0].plot(x, betaprime.pdf(x, 2, 3, scale=1000)/old_mass, color='#8a5fa8', linestyle=':', label='Прежний с теми же границами')
    axes[0].set_title('Плотность рейтингов')
    axes[0].set_xlabel('Рейтинг')
    axes[0].legend(fontsize=8)
    axes[1].step(np.sort(ratings), np.arange(1, len(ratings)+1)/len(ratings), where='post', color='#333', label='Реальные данные')
    axes[1].plot(x, new_cdf, color='#216eac', label='Новая модель')
    axes[1].plot(x, (betaprime.cdf(x, 2, 3, scale=1000)-betaprime.cdf(bounds[0], 2, 3, scale=1000))/old_mass,
                 color='#8a5fa8', linestyle=':', label='Прежняя, те же границы')
    axes[1].set_title('Эмпирическая и модельная CDF')
    axes[1].set_xlabel('Рейтинг')
    axes[1].legend(fontsize=8)
    probabilities = (np.arange(len(ratings))+.5)/len(ratings)
    expected = betaprime.ppf(low+probabilities*(high-low), alpha, beta, scale=scale)
    axes[2].scatter(expected, np.sort(ratings), s=10, color='#216eac')
    axes[2].plot(bounds, bounds, color='#666', linestyle='--')
    axes[2].set_title('Q–Q: новая модель и данные')
    axes[2].set_xlabel('Квантили модели')
    axes[2].set_ylabel('Квантили реальных рейтингов')
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle(f'Beta-prime: α={alpha:g}, β={beta:g}, scale={scale:g}; [{bounds[0]:g},{bounds[1]:g})')
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        destination = figures/f'beta-prime-calibration.{extension}'
        figure.savefig(destination, dpi=150)
        if extension == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(figure)
    parameters = fit['mle_parameters']
    lines = ['# Калибровка beta-prime по реальным рейтингам', '',
             f'Источник: [{source.name}](data/{source.name}), **{len(ratings)} наблюдений**.',
             'Все записи, включая последние пять повторяющихся значений, учтены по указанию пользователя.',
             'Копия сохраняет порядок и значения исходного файла «Для распределения.txt»; окончания строк приведены к LF.',
             f'SHA256 копии: `{fit["source_sha256"]}`.', '',
             'Прежняя генерация отношения двух гамма-величин математически соответствует beta-prime.',
             'Проблема была в неподходящих коэффициентах и неограниченной области рейтингов.',
             'При α=2, β=3, scale=1000 среднее равно 1000, стандартное отклонение ≈1414;',
             f'в данных среднее {ratings.mean():.3f}, стандартное отклонение {ratings.std(ddof=0):.3f}.', '',
             '## Метод подбора', '',
             'R = scale · G₁/G₂, G₁~Gamma(α,1), G₂~Gamma(β,1), location=0.',
             f'Рабочий диапазон **[{bounds[0]:g},{bounds[1]:g})** выбран пользователем; это границы эксперимента,',
             'а не оценка универсальной шкалы рейтингов. Значения вне диапазона отбрасываются и выбираются заново.',
             'Рейтинги не обрезаются к границе и не растягиваются. Наблюдаемые минимум и максимум',
             f'равны {ratings.min():g} и {ratings.max():g}; они не подставлены в качестве ограничений генерации.', '',
             'Минимизируется отрицательное логарифмическое правдоподобие **усечённого** распределения:', '',
             '```text', 'NLL = −Σ log f(Rᵢ; α,β,scale) + n·log[F(U)−F(L)]',
             'f(R) = (R/scale)^(α−1) · (1+R/scale)^(−α−β) / [scale·B(α,β)]', '```', '',
             'Положительность параметров обеспечена оптимизацией их логарифмов. Проверены три',
             'детерминированные начальные точки методом Nelder–Mead; сохранены статусы сходимости.',
             'Коэффициенты округлены до 0.1 для устойчивого воспроизведения; исходная оценка сохранена в JSON.', '',
             '| Параметр | MLE | Применённое значение |', '|---|---:|---:|',
             f'| alpha | {parameters["alpha"]:.6f} | {alpha:g} |',
             f'| beta | {parameters["beta"]:.6f} | {beta:g} |',
             f'| scale | {parameters["scale"]:.6f} | {scale:g} |', '',
             f'Потеря NLL при округлении: {fit["rounded_negative_log_likelihood"]-fit["negative_log_likelihood"]:.6f}.',
             f'Вероятность принятия кандидата: {fit["diagnostics"]["acceptance_probability"]:.8f}.', '',
             '## Соответствие данным', '',
             '| Квантиль | Реальные рейтинги | Новая модель | Прежняя с теми же границами |', '|---|---:|---:|---:|']
    for index, probability in enumerate((5, 25, 50, 75, 95)):
        lines.append(f'| {probability}% | {fit["observed"]["quantiles"][index]:.3f} | '
                     f'{fit["diagnostics"]["quantiles"][index]:.3f} | '
                     f'{fit["legacy_truncated_diagnostics"]["quantiles"][index]:.3f} |')
    lines += ['', f'Расстояние эмпирической CDF до новой модели: {fit["diagnostics"]["ks_distance"]:.6f};',
              f'до прежней с теми же границами: {fit["legacy_truncated_diagnostics"]["ks_distance"]:.6f}.',
              'Это описательная оценка на данных подбора, не независимая проверка и не p-value.',
              'Связи между записями и происхождение групп в исходном файле не известны;',
              'калибровка описывает именно предоставленную выборку.', '',
              '![Калибровка beta-prime](figures/beta-prime-calibration.png)', '',
              '## Воспроизведение и диапазоны', '',
              '```bash', '.venv/bin/python -m pip install -r requirements-dev.txt',
              f'.venv/bin/python scripts/fit_rating_distribution.py --rating-min {bounds[0]:g} --rating-max {bounds[1]:g}',
              '.venv/bin/python -m simulation --elo --elo-tire --grin-cycle --grin-elo \\',
              f'  --distribution beta-prime --alpha {alpha:g} --beta {beta:g} --scale {scale:g} --rating-min {bounds[0]:g} --rating-max {bounds[1]:g}',
              '# Историческая неограниченная модель:',
              '.venv/bin/python -m simulation --elo --grin-cycle --alpha 2 --beta 3 --scale 1000 --unbounded-beta-prime',
              '```', '',
              f'SciPy {scipy.__version__}, NumPy {np.__version__}. Подбор выполняется отдельно от моделирования турниров.',
              '[JSON с параметрами и диагностикой](beta-prime-fit.json).',
              '[Формула beta-prime](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.betaprime.html),',
              '[методы оптимизации](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.minimize.html).', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('docs/data/real-ratings.txt'))
    parser.add_argument('--rating-min', type=float, default=600.)
    parser.add_argument('--rating-max', type=float, default=1900.)
    parser.add_argument('--output', type=Path, default=Path('docs/beta-prime-fit.json'))
    parser.add_argument('--report', type=Path, default=Path('docs/beta-prime-calibration.md'))
    parser.add_argument('--figures', type=Path, default=Path('docs/figures'))
    args = parser.parse_args()
    ratings = read_ratings(args.input)
    bounds = (args.rating_min, args.rating_max)
    fit = fit_beta_prime(ratings, bounds)
    write_outputs(args.input, args.output, args.report, args.figures, ratings, bounds, fit)
    print(json.dumps({'observations': len(ratings), 'bounds': bounds, 'parameters': fit['parameters'],
                      'ks_distance': fit['diagnostics']['ks_distance']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
