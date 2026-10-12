import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import betaprime

from scripts.fit_rating_distribution import (diagnostics, fit_beta_prime,
                                              negative_log_likelihood, read_ratings)
from simulation import Config, generate_field
from simulation.experiments import run_experiments
from simulation.verify import verify_artifacts


class RatingCalibrationTests(unittest.TestCase):
    def test_real_data_fit_and_applied_defaults(self):
        source = Path('docs/data/real-ratings.txt')
        ratings = read_ratings(source)
        self.assertEqual(len(ratings), 187)
        self.assertEqual(ratings[-5:].tolist(), ratings[-10:-5].tolist())
        self.assertEqual((ratings.min(), ratings.max()), (763, 1607))
        fit = fit_beta_prime(ratings)
        config = Config()
        self.assertEqual(fit['parameters'], {'alpha': config.alpha, 'beta': config.beta, 'scale': config.scale})
        archived = json.loads(Path('docs/beta-prime-fit.json').read_text())
        self.assertEqual(archived['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(archived['bounds'], list(config.rating_bounds()))
        self.assertEqual(archived['parameters'], fit['parameters'])
        quality = diagnostics(ratings, fit['parameters'], config.rating_bounds())
        self.assertLess(quality['ks_distance'], .08)
        self.assertLess(abs(quality['mean']-ratings.mean()), 3)
        self.assertLess(abs(quality['std_ddof0']-ratings.std(ddof=0)), 3)
        # Even truncating the old parameters cannot reproduce the observed CDF.
        old_nll = negative_log_likelihood((2, 3, 1000), ratings, config.rating_bounds())
        self.assertLess(fit['rounded_negative_log_likelihood'], old_nll-100)
        self.assertLess(fit['rounded_negative_log_likelihood']-fit['negative_log_likelihood'], .01)

    def test_gamma_sampling_matches_independent_truncated_cdf(self):
        config = Config()
        ratings = np.sort([value for repeat in range(250) for value in generate_field(config, repeat)[0]])
        lower, upper = config.rating_bounds()
        self.assertTrue(np.all((ratings >= lower) & (ratings < upper)))
        mass = betaprime.cdf(upper, config.alpha, config.beta, scale=config.scale)-betaprime.cdf(lower, config.alpha, config.beta, scale=config.scale)
        def density(value):
            return betaprime.pdf(value, config.alpha, config.beta, scale=config.scale)/mass
        self.assertAlmostEqual(quad(density, lower, upper)[0], 1, places=9)
        cdf = (betaprime.cdf(ratings, config.alpha, config.beta, scale=config.scale)-betaprime.cdf(lower, config.alpha, config.beta, scale=config.scale))/mass
        n = len(ratings)
        distance = max(np.max(np.arange(1, n+1)/n-cdf), np.max(cdf-np.arange(n)/n))
        self.assertLess(distance, .025)

    def test_configurable_bounds_and_legacy_mode(self):
        for distribution in ('beta-prime', 'uniform'):
            config = Config(distribution=distribution)
            self.assertEqual(config.rating_bounds(), (600, 1900))
            changed = replace(config, rating_min=700, rating_max=1500)
            self.assertEqual(changed.rating_bounds(), (700, 1500))
            for repeat in range(10):
                self.assertTrue(all(700 <= value < 1500 for value in generate_field(changed, repeat)[0]))
        self.assertIsNone(Config(distribution='weibull').rating_bounds())
        self.assertIsNone(Config(unbounded_beta_prime=True).rating_bounds())
        for config in (Config(unbounded_beta_prime='yes'),
                       Config(unbounded_beta_prime=True, rating_min=600, rating_max=1900),
                       Config(distribution='uniform', unbounded_beta_prime=True)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_invalid_calibration_inputs(self):
        ratings = read_ratings('docs/data/real-ratings.txt')
        for values, bounds in ((ratings, (0, 1900)), (ratings, (600, 600)),
                               (ratings, (600, float('inf'))), (ratings, (1000, 1900)),
                               (ratings[:10], (600, 1900)), (np.full(30, np.nan), (600, 1900))):
            with self.assertRaises(ValueError):
                fit_beta_prime(values, bounds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ratings.txt'
            for text in ('\n'.join(['1000']*20+['execute something']), '\n'.join(['1000']*20+['nan'])):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_ratings(path)

    def test_cli_supports_old_unbounded_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-m', 'simulation', '--elo', '--grin-cycle',
                                     '--participants', '2', '--repeats', '1', '--alpha', '2', '--beta', '3',
                                     '--scale', '1000', '--unbounded-beta-prime', '--output', directory],
                                    check=True, capture_output=True)
            self.assertEqual(result.returncode, 0)
            manifest = json.loads((Path(directory)/'config.json').read_text())
            self.assertIsNone(manifest['rating_bounds'])
            self.assertEqual(manifest['config']['alpha'], 2)
            self.assertTrue(manifest['config']['unbounded_beta_prime'])

    def test_precalibration_v5_exports_remain_readable(self):
        cases = (Config(participants=2, repeats=1, alpha=2, beta=3, scale=1000, unbounded_beta_prime=True),
                 Config(participants=2, repeats=1, alpha=2, beta=3, scale=1000, rating_min=900, rating_max=1300),
                 Config(participants=2, repeats=1, distribution='uniform', rating_min=800, rating_max=2400),
                 Config(participants=2, repeats=1, distribution='weibull'))
        with tempfile.TemporaryDirectory() as directory:
            for index, config in enumerate(cases):
                with self.subTest(config=config):
                    output = Path(directory)/str(index)
                    run_experiments(config, ['elo'], True, output, grin_cycle_score=True)
                    path = output/'config.json'
                    manifest = json.loads(path.read_text())
                    manifest['config'].pop('unbounded_beta_prime')
                    if config.distribution == 'uniform':
                        manifest['config']['rating_min'] = manifest['config']['rating_max'] = None
                    path.write_text(json.dumps(manifest))
                    self.assertTrue(verify_artifacts(output))
                    if index == 0:
                        manifest['rating_bounds'] = [600, 1900]
                        path.write_text(json.dumps(manifest))
                        with self.assertRaises(ValueError):
                            verify_artifacts(output)


if __name__ == '__main__':
    unittest.main()
