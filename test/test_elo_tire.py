import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simulation import Config, generate_field, simulate, tire_effective_rating, validate_result
from simulation.core import MAX_RATING_CANDIDATES
from simulation.experiments import run_experiments
from simulation.place_statistics import place_errors, summarize_places
from simulation.verify import verify_artifacts
from test.test_simulation import FixedRandom


class DistributionTests(unittest.TestCase):
    def test_original_generation_is_unchanged(self):
        config = Config(participants=17)
        rng = np.random.default_rng(np.random.SeedSequence([42, 3, 0]))
        expected = 1000*(rng.gamma(2, size=17)/rng.gamma(3, size=17))
        ratings, draw = generate_field(config, 3)
        self.assertEqual(ratings, expected.tolist())
        expected_draw = np.random.default_rng(np.random.SeedSequence([42, 3, 1])).permutation(17)
        self.assertEqual(draw, expected_draw.tolist())

    def test_distributions_bounds_and_repeatability(self):
        for distribution in ('beta-prime', 'weibull', 'uniform'):
            config = Config(distribution=distribution, rating_min=900, rating_max=1300)
            for repeat in range(3):
                field = generate_field(config, repeat)
                self.assertEqual(field, generate_field(config, repeat))
                self.assertTrue(all(900 <= value < 1300 for value in field[0]))
                self.assertEqual(field[1], generate_field(Config(), repeat)[1])
        self.assertEqual(Config(distribution='uniform').rating_bounds(), (800, 2400))
        self.assertTrue(all(800 <= value < 2400 for value in generate_field(Config(distribution='uniform'), 0)[0]))
        self.assertTrue(all(value > 852 for value in generate_field(Config(distribution='weibull'), 0)[0]))

    def test_truncation_rejects_instead_of_clipping_or_rescaling(self):
        config = Config(participants=2, rating_min=1, rating_max=3)
        with patch('simulation.core.sample_ratings', return_value=np.array([.5, 1.5, 2.5, 4.])):
            self.assertEqual(generate_field(config, 0)[0], [1.5, 2.5])

    def test_candidate_budget(self):
        drawn = []
        def outside(config, rng, size):
            drawn.append(size)
            return np.ones(size)
        with patch('simulation.core.sample_ratings', side_effect=outside):
            with self.assertRaisesRegex(ValueError, 'too improbable'):
                generate_field(Config(rating_min=2, rating_max=3), 0)
        self.assertEqual(sum(drawn), MAX_RATING_CANDIDATES)

    def test_invalid_parameters(self):
        for config in (Config(distribution='unknown'), Config(rating_min=800),
                       Config(rating_min=0, rating_max=1000), Config(rating_min=1000, rating_max=800),
                       Config(rating_min=800, rating_max=float('inf')),
                       Config(distribution='weibull', rating_min=100, rating_max=852),
                       Config(weibull_shape=0), Config(weibull_scale=0), Config(weibull_location=-1),
                       Config(tire_floor=.9, tire_reference=.85), Config(tire_reference=1.1),
                       Config(tire_floor=0), Config(tire_c=0), Config(tire_bouts=0), Config(tire_bouts=1.5)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


class TireTests(unittest.TestCase):
    def test_curve_control_points_and_limit(self):
        config = Config()
        self.assertEqual(tire_effective_rating(1000, 0, config), 1000)
        self.assertAlmostEqual(tire_effective_rating(1000, 1200, config), 900)
        self.assertAlmostEqual(tire_effective_rating(1000, 1e300, config), 850)
        values = [tire_effective_rating(1000, s, config) for s in (0, 400, 800, 1200, 5000)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_equal_bouts_and_signed_load(self):
        config = Config(participants=2)
        result = simulate(config, 'elo-tire', [1000, 1000], [0, 1], rng=FixedRandom([.1, .9, .1]))
        self.assertTrue(result['reset'])
        for index, bout in enumerate(result['bouts']):
            self.assertEqual(bout['p_a'], .5)
            self.assertEqual(bout['fatigue_before_a'], 400*index)
            self.assertEqual(bout['fatigue_after_a'], 400*(index+1))
            self.assertEqual(bout['fatigue_after_a'], bout['fatigue_after_b'])
        forward = simulate(config, 'elo-tire', [1000, 1100], [0, 1], rng=FixedRandom([.1]*3))
        backward = simulate(config, 'elo-tire', [1000, 1100], [1, 0], rng=FixedRandom([.1]*3))
        self.assertEqual(forward['bouts'][0]['fatigue_after_a'], 500)
        self.assertEqual(backward['bouts'][0]['fatigue_after_a'], 300)

    def test_disabled_fatigue_matches_elo_and_all_sizes(self):
        for n in range(1, 33):
            config = Config(participants=n, tire_reference=1)
            ratings, draw = generate_field(config, 2)
            elo = simulate(config, 'elo', ratings, draw, repeat=2)
            tire = simulate(config, 'elo-tire', ratings, draw, repeat=2)
            self.assertEqual(elo['places'], tire['places'])
            self.assertEqual(elo['new_ratings'], tire['new_ratings'])
            for a, b in zip(elo['bouts'], tire['bouts'], strict=True):
                self.assertEqual({key: value for key, value in a.items() if not key.startswith('fatigue_')},
                                 {key: value for key, value in b.items() if not key.startswith('fatigue_')})
            validate_result(simulate(replace(config, tire_reference=.9), 'elo-tire', ratings, draw, repeat=2))

    def test_hand_calculated_place_statistics_and_rating_ties(self):
        self.assertEqual(place_errors([100, 100, 50], {0: 2, 1: 1, 2: 3}), [1, -1, 0])
        samples = [place_errors([300, 200, 100], places) for places in
                   ({0: 2, 1: 1, 2: 3}, {0: 3, 1: 2, 2: 1})]
        rows = summarize_places(samples, 3)
        self.assertEqual(rows[0], {'place': 1, 'variance': .25, 'mean_abs_deviation': 1.5,
                                   'max_down': 2, 'max_up': 0})
        self.assertEqual(rows[1]['max_up'], 1)
        self.assertEqual(rows[2]['variance'], 1)
        self.assertIsNone(summarize_places([], 1)[0]['variance'])


class TireArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)/'source'
        run_experiments(Config(participants=4, repeats=2), ['elo', 'elo-tire'], True, cls.path,
                        grin_cycle_score=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exports(self):
        self.assertTrue(verify_artifacts(self.path))
        manifest = json.loads((self.path/'config.json').read_text())
        self.assertEqual(manifest['schema_version'], 5)
        self.assertEqual(manifest['ranking_names']['grin_cycle_score'], 'grin-cycle')
        for model in ('elo', 'elo-tire'):
            for method in ('grin-elo', 'grin-cycle'):
                for extension in ('csv', 'png', 'svg'):
                    self.assertGreater((self.path/f'place-deviations-{model}-{method}.{extension}').stat().st_size, 0)

    def test_corrupt_bouts_and_statistics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'corrupt'
            shutil.copytree(self.path, path)
            bouts = path/'bouts.csv'
            original = bouts.read_text()
            for field in ('fatigue_after_a', 'effective_a', 'p_a'):
                with bouts.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields, rows = reader.fieldnames, list(reader)
                row = next(row for row in rows if row['model'] == 'elo-tire')
                row[field] = str(float(row[field])+.01)
                with bouts.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises((ValueError, RuntimeError)):
                    verify_artifacts(path)
                bouts.write_text(original)
            summary = json.loads((path/'summary.json').read_text())
            summary['elo-tire']['place_deviations']['grin-cycle']['places'][0]['variance'] += 1
            (path/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'per-place'):
                verify_artifacts(path)

    def test_schema_four_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy'
            shutil.copytree(self.path, path)
            manifest = json.loads((path/'config.json').read_text())
            manifest['schema_version'] = 4
            manifest['models'] = ['elo']
            old_fields = ('participants', 'repeats', 'seed', 'alpha', 'beta', 'scale', 'dr', 'k', 'fatigue_rate')
            manifest['config'] = {key: manifest['config'][key] for key in old_fields}
            manifest.pop('ranking_names')
            manifest.pop('rating_bounds')
            (path/'config.json').write_text(json.dumps(manifest))
            summary = json.loads((path/'summary.json').read_text())['elo']
            summary.pop('place_deviations')
            summary.pop('rating_fallback')
            (path/'summary.json').write_text(json.dumps({'elo': summary}))
            for name in ('participants.csv', 'bouts.csv'):
                with (path/name).open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields = [key for key in reader.fieldnames if not key.startswith('fatigue_')]
                    rows = [{key: row[key] for key in fields} for row in reader if row['model'] == 'elo']
                with (path/name).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerows(rows)
            records = [line for line in (path/'tournaments.jsonl').read_text().splitlines()
                       if json.loads(line)['model'] == 'elo']
            (path/'tournaments.jsonl').write_text('\n'.join(records)+'\n')
            self.assertTrue(verify_artifacts(path))

    def test_corrupt_csv_and_coverage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'corrupt'
            shutil.copytree(self.path, path)
            table = path/'place-deviations-elo-tire-grin-cycle.csv'
            original = table.read_text()
            lines = original.splitlines()
            row = lines[1].split(',')
            row[3] = str(float(row[3])+1)
            lines[1] = ','.join(row)
            table.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError, 'per-place'):
                verify_artifacts(path)
            table.write_text(original)
            summary = json.loads((path/'summary.json').read_text())
            summary['elo-tire']['place_deviations']['grin-cycle']['skipped'] += 1
            (path/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                verify_artifacts(path)

    def test_new_flags_are_exact_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'new', Path(directory)/'old'
            command = [sys.executable, '-m', 'simulation', '--elo-tire', '--participants', '2', '--repeats', '1',
                       '--tire-bouts', '4', '--tire-floor', '.8', '--tire-reference', '.95', '--tire-c', '300',
                       '--distribution', 'weibull', '--weibull-shape', '2.5', '--weibull-scale', '300',
                       '--weibull-location', '700', '--rating-min', '800', '--rating-max', '1600']
            subprocess.run(command+['--grin-elo', '--grin-cycle', '--output', str(a)], check=True, capture_output=True)
            subprocess.run(command+['--grin-tour', '--grin-cycle-score', '--output', str(b)], check=True, capture_output=True)
            for name in ('summary.json', 'participants.csv', 'bouts.csv', 'tournaments.jsonl'):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            manifest = json.loads((a/'config.json').read_text())
            self.assertEqual(manifest['config']['tire_bouts'], 4)
            self.assertEqual(manifest['rating_bounds'], [800, 1600])

    def test_cli_rejects_incomplete_bounds(self):
        result = subprocess.run([sys.executable, '-m', 'simulation', '--elo-tire', '--rating-min', '800'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('supplied together', result.stderr)
