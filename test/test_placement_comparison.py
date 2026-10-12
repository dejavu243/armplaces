import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simulation import Config
from simulation.comparison_statistics import individual_metrics, metric_details, paired_comparisons
from simulation.experiments import run_experiments
from simulation.verify import verify_artifacts


class ComparisonMetricTests(unittest.TestCase):
    def test_hand_calculated_individual_metrics_and_ties(self):
        self.assertEqual(individual_metrics([100, 100, 50], {0: 2, 1: 1, 2: 3}),
                         {'mae': 2/3, 'spearman': .5})
        details = metric_details([0., 1., 2.])
        self.assertEqual(details['mae_std'], 1)
        self.assertAlmostEqual(details['mae_se'], 1/3**.5)
        self.assertAlmostEqual(details['mae_p95'], 1.9)
        self.assertEqual(details['mae_max'], 2)
        self.assertEqual(details['perfect_order_rate'], 1/3)
        self.assertIsNone(metric_details([])['mae_ci95'])
        self.assertIsNone(metric_details([0.])['mae_ci95'])

    def test_paired_statistics_use_only_common_repeats(self):
        result = paired_comparisons({'a': [0., 1., None, 3.], 'b': [1., 0., 2., 3.]})['a__vs__b']
        self.assertEqual((result['samples'], result['skipped']), (3, 1))
        self.assertEqual((result['left_better'], result['tied'], result['right_better']), (1, 1, 1))
        self.assertEqual(result['mean_difference'], 0)
        self.assertAlmostEqual(result['std_difference'], 1)
        self.assertAlmostEqual(result['ci95'][1], 1.96/3**.5)


class ComparisonArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)/'source'
        run_experiments(Config(participants=16, repeats=3), ['elo-tire'], True, cls.path,
                        grin_cycle_score=True, far_tour=True, zhivora_tour=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_export_schema_and_all_methods(self):
        self.assertTrue(verify_artifacts(self.path))
        manifest = json.loads((self.path/'config.json').read_text())
        self.assertEqual(manifest['schema_version'], 6)
        self.assertEqual(manifest['zhivora_fixed_top'], 4)
        summary = json.loads((self.path/'summary.json').read_text())['elo-tire']
        self.assertEqual(len(summary['paired_comparisons']), 6)
        for internal, name in manifest['ranking_names'].items():
            self.assertEqual(summary[internal]['samples'], 3)
            self.assertEqual(summary['place_deviations'][name]['skipped'], 0)
            for extension in ('csv', 'png', 'svg'):
                self.assertTrue((self.path/f'place-deviations-elo-tire-{name}.{extension}').is_file())
        with (self.path/'ranking-metrics.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)

    def test_corrupt_new_places_rejected_after_csv_is_also_changed(self):
        for internal, prefix in (('far_tour', 'far'), ('zhivora_tour', 'zhivora')):
            with self.subTest(method=internal), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                shutil.copytree(self.path, path, dirs_exist_ok=True)
                records = [json.loads(line) for line in (path/'tournaments.jsonl').read_text().splitlines()]
                places = records[0][internal]['places']
                names = [who for who, place in places.items() if place > 4]
                a, b = names[:2]
                places[a], places[b] = places[b], places[a]
                (path/'tournaments.jsonl').write_text('\n'.join(json.dumps(record) for record in records)+'\n')
                file = path/'participants.csv'
                with file.open() as stream:
                    reader = csv.DictReader(stream)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    if row['repeat'] == '0' and row['participant'] in (a, b):
                        row[prefix+'_place'] = places[row['participant']]
                with file.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, 'independently replayed'):
                    verify_artifacts(path)

    def test_corrupt_pair_confidence_or_tournament_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shutil.copytree(self.path, path, dirs_exist_ok=True)
            file = path/'summary.json'
            original = file.read_text()
            summary = json.loads(original)
            next(iter(summary['elo-tire']['paired_comparisons'].values()))['ci95'][0] += .01
            file.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'paired confidence'):
                verify_artifacts(path)
            file.write_text(original)
            file = path/'ranking-metrics.csv'
            with file.open() as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]['mae'] = str(float(rows[0]['mae'])+.1)
            with file.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'per-tournament mae'):
                verify_artifacts(path)

    def test_singleton_and_pascal_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            run_experiments(Config(participants=1, repeats=2), ['elo-tire'], True, directory,
                            grin_cycle_score=True, far_tour=True, zhivora_tour=True)
            self.assertTrue(verify_artifacts(directory))
        with tempfile.TemporaryDirectory() as directory:
            run_experiments(Config(participants=16, repeats=1), ['elo-tire'], True, directory,
                            grin_cycle_score=True, far_tour=True, zhivora_tour=True, zhivora_fixed_top=6)
            self.assertTrue(verify_artifacts(directory))

    def test_cli_flags_and_legacy_schema_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            subprocess.run([sys.executable, '-m', 'simulation', '--elo-tire', '--far-tour',
                            '--zhivora-tour', '--grin-tour', '--grin-cycle-score', '--participants', '8',
                            '--repeats', '1', '--output', str(path/'new')], check=True, capture_output=True)
            self.assertTrue(verify_artifacts(path/'new'))
            run_experiments(Config(participants=2, repeats=1), ['elo'], True, path/'legacy')
            self.assertEqual(json.loads((path/'legacy/config.json').read_text())['schema_version'], 5)
            self.assertTrue(verify_artifacts(path/'legacy'))


if __name__ == '__main__':
    unittest.main()
