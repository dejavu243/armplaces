import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_formula_experiment import read_category


class FormulaExperimentTests(unittest.TestCase):
    def test_three_examples_and_multiple_sizes_export_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, report = root/'results', root/'docs/report.md'
            subprocess.run([sys.executable, 'scripts/run_formula_experiment.py',
                            '--sizes', '2', '3', '--repeats', '2', '--output', str(output),
                            '--report', str(report)], check=True, capture_output=True)
            data = json.loads((root/'docs/data/report.json').read_text())
            self.assertEqual(len(data['examples']), 3)
            self.assertEqual(len(data['categories']), 6)
            self.assertIn('все турниры соответствующей категории (2)', report.read_text())
            self.assertNotIn('все 1000 турниров', report.read_text())
            for item in data['categories']:
                config = item['manifest']['config']
                self.assertEqual((config['dr'], config['tire_reference'], config['tire_c'],
                                  config['tire_bouts'], config['tire_exponent_sign']),
                                 (167, .85, 334, 3, 1))
                folder = output/item['distribution']/f'n{item["participants"]}'
                _, results, rows, summary = read_category(folder)
                self.assertEqual(summary, item['statistics'])
                self.assertEqual(len(results), 2)
                with (folder/'finals.csv').open(newline='') as stream:
                    saved = list(csv.DictReader(stream))
                self.assertEqual(len(saved), 2)
                self.assertEqual([float(row['gap']) for row in saved], [row['gap'] for row in rows])
                self.assertEqual(summary['final']['samples'], 2)
                self.assertEqual(summary['reset_rate'], sum(row['reset'] for row in rows)/2)
            for distribution in ('beta-prime', 'weibull', 'uniform'):
                example = root/f'docs/report-n22-{distribution}.md'
                self.assertIn('R(S)=R₀·0.85^(S/1002)', example.read_text())
            self.assertGreater((root/'docs/figures/report-finals.png').stat().st_size, 0)
            self.assertGreater((root/'docs/figures/report-finals.svg').stat().st_size, 0)
