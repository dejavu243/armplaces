import csv
import json
from math import isclose
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simulation.verify import verify_artifacts


class N22ExportTests(unittest.TestCase):
    def assert_fatigue_calculation(self, row):
        difference = abs(float(row['effective_a'])-float(row['effective_b']))
        expected_increment = 334-difference if difference < 334 else 0.
        self.assertAlmostEqual(float(row['rating_gap_abs']), difference)
        self.assertAlmostEqual(float(row['fatigue_increment']), expected_increment)
        for side in ('a', 'b'):
            before, after = (float(row[f'fatigue_{when}_{side}']) for when in ('before', 'after'))
            self.assertTrue(isclose(after-before, expected_increment, rel_tol=1e-12, abs_tol=1e-9))
            self.assertAlmostEqual(float(row['rating_exponent_'+side]), before/1002)
            expected_rating = float(row['rating_initial_'+side])*.85**(before/1002)
            self.assertTrue(isclose(expected_rating, float(row['effective_'+side]), rel_tol=1e-12, abs_tol=1e-9))
            if difference >= 334:
                self.assertEqual(before, after)

    def test_export_preserves_verified_bouts_and_four_rankings_with_one_based_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts, output = Path(directory)/'artifacts', Path(directory)/'export'
            subprocess.run([sys.executable, 'scripts/render_n22_examples.py', '--artifacts', str(artifacts),
                            '--output', str(output)], check=True, capture_output=True)
            self.assertTrue(verify_artifacts(artifacts))
            manifest = json.loads((artifacts/'config.json').read_text())
            self.assertEqual((manifest['config']['tire_reference'], manifest['config']['tire_c'],
                              manifest['config']['dr'], manifest['config']['tire_exponent_sign']),
                             (.85, 334, 167, 1))
            records = [json.loads(line) for line in (artifacts/'tournaments.jsonl').read_text().splitlines()]
            tables = {}
            for name in ('ratings', 'places', 'bouts', 'finals'):
                with (output/f'{name}.csv').open(newline='') as stream:
                    tables[name] = list(csv.DictReader(stream))
            with (artifacts/'bouts.csv').open(newline='') as stream:
                raw_bouts = list(csv.DictReader(stream))
            self.assertEqual(len(tables['ratings']), 66)
            self.assertEqual(len(tables['places']), 66)
            self.assertEqual(len(tables['bouts']), len(raw_bouts))
            self.assertEqual(len(tables['finals']), 3)
            for record in records:
                example = str(record['repeat']+1)
                places = [row for row in tables['places'] if row['example'] == example]
                ratings = [row for row in tables['ratings'] if row['example'] == example]
                self.assertEqual(sorted(int(row['number']) for row in ratings), list(range(1, 23)))
                for method in ('far_tour', 'zhivora_tour', 'grin_cycle_score', 'grin_tour'):
                    actual = {str(int(row[method])-1): int(row['place']) for row in places}
                    self.assertEqual(actual, record[method]['places'])
                final = next(row for row in tables['finals'] if row['example'] == example)
                self.assertEqual(int(final['champion']), record['champion']+1)
                self.assertEqual(final['reset'], str(record['reset']))
                self.assertTrue((output/f'example-{example}.md').is_file())
            for raw, exported in zip(raw_bouts, tables['bouts'], strict=True):
                self.assert_fatigue_calculation(exported)
                for field in ('match', 'a', 'b', 'winner', 'loser'):
                    self.assertEqual(int(exported[field]), int(raw[field])+1)
                for field in ('p_a', 'effective_a', 'effective_b', 'fatigue_after_a', 'fatigue_after_b'):
                    self.assertEqual(exported[field], raw[field])
            html = (output/'all-examples.html').read_text()
            self.assertIn('R_eff=R₀·0.85^(S/1002)', html)
            self.assertIn('Деление S на N_f·C=3·334=1002 применяется только', html)
            self.assertIn('Расчёт накопленной усталости', html)
            second_bout = tables['bouts'][1]
            self.assertAlmostEqual(float(second_bout['fatigue_increment']), 72.15984771686931)
            for example in range(1, 4):
                self.assertIn(f'id="example-{example}"', html)
            self.assertIn('[финалы](finals.csv)', (output/'index.md').read_text())

    def test_ten_examples_for_seventeen_and_twenty_seven_participants(self):
        for n in (17, 27):
            with self.subTest(participants=n), tempfile.TemporaryDirectory() as directory:
                artifacts, output = Path(directory)/'artifacts', Path(directory)/'export'
                subprocess.run([sys.executable, 'scripts/render_n22_examples.py', '--participants', str(n),
                                '--repeats', '10', '--artifacts', str(artifacts), '--output', str(output)],
                               check=True, capture_output=True)
                self.assertTrue(verify_artifacts(artifacts))
                manifest = json.loads((artifacts/'config.json').read_text())
                self.assertEqual(manifest['config']['participants'], n)
                self.assertEqual(manifest['config']['repeats'], 10)
                self.assertEqual(manifest['zhivora_fixed_top'], 6)
                records = [json.loads(line) for line in (artifacts/'tournaments.jsonl').read_text().splitlines()]
                self.assertEqual(len(records), 10)
                for record in records:
                    for method in ('far_tour', 'zhivora_tour', 'grin_cycle_score', 'grin_tour'):
                        self.assertEqual(sorted(record[method]['places'].values()), list(range(1, n+1)))
                for name in ('ratings', 'places'):
                    with (output/f'{name}.csv').open(newline='') as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 10*n)
                with (output/'finals.csv').open(newline='') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 10)
                with (output/'bouts.csv').open(newline='') as stream:
                    bouts = list(csv.DictReader(stream))
                self.assertEqual(len(bouts), 10*(2*n-2)+sum(record['reset'] for record in records))
                self.assertTrue(any(float(row['fatigue_increment']) > 100 for row in bouts))
                for row in bouts:
                    self.assert_fatigue_calculation(row)
                html = (output/'all-examples.html').read_text()
                for example in range(1, 11):
                    self.assertIn(f'id="example-{example}"', html)
                    self.assertTrue((output/f'example-{example}.md').is_file())
                self.assertIn(f'Примеры турниров для {n} участников', html)
