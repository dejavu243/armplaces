import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simulation.verify import verify_artifacts


class N22ExportTests(unittest.TestCase):
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
                for field in ('match', 'a', 'b', 'winner', 'loser'):
                    self.assertEqual(int(exported[field]), int(raw[field])+1)
                for field in ('p_a', 'effective_a', 'effective_b', 'fatigue_after_a', 'fatigue_after_b'):
                    self.assertEqual(exported[field], raw[field])
            html = (output/'all-examples.html').read_text()
            self.assertIn('R_eff=R₀·0.85^(S/1002)', html)
            for example in range(1, 4):
                self.assertIn(f'id="example-{example}"', html)
            self.assertIn('[финалы](finals.csv)', (output/'index.md').read_text())
