import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simulation import Config
from simulation.experiments import rank_metrics, run_experiments
from simulation.verify import verify_artifacts


class ExperimentTests(unittest.TestCase):
    def test_all_flags_exports_and_order_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'
            command = [sys.executable, '-m', 'simulation', '--participants', '5', '--repeats', '3', '--grin-tour']
            subprocess.run(command+['--strong-win', '--elo', '--elo-stamina', '--output', str(a)], check=True, capture_output=True)
            subprocess.run(command+['--elo-stamina', '--strong-win', '--elo', '--output', str(b)], check=True, capture_output=True)
            self.assertTrue(verify_artifacts(a))
            for name in ('summary.json', 'participants.csv', 'bouts.csv', 'tournaments.jsonl'):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            (a/'bouts.csv').write_text((a/'bouts.csv').read_text().splitlines()[0]+'\n')
            with self.assertRaises(ValueError):
                verify_artifacts(a)

    def test_disabled_and_singleton(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiments(Config(participants=1, repeats=2), ['elo'], False, directory)
            self.assertEqual(summary['elo']['grin_tour']['samples'], 0)
            self.assertIsNone(summary['elo']['bracket']['spearman'])
            self.assertTrue(verify_artifacts(directory))
            config = json.loads(Path(directory, 'config.json').read_text())
            self.assertFalse(config['grin_tour'])

    def test_installed_cli_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-m', 'simulation', '--elo',
                                     '--participants', '2', '--repeats', '1', '--output', 'out'],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(verify_artifacts(Path(directory)/'out'))
            historical = subprocess.run([sys.executable, '-m', 'armplaces', '--input',
                                         str(Path('data/left_hand_75kg').resolve()),
                                         '--output', 'historical'], cwd=directory,
                                        capture_output=True, text=True)
            self.assertEqual(historical.returncode, 0, historical.stderr)
            self.assertTrue(Path(directory, 'historical', 'places.txt').is_file())

    def test_metrics(self):
        self.assertEqual(rank_metrics([3.,2.,1.], {0:1,1:2,2:3})['mae'], 0.)
        self.assertAlmostEqual(rank_metrics([3.,2.,1.], {0:1,1:2,2:3})['spearman'], 1.)
        self.assertAlmostEqual(rank_metrics([3.,2.,1.], {0:3,1:2,2:1})['spearman'], -1.)
        self.assertAlmostEqual(rank_metrics([4.,3.,2.,1.], {0:1,1:2,2:3,3:3})['mae'], .25)

    def test_modified_metrics_and_counters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            run_experiments(Config(participants=4, repeats=2), ['strong-win'], True, path)
            summary_path = path/'summary.json'
            original = summary_path.read_text()
            summary = json.loads(original)
            summary['strong-win']['bracket']['mae'] += .1
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verify_artifacts(path)
            summary_path.write_text(original)
            import csv
            with (path/'participants.csv').open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows[0]['wins'] = str(int(rows[0]['wins'])+1)
            rows[0]['played'] = str(int(rows[0]['played'])+1)
            with (path/'participants.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(RuntimeError):
                verify_artifacts(path)

    def test_cli_errors_and_help(self):
        for args in ([], ['--grin-tour'], ['--elo', '--participants', '129'],
                     ['--elo', '--dr', 'nan'], ['--elo', '--repeats', '0']):
            result = subprocess.run([sys.executable, '-m', 'simulation', *args], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, '-m', 'simulation', '--help'], capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b'--elo-stamina', result.stdout)
