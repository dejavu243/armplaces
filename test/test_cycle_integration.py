"""Cycle-scoring flags, common bouts and independently checked artifacts."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simulation import Config, generate_field, simulate
from simulation.experiments import run_experiments
from simulation.verify import verify_artifacts


class CycleIntegrationTests(unittest.TestCase):
    def test_flags_preserve_random_bouts_and_standard_ranking(self):
        config=Config(participants=16)
        ratings,draw=generate_field(config,10)
        baseline=simulate(config,'elo',ratings,draw,repeat=10,grin_tour=True)
        combined=simulate(config,'elo',ratings,draw,repeat=10,grin_tour=True,
                          grin_cycle_score=True)
        self.assertEqual(baseline['bouts'],combined['bouts'])
        self.assertEqual(baseline['grin_tour'],combined['grin_tour'])
        self.assertEqual(baseline['sequence'],combined['sequence'])
        self.assertEqual(combined['grin_cycle_score']['status'],'ok')
        self.assertTrue(combined['grin_cycle_score']['diagnostics']['activated'])
        singleton=simulate(Config(participants=1),'elo',[1.],[0],grin_cycle_score=True)
        self.assertFalse(singleton['grin_cycle_score']['diagnostics']['activated'])
        self.assertEqual(singleton['grin_cycle_score']['places'],{'0':1})

    def test_cli_invalid_weight_combinations(self):
        for args in (['--elo','--cycle-weights','1','1','1'],
                     ['--elo','--grin-cycle-score','--cycle-weights','0','0','0'],
                     ['--elo','--grin-cycle-score','--cycle-weights','nan','1','1']):
            result=subprocess.run([sys.executable,'-m','simulation',*args],capture_output=True)
            self.assertNotEqual(result.returncode,0)
        self.assertEqual(subprocess.run([sys.executable,'-m','simulation','--help'],
                                        capture_output=True).returncode,0)

    def test_export_verifier_recomputes_scores_and_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            summary=run_experiments(Config(participants=16,repeats=12),['elo'],True,path,
                                    grin_cycle_score=True)
            self.assertEqual(summary['elo']['cycle_activated'],2)
            self.assertEqual(summary['elo']['cycle_success'],2)
            self.assertTrue(verify_artifacts(path))
            with (path/'participants.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(row['cycle_place'] for row in rows))
            source=path/'tournaments.jsonl'
            records=[json.loads(line) for line in source.read_text().splitlines()]
            triggered=next(record for record in records if
                           record['grin_cycle_score']['diagnostics']['activated'])
            index=records.index(triggered)
            for change in ('score','criterion','preliminary','activation','order','edge','place'):
                changed=json.loads(json.dumps(records))
                diag=changed[index]['grin_cycle_score']['diagnostics']
                if change=='score':
                    member=diag['components'][0]['members'][0]
                    diag['components'][0]['scores'][member]['total']+=0.1
                elif change=='criterion':
                    member=diag['components'][0]['members'][0]
                    diag['components'][0]['scores'][member]['values']['loss_place']+=1
                elif change=='preliminary':
                    member=diag['components'][0]['members'][0]
                    diag['preliminary_places'][member]+=1
                elif change=='activation':
                    diag['activated']=False
                elif change=='order':
                    diag['components'][0]['order'].reverse()
                elif change=='edge':
                    diag['added_edges'].append(['0','1'])
                else:
                    member=diag['components'][0]['members'][0]
                    changed[index]['grin_cycle_score']['places'][member]=99
                source.write_text(''.join(json.dumps(record)+'\n' for record in changed))
                with self.assertRaises((ValueError,RuntimeError)):
                    verify_artifacts(path)
            source.write_text(''.join(json.dumps(record)+'\n' for record in records))
            config_file=path/'config.json'
            manifest=json.loads(config_file.read_text())
            changed=json.loads(json.dumps(manifest))
            changed['normalized_cycle_weights']=[.5,.25,.25]
            config_file.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                verify_artifacts(path)
