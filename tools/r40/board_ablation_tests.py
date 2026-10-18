#!/usr/bin/env python3
"""Freeze the physical experiment schedule and refuse an unbound execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'tools/r40/board_ablation_collect.py'
spec=importlib.util.spec_from_file_location('board_ablation_collect',path)
collect=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect)


class BoardCollectionTests(unittest.TestCase):
    def test_original_complete_three_block_schedule(self):
        for experiment in collect.CONTROLS:
            with self.subTest(experiment=experiment):
                exp,entries=collect.schedule(experiment,'python',{},Path('/out'))
                frozen=json.loads((ROOT/exp['subset']).read_text())['points']
                expected={(p['id'],p['sha256']) for p in frozen}
                self.assertEqual(len(entries),6*len(expected))
                for block in range(3):
                    for side in ('control','candidate'):
                        selected=[e for e in entries if e['block']==block and e['side']==side]
                        self.assertEqual({(e['point']['id'],e['point']['sha256']) for e in selected},expected)
                        for entry in selected:
                            command=entry['command']
                            self.assertEqual(command[command.index('--warmup')+1],'5')
                            self.assertEqual(command[command.index('--timed')+1],'30')
                self.assertEqual(entries,collect.schedule(experiment,'python',{},Path('/out'))[1])

    def test_wrong_source_cannot_start_a_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'refused'
            done=subprocess.run([sys.executable,str(path),'--experiment','R40-05.three-cores',
                '--xclbin',str(Path(directory)/'absent.xclbin'),'--image-sha256','0'*64,
                '--software-tree-sha256','files:wrong-source',
                '--operating-clock-evidence',str(Path(directory)/'absent-clock.json'),
                '--out',str(out)],capture_output=True,text=True)
            self.assertEqual(done.returncode,1,done.stderr)
            report=json.loads((out/'collection.json').read_text())
            self.assertEqual(report['status'],'failed-preserved')
            self.assertIn('software identity mismatch',report['error'])
            self.assertEqual(report['completed'],[])
            self.assertEqual(report['required_gate_promotions'],[])
            self.assertFalse(list(out.glob('*.jsonl')))
            self.assertFalse(list(out.glob('*.log')))


if __name__=='__main__':unittest.main()
