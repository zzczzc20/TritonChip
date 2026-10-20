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

    def test_frozen_baselines_and_single_factor(self):
        for experiment in collect.CONTROLS:
            exp, entries=collect.schedule(experiment,'python',{},Path('/out'))
            pair={e['side']:e for e in entries if e['block']==0 and e['point']==entries[0]['point']}
            a,b=pair['control']['knobs'],pair['candidate']['knobs']
            self.assertEqual({k for k in a if a[k]!=b[k]},set(exp['vary']))
            if exp['baseline']=='B40-S':
                self.assertEqual(a['TNCA_ACTIVE_CORES'],'1')
                self.assertEqual(a['TNCA_TRANSFER_POLICY'],'per-kernel')
                self.assertEqual(a['TNCA_PIPELINE'],'1')
                self.assertEqual(a['TNCA_RTL_OUTSTANDING'],'1')
                self.assertEqual(a['TNCA_RTL_MEM_PORTS'],'1')
            else:
                self.assertEqual(a['TNCA_RTL_MEM_PORTS'],b['TNCA_RTL_MEM_PORTS'])
                self.assertEqual(a['TNCA_RTL_OUTSTANDING'],b['TNCA_RTL_OUTSTANDING'])

    def test_required_two_core_control_keeps_original_compute_points(self):
        exp,two=collect.schedule('R40-05.three-cores','python',{},Path('/out'),compute_cores=2)
        _,three=collect.schedule('R40-05.three-cores','python',{},Path('/out'))
        self.assertEqual(exp['threshold'],1.5)
        self.assertEqual([e['point'] for e in two],[e['point'] for e in three])
        for entry in two:
            self.assertEqual(entry['knobs']['TNCA_ACTIVE_CORES'],'1' if entry['side']=='control' else '2')
        with self.assertRaises(ValueError):
            collect.schedule('R40-06.ports','python',{},Path('/out'),compute_cores=2)

    def test_prospective_long_task_rule(self):
        sampling=json.loads((ROOT/'spec/r40/v2/plan.json').read_text())['sample_plan']['board']
        self.assertEqual(collect.sample_count(dict(control=1.,candidate=.1),sampling),30)
        self.assertEqual(collect.sample_count(dict(control=1.00001,candidate=.1),sampling),10)
        self.assertEqual(collect.sample_count(dict(control=.1,candidate=1.00001),sampling),10)
        for durations in [{},dict(control=2),dict(control=0,candidate=2),
                          dict(control=float('nan'),candidate=2),dict(control=2,candidate=float('inf'))]:
            with self.assertRaises(ValueError): collect.sample_count(durations,sampling)
        _,entries=collect.schedule('R40-05.three-cores','python',{},Path('/out'))
        entry=entries[0]; original=list(entry['command'])
        command=collect.sampling_command(entry,warmup=1,timed=1,stem='qualify',out=Path('/out'))
        self.assertEqual(entry['command'],original)
        self.assertEqual(command[command.index('--point')+1],entry['point']['id'])
        self.assertEqual(command[command.index('--out')+1],'/out/qualify.jsonl')
        self.assertEqual(command[command.index('--bundle-dir')+1],'/out/bundles/qualify')

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
