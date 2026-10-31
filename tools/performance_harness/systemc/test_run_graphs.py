"""Timeouts must stop the real graph worker and its model process; results
must carry the knobs the run actually used."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_graphs
from run_graphs import bounded_child, complete_run, knobs_in_effect

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parents[2]/'evidence/r41/stage1/kickoff/prospective-wide'
# archbench/platforms/tnca.py NOT_KNOBS, as a real child reports it.
NOT_KNOBS = ['TNCA_BACKEND', 'TNCA_BACKEND_ARGS', 'TNCA_COUNTERS', 'TNCA_DEADLINE', 'TNCA_DEVICE',
             'TNCA_DUMP_DIR', 'TNCA_HOST_TIMEOUT', 'TNCA_KERNEL_DUMP', 'TNCA_LAUNCH', 'TNCA_OPT_LEVEL',
             'TNCA_PROFILE', 'TNCA_TRITON']


def fake_child(mutate=None):
    """Stand in for the graph child: record the knobs its environment has."""
    def child(cmd, *, cwd, env, stderr, timeout):
        knobs = {k: v for k, v in sorted(env.items()) if k.startswith('TNCA_') and k not in NOT_KNOBS}
        if mutate:
            mutate(knobs)
        point = cmd[cmd.index('--point') + 1]
        rec = dict(point_id=point, status='pass', problems=[], fallbacks=[], platform=dict(knobs=knobs),
                   harness_not_knobs=NOT_KNOBS, evidence_level='model-prediction-uncalibrated',
                   model_timing_status='uncalibrated',
                   model_graph=dict(graph_cycles=100, device_cycles=100, measured_call='steady',
                                    launches=[dict(kernel='k', cycles=100)]))
        return 'ARCHBENCH-RECORD ' + json.dumps(rec) + '\n', 0
    return child


class GraphWorkerTests(unittest.TestCase):
    def test_partial_success_record_cannot_hide_a_failed_worker(self):
        self.assertFalse(complete_run(dict(status='pass', returncode='timeout')))
        self.assertFalse(complete_run(dict(status='pass', returncode=7)))
        self.assertFalse(complete_run(dict(status='pass', returncode=0, fallbacks=['cpu'])))
        self.assertTrue(complete_run(dict(status='pass', returncode=0, fallbacks=[])))

    def test_success_keeps_stdout(self):
        stdout, code = bounded_child([sys.executable, '-c', 'print("complete")'],
                                      cwd='.', env=dict(os.environ), stderr=subprocess.DEVNULL, timeout=3)
        self.assertEqual((stdout, code), ('complete\n', 0))

    def test_timeout_kills_worker_and_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'descendant'
            program = ('import subprocess,sys,time; '
                       'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                       f'open({str(pidfile)!r},"w").write(str(p.pid)); '
                       'print("started",flush=True); time.sleep(30)')
            stdout, code = bounded_child([sys.executable, '-c', program], cwd=directory,
                                          env=dict(os.environ), stderr=subprocess.DEVNULL, timeout=.5)
            self.assertEqual(code, 'timeout')
            self.assertEqual(stdout, 'started\n')
            status = Path('/proc', pidfile.read_text(), 'stat')
            for _ in range(50):
                try:
                    state = status.read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    break
                if state == 'Z':
                    break
                time.sleep(.01)
            else:
                self.fail('model/compile descendant survived the graph timeout')


class EffectiveKnobTests(unittest.TestCase):
    """The audit: --env TNCA_GEMM_TILE=16,32,16 ran, results said 16,16,16."""

    def graphs(self, directory, mutate=None, overrides=('TNCA_GEMM_TILE=16,32,16',)):
        out = Path(directory)/'run'
        argv = ['run_graphs.py', '--source-root', str(HERE.parents[2]), '--binary', str(Path(directory)/'tnca-chip'),
                '--out', str(out), '--point', 'op.matmul/medium/fp16-m128-k192-n128', '--jobs', '1',
                '--active-cores', '3']
        for kv in overrides:
            argv += ['--env', kv]
        with patch.object(run_graphs, 'bounded_child', fake_child(mutate)), patch.object(sys, 'argv', argv), \
                patch.dict(os.environ, {'TNCA_GEMM_TILE': '64,64,64'}):
            code = run_graphs.main()
        return code, out, json.loads((out/'results.json').read_text())['points'][0]

    def test_results_record_the_overridden_knobs_actually_in_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            code, out, row = self.graphs(directory)
            self.assertEqual(code, 0)
            self.assertTrue(row['knobs_verified'])
            self.assertEqual(row['knobs']['TNCA_GEMM_TILE'], '16,32,16')
            self.assertEqual(row['requested_knobs']['TNCA_GEMM_TILE'], '16,16,16')
            self.assertEqual(row['knobs']['TNCA_ACTIVE_CORES'], '3')
            self.assertEqual(row['knobs']['TNCA_TRANSFER_POLICY'], 'per-kernel')
            self.assertFalse(any(k.startswith('TNCA_TLM_') for k in row['knobs']))
            self.assertIn('TNCA_TLM_BINARY', row['model_knobs'])
            recorded = json.loads(Path(row['record']).read_text())['platform']['knobs']
            self.assertEqual({**row['knobs'], **row['model_knobs']}, recorded)
            # predictions.py copies exactly these verified knobs.
            (Path(directory)/'tnca-chip').write_text('binary')
            (Path(directory)/'model.cfg').write_text('outlets=3\n')
            check = subprocess.run([sys.executable, str(HERE/'predictions.py'), '--results', str(out/'results.json'),
                                    '--config', str(Path(directory)/'model.cfg'), '--binary', str(Path(directory)/'tnca-chip'),
                                    '--out', str(Path(directory)/'p.jsonl')], capture_output=True, text=True, check=True)
            prediction = json.loads((Path(directory)/'p.jsonl').read_text())
            self.assertEqual(prediction['knobs'], row['knobs'])
            self.assertEqual(prediction['knobs_source'], 'record-platform-knobs')

    def test_a_record_that_ran_other_knobs_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            code, out, row = self.graphs(directory, mutate=lambda k: k.update(TNCA_GEMM_TILE='16,16,16'))
            self.assertEqual(code, 1)
            self.assertEqual(row['status'], 'knob-mismatch')
            self.assertFalse(row['knobs_verified'])
            self.assertIsNone(row['knobs'])
            self.assertIn('TNCA_GEMM_TILE', row['knob_problem'])
            self.assertFalse(complete_run(row))
        with tempfile.TemporaryDirectory() as directory:
            code, out, row = self.graphs(directory, mutate=lambda k: k.pop('TNCA_SUBMIT'))
            self.assertEqual(row['status'], 'knob-mismatch')

    def test_record_without_exclusions_is_not_trusted(self):
        rec = dict(platform=dict(knobs=dict(TNCA_TUNE='0')))
        self.assertIsNone(knobs_in_effect(rec, dict(TNCA_TUNE='0'))[0])

    @unittest.skipUnless((EVIDENCE/'results.json').is_file(), 'source root without the kickoff evidence')
    def test_preserved_prospective_row_is_refused_by_predictions(self):
        """The original results.json stays as evidence; its knobs cannot be copied."""
        results = EVIDENCE/'results.json'
        row = json.loads(results.read_text())['points'][0]
        record = json.loads((EVIDENCE/'runs/op.matmul__medium__fp16-m128-k192-n128/record.json').read_text())
        self.assertEqual(row['knobs']['TNCA_GEMM_TILE'], '16,16,16')
        self.assertEqual(record['platform']['knobs']['TNCA_GEMM_TILE'], '16,32,16')
        sys.path.insert(0, str(HERE))
        from predictions import row_knobs
        self.assertEqual(row_knobs(row, results), (None, 'knob-mismatch'))
        honest = dict(row, knobs={k: record['platform']['knobs'][k] for k in row['knobs']})
        self.assertEqual(row_knobs(honest, results)[1], 'legacy-row-checked-against-record')


if __name__ == '__main__':
    unittest.main()
