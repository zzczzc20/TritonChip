"""Board-batch tooling: sampling rule, identity refusals, residuals, tuning judgement.

Nothing here touches a card: ArchBench is replaced by a fake that writes
records in the board schema from its own arguments.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import board_batch  # noqa: E402
import experiment  # noqa: E402
import prediction_residuals  # noqa: E402
import prospective_residual  # noqa: E402
import tune_board_qualification as tune  # noqa: E402

UUID = '1bada9db-78f3-5bc8-7ba6-cce062e7ade8'
CLOCK_HZ = 90000000

FAKE_ARCHBENCH = r'''
import argparse, hashlib, json, os, sys
ap = argparse.ArgumentParser()
for a in ('--platform', '--backend', '--backend-args', '--profile', '--point', '--jobs', '--seed',
          '--warmup', '--timed', '--timeout', '--bundle-dir', '--out'):
    ap.add_argument(a)
ap.add_argument('--set', action='append', default=[])
a = ap.parse_args()
fake = json.loads(os.environ['FAKE_BOARD'])
knobs = dict(kv.split('=', 1) for kv in a.set)
knobs['TNCA_OPERATING_CLOCK_EVIDENCE'] = os.environ['TNCA_OPERATING_CLOCK_EVIDENCE']
assert not any(k.startswith('TNCA_') and k not in ('TNCA_OPERATING_CLOCK_EVIDENCE', 'TNCA_DEADLINE',
               'TNCA_HOST_TIMEOUT') for k in os.environ), 'ambient knob leaked into the run'
cycles = fake['cycles'][a.point]
timed = int(a.timed)
clock_sha = hashlib.sha256(open(os.environ['TNCA_OPERATING_CLOCK_EVIDENCE'], 'rb').read()).hexdigest()
status = 'fail' if a.point in fake.get('fail', []) else 'pass'
cores = int(knobs.get('TNCA_ACTIVE_CORES', '1'))
rec = dict(point_id=a.point, point_sha256=fake['definitions'][a.point], status=status, problems=[], fallbacks=[],
           correctness=dict(max_rel_err=1e-4, tol=1e-2), evidence_level='measured-board', timestamp='2026-10-22T00:00:00+00:00',
           platform=dict(backend=a.backend, backend_args=json.loads(a.backend_args),
                         image_sha256=fake['image'].get(a.point, fake['image']['*']), clock_hz=90000000,
                         operating_clock_evidence_sha256=clock_sha, software_tree=fake['tree'], knobs=knobs,
                         inductor=dict(tensor_descriptors=True)),
           provenance=dict(clock_hz=90000000, artifacts=['a'*64], launch_map=dict(cores=list(range(cores)), launches=[{}])),
           native_memory=dict(window=dict(base=0, size=4294967296, local_limit=131072)),
           time=dict(launches=1),
           timings={'device-cycles-serial-grid': dict(warmup=int(a.warmup), samples=[cycles]*timed, median=cycles),
                    'device-graph-makespan': dict(samples=[cycles+2]*timed, median=cycles+2,
                                                  boundaries='first accepted launch to last device completion')})
with open(a.out, 'a') as fh:
    fh.write(json.dumps(rec) + '\n')
sys.exit(0 if status == 'pass' else 1)
'''


class BoardBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'src'
        (self.root/'archbench').mkdir(parents=True)
        shutil.copyfile(REPO/'archbench/provenance.py', self.root/'archbench/provenance.py')
        (self.root/'spec/r41/stage1').mkdir(parents=True)
        shutil.copyfile(REPO/'spec/r41/stage1/plan.json', self.root/'spec/r41/stage1/plan.json')
        self.plan = json.loads((self.root/'spec/r41/stage1/plan.json').read_text())
        self.points = {'op.short/a': 'a'*64, 'op.short/b': 'b'*64, 'op.long/c': 'c'*64}
        (self.root/'spec/suite.json').write_text(json.dumps(dict(points=[
            dict(id=p, sha256=s) for p, s in self.points.items()])))
        image = self.root/'image.xclbin'
        image.write_bytes(b'xclbin2\0' + bytes.fromhex(UUID.replace('-', '')) + b'image')
        self.image_sha = experiment.digest(image)
        clock = self.root/'clock.json'
        clock.write_text(json.dumps(dict(image_sha256=self.image_sha, clock_hz=CLOCK_HZ)))
        self.runner_path = Path(self.tmp.name)/'fake_archbench.py'
        self.runner_path.write_text(FAKE_ARCHBENCH)
        self.tree = experiment.software_tree(self.root)
        self.fake = dict(cycles={'op.short/a': 9000000, 'op.short/b': 18000000, 'op.long/c': 900000000},
                         definitions=self.points, image={'*': self.image_sha}, tree=self.tree)
        self.out = Path(self.tmp.name)/'out'
        self.loaded = UUID

    def manifest(self, batches, **overrides):
        board = dict(image=str(self.root/'image.xclbin'), image_sha256=self.image_sha, image_uuid=UUID,
                     clock_evidence=str(self.root/'clock.json'), clock_hz=CLOCK_HZ, device='0000:04:00.1',
                     backend='fpga-u280-multicore', profile='core-vrf-sync-v5',
                     backend_args=dict(xclbin=str(self.root/'image.xclbin'), core_count=3),
                     transport=dict(deadline_cycles=200000000000, host_timeout_seconds=28800),
                     memory_map=dict(window_base=0, window_bytes=4294967296, local_limit=131072))
        spec = dict(plan='spec/r41/stage1/plan.json',
                    plan_sha256=experiment.digest(self.root/'spec/r41/stage1/plan.json'),
                    measurement=copy.deepcopy(self.plan['measurement']),
                    suites=[dict(path='spec/suite.json', sha256=experiment.digest(self.root/'spec/suite.json'))],
                    board=board, board_batches=batches)
        spec.update(overrides)
        path = Path(self.tmp.name)/f'manifest-{len(list(Path(self.tmp.name).glob("manifest-*")))}.json'
        path.write_text(json.dumps(spec))
        return path

    def configs(self, *names, knobs=None):
        return [dict(id=n, point=n, point_sha256=self.points[n], knobs=knobs or dict(TNCA_ACTIVE_CORES='3'))
                for n in names]

    def probe(self, board):
        return dict(loaded_uuid=self.loaded, clock_counter_mhz=[90], contexts=0, client_pid=None)

    def batch(self, manifest, name, **kwargs):
        reg = board_batch.load(manifest, name, self.root, self.out, sys.executable)
        reg['batch'].setdefault('name', name)
        return board_batch.Batch(reg, root=self.root, out=self.out, python=Path(sys.executable),
                                 runner=[sys.executable, str(self.runner_path)], tree=self.tree,
                                 probe=self.probe, **kwargs)

    def run_batch(self, manifest, name, **kwargs):
        env = {'FAKE_BOARD': json.dumps(self.fake), 'TNCA_GEMM_TILE': '64,64,64'}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return self.batch(manifest, name, **kwargs).execute()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_device_rule_short_groups_get_three_formal_samples(self):
        manifest = self.manifest(dict(b=dict(sampling='device-rule',
            configurations=self.configs('op.short/a', 'op.short/b', 'op.long/c'),
            groups=[['op.short/a', 'op.short/b'], ['op.long/c']])))
        summary = self.run_batch(manifest, 'b')
        self.assertEqual(summary['status'], 'complete')
        runs = json.loads((self.out/'state.json').read_text())['runs']
        self.assertEqual(sorted(runs), sorted(['op.short__a-q', 'op.short__b-q', 'op.short__a-f3',
                                               'op.short__b-f3', 'op.long__c-q']))
        formal = [json.loads(l) for l in (self.out/'formal.jsonl').read_text().splitlines()]
        self.assertEqual([len(r['timings']['device-cycles-serial-grid']['samples']) for r in formal], [3, 3, 1])
        self.assertEqual(summary['samples']['op.short/a']['serial_grid_samples'], [9000000]*3)
        # Exact resume: nothing reruns.
        again = self.run_batch(manifest, 'b')
        self.assertEqual(again['status'], 'complete')
        self.assertEqual(json.loads((self.out/'state.json').read_text())['runs'], runs)

    def test_one_long_arm_keeps_both_qualification_records(self):
        self.fake['cycles']['op.short/b'] = 180000000
        manifest = self.manifest(dict(b=dict(sampling='device-rule',
            configurations=self.configs('op.short/a', 'op.short/b'), groups=[['op.short/a', 'op.short/b']])))
        self.run_batch(manifest, 'b')
        self.assertEqual(sorted(json.loads((self.out/'state.json').read_text())['runs']),
                         ['op.short__a-q', 'op.short__b-q'])

    def test_anchor_stability_interleaves_independent_processes(self):
        manifest = self.manifest(dict(a=dict(sampling='anchor-stability',
            configurations=self.configs('op.short/a', 'op.long/c'))))
        reg = board_batch.load(manifest, 'a', self.root, self.out, sys.executable)
        order = [(c, t) for c, t, _w, _n in board_batch.schedule(reg)]
        self.assertEqual(order, [('op.short/a', 's1'), ('op.long/c', 's1'), ('op.long/c', 's2'),
                                 ('op.short/a', 's2'), ('op.short/a', 's3'), ('op.long/c', 's3')])
        summary = self.run_batch(manifest, 'a')
        self.assertEqual(summary['samples']['op.long/c']['processes'], 3)
        self.assertEqual(summary['samples']['op.long/c']['graph_makespan_relative_spread'], 0)

    def test_failures_are_preserved_and_wrong_image_refused(self):
        self.fake['fail'] = ['op.short/a']
        self.fake['image']['op.short/b'] = 'f'*64
        manifest = self.manifest(dict(b=dict(sampling='device-rule',
            configurations=self.configs('op.short/a', 'op.short/b', 'op.long/c'))))
        summary = self.run_batch(manifest, 'b')
        self.assertEqual(summary['status'], 'failed-preserved')
        self.assertEqual(summary['failed_configurations'], ['op.short/a', 'op.short/b'])
        runs = json.loads((self.out/'state.json').read_text())['runs']
        self.assertTrue(any(p.startswith('image') for p in runs['op.short__b-q']['problems']))
        self.assertTrue(runs['op.long__c-q']['ok'])
        # Without --retry-failed nothing reruns; with it the old output is kept.
        self.fake['fail'] = []
        self.run_batch(manifest, 'b')
        self.assertFalse(list((self.out/'runs').glob('*.failed*')))
        self.run_batch(manifest, 'b', retry_failed=True)
        self.assertTrue((self.out/'runs/op.short__a-q.failed0').exists())

    def test_registration_refusals(self):
        batches = dict(b=dict(sampling='device-rule', configurations=self.configs('op.short/a')))
        drifted = copy.deepcopy(self.plan['measurement'])
        drifted['long_device_tasks']['short_timed_samples'] = 5
        for manifest, message in (
                (self.manifest(batches, measurement=drifted), 'frozen plan measurement'),
                (self.manifest(batches, plan_sha256='0'*64), 'frozen plan identity'),
                (self.manifest(dict(b=dict(sampling='device-rule', configurations=[
                    dict(self.configs('op.short/a')[0], point_sha256='e'*64)]))), 'not the frozen suite definition'),
                (self.manifest(dict(b=dict(sampling='device-rule', configurations=self.configs('op.short/a'),
                    coverage=dict(suite='spec/suite.json', sha256=experiment.digest(self.root/'spec/suite.json'))))),
                 'cover its registered suite'),
                (self.manifest(dict(b=dict(sampling='device-rule', configurations=self.configs(
                    'op.short/a', knobs=dict(TNCA_OPERATING_CLOCK_EVIDENCE='x'))))), 'board registration'),
                (self.manifest(dict(b=dict(sampling='device-rule', configurations=self.configs(
                    'op.short/a', knobs=dict(TNCA_TUNE_DIR='{tune_dir}'))))), 'unexpanded')):
            with self.assertRaisesRegex(board_batch.Refused, message):
                board_batch.load(manifest, 'b', self.root, self.out, sys.executable)

    def test_wrong_loaded_image_refuses_before_any_run(self):
        manifest = self.manifest(dict(b=dict(sampling='device-rule', configurations=self.configs('op.short/a'))))
        self.loaded = '428c00e5-6996-92c7-1f8f-ec41f2d037b3'
        with self.assertRaisesRegex(board_batch.Refused, 'loaded image'):
            self.run_batch(manifest, 'b')
        self.assertFalse((self.out/'runs').exists())


class ProspectiveResidualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.model = json.loads((REPO/'evidence/r41/stage1/kickoff/prospective-wide/runs/'
                                 'op.matmul__medium__fp16-m128-k192-n128/record.json').read_text())
        self.record = self.dir/'record.json'
        self.record.write_text(json.dumps(self.model))
        self.prediction = self.dir/'prediction.json'
        self.prediction.write_text(json.dumps(dict(
            point=self.model['point_id'], record_sha256=experiment.digest(self.record),
            sealed_before_board_utc='2026-10-20T18:19:48+00:00', evidence='model-prediction-uncalibrated',
            model_graph=dict(graph_cycles=911645, device_cycles=911645))))
        self.config = prospective_residual.configuration(self.model['platform']['knobs'])

    def board(self, rows):
        path = self.dir/'board.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def row(self, **changes):
        knobs = dict(self.config, TNCA_OPERATING_CLOCK_EVIDENCE='/x/qualified_clock.json')
        row = dict(point_id=self.model['point_id'], point_sha256=self.model['point_sha256'], status='pass',
                   fallbacks=[], evidence_level='measured-board', timestamp='2026-10-22T01:00:00+00:00',
                   platform=dict(backend='fpga-u280-multicore', image_sha256='a'*64, clock_hz=CLOCK_HZ,
                                 knobs=knobs, software_tree='files:new'),
                   provenance=dict(artifacts=list(self.model['provenance']['artifacts'])),
                   timings={'device-graph-makespan': dict(samples=[800000, 810000, 805000],
                                                          boundaries='first accepted launch to last device completion'),
                            'device-cycles-serial-grid': dict(samples=[800000, 810000, 805000])})
        for key, value in changes.items():
            if key == 'knobs':
                row['platform']['knobs'] = dict(row['platform']['knobs'], **value)
            elif key in ('image_sha256', 'clock_hz', 'backend'):
                row['platform'][key] = value
            else:
                row[key] = value
        return row

    def compare(self, rows, **kw):
        return prospective_residual.compare(self.prediction, self.record, self.board(rows), image='a'*64,
                                            clock_hz=CLOCK_HZ, **kw)

    def test_residual_only_for_the_same_artifacts(self):
        result = self.compare([self.row()])
        self.assertEqual(result['status'], 'residual-recorded')
        self.assertAlmostEqual(result['residual']['graph_makespan_relative'], (911645-805000)/805000)
        other = self.row(provenance=dict(artifacts=['b'*64]))
        self.assertEqual(self.compare([other])['status'], 'not-comparable')

    def test_control_context_and_refusals(self):
        control = self.row(knobs=dict(TNCA_GEMM_TILE='16,16,16'))
        result = self.compare([self.row(), control], control=self.board([self.row(), control]),
                              control_set=dict(TNCA_GEMM_TILE='16,16,16'))
        self.assertIn('candidate_over_control', result['board_control'])
        for bad, message in ((self.row(timestamp='2026-10-19T00:00:00+00:00'), 'precedes'),
                             (self.row(evidence_level='model-prediction-uncalibrated'), 'cannot stand for a board'),
                             (self.row(backend='systemc-tlm-grid'), 'cannot stand for a board'),
                             (self.row(clock_hz=100000000), 'not the declared'),
                             (self.row(image_sha256='b'*64), 'ran image'),
                             (self.row(fallbacks=['cpu']), 'fallback'),
                             (self.row(knobs=dict(TNCA_GEMM_TILE='16,16,16')), 'no board record')):
            with self.assertRaisesRegex(ValueError, message):
                self.compare([bad])
        submission = self.row()
        submission['timings']['device-graph-makespan']['boundaries'] = 'first submission to last submission'
        with self.assertRaisesRegex(ValueError, 'completion'):
            self.compare([submission])
        self.prediction.write_text(json.dumps(dict(json.loads(self.prediction.read_text()), record_sha256='0'*64)))
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            self.compare([self.row()])


class PredictionSetResidualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.knobs = dict(TNCA_GEMM_TILE='16,32,16', TNCA_ACTIVE_CORES='1')

    def entry(self, path, cycles, status='sealed', timing='uncalibrated', program='x'*64, knobs=None):
        return dict(point='op.matmul/medium/fp16-m128-k192-n128', point_sha256='p'*64, arm='candidate', path=path,
                    configuration=knobs or self.knobs, status=status, graph_cycles=cycles,
                    model_timing_status=timing, record_sha256='r'*64,
                    launches=[dict(kernel='triton_tem_fused_mm_0', artifact_sha256='a'*64, executable_sha256=program)])

    def write(self, name, value, lines=False):
        path = self.dir/name
        path.write_text(''.join(json.dumps(v)+'\n' for v in value) if lines else json.dumps(value))
        return path

    def board(self, **changes):
        rec = dict(point_id='op.matmul/medium/fp16-m128-k192-n128', point_sha256='p'*64, status='pass', fallbacks=[],
                   evidence_level='measured-board', timestamp='2026-10-22T00:00:00+00:00',
                   platform=dict(backend='fpga-u280-multicore', image_sha256='i'*64, clock_hz=CLOCK_HZ,
                                 knobs=dict(self.knobs, TNCA_OPERATING_CLOCK_EVIDENCE='/q.json')),
                   provenance=dict(artifacts=['b'*64], program=[['triton_tem_fused_mm_0', 'x'*64]]),
                   timings={'device-graph-makespan': dict(samples=[1000, 1010, 990],
                                                          boundaries='first accepted launch to last device completion'),
                            'device-cycles-serial-grid': dict(samples=[1000, 1010, 990])})
        rec.update(changes)
        return rec

    def compare(self, predictions, boards):
        sealed = self.write('set.json', dict(sealed_before_board_utc='2026-10-21T12:00:00+00:00',
                                             predictions=predictions))
        return prediction_residuals.compare([sealed], [self.write('b.jsonl', boards, lines=True)],
                                            image='i'*64, clock_hz=CLOCK_HZ,
                                            executables=lambda r: r['provenance'].get('program'))

    def test_fast_and_slow_residuals_per_configuration(self):
        report = self.compare([self.entry('fast', 1100), self.entry('slow', 900, timing='inconclusive-fixed-point')],
                              [self.board()])
        row = report['rows'][0]
        self.assertAlmostEqual(row['fast']['residual'], .1)
        self.assertEqual(row['fast']['status'], 'residual-recorded')
        self.assertEqual(row['slow']['status'], 'residual-recorded-inconclusive-prediction')
        self.assertFalse(row['slow']['qualified'])
        self.assertEqual(report['summary']['fast']['qualified_residuals'], 1)
        self.assertEqual(report['summary']['slow']['inconclusive'], 1)
        report = self.compare([self.entry('fast', 1100, program='y'*64),
                               dict(self.entry('slow', 0), status='not-run', reason='over 3 h')], [self.board()])
        self.assertEqual(report['rows'][0]['fast']['status'], 'not-comparable')
        self.assertEqual(report['rows'][0]['slow'], dict(status='prediction-not-run', reason='over 3 h'))
        other = dict(self.knobs, TNCA_GEMM_TILE='16,16,16')
        report = self.compare([self.entry('fast', 1100, knobs=other)], [self.board()])
        self.assertEqual(report['rows'][0]['status'], 'no-board-record')
        unbundled = self.board()
        del unbundled['provenance']['program']
        report = self.compare([self.entry('fast', 1100)], [unbundled])
        self.assertIn('unavailable', report['rows'][0]['fast']['reason'])
        for bad, message in ((self.board(evidence_level='model-prediction-uncalibrated'), 'cannot stand'),
                             (self.board(timestamp='2026-10-21T11:00:00+00:00'), 'precedes'),
                             (self.board(point_sha256='q'*64), 'another point definition')):
            with self.assertRaisesRegex(ValueError, message):
                self.compare([self.entry('fast', 1100)], [bad])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.compare([self.entry('fast', 1100), self.entry('fast', 1200)], [self.board()])


    def test_executable_identity_ignores_path_metadata(self):
        artifacts = sorted((REPO/'evidence/r41/stage1/kickoff/predictions-v2/artifacts').glob('*.tnca'))
        if not artifacts:
            self.skipTest('no sealed artifacts')
        identities = {}
        for path in artifacts:
            identity = prediction_residuals.executable_identity(path)
            identities.setdefault(identity['executable_sha256'], set()).add(experiment.digest(path))
        # Byte-different files of one program (different run directories) share one identity.
        self.assertTrue(any(len(files) > 1 for files in identities.values()))


class TuneJudgementTests(unittest.TestCase):
    def record(self, point, **changes):
        row = dict(point_id=point, status='pass', fallbacks=[], evidence_level='measured-board',
                   platform=dict(backend='fpga-u280-multicore', knobs=dict(TNCA_TUNE='1', TNCA_MAX_BLOCK='32',
                                                                              TNCA_TUNE_CANDIDATES='4')),
                   provenance=dict(launch_map=dict(launches=[{}])), time=dict(launches=1))
        row.update(changes)
        return row

    def selected(self, key, cache_hit, winner='x16', samples=3):
        rows = [dict(status='measured', candidate=c, device_samples=[[{}]]*samples) for c in ('x16', 'x8')]
        return dict(kernel='k', status='selected', tuning_seconds=12.0,
                    compilation=[dict(status='compiled')],
                    selection=dict(identity_key=key, cache_hit=cache_hit, selected=winner, rows=rows))

    def judge(self, first, rerun, first_records=None, rerun_records=None):
        records = [self.record('p')]
        return tune.judge(first, rerun, first_records or records, rerun_records or records,
                          knobs=dict(TNCA_TUNE='1', TNCA_MAX_BLOCK='32'), max_candidates=4, max_seconds=600)

    def test_qualified_and_refused_passes(self):
        self.assertEqual(self.judge({'r1': self.selected('k1', False)}, {'r2': self.selected('k1', True)})[0], [])
        cases = (
            (({'r1': self.selected('k1', False)}, {'r2': self.selected('k1', False)}), 'measured again'),
            (({'r1': self.selected('k1', False)}, {}), 'not reused'),
            (({'r1': self.selected('k1', False)}, {'r2': self.selected('k1', True, winner='x8')}), 'winner differs'),
            (({'r1': self.selected('k1', False, samples=2)}, {'r2': self.selected('k1', True)}), 'three independent'),
            (({'r1': dict(kernel='k', status='conservative', reason='image-bound operating clock evidence required')}, {}),
             'all conservative'),
            (({}, {}), 'did not run'))
        for (first, rerun), message in cases:
            problems, _ = self.judge(first, rerun)
            self.assertTrue(any(message in p for p in problems), (message, problems))
        many = self.selected('k1', False)
        many['compilation'] = [dict(status='compiled')]*4
        self.assertTrue(any('exceed' in p for p in self.judge({'r1': many}, {'r2': self.selected('k1', True)})[0]))
        leaked = self.record('p', time=dict(launches=5))
        problems, _ = self.judge({'r1': self.selected('k1', False)}, {'r2': self.selected('k1', True)},
                                 first_records=[leaked])
        self.assertTrue(any('outside the generated graph' in p for p in problems))

    def test_expiry_scope_and_classification(self):
        wall, deadline = tune.expiry_scope(240000000, CLOCK_HZ)
        self.assertEqual(deadline, 30000000)
        self.assertAlmostEqual(wall/2, 2*deadline/CLOCK_HZ)
        self.assertEqual(tune.classify(RuntimeError('the device deadline expired with work outstanding')),
                         'device-deadline')
        self.assertEqual(tune.classify(RuntimeError('bounded measurement could not drain')), 'bounded-drain-timeout')
        with self.assertRaises(ValueError):
            tune.expiry_scope(0, CLOCK_HZ)


class ComparatorBoardDomainTests(unittest.TestCase):
    def run_tool(self, tool, cases, domains=None):
        with tempfile.TemporaryDirectory() as directory:
            argv = [sys.executable, str(HERE/tool), '--out', str(Path(directory)/'out.json')]
            if domains:
                argv += ['--domains', domains]
            for label, report in cases.items():
                path = Path(directory)/f'{label}.json'
                path.write_text(json.dumps(report))
                argv += ['--case', f'{label}={path}']
            result = subprocess.run(argv, capture_output=True, text=True)
            out = json.loads((Path(directory)/'out.json').read_text()) if result.returncode == 0 else None
            return result, out

    def memory(self, ports, outstanding, cycles):
        return dict(status='functional-pass', ports=ports, outstanding=outstanding, software_tree='files:t',
                    device_cycles=cycles, elements=4097, inputs=[dict(sha256='i')], backend='fpga-u280-multicore',
                    artifacts=[dict(code_sha256='c')], placements=[1], output_placement=2,
                    outlets=[], counter_delta={})

    def test_memory_board_cohort(self):
        cases = {f'board-p{p}o{o}': self.memory(p, o, c) for (p, o), c in
                 (((1, 1), 37000), ((1, 8), 19000), ((3, 8), 21000))}
        result, out = self.run_tool('memory_service_compare.py', cases, 'board')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(out['status'], 'board-cohort-recorded')
        result, _ = self.run_tool('memory_service_compare.py', cases)
        self.assertNotEqual(result.returncode, 0)
        cases['board-p3o8']['backend'] = 'fpga-multicore-sim'
        result, _ = self.run_tool('memory_service_compare.py', cases, 'board')
        self.assertIn('physical multicore image', result.stderr)

    def pipeline(self, stage):
        row = dict(graph='mm', shape=[16, 16, 48], inputs=[dict(sha256='i')], ring_stages=stage, device_cycles=2000,
                   status='pass', artifacts=[dict(code_sha256='c')], model_counter_deltas={}, rtl_sequences=[],
                   input_placements=[1], output_placement=2)
        return dict(status='functional-pass', backend='fpga-u280-multicore', rows=[row])

    def test_pipeline_board_cohort(self):
        cases = {'board-baseline': self.pipeline(None), **{f'board-stage{s}': self.pipeline(s) for s in (1, 2, 3)}}
        result, out = self.run_tool('pipeline_admission_compare.py', cases, 'board')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(out['status'], 'board-cohort-recorded')
        self.assertEqual(out['rows'][1]['board_cycle_ratio_to_baseline'], 1.0)
        del cases['board-stage3']
        self.assertNotEqual(self.run_tool('pipeline_admission_compare.py', cases, 'board')[0].returncode, 0)


class RegisteredManifestTests(unittest.TestCase):
    """The 90 MHz board manifests are registered completely before any run."""
    NAMES = ('board-anchors', 'board-original-suite', 'board-gemm-tile', 'board-prospective',
             'board-memory-service', 'board-pipeline-admission', 'board-tune-qualification')

    def test_registration_is_complete_and_frozen(self):
        plan = json.loads((REPO/'spec/r41/stage1/plan.json').read_text())
        for name in self.NAMES:
            path = REPO/f'spec/r41/stage1/experiments/{name}.json'
            spec = json.loads(path.read_text())
            self.assertEqual(spec['allowed_hosts'], ['acf3019'], name)
            self.assertEqual(spec['measurement'], plan['measurement'], name)
            self.assertEqual(spec['plan_sha256'], experiment.digest(REPO/'spec/r41/stage1/plan.json'), name)
            self.assertEqual(spec['registration']['status'], 'registered-not-run', name)
            self.assertIsInstance(spec['registration']['expected_board_hours'], float, name)
            self.assertTrue(spec['expected_counter_changes'], name)
            board = spec['board']
            self.assertEqual(board['image_sha256'], 'a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083')
            self.assertEqual((board['clock_hz'], board['backend']), (90000000, 'fpga-u280-multicore'))
            self.assertEqual(board['memory_map']['source_sha256'], experiment.digest(REPO/board['memory_map']['source']))
            outputs = set()
            for step in spec['steps']:
                self.assertIn(step['kind'], experiment.KINDS)
                if step['kind'] == 'fpga':
                    self.assertTrue(step.get('timing_boundary') and step.get('cache_state'), (name, step['id']))
                tool = next((a for a in step['argv'] if a.startswith('tools/')), None)
                self.assertTrue((REPO/tool).is_file(), (name, tool))
                outputs |= set(step.get('outputs', []))
            if spec.get('conclusion_source'):
                self.assertIn(spec['conclusion_source'], outputs, name)
            for batch in spec.get('board_batches', {}):
                reg = board_batch.load(path, batch, REPO, REPO/'sim/build/test', sys.executable, dict(tune_dir='/t'))
                self.assertTrue(reg['order'], (name, batch))
        anchors = json.loads((REPO/'spec/r41/stage1/experiments/board-anchors.json').read_text())
        self.assertEqual(anchors['registration']['anchors'], ['op.add/hbm/fp16-n67108864',
                         'op.matmul/medium/fp16-m128-k192-n128', 'model.llama_decode/hbm/fp16-b1-p512'])
        original = board_batch.load(REPO/'spec/r41/stage1/experiments/board-original-suite.json', 'original',
                                    REPO, REPO/'sim/build/test', sys.executable)
        self.assertEqual(len({c['point'] for c in original['configs'].values()}), 113)
        gemm = board_batch.load(REPO/'spec/r41/stage1/experiments/board-gemm-tile.json', 'gemm',
                                REPO, REPO/'sim/build/test', sys.executable)
        self.assertEqual(sorted({c['knobs']['TNCA_GEMM_TILE'] for c in gemm['configs'].values()}),
                         ['16,16,16', '16,32,16'])
        self.assertEqual(len(gemm['configs']), 18)
        prospective = json.loads((REPO/'spec/r41/stage1/experiments/board-prospective.json').read_text())
        erratum = json.loads((REPO/'evidence/r41/stage1/kickoff/prospective-wide-erratum.json').read_text())
        candidate = prospective['board_batches']['prospective']['configurations'][0]
        self.assertEqual(candidate['knobs'], erratum['true_configuration_knobs'])


if __name__ == '__main__':
    unittest.main()
