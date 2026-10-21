"""Evidence reuse must not hide failed, changed or timed-out experiments."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiment import run

HERE = Path(__file__).resolve().parent
UUID_A = '1bada9db-78f3-5bc8-7ba6-cce062e7ade8'
UUID_B = '428c00e5-6996-92c7-1f8f-ec41f2d037b3'


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'source').write_text('frozen source\n')
        self.manifest = self.root/'experiment.json'

    def spec(self, command, *, timeout=3, outputs=(), **step):
        self.manifest.write_text(json.dumps(dict(
            id='proof-test', allowed_hosts=[socket.gethostname()], hypothesis='test actual completion',
            competing_explanations=['missing or stale evidence'], expected_observations=['bounded command'],
            sources=['source'], steps=[dict(id='command', kind='compile', argv=command,
                                           timeout_seconds=timeout, outputs=list(outputs), **step)])))

    def execute(self, **kwargs):
        return run(self.manifest, self.root, self.root/'results', python=Path(sys.executable), **kwargs)

    def test_exact_resume_and_changed_output_refusal(self):
        code = ("from pathlib import Path; p=Path('count'); "
                "p.write_text(str(int(p.read_text())+1) if p.exists() else '1')")
        self.spec([sys.executable, '-c', code], outputs=['count'])
        with patch.dict(os.environ, {'UNRELATED_SECRET': 'must-not-enter-evidence'}):
            first = self.execute()
        second = self.execute()
        self.assertEqual(first['status'], 'completed-experiment')
        self.assertTrue(second['steps'][0]['reused'])
        self.assertEqual((self.root/'count').read_text(), '1')
        self.assertNotIn('UNRELATED_SECRET', first['steps'][0]['identity']['environment'])
        (self.root/'count').write_text('forged')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.execute()

    def test_failure_is_preserved_without_retry(self):
        self.spec([sys.executable, '-c', 'raise SystemExit(7)'])
        first = self.execute()
        self.assertEqual(first['status'], 'stopped-on-failure')
        second = self.execute()
        self.assertEqual(second['status'], 'stopped-on-preserved-failure')
        self.assertEqual(first['steps'][0]['attempt'], second['steps'][0]['attempt'])

    def test_timeout_terminates_the_actual_worker(self):
        self.spec([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.1)
        report = self.execute()
        row = report['steps'][0]
        self.assertEqual(row['status'], 'timeout')
        self.assertLess(row['returncode'], 0)
        self.assertFalse(Path('/proc', str(row['pid'])).exists())

    def test_missing_success_output_is_an_error(self):
        self.spec([sys.executable, '-c', 'pass'], outputs=['not-produced'])
        self.assertEqual(self.execute()['steps'][0]['status'], 'error')

    def test_cache_state_and_timing_boundary_never_dedup_together(self):
        code = "open('count','a').write('x')"
        self.spec([sys.executable, '-c', code], cache_state='warm', timing_boundary='steady-call')
        first = self.execute()
        self.spec([sys.executable, '-c', code], cache_state='fresh', timing_boundary='steady-call')
        second = self.execute()
        self.spec([sys.executable, '-c', code], cache_state='fresh', timing_boundary='cold-call')
        third = self.execute()
        self.assertEqual([r['steps'][0]['reused'] for r in (first, second, third)], [False]*3)
        self.assertEqual(len({r['steps'][0]['fingerprint'] for r in (first, second, third)}), 3)
        self.assertEqual((self.root/'count').read_text(), 'xxx')

    def test_explicit_report_fields_from_declared_conclusion(self):
        record = dict(result='3 of 3 numerically correct', residual={'median': .12},
                      conclusion='diagnostic only', next_step='board closure')
        code = f"import json; open('cmp.json','w').write(json.dumps({record!r}))"
        self.spec([sys.executable, '-c', code], outputs=['cmp.json'])
        run_spec = json.loads(self.manifest.read_text())
        report = self.execute()
        for field in ('hypothesis', 'competing_explanations', 'expected_counter_changes'):
            self.assertIn(field, report)
        self.assertEqual(report['expected_counter_changes_source'], 'expected_observations')
        self.assertEqual([report[k] for k in ('result', 'residual', 'conclusion', 'next_step')],
                         ['not-recorded']*4)
        run_spec['conclusion_source'] = 'cmp.json'
        run_spec['expected_counter_changes'] = ['fewer accumulator row commands']
        self.manifest.write_text(json.dumps(run_spec))
        report = self.execute()
        self.assertEqual(report['residual'], {'median': .12})
        self.assertEqual(report['next_step'], 'board closure')
        self.assertEqual(report['expected_counter_changes'], ['fewer accumulator row commands'])
        reviewed = self.root/'reviewed.json'
        reviewed.write_text(json.dumps(dict(record, conclusion='reviewed')))
        self.assertEqual(self.execute(conclusion=reviewed)['conclusion'], 'reviewed')
        reviewed.write_text(json.dumps(dict(result='x')))
        with self.assertRaisesRegex(ValueError, 'lacks'):
            self.execute(conclusion=reviewed)


class BoardStepTests(unittest.TestCase):
    """fpga steps: identity-complete keys, read-only probes, refusals."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'archbench').mkdir()
        shutil.copyfile(HERE.parents[1]/'archbench/provenance.py', self.root/'archbench/provenance.py')
        (self.root/'source').write_text('frozen source\n')
        self.manifest = self.root/'experiment.json'
        self.probes = []
        self.loaded = UUID_A
        self.after = None

    def image(self, name, uuid, filler):
        path = self.root/name
        path.write_bytes(b'xclbin2\0' + b'\0'*408 + bytes.fromhex(uuid.replace('-', '')) + filler)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        clock = self.root/(name+'.clock.json')
        clock.write_text(json.dumps(dict(schema='tnca-qualified-operating-clock/1', image_sha256=sha,
                                         clock_hz=90000000, qualification='test')))
        return dict(image=name, image_sha256=sha, image_uuid=uuid, clock_evidence=clock.name,
                    clock_hz=90000000, device='0000:04:00.1',
                    memory_map=dict(hbm_bytes=4 << 30, memory_ports=3, window_base=0))

    def probe(self, board):
        self.probes.append(board)
        uuid = self.after if (self.after and len(self.probes) % 2 == 0) else self.loaded
        return dict(host=socket.gethostname(), loaded_uuid=uuid, clock_counter_mhz=[90, 450],
                    contexts=0, client_pid=None)

    def spec(self, board, *, counters=True, **step):
        code = "open('board-count','a').write('x')"
        step = dict(dict(id='measure', kind='fpga', argv=[sys.executable, '-c', code], timeout_seconds=5,
                         timing_boundary='device-graph-makespan steady call', cache_state='fresh'), **step)
        spec = dict(id='board-test', allowed_hosts=[socket.gethostname()], hypothesis='h',
                    competing_explanations=['c'], expected_observations=['o'], sources=['source'],
                    board=board, steps=[step])
        if counters:
            spec['expected_counter_changes'] = ['outlet lines unchanged']
        self.manifest.write_text(json.dumps(spec))

    def execute(self, **kwargs):
        return run(self.manifest, self.root, self.root/'results', python=Path(sys.executable),
                   board_probe=self.probe, **kwargs)

    def test_two_images_differing_only_in_sha_never_dedup(self):
        a = self.image('a.xclbin', UUID_A, b'build-1')
        b = self.image('b.xclbin', UUID_A, b'build-2')
        self.assertNotEqual(a['image_sha256'], b['image_sha256'])
        self.spec(a)
        first = self.execute()
        again = self.execute()
        self.spec(b)
        other = self.execute()
        self.assertEqual(first['status'], 'completed-experiment')
        self.assertTrue(again['steps'][0]['reused'])
        self.assertFalse(other['steps'][0]['reused'])
        self.assertNotEqual(first['steps'][0]['fingerprint'], other['steps'][0]['fingerprint'])
        self.assertEqual((self.root/'board-count').read_text(), 'xx')
        row = first['steps'][0]
        self.assertEqual(row['board']['image_uuid'], UUID_A)
        self.assertEqual(row['board']['image_sha256'], a['image_sha256'])
        self.assertEqual(row['board']['clock_hz'], 90000000)
        self.assertEqual(row['board']['host'], socket.gethostname())
        self.assertEqual(row['board']['probe_before']['clock_counter_mhz'][0], 90)
        self.assertIn('probe_after', row['board'])
        self.assertTrue(first['identity']['software_tree'].startswith('files:'))

    def test_clock_evidence_and_memory_map_enter_the_key(self):
        a = self.image('a.xclbin', UUID_A, b'build-1')
        self.spec(a)
        first = self.execute()
        clock = self.root/a['clock_evidence']
        clock.write_text(json.dumps(dict(json.loads(clock.read_text()), qualification='requalified')))
        second = self.execute()
        self.spec(dict(a, memory_map=dict(a['memory_map'], hbm_bytes=2 << 30)))
        third = self.execute()
        self.assertEqual(len({r['steps'][0]['fingerprint'] for r in (first, second, third)}), 3)
        self.assertFalse(any(r['steps'][0]['reused'] for r in (second, third)))

    def test_wrong_loaded_image_refuses_without_running(self):
        self.spec(self.image('a.xclbin', UUID_A, b'build-1'))
        self.loaded = UUID_B
        report = self.execute()
        self.assertEqual(report['status'], 'refused-board-identity')
        self.assertIn(UUID_B, report['refusal']['reason'])
        self.assertFalse((self.root/'board-count').exists())
        self.assertFalse(list((self.root/'results/steps').rglob('attempt-*')))

    def test_busy_card_and_other_clock_refuse(self):
        self.spec(self.image('a.xclbin', UUID_A, b'build-1'))
        busy = lambda board: dict(loaded_uuid=UUID_A, clock_counter_mhz=[90], contexts=1, client_pid=None)
        report = run(self.manifest, self.root, self.root/'results', python=Path(sys.executable), board_probe=busy)
        self.assertIn('in use', report['refusal']['reason'])
        fast = lambda board: dict(loaded_uuid=UUID_A, clock_counter_mhz=[100], contexts=0, client_pid=None)
        report = run(self.manifest, self.root, self.root/'results', python=Path(sys.executable), board_probe=fast)
        self.assertIn('clock', report['refusal']['reason'])
        self.assertFalse((self.root/'board-count').exists())

    def test_image_swapped_during_step_fails_it(self):
        self.spec(self.image('a.xclbin', UUID_A, b'build-1'))
        self.after = UUID_B
        report = self.execute()
        self.assertEqual(report['status'], 'stopped-on-failure')
        self.assertEqual(report['steps'][0]['status'], 'board-identity-changed')

    def test_product_tree_change_during_a_board_step_fails_it(self):
        self.spec(self.image('a.xclbin', UUID_A, b'build-1'),
                  argv=[sys.executable, '-c', "open('archbench/edited.py','w').write('x=1')"])
        report = self.execute()
        self.assertEqual(report['steps'][0]['status'], 'source-changed')

    def test_declared_identity_must_match_the_files(self):
        a = self.image('a.xclbin', UUID_A, b'build-1')
        for bad, message in ((dict(a, image_sha256='0'*64), 'does not match'),
                             (dict(a, image_uuid=UUID_B), 'does not carry'),
                             (dict(a, clock_hz=100000000), 'qualified clock'),
                             ({k: v for k, v in a.items() if k != 'memory_map'}, 'lacks board identity')):
            self.spec(bad)
            with self.assertRaisesRegex(ValueError, message):
                self.execute()
        other = self.image('b.xclbin', UUID_A, b'build-2')
        self.spec(dict(a, clock_evidence=other['clock_evidence']))
        with self.assertRaisesRegex(ValueError, 'another image'):
            self.execute()
        self.spec(a, timing_boundary=None)
        with self.assertRaisesRegex(ValueError, 'timing boundary'):
            self.execute()
        self.spec(a, counters=False)
        with self.assertRaisesRegex(ValueError, 'expected counter changes'):
            self.execute()
        self.assertFalse((self.root/'board-count').exists())


if __name__ == '__main__':
    unittest.main()
