#!/usr/bin/env python3
"""Local dry runs of the R41 Stage 2 board batch: manifests, clock qualification, endurance, queue.

Nothing here touches a card: the board is a fake image file, a fake clock
evidence file, a fake sysfs tree and a stub for every fpga command.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'python'))
import clock_qualify          # noqa: E402
import endurance_mixed        # noqa: E402
import experiment             # noqa: E402
import stage2_board           # noqa: E402

UUID = stage2_board.UUID
QUEUE = ROOT / 'evidence/r41/stage2/board/launches/board_queue.sh'
RUN1 = ROOT / 'evidence/r41/stage2/run1/run1-reports.tgz'

STUB = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ['STUB_LOG'], 'a') as fh:
    fh.write(json.dumps(dict(argv=sys.argv[1:], backend=os.environ.get('TNCA_BACKEND'),
                             args=os.environ.get('TNCA_BACKEND_ARGS'), profile=os.environ.get('TNCA_PROFILE'),
                             clock=os.environ.get('TNCA_OPERATING_CLOCK_EVIDENCE'))) + '\n')
for out in json.loads(os.environ.get('STUB_OUTPUTS', '[]')):
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(status='complete', cases=[dict(case='c', status='pass')], result='pass', residual='none',
               conclusion='stub', next_step='none')
    if 'hbm8_case.py' in ' '.join(sys.argv):     # the hbm8 report shape hbm8_summary.py reads
        row.update(status='functional-pass', mode='board', software_tree='stub', checks=[dict(name='c', ok=True)],
                   axi_status_at_end=dict(error=False, stray=False))
    p.write_text(json.dumps(row) + '\n')
    k = p.parent / 'dump' / 'k'; k.mkdir(parents=True, exist_ok=True)
    (k / 'k.s').write_text('    t.ld.async.b.t v0, a0\n')
sys.exit(int(os.environ.get('STUB_EXIT', '0')))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_image(directory):
    image = Path(directory) / 'fake.xclbin'
    image.write_bytes(b'xclbin2\0' + bytes(64) + bytes.fromhex(UUID.replace('-', '')) + bytes(256))
    return image


def fake_sysfs(directory, uuid=UUID, mhz=100, contexts=0, client=''):
    base = Path(directory)
    dev, mgmt = base / '0000:04:00.1', base / '0000:04:00.0/clock_freq_counter.m.28311552'
    dev.mkdir(parents=True, exist_ok=True)
    mgmt.mkdir(parents=True, exist_ok=True)
    (dev / 'xclbinuuid').write_text(uuid + '\n')
    (mgmt / 'clock_counter_freqs').write_text(f'{mhz}\n500\n450\n0\n')
    (dev / 'kdsstat').write_text(f'xclbin:\t\t\t{uuid}\noutstanding execs:\t0\ntotal execs:\t\t0\n'
                                 f'contexts:\t\t{contexts}\nclient pid:\n\t\t\t{client}\n')
    return base


class Manifests(unittest.TestCase):
    def test_registered_equals_generated(self):
        for name in stage2_board.ORDER:
            registered = json.loads((stage2_board.SPEC / f'{name}.json').read_text())
            self.assertEqual(registered, stage2_board.manifest(name), name)

    def test_board_blocks(self):
        for name in stage2_board.ORDER:
            spec = stage2_board.manifest(name)
            b = spec['board']
            self.assertEqual((b['image_sha256'], b['image_uuid'], b['clock_hz'], b['profile']),
                             (stage2_board.IMAGE_SHA256, UUID, 100_000_000, 'core-vrf-sync-v5-x8'), name)
            self.assertEqual(b['backend_args']['image_profile'], 'core-vrf-sync-v5-x8')
            for step in spec['steps']:
                if step['kind'] == 'fpga' and 'env' in step:
                    env = step['env']
                    self.assertEqual(env['TNCA_PROFILE'], 'core-vrf-sync-v5-x8', (name, step['id']))
                    self.assertEqual(env['TNCA_OPERATING_CLOCK_EVIDENCE'], stage2_board.CLOCK)
                    self.assertEqual(json.loads(env['TNCA_BACKEND_ARGS'])['xclbin'], stage2_board.IMAGE)

    def test_image_source_matches_tree(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(stage2_board.source_check(ROOT, Path(d) / 's.json'), 0)


class ExperimentDryRun(unittest.TestCase):
    """experiment.py over every manifest: fake board, stubbed fpga commands, real report steps."""

    def prepare(self, d, name):
        d = Path(d)
        image = fake_image(d)
        clock = d / 'qualified_clock.json'
        clock.write_text(json.dumps(dict(schema='tnca-qualified-operating-clock/1', image_sha256=sha(image),
                                         qualification='r41s2-run1-100mhz', clock_hz=100_000_000)))
        stub = d / 'stub.py'
        stub.write_text(STUB)
        stub.chmod(0o755)
        spec = copy.deepcopy(stage2_board.manifest(name))
        spec['board'].update(image=str(image), image_sha256=sha(image), clock_evidence=str(clock))
        spec['board']['backend_args']['xclbin'] = str(image)
        for step in spec['steps']:
            if step['kind'] == 'fpga':
                step['argv'] = [str(stub)] + step['argv'][1:]
                step.setdefault('env', {})['STUB_OUTPUTS'] = json.dumps(step.get('outputs', []))
                if 'board' in step and 'backend_args' in step['board']:
                    step['board']['backend_args']['xclbin'] = str(image)
            else:
                step['argv'] = [sys.executable] + step['argv'][1:]
        path = d / f'{name}.json'
        path.write_text(json.dumps(spec))
        return path, d / 'out', d / 'stub.log'

    def run_one(self, name, probe):
        with tempfile.TemporaryDirectory() as d:
            path, out, log = self.prepare(d, name)
            with mock.patch.object(socket, 'gethostname', return_value='acf3019'), \
                    mock.patch.dict(os.environ, STUB_LOG=str(log)):
                report = experiment.run(path, ROOT, out, python=Path(sys.executable), board_probe=probe)
            calls = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
            return report, calls

    def test_every_manifest_completes_on_the_fake_board(self):
        good = lambda board: dict(loaded_uuid=UUID, clock_counter_mhz=[100, 500, 450, 0], contexts=0, client_pid=None)
        for name in stage2_board.ORDER:
            report, calls = self.run_one(name, good)
            self.assertEqual(report['status'], 'completed-experiment', (name, report.get('refusal'),
                                                                        [s.get('status') for s in report['steps']]))
            fpga = [s for s in report['steps'] if s['kind'] == 'fpga']
            self.assertTrue(fpga, name)
            for s in fpga:
                self.assertEqual(s['board']['clock_hz'], 100_000_000)
                self.assertEqual(s['board']['image_uuid'], UUID)
            if name not in ('s2-identity', 's2-anchors', 's2-original-suite', 's2-endurance'):
                self.assertTrue(all(c['profile'] == 'core-vrf-sync-v5-x8' for c in calls), name)
            if name == 's2-fp8-cases':
                self.assertEqual(report['result'], 'pass')

    def test_refusals(self):
        cases = [(dict(loaded_uuid=UUID, clock_counter_mhz=[90, 500, 450, 0], contexts=0), 'clock counter'),
                 (dict(loaded_uuid='1bada9db-78f3-5bc8-7ba6-cce062e7ade8', clock_counter_mhz=[90], contexts=0),
                  'loaded image'),
                 (dict(loaded_uuid=UUID, clock_counter_mhz=[100], contexts=1, client_pid='77'), 'card in use')]
        for row, why in cases:
            report, calls = self.run_one('s2-fp8-cases', lambda board, row=row: row)
            self.assertEqual(report['status'], 'refused-board-identity')
            self.assertIn(why, report['refusal']['reason'])
            self.assertEqual(calls, [])


class ClockQualify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.image = fake_image(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self):
        audit = self.d / 'audit'
        audit.mkdir()
        with tarfile.open(RUN1) as t:
            for n in ('timing.rpt', 'route_status.rpt', 'cdc.rpt'):
                (audit / n).write_bytes(t.extractfile(f'run1-reports/analysis/{n}').read())
        (audit / 'analysis.tsv').write_text('checkpoint\tx.dcp\nrequested_mhz\t100\noriginal_period_ns\t10.000\n'
                                            'analysis_period_ns\t10.000\n')
        (audit / 'actual_cores.rpt').write_text(''.join(
            f'core {i} primitive_count 1 placed 1 unplaced 0 excluded_constants 0 actual_slrs {{SLR{i}}} '
            f'expected SLR{i} pass 1\n' for i in range(3)) + 'failing_cores {}\n')
        return audit

    def test_capture_and_assemble(self):
        sysfs = fake_sysfs(self.d / 'sys')
        cap = clock_qualify.capture(sysfs, self.image, UUID, interval=0)
        self.assertEqual(cap['clock_hz'], 100_000_000)
        (self.d / 'cap.json').write_text(json.dumps(cap))
        with mock.patch.object(clock_qualify, 'IMAGE_SHA256', sha(self.image)):
            row = clock_qualify.assemble(self.image, self.audit(), self.d / 'cap.json')
        self.assertEqual((row['setup_wns_ns'], row['hold_whs_ns'], row['clock_hz']), (0.016, 0.003, 100_000_000))
        self.assertEqual(row['physical_core_slrs'], {'0': 0, '1': 1, '2': 2})
        self.assertEqual(row['cdc_review_scope']['critical'], 56)
        out = self.d / 'q.json'
        out.write_text(json.dumps(row))
        # The 100 MHz review is registered for the run-1 image only.
        with self.assertRaisesRegex(ValueError, 'another image'):
            clock_qualify.check(out, self.image)
        from tnca import operating_clock
        with mock.patch.dict(operating_clock.QUALIFICATIONS, {'r41s2-run1-100mhz': dict(
                image_sha256=sha(self.image), max_hz=100_000_000, exact_hz=100_000_000)}):
            self.assertEqual(clock_qualify.check(out, self.image)['clock_hz'], 100_000_000)

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            clock_qualify.capture(fake_sysfs(self.d / 'a', uuid='1bada9db-78f3-5bc8-7ba6-cce062e7ade8'),
                                  self.image, UUID, interval=0)
        cap = clock_qualify.capture(fake_sysfs(self.d / 'b', mhz=90), self.image, UUID, interval=0)
        (self.d / 'cap.json').write_text(json.dumps(cap))
        audit = self.audit()
        with mock.patch.object(clock_qualify, 'IMAGE_SHA256', sha(self.image)):
            with self.assertRaisesRegex(SystemExit, 'measured clock 90000000'):
                clock_qualify.assemble(self.image, audit, self.d / 'cap.json')
            cap['clock_hz'] = 100_000_000
            (self.d / 'cap.json').write_text(json.dumps(cap))
            text = (audit / 'cdc.rpt').read_text().splitlines()
            drop = next(i for i, l in enumerate(text) if ' CDC-11  Critical' in l and l.strip()[0].isdigit())
            (audit / 'cdc.rpt').write_text('\n'.join(text[:drop] + text[drop + 1:]))
            with self.assertRaisesRegex(SystemExit, 'attempt-3 review'):
                clock_qualify.assemble(self.image, audit, self.d / 'cap.json')


class Endurance(unittest.TestCase):
    def run_fake(self, refuse_at=None, duration=1.0):
        import board_batch

        class FakeBatch:
            def __init__(self, reg, **kw):
                self.runner, self.n = None, 0

            def check_card(self):
                pass

            def run_one(self, cfg, tag, warmup, timed):
                self.n += 1
                if refuse_at and self.n >= refuse_at:
                    raise board_batch.Refused('loaded image 1bada9db is not the declared image')
                cut = self.runner.index('--') + 1
                subprocess.run(self.runner[:cut] + [sys.executable, '-c', 'x = bytearray(20 << 20)'], check=True)
                return dict(ok=True, problems=[], raw='r', seconds=0.01, serial_grid_samples=[1000])

        with tempfile.TemporaryDirectory() as d:
            args = types.SimpleNamespace(manifest=stage2_board.SPEC / 's2-endurance.json', batch='mix', root=ROOT,
                                         out=Path(d) / 'o', python=Path(sys.executable), duration=duration,
                                         fp8_every=1, seed=1, sample_seconds=0.2)
            rec = endurance_mixed.run(args, probe=None, batch_cls=FakeBatch,
                                      fp8_runner=lambda *a: dict(ok=True, problems=[], cases=4))
            sessions = [json.loads(l) for l in (Path(d) / 'o/sessions.jsonl').read_text().splitlines()]
            return rec, sessions

    def test_sessions_and_rss(self):
        rec, sessions = self.run_fake()
        self.assertEqual(rec['result'], 'pass')
        self.assertGreaterEqual(rec['sessions'], 9)
        self.assertTrue(any(s['kind'] == 'fp8' for s in sessions))
        rss = [s['max_rss_kib'] for s in sessions if s['kind'] == 'point']
        self.assertTrue(all(r and r > 20 * 1024 for r in rss))

    def test_stops_on_card_change(self):
        rec, sessions = self.run_fake(refuse_at=3, duration=60)
        self.assertEqual(rec['result'], 'fail')
        self.assertIn('stopped', ' '.join(rec['residual']))
        self.assertEqual(sum(1 for s in sessions if s['kind'] == 'point'), 3)


FAKE_PY = r'''#!/bin/bash
echo "$*" >> $T/py.calls
case "$1" in
  tools/performance_harness/clock_qualify.py)
    out=$(echo "$*" | sed 's/.*--out \([^ ]*\).*/\1/'); echo '{}' > $out
    [ "$2" = assemble ] && [ -n "${ASSEMBLE_FAIL:-}" ] && exit 1; exit 0 ;;
  tools/performance_harness/experiment.py)
    m=$(basename $2 .json); echo $m >> $T/manifests
    case " ${FAIL_MANIFESTS:-} " in *" $m "*) echo stopped-on-failure; exit 1;; esac
    echo completed-experiment; exit 0 ;;
esac
exit 0
'''
FAKE_PROGRAM = r'''#!/bin/bash
echo "$*" >> $T/program.calls
if [ "$1" = --go ]; then
  echo 49964b17-e0b4-5bd2-dd40-7fc37b1205d3 > $T/sys/0000:04:00.1/xclbinuuid
  printf "${LOAD_MHZ:-100}\n500\n450\n0\n" > $T/sys/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs
fi
exit 0
'''
FAKE_XCLBINUTIL = r'''#!/bin/bash
cat <<EOF
   UUID (xclbin):          49964b17-e0b4-5bd2-dd40-7fc37b1205d3
   Name:      DATA_CLK
   Index:     0
   Type:      DATA
   Frequency: 100 MHz
EOF
'''
FAKE_VIVADO = r'''#!/bin/bash
args=("$@"); for i in "${!args[@]}"; do [ "${args[$i]}" = -tclargs ] && out=${args[$((i+2))]}; done
mkdir -p $out && echo 'core 0 ...' > $out/actual_cores.rpt && echo R40_OPERATING_CLOCK_ANALYSIS_DONE
'''


class Queue(unittest.TestCase):
    """board_queue.sh on a fake home and a fake sysfs; programs are stubs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.t = Path(self.tmp.name)
        (t / 'bin').mkdir()
        for name, text in (('py', FAKE_PY), ('program.sh', FAKE_PROGRAM), ('xclbinutil', FAKE_XCLBINUTIL),
                           ('vivado', FAKE_VIVADO)):
            (t / 'bin' / name).write_text(text)
            (t / 'bin' / name).chmod(0o755)
        img = t / 'tnca_images/r41s2_run1_c9f25787494a'
        img.mkdir(parents=True)
        (img / 'tnca_u280.xclbin').write_bytes(b'x')
        (img / 'level0_wrapper_routed.dcp').write_bytes(b'd')
        src = t / 'archive'
        (src / 'flow/fpga').mkdir(parents=True)
        (src / 'flow/fpga/audit_r40_operating_clock.tcl').write_text('# stub\n')
        with tarfile.open(t / 'r41s2_board_abc.tar.gz', 'w:gz') as tf:
            tf.add(src, arcname='.')
        (t / 'stage1.log').write_text('... BOARD-QUEUE-DONE\n')
        (t / 'hbm8.log').write_text('hbm8 queue start\n')
        self.sys = fake_sysfs(t / 'sys', uuid='428c00e5-6996-92c7-1f8f-ec41f2d037b3', mhz=300)
        self.env = dict(os.environ, T=str(t), S2_HOME=str(t), S2_SYSFS=str(self.sys), S2_PYTHON=str(t / 'bin/py'),
                        S2_PROGRAM=str(t / 'bin/program.sh'), S2_VIVADO=str(t / 'bin/vivado'),
                        S2_STAGE1_LOG=str(t / 'stage1.log'), S2_HBM8_LOG=str(t / 'hbm8.log'),
                        S2_STAGE1_PATTERN='^no-such-process-r41s1$', S2_HBM8_PATTERN='^no-such-process-hbm8$',
                        S2_IDLE_MINUTES='2', S2_POLL_SECONDS='0', S2_LOAD_WAIT_SECONDS='2',
                        S2_TEST_XSHA=sha(img / 'tnca_u280.xclbin'),
                        S2_TEST_DCPSHA=sha(img / 'level0_wrapper_routed.dcp'),
                        S2_MEMINFO=str(t / 'meminfo'), S2_AUDIT_MIN_KB='1',
                        S2_MANIFESTS='s2-identity s2-anchors s2-fp8-hub s2-fp8-cases',
                        PATH=f'{t / "bin"}:{os.environ["PATH"]}')
        (t / 'meminfo').write_text('MemAvailable:   40000000 kB\n')

    def tearDown(self):
        self.tmp.cleanup()

    def queue(self, timeout=60, **env):
        return subprocess.run(['bash', str(QUEUE), 'abc'], env=dict(self.env, **env), timeout=timeout,
                              capture_output=True, text=True)

    def lines(self, name):
        p = self.t / name
        return p.read_text().split() if p.exists() else []

    def log(self):
        return (self.t / 'r41s2_board_abc/queue.log').read_text()

    def test_load_qualify_and_run(self):
        r = self.queue()
        self.assertEqual(r.returncode, 0, self.log())
        self.assertIn('S2-BOARD-QUEUE-DONE', self.log())
        calls = (self.t / 'program.calls').read_text().splitlines()
        self.assertEqual([c.split()[0] for c in calls], ['--check', '--go', '--status'])
        self.assertEqual(self.lines('manifests'), ['s2-identity', 's2-anchors', 's2-fp8-cases'])
        self.assertIn('manifest s2-fp8-hub skipped', self.log())
        self.assertIn('loaded uuid 49964b17-e0b4-5bd2-dd40-7fc37b1205d3 clock counter 100 MHz', self.log())
        # A rerun does not load again and does not redo the audit.
        r = self.queue()
        self.assertEqual(r.returncode, 0, self.log())
        self.assertEqual(len((self.t / 'program.calls').read_text().splitlines()), 3)

    def test_wrong_clock_after_load_stops(self):
        r = self.queue(LOAD_MHZ='90')
        self.assertEqual(r.returncode, 5, self.log())
        self.assertEqual(self.lines('manifests'), [])

    def test_identity_failure_stops_and_other_failure_continues(self):
        r = self.queue(FAIL_MANIFESTS='s2-anchors')
        self.assertEqual(r.returncode, 0, self.log())
        self.assertEqual(self.lines('manifests'), ['s2-identity', 's2-anchors', 's2-fp8-cases'])
        (self.t / 'manifests').unlink()
        r = self.queue(FAIL_MANIFESTS='s2-identity')
        self.assertEqual(r.returncode, 7, self.log())
        self.assertEqual(self.lines('manifests'), ['s2-identity'])

    def test_busy_card_never_loads(self):
        fake_sysfs(self.sys, uuid='428c00e5-6996-92c7-1f8f-ec41f2d037b3', mhz=300, contexts=1, client='4242')
        with self.assertRaises(subprocess.TimeoutExpired):
            self.queue(timeout=5)
        self.assertFalse((self.t / 'program.calls').exists())

    def test_waits_for_running_queue(self):
        (self.t / 'stage1.log').write_text('queue start\n')
        sleeper = subprocess.Popen(['bash', '-c', 'exec -a fake-r41s1-queue sleep 30'])
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.queue(timeout=5, S2_STAGE1_PATTERN='^fake-r41s1-queue')
        finally:
            sleeper.kill()
            sleeper.wait()
        self.assertFalse((self.t / 'program.calls').exists())
        self.assertNotIn('card idle', self.log())


if __name__ == '__main__':
    unittest.main()
