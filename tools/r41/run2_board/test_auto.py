import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('run2_auto', Path(__file__).with_name('auto.py'))
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


def row(uuid='1c36f315-c7c1-c7f5-0de1-8c3e8d80e1dd', total=0, contexts=0):
    return {'uuid': uuid, 'total execs': total, 'contexts': contexts, 'outstanding execs': 0, 'client pid': ''}


class IdleTests(unittest.TestCase):
    def test_900_seconds_and_activity_reset(self):
        t = auto.IdleTracker(seconds=900, max_gap=15)
        for n in range(0, 901, 5):
            self.assertEqual(t.observe(n, row()), n == 900)
        self.assertFalse(t.observe(905, row(total=1)))
        self.assertFalse(t.observe(910, row(contexts=1)))
        self.assertFalse(t.observe(915, row(total=1)))

    def test_gap_and_image_change_reset(self):
        t = auto.IdleTracker(seconds=20, max_gap=10)
        self.assertFalse(t.observe(0, row()))
        self.assertFalse(t.observe(11, row()))
        self.assertFalse(t.observe(16, row(uuid=auto.IMAGE_UUID)))
        self.assertFalse(t.observe(21, row(uuid=auto.IMAGE_UUID)))

    def test_sysfs_missing_or_busy_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp)/'0000:04:00.1'
            dev.mkdir()
            (dev/'xclbinuuid').write_text(auto.IMAGE_UUID)
            (dev/'kdsstat').write_text('contexts: 0\noutstanding execs: 0\ntotal execs: 4\nclient pid:\n 1234\n')
            self.assertFalse(auto.idle(auto.probe(tmp)))
            (dev/'kdsstat').write_text('contexts: 0\n')
            with self.assertRaises(ValueError):
                auto.probe(tmp)

    def test_incomplete_plan_never_reaches_programming(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'bad.xclbin'
            image.write_bytes(b'not the approved image')
            with self.assertRaises(ValueError):
                auto.validate_plan({'schema': 'tnca-r41-run2-board-plan/1', 'stages': []}, image)


class ClockReviewTests(unittest.TestCase):
    def test_unapproved_image_is_rejected_before_running_tools(self):
        import sys
        import importlib.util
        from unittest.mock import patch
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location('clock_review_test', folder/'clock_review.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'image'
            image.write_bytes(b'not the approved xclbin')
            with patch.object(mod.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'explicitly approved'):
                    mod.review(image, tmp, 'xclbinutil')
                run.assert_not_called()


class QueuePreflightTests(unittest.TestCase):
    def test_incomplete_plan_stops_before_programming_even_with_approved_hash(self):
        import json
        import sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, plan, program = root/'image', root/'plan.json', root/'program.sh'
            image.write_bytes(b'fixture')
            program.write_text('exit 99\n')
            plan.write_text(json.dumps({'schema': 'tnca-r41-run2-board-plan/1', 'stages': []}))
            argv = ['auto.py', '--plan', str(plan), '--image', str(image),
                    '--program', str(program), '--out', str(root/'out')]
            with patch.object(sys, 'argv', argv), patch.object(auto, 'sha', return_value=auto.IMAGE_SHA), \
                 patch.object(auto.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'all required stages'):
                    auto.main()
                run.assert_not_called()
                self.assertFalse((root/'out').exists())


class SourceReviewTests(unittest.TestCase):
    def test_changed_frozen_source_is_rejected(self):
        import io
        import sys
        import tarfile
        from unittest.mock import patch
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('source_review_test', folder/'source_review.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'tree'
            (root/'rtl').mkdir(parents=True)
            (root/'rtl/a').write_bytes(b'changed')
            (root/'image.tcl').write_bytes(b'image')
            archive = Path(tmp)/'source.tgz'
            with tarfile.open(archive, 'w:gz') as tar:
                for name, contents in [('rtl/a', b'original'), ('image.tcl', b'image')]:
                    info = tarfile.TarInfo(name)
                    info.size = len(contents)
                    tar.addfile(info, io.BytesIO(contents))
            with patch.object(mod, 'PATHS', ('rtl', 'image.tcl')), \
                 patch.object(mod, 'ARCHIVE_SHA', auto.sha(archive)):
                with self.assertRaisesRegex(ValueError, '1 changed'):
                    mod.review(archive, root)


class HbmReviewTests(unittest.TestCase):
    def test_original_mapping_passes_and_altered_map_is_rejected(self):
        import sys
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('hbm_review_test', folder/'hbm_review.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        root = folder.resolve().parents[2]
        mapping = root/'evidence/r41/hbm_bw/g4/run2/link_1f0a5aff/xclbin_map.json'
        self.assertEqual(mod.review(mapping)['allocated_bytes'], 8 << 30)
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp)/'map.json'
            altered.write_bytes(mapping.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                mod.review(altered)


class StageTests(unittest.TestCase):
    def test_failed_result_never_publishes_a_passing_stage(self):
        import json
        import sys
        from unittest.mock import patch
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('run2_stage', folder/'stage.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'root'
            root.mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'id': 'smoke', 'board': {
                'image_sha256': auto.IMAGE_SHA, 'image_uuid': auto.IMAGE_UUID,
                'clock_hz': 100_000_000, 'profile': mod.PROFILE,
                'backend_args': {'image_profile': mod.PROFILE, 'core_count': 3,
                    'memory_ports': 3, 'hbm_bytes': 8 << 30, 'hbm_dma': True}}}))
            out = Path(tmp)/'output'
            registration = {'experiments': [{'id': 'smoke', 'manifest': 'manifest.json',
                'sha256': mod.sha(manifest), 'checks': [{'path': 'batch.json',
                    'field': ['status'], 'equals': 'complete'}]}]}
            def fake_run(command, **kwargs):
                target = out/'smoke'
                target.mkdir()
                (target/'report.json').write_text(json.dumps({'status': 'completed-experiment'}))
                (target/'batch.json').write_text(json.dumps({'status': 'failed-preserved'}))
            with patch.object(mod.subprocess, 'run', side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, 'result check failed'):
                    mod.run(root, out, '/usr/bin/python3', registration)
            self.assertFalse((out/'stage.json').exists())
            wrong = json.loads(manifest.read_text())
            wrong['board']['image_sha256'] = '30e79769' * 8
            manifest.write_text(json.dumps(wrong))
            registration['experiments'][0]['sha256'] = mod.sha(manifest)
            with patch.object(mod.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'run-2 board configuration'):
                    mod.run(root, Path(tmp)/'other', '/usr/bin/python3', registration)
                run.assert_not_called()

    def test_suite_coverage_refuses_duplicate_points(self):
        import json
        import sys
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('run2_stage_coverage', folder/'stage.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root/'suite.json'
            suite.write_text(json.dumps({'points': [{'id': 'p', 'sha256': 'digest'}]}))
            case = {'point': 'p', 'point_sha256': 'digest', 'knobs': {}}
            (root/'manifest.json').write_text(json.dumps({'board_batches': {'b': {'configurations': [case]}}}))
            config = {'suite': 'suite.json', 'suite_sha256': mod.sha(suite), 'count': 1, 'tiled_pointwise': '0'}
            rows = [{'manifest': 'manifest.json'}]
            mod.check_coverage(root, rows, config)
            (root/'manifest.json').write_text(json.dumps({'board_batches': {'b': {'configurations': [case, case]}}}))
            with self.assertRaisesRegex(ValueError, 'more than once'):
                mod.check_coverage(root, rows, config)


class CdcIdentityTests(unittest.TestCase):
    def test_equal_count_does_not_hide_a_changed_endpoint(self):
        import json
        module = Path(__file__).resolve().parents[1]/'run2_postlink/check_run2.py'
        spec = importlib.util.spec_from_file_location('run2_postlink_test', module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            review = Path(tmp)/'review.json'
            review.write_text(json.dumps({'qualified_clock': {'cdc_review_scope': {
                'reviewed_critical': [{'rule': 'CDC-10', 'source': 'old/C', 'destination': 'irq/D'}]}}}))
            got = mod.cdc_identity(' 1 CDC-10 Critical combinational 2 False Path new/C irq/D', review)
            self.assertEqual(got['reported'], got['reviewed'])
            self.assertEqual(got['added'], [('CDC-10', 'new/C', 'irq/D')])
            self.assertEqual(got['absent'], [('CDC-10', 'old/C', 'irq/D')])


class WatchTests(unittest.TestCase):
    def test_missing_plan_only_waits(self):
        import subprocess
        import sys
        import time
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'watch'
            proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('watch_plan.py')),
                    '--plan', str(Path(tmp)/'missing.json'), '--image', str(Path(tmp)/'missing.xclbin'),
                    '--out', str(out), '--poll', '1'])
            try:
                for _ in range(50):
                    if (out/'watch.jsonl').exists() and (out/'watch.jsonl').stat().st_size:
                        break
                    time.sleep(.02)
                self.assertIn('waiting', (out/'watch.jsonl').read_text())
                self.assertFalse((out/'queue-process.log').exists())
            finally:
                proc.terminate()
                proc.wait(timeout=5)


class Run2QualificationTests(unittest.TestCase):
    def test_exception_cannot_be_misrepresented_as_full_design_closure(self):
        import json
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'python'))
        from tnca.operating_clock import qualified_clock
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'qualified.json'
            path.write_text(json.dumps({'schema': 'tnca-qualified-operating-clock/1',
                'image_sha256': auto.IMAGE_SHA, 'qualification': 'r41-run2-100mhz-hbm441',
                'clock_hz': 100_000_000, 'setup_scope': 'full-design',
                'original_design_setup_wns_ns': -0.045, 'hbm_metadata_mhz': 441,
                'setup_wns_ns': 0.038}))
            with self.assertRaisesRegex(ValueError, 'kernel-only'):
                qualified_clock(path, auto.IMAGE_SHA)
            with self.assertRaisesRegex(ValueError, 'another image'):
                qualified_clock(path, '0'*64)

    def test_identity_refuses_unapproved_image_before_clock_capture(self):
        import sys
        from unittest.mock import patch
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('run2_identity_test', folder/'identity.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'wrong.xclbin'
            image.write_bytes(b'wrong image')
            with patch.object(mod.clock_qualify, 'capture') as capture:
                with self.assertRaisesRegex(ValueError, 'explicitly approved'):
                    mod.capture_and_qualify(image, root, root/'cdc.json', root/'capture.json',
                                            root/'qualified.json', root/'sysfs', 'xclbinutil')
                capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
