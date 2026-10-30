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
        from unittest.mock import patch
        folder = Path(__file__).parent
        spec = importlib.util.spec_from_file_location('run2_stage', folder/'stage.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'root'
            root.mkdir()
            manifest = root/'manifest.json'
            manifest.write_text('{}')
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


if __name__ == '__main__':
    unittest.main()
