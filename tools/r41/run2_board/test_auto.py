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


class CandidateQueueTests(unittest.TestCase):
    def test_candidate_cannot_use_approval_for_another_image(self):
        import json
        import sys
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('candidate_queue_approval_test', folder/'candidate_queue.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'approval.json'
            path.write_text(json.dumps(dict(status='accepted', image_sha256=auto.IMAGE_SHA,
                                             image_uuid=auto.IMAGE_UUID, author_decision='original image only')))
            with self.assertRaisesRegex(ValueError, 'exact candidate image'):
                mod.approval(path, 'f'*64, auto.IMAGE_UUID)

    def test_new_client_before_go_stops_programming(self):
        import sys
        from unittest.mock import patch
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('candidate_queue_test', folder/'candidate_queue.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image'
            image.write_bytes(b'candidate')
            idle_row = row()
            busy_row = row(contexts=1)
            state = dict(image_sha256=auto.sha(image), image_uuid=auto.IMAGE_UUID,
                         evidence={}, registrations=[])
            argv = ['candidate_queue.py','--root',str(root),'--work',str(root),'--image',str(image),
                    '--approval',str(root/'approval.json'),'--out',str(root/'output'),'--python',sys.executable]
            tracker = mod.auto.IdleTracker
            with patch.object(sys, 'argv', argv), patch.object(mod, 'preflight', return_value=state), \
                 patch.object(mod.auto, 'board_environment', return_value={}), \
                 patch.object(mod.auto, 'IdleTracker', side_effect=lambda **kw: tracker(seconds=0, **kw)), \
                 patch.object(mod.auto, 'probe', side_effect=[idle_row,idle_row,busy_row]), \
                 patch.object(mod.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'busy before programming'):
                    mod.main()
                self.assertEqual(run.call_count, 1)
                self.assertIn('--check', run.call_args.args[0])
                self.assertNotIn('--go', run.call_args.args[0])


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

    def test_candidate_clock_rejects_unreviewed_cdc_endpoint(self):
        import json
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'python'))
        from tnca.operating_clock import qualified_clock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_sha = 'a'*64
            uid = '11111111-2222-3333-4444-555555555555'
            names = ('timing', 'route', 'cdc', 'board_clock', 'placement', 'clock_review',
                     'cdc_review', 'bus_skew', 'failing_setup')
            paths = {n: root/(n+'.json') for n in names}
            for name in ('timing', 'route', 'bus_skew', 'failing_setup'):
                paths[name].write_text(name)
            paths['cdc'].write_text(' 1 CDC-10 Critical combinational 2 False Path source/C dest/D\n')
            paths['placement'].write_text(''.join(f'core {i} actual_slrs {{SLR{i}}} expected SLR{i} pass 1\n' for i in range(3))+'failing_cores {}\n')
            paths['board_clock'].write_text(json.dumps(dict(image_sha256=image_sha, loaded_xclbin_uuid=uid,
                clock_hz=100_000_000, counter_samples=[{'frequencies_mhz': [100]}]*3)))
            cdc = dict(schema='tnca-candidate-cdc-review/1', kind='cdc-review', status='pass',
                image_sha256=image_sha, cdc_report_sha256=auto.sha(paths['cdc']),
                reviewed_critical=[dict(rule='CDC-10', source='source/C', destination='dest/D',
                                        disposition='accepted-platform-scoped', review='reviewed')])
            paths['cdc_review'].write_text(json.dumps(cdc))
            review = dict(schema='tnca-candidate-clock-review/1', kind='clock-review', status='pass',
                image_sha256=image_sha, image_uuid=uid, clocks_mhz={'DATA_CLK': 100, 'hbm_aclk': 450},
                original_design_timing=dict(wns=0.02, whs=0.001, setup_failing=0, hold_failing=0),
                kernel_timing=dict(wns=0.04, whs=0.001, setup_failing=0, hold_failing=0),
                pulse_width_timing=dict(wpws=0.0, tpws_failing=0),
                route=dict(routable=1, fully_routed=1, errors=0, unrouted=0),
                report_sha256={raw: auto.sha(paths[name]) for name,raw in
                    (('timing','timing.rpt'), ('route','route_status.rpt'), ('bus_skew','bus_skew.rpt'),
                     ('failing_setup','failing_setup_all_groups.tsv'))})
            paths['clock_review'].write_text(json.dumps(review))
            data = dict(schema='tnca-qualified-operating-clock/1', image_sha256=image_sha,
                qualification='r41-corrected-candidate-100mhz', clock_hz=100_000_000,
                setup_scope='full-design', setup_wns_ns=0.02, hold_whs_ns=0.001, pulse_width_wpws_ns=0.0,
                route_legal=True, cdc_review='passed', physical_core_slrs={'0': 0, '1': 1, '2': 2},
                clock_source='measured-board-counter', evidence_files={n: dict(path=str(p), sha256=auto.sha(p)) for n,p in paths.items()})
            qualified = root/'qualified.json'
            qualified.write_text(json.dumps(data))
            self.assertEqual(qualified_clock(qualified, image_sha)['clock_hz'], 100_000_000)
            cdc['reviewed_critical'][0]['destination'] = 'other/D'
            paths['cdc_review'].write_text(json.dumps(cdc))
            data['evidence_files']['cdc_review']['sha256'] = auto.sha(paths['cdc_review'])
            qualified.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'exact critical endpoints'):
                qualified_clock(qualified, image_sha)

    def test_candidate_identity_does_not_capture_without_cdc_review(self):
        import json
        import sys
        from unittest.mock import patch
        folder = Path(__file__).parent
        sys.path.insert(0, str(folder))
        sys.modules.setdefault('auto', auto)
        spec = importlib.util.spec_from_file_location('candidate_identity_test', folder/'candidate_identity.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image'
            image.write_bytes(b'candidate')
            review = root/'clock.json'
            review.write_text(json.dumps({'status': 'pass'}))
            with patch.object(mod.candidate_clock, 'review', return_value={'status': 'pass'}), \
                 patch.object(mod.clock_qualify, 'capture') as capture:
                with self.assertRaises(FileNotFoundError):
                    mod.qualify(image, root, review, root/'missing-cdc.json', root/'capture.json',
                                root/'qualified.json', root/'sysfs', 'xclbinutil')
                capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
