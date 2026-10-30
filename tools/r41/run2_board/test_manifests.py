import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import auto
import manifests
import registrations
import stage


class Run2Manifests(unittest.TestCase):
    def test_registered_manifests_and_machine_identity(self):
        jobs = manifests.jobs()
        self.assertEqual(len(jobs), 23)
        for source, name in jobs:
            spec = manifests.derive(source, name)
            path = manifests.ROOT / manifests.OUT / (name + '.json')
            self.assertEqual(json.loads(path.read_text()), spec)
            board = spec['board']
            self.assertEqual((board['image_sha256'], board['image_uuid'], board['profile']),
                             (auto.IMAGE_SHA, auto.IMAGE_UUID, stage.PROFILE))
            self.assertEqual(board['backend_args'], manifests.backend())
            for step in spec['steps']:
                env = step.get('env', {})
                if 'TNCA_BACKEND_ARGS' in env:
                    args = json.loads(env['TNCA_BACKEND_ARGS'])
                    self.assertEqual(args['xclbin'], manifests.IMAGE)
                    self.assertEqual(args['image_profile'], stage.PROFILE)
                    self.assertEqual(args['hbm_bytes'], 16 << 30 if step['id'] == 'clamp' else 8 << 30)

    def test_both_bandwidth_suites_are_complete(self):
        suite = manifests.ROOT/'archbench/suites/r41_membound.json'
        for mode, tiled in (('default', '0'), ('tiled2', '2')):
            rows = [{'manifest': str(p.relative_to(manifests.ROOT))} for p in
                    sorted((manifests.ROOT/manifests.OUT).glob('r2-membound-*-' + mode + '.json'))]
            self.assertEqual(len(rows), 6)
            stage.check_coverage(manifests.ROOT, rows, dict(suite=str(suite.relative_to(manifests.ROOT)),
                                 suite_sha256=stage.sha(suite), count=513, tiled_pointwise=tiled))

    def test_fp8_workloads_require_both_run2_instructions(self):
        for name in ('r2-s2-fp8-cases', 'r2-s2-fp8-blockwise'):
            spec = json.loads((manifests.ROOT/manifests.OUT/(name+'.json')).read_text())
            args = next(step['argv'] for step in spec['steps'] if step['id'] == 'summary')
            self.assertIn('--require-x8', args)
            self.assertIn('--require-dp', args)
            self.assertIn('run-2', args)

    def test_every_required_stage_has_registered_results_and_current_hashes(self):
        for name in auto.REQUIRED:
            registration = registrations.group(name)
            self.assertEqual(json.loads((manifests.ROOT/registrations.OUT/(name+'.json')).read_text()), registration)
            self.assertTrue(registration['experiments'])
            for experiment in registration['experiments']:
                self.assertTrue(experiment['checks'])
                self.assertEqual(experiment['sha256'], stage.sha(manifests.ROOT/experiment['manifest']))


if __name__ == '__main__':
    unittest.main()
