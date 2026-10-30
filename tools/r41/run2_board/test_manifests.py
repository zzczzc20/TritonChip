import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import auto
import manifests
import registrations
import stage
import candidate_manifests


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
        used = []
        for name in auto.REQUIRED:
            registration = registrations.group(name)
            self.assertEqual(json.loads((manifests.ROOT/registrations.OUT/(name+'.json')).read_text()), registration)
            self.assertTrue(registration['experiments'])
            for experiment in registration['experiments']:
                used.append(experiment['id'])
                self.assertTrue(experiment['checks'])
                self.assertEqual(experiment['sha256'], stage.sha(manifests.ROOT/experiment['manifest']))
        self.assertCountEqual(used, [name for _, name in manifests.jobs()])
        self.assertEqual(len(used), len(set(used)))

    def test_candidate_rebinds_every_workload_without_board_access(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'candidate.xclbin'
            image.write_bytes(b'candidate')
            digest = stage.sha(image)
            uid = '11111111-2222-3333-4444-555555555555'
            mapping = root/'candidate-map.json'
            data = json.loads((manifests.ROOT/manifests.MAP).read_text())
            data['image_sha256'] = digest
            mapping.write_text(json.dumps(data))
            info = 'UUID (xclbin): ' + uid + '\nName: DATA_CLK\n Index: 0\n Type: DATA\n Frequency: 100 MHz\n'
            class Result:
                stdout = info
            with patch.object(candidate_manifests, 'ROOT', root), \
                 patch.object(candidate_manifests.subprocess, 'run', return_value=Result()):
                self.assertEqual(candidate_manifests.prepare(image, digest, uid, mapping, 'xclbinutil'), 23)
            for path in (root/candidate_manifests.OUT/'board').glob('*.json'):
                spec = json.loads(path.read_text())
                self.assertEqual(spec['board']['image_sha256'], digest)
                self.assertEqual(spec['board']['image_uuid'], uid)
                self.assertTrue(spec['id'].startswith('r3-'))
                self.assertNotIn(auto.IMAGE_SHA, json.dumps(spec))
            stages = list((root/candidate_manifests.OUT/'stages').glob('*.json'))
            self.assertEqual(len(stages), len(auto.REQUIRED))
            for path in stages:
                reg = json.loads(path.read_text())
                self.assertEqual(reg['image_identity'], {'sha256': digest, 'uuid': uid})
                for row in reg['experiments']:
                    self.assertEqual(row['sha256'], stage.sha(root/row['manifest']))


if __name__ == '__main__':
    unittest.main()
