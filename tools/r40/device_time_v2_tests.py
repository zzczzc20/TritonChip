#!/usr/bin/env python3
"""Exercise matched capability evidence and reject duplicates/reused sessions."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import contract_v2
import device_time_v2 as runner
from protocol_tests import record


class DeviceTimeTests(unittest.TestCase):
    def test_native_resident_inventory_is_separate_from_timing_controls(self):
        from native_inventory import audit
        with tempfile.TemporaryDirectory() as d:
            paths=self.fixture(Path(d))
            path=paths[1][0]
            self.assertFalse(all(c['ok'] for c in audit([path])))
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            for r in rows:
                r['platform']['knobs']['TNCA_TRANSFER_POLICY']='resident'
                r['host']={'upload_bytes':0,'download_bytes':0}
                r['time']['launches']=1
                r['native_memory']={'tensor_peak_bytes':1024}
            def save():path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save()
            self.assertTrue(all(c['ok'] for c in audit([path])))
            for r in rows:r['platform']['backend']='fpga-u280-multicore'
            save()
            self.assertTrue(all(c['ok'] for c in audit([path])))
            rows[0]['platform']['backend']='fpga-multicore-sim';save()
            self.assertFalse(all(c['ok'] for c in audit([path])))
            rows[0]['platform']['backend']='fpga-u280-multicore'
            rows[0]['host']['upload_bytes']=64;save()
            self.assertFalse(all(c['ok'] for c in audit([path])))
            rows[0]['host']['upload_bytes']=0
            rows[0]['point_sha256']='changed';save()
            self.assertFalse(all(c['ok'] for c in audit([path])))

    def fixture(self,root,slowdown=1.0):
        plan=json.loads((contract_v2.ROOT/'spec/r40/v2/plan.json').read_text())
        pts=json.loads((contract_v2.ROOT/'spec/r40/suites/existing.json').read_text())['points']
        pts.sort(key=lambda p:p['id'])
        rng=random.Random(plan['sample_plan']['bootstrap']['seed'])
        paths=[[],[]]
        for block in range(3):
            records=[[],[]]
            for i,pt in enumerate(pts):
                first=rng.randrange(2)
                for side in (0,1):
                    r=record(pt,.001*(slowdown if side else 1),kind='device-cycles-serial-grid')
                    r['platform']['knobs']={'TNCA_TORCH_DEVICE':'native' if side else 'cpu','TNCA_TRANSFER_POLICY':'per-kernel'}
                    r['platform']['inductor']['device']='tnca' if side else 'cpu'
                    r['time'].update(warmup=5,deterministic=False)
                    start=block*100000+i*100+(0 if side==first else 20)+1
                    r['measurement']=dict(block_id=str(block),process_session=f'{block}-{side}',boot_id='boot',started_ns=start,ended_ns=start+10)
                    records[side].append(r)
            for side in (0,1):
                path=root/f'{block}-{side}.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in records[side]))
                paths[side].append(path)
        return paths

    def test_identity_is_capability_pass_and_slowdown_is_failure(self):
        with tempfile.TemporaryDirectory() as d,patch('ablate.BOOT',40):
            paths=self.fixture(Path(d))
            result,checks=runner.judge(*paths)
            self.assertTrue(all(c['ok'] for c in checks))
            self.assertEqual(result['s_geo'],1)
            self.assertEqual(result['verdict'],'no-significant-gain')
            paths=self.fixture(Path(d),slowdown=1.06)
            _,checks=runner.judge(*paths)
            self.assertFalse(all(c['ok'] for c in checks))

    def test_final_multicore_board_backend_and_simulator_refusal(self):
        with tempfile.TemporaryDirectory() as d,patch('ablate.BOOT',40):
            paths=self.fixture(Path(d))
            for side in paths:
                for path in side:
                    rows=[json.loads(line) for line in path.read_text().splitlines()]
                    for row in rows:
                        row['platform']['backend']='fpga-u280-multicore'
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            _,checks=runner.judge(*paths)
            self.assertTrue(all(c['ok'] for c in checks))
            path=paths[1][0]
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['platform']['backend']='fpga-multicore-sim'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'actual U280'):
                runner.judge(*paths)

    def test_bad_inventory_sessions_order_and_config_are_refused(self):
        for mutation,expected in [('duplicate','exactly once'),('session','reused'),('order','random order'),('compiler','mixed')]:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d,patch('ablate.BOOT',40):
                paths=self.fixture(Path(d))
                path=paths[1][1]
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                if mutation=='duplicate':rows.append(rows[0])
                if mutation=='session':rows[0]['measurement']['process_session']='0-1'
                if mutation=='order':rows[0]['measurement'].update(started_ns=1,ended_ns=2)
                if mutation=='compiler':rows[0]['platform']['inductor']['max_block']=64
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaisesRegex((ValueError,runner.ablate.Refused),expected):runner.judge(*paths)


if __name__=='__main__':unittest.main()
