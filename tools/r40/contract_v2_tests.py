#!/usr/bin/env python3
"""Frozen-contract corruption must fail, including replacement of its lock."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import contract_v2 as contract


class ContractTests(unittest.TestCase):
    def test_every_contract_file_and_lock_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in contract.FILES:
                file=root/path;file.parent.mkdir(parents=True,exist_ok=True)
                file.write_text(path+'\n')
            lock=dict(files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in contract.FILES})
            raw=json.dumps(lock,sort_keys=True).encode()
            (root/contract.LOCK).write_bytes(raw)
            expected=hashlib.sha256(raw).hexdigest()
            contract.verify(root,expected=expected)
            for path in contract.FILES:
                original=(root/path).read_bytes()
                (root/path).write_bytes(original+b'amendment')
                with self.assertRaisesRegex(ValueError,'contract changed'):
                    contract.verify(root,expected=expected)
                (root/path).write_bytes(original)
            (root/contract.LOCK).write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'lock changed'):
                contract.verify(root,expected=expected)

    def test_real_frozen_contract(self):
        contract.verify()

    def test_original_scope_and_other_thresholds_survive(self):
        root=contract.ROOT
        a=json.loads((root/'spec/r40/plan.json').read_text())
        b=json.loads((root/'spec/r40/v2/plan.json').read_text())
        for field in ('workloads','subsets','sample_plan','formulas','toolchain_lock'):
            self.assertEqual(a[field],b[field])
        gates=lambda p:{g['id']:(g['kind'],g['required']) for w in p['work_packages'] for g in w['gates']}
        self.assertEqual(gates(a),gates(b))
        self.assertEqual(len(gates(b)),36)
        for name,exp in a['experiments'].items():
            if name not in ('R40-01.device-time','R40-04.ingress'):
                self.assertEqual(exp,b['experiments'][name])
        self.assertEqual(b['experiments']['R40-01.device-time']['threshold'],1/1.05)

    def test_nested_device_exemption_keeps_compiler_settings(self):
        from ablate import config_of
        a={'platform':{'inductor':{'device':'cpu','max_block':16}}}
        b={'platform':{'inductor':{'device':'tnca','max_block':16}}}
        vary=['platform.inductor.device']
        self.assertEqual(config_of(a,vary),config_of(b,vary))
        b['platform']['inductor']['max_block']=64
        self.assertNotEqual(config_of(a,vary),config_of(b,vary))
        self.assertEqual(a['platform']['inductor']['device'],'cpu')


if __name__=='__main__':unittest.main()
