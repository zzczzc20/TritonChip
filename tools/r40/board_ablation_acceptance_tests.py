#!/usr/bin/env python3
"""Keep actual raw launch endpoints when pooling independently audited blocks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ablate
import board_ablation_acceptance as audit
ROOT=Path(__file__).resolve().parents[2]


class EndpointPoolingTests(unittest.TestCase):
    def setUp(self):
        self.row=json.loads((ROOT/'evidence/r40/v2/convergence/attempt03/native_endpoint_reference_b59119d.json').read_text())

    def test_actual_endpoint_records_are_pooled_with_samples(self):
        original=copy.deepcopy(self.row)
        pooled=audit.pool([self.row,self.row,self.row],'graph-makespan')
        _,entry=ablate.timing_entry(pooled,'graph-makespan')
        self.assertEqual(len(entry['samples']),3)
        self.assertEqual(len(entry['endpoint_records']),3)
        self.assertNotIn('median',entry)
        self.assertEqual(self.row,original)
        self.assertEqual(len(ablate.samples_of(pooled,'graph-makespan')[1]),3)

    def test_missing_or_changed_raw_endpoints_cannot_enter_statistics(self):
        for mutation in ['missing','cycles','launches','unequal']:
            with self.subTest(mutation=mutation):
                bad=copy.deepcopy(self.row);_,entry=ablate.timing_entry(bad,'graph-makespan')
                if mutation=='missing':entry['endpoint_records']=[]
                if mutation=='cycles':entry['samples'][0]+=1
                if mutation=='launches':bad['time']['launches']+=1
                if mutation=='unequal':
                    entry['samples']*=2;entry['endpoint_records']*=2
                with self.assertRaises((ValueError,ablate.Refused)):
                    audit.pool([self.row,bad,self.row],'graph-makespan')

    def test_partial_collection_cannot_become_an_audited_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'collection.json').write_text(json.dumps({'status':'failed-preserved'}))
            with self.assertRaisesRegex(ValueError,'complete collection required'):
                audit.audit(Path(temp))


if __name__=='__main__':unittest.main()
