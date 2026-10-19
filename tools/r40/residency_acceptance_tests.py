#!/usr/bin/env python3
"""Mutation checks against complete original physical observations."""
import copy
import gzip
import json
import unittest
import residency_acceptance as audit

BASE = audit.ROOT/'evidence/r40/v2/convergence/attempt03'


class ResidencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = json.loads((BASE/'residency_graphs24_6bd07b7_complete.json').read_text())
        cls.launch = json.loads((BASE/'residency_graphs24_6bd07b7_launch.json').read_text())
        cls.model = json.loads(gzip.decompress((BASE/'model_report_6bd07b7.json.gz').read_bytes()))
        cls.tree = cls.graphs['state']['software_tree']
        cls.image = cls.graphs['state']['image_sha256']

    def test_complete_physical_graphs_and_full_model(self):
        state = audit.graphs(self.graphs, self.tree, self.image, self.launch)
        audit.model(self.model, self.tree, self.image, state)

    def test_incomplete_duplicate_changed_source_and_lost_positive_control(self):
        mutations = [lambda r:r['state']['completed'].pop(),
                     lambda r:r['state'].update(status='running'),
                     lambda r:r['state'].update(software_tree='other-source'),
                     lambda r:r['state']['completed'][0].update(exit_code=1),
                     lambda r:r['state']['completed'].__setitem__(1,r['state']['completed'][0])]
        for mutation in mutations:
            r=copy.deepcopy(self.graphs);mutation(r)
            with self.assertRaises(ValueError):audit.graphs(r,self.tree,self.image,self.launch)
        r=copy.deepcopy(self.graphs)
        r['files']['point00.log']=r['files']['point00.log'].replace("'h2d': 64000", "'h2d': 0")
        # Change every positive count rather than relying on a particular graph size.
        import re
        r['files']['point00.log']=re.sub(r"'h2d': [1-9][0-9]*", "'h2d': 0",r['files']['point00.log'])
        r['file_sha256']['point00.log']=audit.sha(r['files']['point00.log'].encode())
        with self.assertRaises(ValueError):audit.graphs(r,self.tree,self.image,self.launch)

    def test_model_numerics_staging_completion_and_length_are_mandatory(self):
        mutations = [lambda r:r['steps'].pop(),
                     lambda r:r.update(actual_decode_calls=63),
                     lambda r:r.update(initial_weight_upload_bytes=1),
                     lambda r:r.update(tokens_match_independent_reference=False),
                     lambda r:r['steps'][0]['correctness'].update(max_rel_err=.051),
                     lambda r:r['steps'][0]['correctness'].update(max_rel_err=float('nan')),
                     lambda r:r['steps'][0]['launches'][0].update(upload_bytes=64),
                     lambda r:r['steps'][0]['transfers'].append(dict(tensor=True,staging=False,bytes=64)),
                     lambda r:r['steps'][0]['events'][0]['history'].pop(-2),
                     lambda r:r['steps'][0].update(host_visibility_history=[]),
                     lambda r:r['steps'][0].update(validation_transfers=[]),
                     lambda r:r['steps'][0].update(completion_history=[])]
        for mutation in mutations:
            r=copy.deepcopy(self.model);mutation(r)
            with self.assertRaises(ValueError):audit.model(r,self.tree,self.image,self.graphs['state'])


if __name__=='__main__':unittest.main()
