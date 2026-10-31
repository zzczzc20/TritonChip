#!/usr/bin/env python3
"""Safety checks for the Qwen3.5 timeout-only recovery decision."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import q35_timeout_recovery as queue


def prompt(name, passed=True):
    return dict(id=name, passed=passed, turns=[dict(device_evidence=dict(
        cpu_numerical_fallback=False, compiled_wrapper_roles_covered=True,
        compiled_wrapper_fallbacks={'prefill': [], 'decode': [], 'selection': []},
        model_tensor_placements={'parameters': {'tnca:0': 1}, 'buffers': {'tnca:0': 1}}),
        steps=[dict(kind='decode', transfers_inside_step={'host_read_calls': 0})])])

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name); self.orig = self.work/'orig'; self.root = self.work/'resume'
        self.out = self.work/'out'; self.original_out = self.work/'original-output'
        board=self.orig/'spec/r41/run3/board'; stages=self.orig/'spec/r41/run3/stages'
        board.mkdir(parents=True);stages.mkdir(parents=True)
        spec=dict(id=queue.QUALITY_ID,registration={},steps=[dict(argv=['chat_quality.py','--out','x'])])
        path=board/'quality.json';path.write_text(json.dumps(spec));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        registration=dict(schema='test',image_identity={},experiments=[dict(id=queue.QUALITY_ID,
            manifest=str(path.relative_to(self.orig)),sha256=sha)])
        (stages/'chat-q35-nohost-v5-20261031.json').write_text(json.dumps(registration))
        quality=self.original_out/queue.QUALITY_ID/'q35-quality';quality.mkdir(parents=True)
        (quality/'quality.json').write_text(json.dumps(dict(prompts=[prompt('fact'),prompt('arith'),prompt('reason')])))
        (quality.parent/'report.json').write_text(json.dumps(dict(steps=[dict(id='q35-quality',status='timeout')])))
        for name in ('event_index','equalrow_event','pipeline_equalrow'):
            d=self.work/('runtime_'+name+'_20261031')/'spec/r41/run3/stages';d.mkdir(parents=True)
            (d/'chat-q35-nohost-v5-20261031.json').write_text(json.dumps(dict(experiments=[{'id':'session'}])))
        self.env=patch.multiple(queue,WORK=self.work,ORIG=self.orig,ROOT=self.root,OUT=self.out,
            ORIGINAL_OUT=self.original_out,PIDS=[1,2,3,4]);self.env.start();self.addCleanup(self.env.stop)
        self.wait=patch.object(queue,'wait_pid');self.wait.start();self.addCleanup(self.wait.stop)
        self.calls=[]
        def stage(root,target,registration,log):
            self.calls.append(target.name)
            if target.name in queue.IDS:
                name='r41-q35-segment-'+target.name+'-20261031';d=target/name/'q35-quality';d.mkdir(parents=True)
                (d/'quality.json').write_text(json.dumps(dict(reduced_proof=True,prompts=[prompt(target.name)])))
        self.stage=patch.object(queue,'stage',side_effect=stage);self.stage.start();self.addCleanup(self.stage.stop)

    def test_only_missing_prompts_then_speed_stages(self):
        queue.main()
        self.assertEqual(self.calls,['instruct','multiturn','speed-event_index','speed-equalrow_event','speed-pipeline_equalrow'])
        coverage=json.loads((self.out/'segmented-coverage.json').read_text())
        self.assertEqual(set(coverage['prompts']),set(queue.IDS))
        self.assertIn('timed out',coverage['scope'])
        self.assertFalse((self.original_out/'stage.json').exists())

    def test_original_success_does_not_run_recovery(self):
        (self.original_out/'stage.json').write_text(json.dumps({'status':'pass'}));queue.main()
        self.assertEqual(self.calls,[])
        self.assertFalse((self.out/'segmented-coverage.json').exists())

    def test_other_failure_does_not_run_recovery(self):
        p=self.original_out/queue.QUALITY_ID/'report.json';p.write_text(json.dumps({'steps':[{'id':'q35-quality','status':'failed'}]}))
        queue.main();self.assertEqual(self.calls,[])

    def test_missing_placement_refuses_recovery(self):
        bad=prompt('fact');bad['turns'][0]['device_evidence']['model_tensor_placements']={}
        p=self.original_out/queue.QUALITY_ID/'q35-quality/quality.json';p.write_text(json.dumps({'prompts':[bad]}))
        with self.assertRaises(ValueError):queue.main()
        self.assertEqual(self.calls,[])

    def test_reattach_requires_unchanged_manifests(self):
        queue.prepare();queue.prepare()
        reg=self.out/'r41-q35-segment-fact-20261031-registration.json';data=json.loads(reg.read_text())
        (self.root/data['experiments'][0]['manifest']).write_text('{}')
        with self.assertRaises(ValueError):queue.prepare()

    def test_failed_prior_prompt_refuses_recovery(self):
        p=self.original_out/queue.QUALITY_ID/'q35-quality/quality.json';p.write_text(json.dumps({'prompts':[prompt('fact',False)]}))
        with self.assertRaises(ValueError):queue.main()
        self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()
