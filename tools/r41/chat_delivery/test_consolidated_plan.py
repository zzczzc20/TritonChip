#!/usr/bin/env python3
"""Exercise campaign registration without touching a board."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import consolidated_plan as plan


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.templates, self.refs = (self.base / name for name in ('root', 'templates', 'refs'))
        for name in plan.SOURCE_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
            self.write(self.root / name / '.fixture', 'source')
        for name in plan.SOURCE_FILES:
            self.write(self.root / name, 'source')
        for name in ('python/tnca/compile_cache.py', 'python/tnca/inductor.py', 'python/tnca/runtime.py',
                     'python/tnca/triton_backend/driver.py', 'tools/models/chat.py', 'tools/models/chat_quality.py',
                     'tools/models/chat_timing.py'):
            self.write(self.root / name, 'source')
        script = b'Capital?\nItaly?\n/reset\nSay hi.\n'
        path = self.root / 'spec/r41/chat/script-text.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(script)
        self.write_json(self.root / 'spec/r41/chat/script-author-20261101.json', dict(
            schema='tnca-chat-script-checks/1', script_sha256=hashlib.sha256(script).hexdigest(),
            checks=[{'all_terms': ['paris']}, {'all_terms': ['rome']}, {'any_terms': ['hi']}]))
        prompts = [dict(id=f'p{i}', turns=[f'prompt{i}']) for i in range(5)]
        prompts.append(dict(id='image', turns=['image'], models=['vl']))
        self.write_json(self.root / 'spec/r41/chat/quality-author-20261101.json',
                        dict(scoring_rule='author-20261101', prompts=prompts))
        for path in plan.EXTERNAL_INPUTS:
            self.write(self.templates / path, 'registered board input')
        for model, (name, _) in plan.MODELS.items():
            selected = [p for p in prompts if model in p.get('models', ['vl', 'q35'])]
            reference = dict(status='reference-recorded', device='cpu', model=name, model_scope='full-checkpoint',
                prompts=[dict(id=p['id'], turns=[dict(user=p['turns'][0], tokens=[1],
                    first_token_reference=dict(top1_token=1, vocab_size=3, top5_tokens=[1, 0, 2],
                                               within_half_logit_tokens=[1]))]) for p in selected])
            self.write_json(self.refs / f'r41_author_ref_{model}_cpu_20261101/quality.json', reference)
            relative = f'spec/r41/run3/board/{model}-quality.json'
            manifest = dict(id=f'{model}-quality', board=dict(image_sha256=plan.IMAGE_SHA,
                image_uuid=plan.IMAGE_UUID, clock_hz=100000000, profile='core-vrf-sync-v5-x8-g2c4-dp',
                backend_args=dict(image_profile='core-vrf-sync-v5-x8-g2c4-dp', core_count=3,
                                  memory_ports=3, hbm_bytes=8 << 30, hbm_dma=True)),
                registration={}, sources=[],
                steps=[dict(kind='fpga', env={}, argv=['{root}/tools/models/chat_quality.py', '--model', model,
                    '--device', 'tnca', '--spec', '{root}/spec/r41/chat/quality-v1.json',
                    '--reference', f'{{root}}/evidence/r41/chat/quality-ref/{model}.json'])])
            self.write_json(self.templates / relative, manifest)
            reg = dict(experiments=[dict(id=f'{model}-quality-', manifest=relative,
                                          sha256=plan.sha(self.templates / relative))])
            self.write_json(self.templates / f'spec/r41/run3/stages/chat-{model}-nohost-v5-20261031.json', reg)

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, path, value):
        self.write(path, json.dumps(value))

    def test_two_single_session_registrations_include_warm_and_script(self):
        out = self.base / 'out'
        result = plan.prepare(self.root, self.templates, self.refs, out)
        self.assertEqual([row['model'] for row in result['models']], ['vl', 'q35'])
        self.assertEqual(result['identity_checks'], 1)
        for model in result['models']:
            manifest = json.loads((out / model['manifest']).read_text())
            self.assertEqual(len(manifest['steps']), 1)
            self.assertIn('--warm-repeat', manifest['steps'][0]['argv'])
            self.assertIn('--script-checks', manifest['steps'][0]['argv'])
            self.assertEqual(plan.sha(out / model['reference']), model['reference_sha256'])
            registration = json.loads((out / model['registration']).read_text())
            self.assertEqual(registration['experiments'][0]['budget_seconds'],
                             plan.MODEL_BUDGET_SECONDS[model['model']])
            self.assertEqual(registration['max_experiment_seconds'], manifest['budget_seconds'])
            self.assertEqual(manifest['budget_seconds'], manifest['steps'][0]['timeout_seconds'])

    def test_rejects_changed_registered_manifest_and_missing_reference(self):
        changed = self.templates / 'spec/r41/run3/board/vl-quality.json'
        changed.write_text(changed.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            plan.prepare(self.root, self.templates, self.refs, self.base / 'bad-template')
        self.setUp_reference_without_evidence()
        with self.assertRaisesRegex(ValueError, 'first-token evidence'):
            plan.prepare(self.root, self.templates, self.refs, self.base / 'bad-reference')

    def setUp_reference_without_evidence(self):
        changed = self.templates / 'spec/r41/run3/board/vl-quality.json'
        regpath = self.templates / 'spec/r41/run3/stages/chat-vl-nohost-v5-20261031.json'
        reg = json.loads(regpath.read_text()); reg['experiments'][0]['sha256'] = plan.sha(changed)
        self.write_json(regpath, reg)
        refpath = self.refs / 'r41_author_ref_vl_cpu_20261101/quality.json'
        ref = json.loads(refpath.read_text()); del ref['prompts'][0]['turns'][0]['first_token_reference']
        self.write_json(refpath, ref)


if __name__ == '__main__':
    unittest.main()
