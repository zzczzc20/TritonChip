#!/usr/bin/env python3
"""Test author rescoring and distinguish missing evidence from a failure."""
import copy
import unittest

import rescore_deadline_quality as scorer


def fixture():
    prompt = dict(id='fact', models=['q35'], turns=['What is the answer?'], keywords=['42'],
                  content_checks=[dict(all_terms=['42'])])
    spec = dict(scoring_rule='author-20261101', prompts=[prompt])
    identity = dict(model='qwen3.5-4b-fp8', model_scope='full-checkpoint', repo='r', revision='v',
                    reduced_depth_layers=None)
    cpu_turn = dict(user=prompt['turns'][0], text='42', tokens=[0], first_token_reference=dict(
        vocab_size=6, top1_token=0, top5_tokens=[0, 1, 2, 3, 4], within_half_logit_tokens=[0, 1]))
    cpu = dict(identity, device='cpu', status='reference-recorded', prompts=[dict(id='fact', turns=[cpu_turn])])
    evidence = dict(cpu_numerical_fallback=False, compiled_wrapper_roles_covered=True,
                    compiled_wrapper_fallbacks=dict(prefill=[], decode=[], selection=[]),
                    model_tensor_placements=dict(parameters={'tnca:0': 1}, buffers={'tnca:0': 1}), vision_calls=[])
    turn = dict(user=prompt['turns'][0], text='42', tokens=[1], device_evidence=evidence,
                steps=[dict(kind='decode', transfers_inside_step=dict(host_read_calls=0, staging_download_bytes=0))])
    device = dict(identity, device='tnca', status='unverified-chat-quality',
                  prompts=[dict(id='fact', turns=[turn], scores=[dict(passed=False)])])
    return device, cpu, spec


class RescoreTests(unittest.TestCase):
    def test_preserves_original_score_and_requires_device_proof(self):
        device, cpu, spec = fixture()
        result = scorer.rescore(device, cpu, spec)
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['prompts'][0]['turns'][0]['original_score']['passed'])
        missing = copy.deepcopy(device)
        del missing['prompts'][0]['turns'][0]['steps'][0]['transfers_inside_step']['host_read_calls']
        self.assertEqual(scorer.rescore(missing, cpu, spec)['status'], 'unverified')
        leaked = copy.deepcopy(device)
        leaked['prompts'][0]['turns'][0]['steps'][0]['transfers_inside_step']['host_read_calls'] = 1
        self.assertEqual(scorer.rescore(leaked, cpu, spec)['status'], 'failed')
        del leaked['prompts'][0]['turns'][0]['steps'][0]['transfers_inside_step']['staging_download_bytes']
        self.assertEqual(scorer.rescore(leaked, cpu, spec)['status'], 'failed')

    def test_partial_or_mismatched_inputs_never_pass(self):
        device, cpu, spec = fixture()
        device['status'] = 'running'
        self.assertEqual(scorer.rescore(device, cpu, spec)['status'], 'unverified')
        device['status'] = 'pass'
        device['prompts'] = []
        self.assertEqual(scorer.rescore(device, cpu, spec)['status'], 'unverified')
        device['revision'] = 'different'
        with self.assertRaises(ValueError):
            scorer.rescore(device, cpu, spec)

    def test_image_prompt_requires_vision_fallback_evidence(self):
        device, cpu, spec = fixture()
        spec['prompts'][0]['images'] = ['image.png']
        self.assertEqual(scorer.rescore(device, cpu, spec)['status'], 'unverified')
        device['prompts'][0]['turns'][0]['device_evidence']['vision_calls'] = [dict(compiled_wrapper_fallbacks=[])]
        self.assertEqual(scorer.rescore(device, cpu, spec)['status'], 'pass')

    def test_legacy_identity_needs_snapshot_name_and_index_hash(self):
        device, cpu, spec = fixture()
        revision, digest = 'a' * 40, 'b' * 64
        cpu.update(repo='author/model', revision=revision,
                   load={'snapshot_view': {'original_index_sha256': digest}})
        device.pop('repo'); device.pop('revision')
        device['load'] = {'snapshot_view': {'view': '/some/view/author--model--' + revision,
                                             'original_index_sha256': digest}}
        result = scorer.rescore(device, cpu, spec)
        self.assertIn('pinned snapshot', result['checkpoint_identity'])
        self.assertTrue(result['prompts'][0]['turns'][0]['prior_assistant_text_matches_cpu'])
        device['load']['snapshot_view']['original_index_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'pinned snapshot'):
            scorer.rescore(device, cpu, spec)

    def test_followup_discloses_different_cpu_conversation_history(self):
        device, cpu, spec = fixture()
        spec['prompts'][0]['turns'].append('Follow up?')
        spec['prompts'][0]['content_checks'].append(dict(all_terms=['42']))
        for record in (device, cpu):
            second = copy.deepcopy(record['prompts'][0]['turns'][0])
            second['user'] = 'Follow up?'
            record['prompts'][0]['turns'].append(second)
        device['prompts'][0]['turns'][0]['text'] = 'The answer is 42.'
        device['prompts'][0]['scores'].append(dict(passed=True))
        result = scorer.rescore(device, cpu, spec)
        self.assertFalse(result['prompts'][0]['turns'][1]['prior_assistant_text_matches_cpu'])


if __name__ == '__main__':
    unittest.main()
