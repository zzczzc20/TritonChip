#!/usr/bin/env python3
import unittest
from unittest import mock

import torch

from chat import compare_logits, generation_stats, select_token, validate_sampling
from chat_quality import first_token_reference, prior_assistant_text_matches, score_author_turn


class ConversationEvidenceTests(unittest.TestCase):
    def test_logit_comparison_rejects_missing_or_incomplete_steps(self):
        with self.assertRaises(ValueError):
            compare_logits([], [])
        with self.assertRaises(ValueError):
            compare_logits([torch.tensor([1.0])], [])
        with self.assertRaises(ValueError):
            compare_logits([torch.tensor([1.0])], [torch.tensor([1.0, 2.0])])

    def test_previous_answers_must_be_present_and_equal(self):
        self.assertTrue(prior_assistant_text_matches([], [], 0))
        self.assertTrue(prior_assistant_text_matches([{'text': 'Alice'}], [{'text': 'Alice'}], 1))
        self.assertFalse(prior_assistant_text_matches([{'text': 'Alice'}], [{'text': 'Hi Alice'}], 1))
        self.assertFalse(prior_assistant_text_matches([{}], [{}], 1))
        with self.assertRaises(ValueError):
            prior_assistant_text_matches([], [], 1)


class SamplingTests(unittest.TestCase):
    def test_tnca_sampling_is_rejected_before_backend_or_model_loading(self):
        from types import SimpleNamespace
        import chat
        with mock.patch.object(chat.L, 'use_tnca') as use_device, \
                mock.patch.object(chat.L, 'load_model') as load_model:
            with self.assertRaisesRegex(ValueError, 'device-side sampling'):
                chat.ChatApplication(SimpleNamespace(device='tnca', sample=True))
        use_device.assert_not_called()
        load_model.assert_not_called()

    def test_repl_rejects_sampling_without_changing_greedy_mode(self):
        import contextlib
        import io
        from types import SimpleNamespace
        import chat
        app = object.__new__(chat.ChatApplication)
        app.args = SimpleNamespace(device='tnca', sample=False, image=[])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(app.repl(lines=['/sample', '/quit']))
        self.assertFalse(app.args.sample)
        self.assertIn('device-side sampling', output.getvalue())

    def test_greedy_and_seeded_sampling(self):
        logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
        token, greedy = select_token(logits, False, 1.0, 0, 1.0, torch.Generator().manual_seed(7))
        self.assertEqual((token, greedy), (2, 2))
        a = select_token(logits, True, 0.8, 3, 0.95, torch.Generator().manual_seed(7))[0]
        b = select_token(logits, True, 0.8, 3, 0.95, torch.Generator().manual_seed(7))[0]
        self.assertEqual(a, b)
        self.assertEqual(select_token(logits, True, 1.0, 1, 1.0, torch.Generator().manual_seed(7))[0], 2)

    def test_sampling_validation(self):
        for values in ((0, 0, 1.0), (-1, 0, 1.0), (1, -1, 1.0), (1, 0, 0), (1, 0, 1.1)):
            with self.assertRaises(ValueError):
                validate_sampling(*values)


class ReportTests(unittest.TestCase):
    def test_code_store_report_counts_loads_and_evictions_since_step_start(self):
        from chat import _counter_delta
        before = dict(loaded_total=3, evicted_total=2, loaded_modules=2)
        after = dict(loaded_total=8, evicted_total=6, loaded_modules=3)
        report = _counter_delta(before, after)
        self.assertEqual((report['loads'], report['evicted'], report['loaded_modules']), (5, 4, 3))
        self.assertIn('process-wide', report['scope'])

    def test_statistics_derive_from_logged_steps(self):
        rows = [dict(kind="prefill", seconds=3.0), dict(kind="decode", seconds=2.0),
                dict(kind="decode", seconds=4.0)]
        stats = generation_stats(rows)
        self.assertEqual(stats["generated_tokens"], 3)
        self.assertEqual(stats["decode_steps"], 2)
        self.assertEqual(stats["mean_decode_seconds_per_token"], 3.0)
        self.assertEqual(stats["decode_tokens_per_second"], 1 / 3)
        self.assertEqual(stats["generation_seconds"], 9.0)

    def test_logits_comparison_uses_reference_tokens(self):
        want = [torch.tensor([0.0, 2.0]), torch.tensor([3.0, 0.0])]
        got = [torch.tensor([0.0, 2.0]), torch.tensor([1.0, 2.0])]
        rows = compare_logits(want, got)
        self.assertEqual(rows[0]["rel_l2"], 0.0)
        self.assertTrue(rows[0]["argmax_equal"])
        self.assertFalse(rows[1]["argmax_equal"])
        self.assertGreater(rows[1]["rel_l2"], 0.0)


class AuthorQualityTests(unittest.TestCase):
    def test_only_reference_and_warm_repeat_need_full_logits(self):
        from chat_quality import needs_turn_logits
        self.assertTrue(needs_turn_logits('cpu', False, None))
        self.assertTrue(needs_turn_logits('tnca', True, None))
        self.assertFalse(needs_turn_logits('tnca', False, None))
        self.assertFalse(needs_turn_logits('tnca', True, object()))

    def test_near_ties_and_top_five_with_legacy_verdict(self):
        logits = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        reference = dict(text='Hello Alice', tokens=[0], first_token_reference=first_token_reference(logits))
        check = dict(all_terms=['Alice'], any_terms=['remembered'])
        top_five = score_author_turn(dict(text='I remembered Alice', tokens=[4]), reference, check, [])
        self.assertTrue(top_five['passed'])
        self.assertFalse(top_five['old_score']['passed'])
        self.assertFalse(score_author_turn(dict(text='I remembered Alice', tokens=[5]), reference, check, [])['passed'])
        self.assertFalse(score_author_turn(dict(text='Hello Bob', tokens=[0]), reference, check, [])['passed'])
        self.assertFalse(score_author_turn(dict(text='I remembered Alice', tokens=[]), reference, check, [])['passed'])

    def test_six_way_near_tie_is_not_limited_to_top_five(self):
        logits = torch.tensor([9.0, 8.9, 8.8, 8.7, 8.6, 8.5])
        evidence = first_token_reference(logits)
        excluded = next(token for token in range(6) if token not in evidence['top5_tokens'])
        reference = dict(text='42', tokens=[0], first_token_reference=evidence)
        score = score_author_turn(dict(text='42', tokens=[excluded]), reference, dict(all_terms=['42']), ['42'])
        self.assertTrue(score['passed'])
        self.assertFalse(score_author_turn(dict(text='142', tokens=[excluded]), reference,
                                           dict(all_terms=['42']), ['42'])['passed'])

    def test_missing_or_invalid_cpu_evidence_does_not_pass(self):
        reference = dict(text='Hello Alice', tokens=[0])
        check = dict(all_terms=['Alice'])
        self.assertIsNone(score_author_turn(dict(text='Alice', tokens=[1]), reference, check, [])['passed'])
        self.assertTrue(score_author_turn(dict(text='Alice', tokens=[0]), reference, check, [])['passed'])
        with self.assertRaises(ValueError):
            first_token_reference(torch.tensor([1.0, float('nan')]))
        reference['first_token_reference'] = dict(vocab_size=2, top1_token=1, top5_tokens=[1],
                                                  within_half_logit_tokens=[1])
        with self.assertRaises(ValueError):
            score_author_turn(dict(text='Alice', tokens=[1]), reference, check, [])

    def test_quality_runner_records_reference_and_old_and_new_verdicts(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import chat_quality
        from chat_timing import user_visible_metrics

        spec = dict(id='test-author', scoring_rule='author-20261101', new_tokens=2, bucket=8,
                    prompts=[dict(id='multiturn', kind='follow-up', keywords=['alice'],
                                  turns=['Remember Alice', 'What is my name?'],
                                  content_checks=[dict(all_terms=['alice'], any_terms=['remembered']),
                                                  dict(all_terms=['alice'])])])
        def report(text, token, logits):
            result = dict(status='pass', generated_text=text, generated_tokens=[token], logits=[torch.tensor(logits)],
                          prompt_tokens=2, statistics={}, session={}, steps=[])
            result['user_visible_timing'] = user_visible_metrics(result, 0.0)
            return result
        def fake_app(answers):
            app = SimpleNamespace(spec=SimpleNamespace(name='test-model', repo='test-repo', revision='test-revision'), load_report={},
                                  reset_conversation=lambda: None)
            app.generate = mock.Mock(side_effect=answers)
            return app
        cpu = fake_app([report('Hello Alice', 0, [9., 8., 7., 6., 5., 4.]),
                        report('Alice', 1, [1., 9., 2., 3., 4., 5.])])
        device_answers = lambda: [report('I remembered Alice', 4, [0.] * 6),
                                  report('Alice', 1, [0.] * 6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / 'spec.json'
            spec_path.write_text(json.dumps(spec))
            def run(device, app, output, reference=None):
                argv = ['chat_quality.py', '--model', 'q35', '--device', device,
                        '--spec', str(spec_path), '--out', str(output)]
                if reference is not None:
                    argv += ['--reference', str(reference)]
                with mock.patch.object(chat_quality.sys, 'argv', argv), \
                     mock.patch.object(chat_quality.chat, 'ChatApplication', return_value=app), \
                     mock.patch.object(chat_quality.chat, 'open_images', return_value=[]), \
                     mock.patch.object(chat_quality.chat.L, 'versions', return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    return chat_quality.main()
            self.assertEqual(run('cpu', cpu, root / 'cpu'), 0)
            reference_path = root / 'cpu/quality.json'
            reference = json.loads(reference_path.read_text())
            self.assertIn(4, reference['prompts'][0]['turns'][0]['first_token_reference']['top5_tokens'])
            self.assertEqual(run('tnca', fake_app(device_answers()), root / 'tnca', reference_path), 0)
            outcome = json.loads((root / 'tnca/quality.json').read_text())
            self.assertTrue(outcome['passed'])
            self.assertFalse(outcome['prompts'][0]['scores'][0]['old_score']['passed'])
            self.assertGreater(outcome['prompts'][0]['turns'][0]['user_visible_timing']['prompt_to_return_seconds'], 0)
            del reference['prompts'][0]['turns'][0]['first_token_reference']
            reference_path.write_text(json.dumps(reference))
            self.assertEqual(run('tnca', fake_app(device_answers()), root / 'unverified', reference_path), 2)
            outcome = json.loads((root / 'unverified/quality.json').read_text())
            self.assertEqual(outcome['status'], 'unverified-chat-quality')
            self.assertIsNone(outcome['passed'])


class ConsolidatedCampaignTests(unittest.TestCase):
    def test_bitwise_repeat_distinguishes_signed_zero(self):
        from chat_quality import bitwise_logits_equal
        self.assertTrue(bitwise_logits_equal([torch.tensor([float('nan')])], [torch.tensor([float('nan')])]))
        self.assertFalse(bitwise_logits_equal([torch.tensor([0.0])], [torch.tensor([-0.0])]))

    def test_script_does_not_pass_when_no_fallback_evidence_is_missing(self):
        from chat_quality import assess_script
        transcript = [dict(assistant='Paris', report={'status': 'pass'})]
        self.assertIsNone(assess_script(transcript, [dict(all_terms=['paris'])], 'tnca')['passed'])
        transcript[0]['report']['cpu_numerical_fallback'] = True
        self.assertIs(assess_script(transcript, [dict(all_terms=['paris'])], 'tnca')['passed'], False)

    def test_quality_warm_repeat_and_script_share_one_session(self):
        import contextlib
        import copy
        import hashlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import chat_quality
        from chat_timing import user_visible_metrics

        spec = dict(id='test-author', scoring_rule='author-20261101', new_tokens=2, bucket=8,
                    prompts=[dict(id='fact', kind='fact', keywords=['paris'], turns=['Capital?'],
                                  content_checks=[dict(all_terms=['paris'])])])
        identity = dict(model='test-model', repo='test-repo', revision='test-revision', model_scope='full-checkpoint')
        reference = dict(identity, device='cpu', status='reference-recorded',
                         prompts=[dict(id='fact', turns=[dict(user='Capital?', text='Paris', tokens=[0])])])
        evidence = dict(cpu_numerical_fallback=False, compiled_wrapper_roles_covered=True,
                        compiled_wrapper_fallbacks=dict(prefill=[], decode=[], selection=[]),
                        steady_decode_host_reads_detected=False,
                        model_tensor_placements=dict(parameters={'tnca:0': 1}, buffers={'tnca:0': 1}),
                        vision_calls=[], images=0)
        steps = [dict(kind='prefill', input_tokens=1, seconds=1, reused_cached_tokens=0),
                 dict(kind='decode', input_tokens=1, seconds=1,
                      transfers_inside_step=dict(host_read_calls=0, staging_download_bytes=0))]
        def report(text='Paris'):
            value = dict(status='pass', generated_text=text, generated_tokens=[0], logits=[torch.tensor([1., -0.])],
                         prompt_tokens=1, statistics={}, session={}, steps=copy.deepcopy(steps), **evidence)
            value['user_visible_timing'] = user_visible_metrics(value, 0.0)
            return value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, ref_path, script_path, checks_path = (root / name for name in
                                                               ('spec.json', 'reference.json', 'script.txt', 'checks.json'))
            spec_path.write_text(json.dumps(spec)); ref_path.write_text(json.dumps(reference))
            script_path.write_text('Capital?\nItaly?\n/reset\nSay hi.\n')
            checks_path.write_text(json.dumps(dict(schema='tnca-chat-script-checks/1',
                script_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest(),
                checks=[dict(all_terms=['paris']), dict(all_terms=['rome']), dict(any_terms=['hello', 'hi'])])))
            app = SimpleNamespace(spec=SimpleNamespace(name='test-model', repo='test-repo', revision='test-revision'),
                                  load_report={}, reset_conversation=mock.Mock(), generate=mock.Mock(side_effect=[report(), report()]))
            def repl(lines, transcript, on_turn):
                self.assertEqual(lines, script_path.read_text().splitlines())
                for text in ('Paris', 'Rome', 'Hello'):
                    transcript.append(dict(user='prompt', assistant=text, report={k: v for k, v in report(text).items()
                                                                                 if k != 'logits'}))
                    on_turn(transcript[-1])
                    partial = json.loads((root / 'out/quality.json').read_text())
                    self.assertEqual(partial['scripted_chat']['status'], 'running')
                return True
            app.repl = mock.Mock(side_effect=repl)
            argv = ['chat_quality.py', '--model', 'q35', '--device', 'tnca', '--spec', str(spec_path),
                    '--reference', str(ref_path), '--out', str(root / 'out'), '--warm-repeat',
                    '--script', str(script_path), '--script-checks', str(checks_path)]
            with mock.patch.object(chat_quality.sys, 'argv', argv), \
                 mock.patch.object(chat_quality.chat, 'ChatApplication', return_value=app) as constructor, \
                 mock.patch.object(chat_quality.chat, 'open_images', return_value=[]), \
                 mock.patch.object(chat_quality.chat.L, 'versions', return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(chat_quality.main(), 0)
            outcome = json.loads((root / 'out/quality.json').read_text())
            constructor.assert_called_once()
            self.assertEqual(app.generate.call_count, 2)
            app.repl.assert_called_once()
            self.assertTrue(outcome['warm_repeat']['logits_bitwise_equal'])
            self.assertTrue(outcome['scripted_chat']['passed'])
            self.assertEqual(outcome['status'], 'pass')

    def test_script_rejects_wrong_hash_before_model_load(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        import chat_quality
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'spec.json').write_text(json.dumps({'scoring_rule': 'quality-v1'}))
            (root / 'script.txt').write_text('hi\n')
            (root / 'checks.json').write_text(json.dumps(dict(schema='tnca-chat-script-checks/1',
                script_sha256='incorrect', checks=[dict(any_terms=['hi'])])))
            argv = ['chat_quality.py', '--model', 'q35', '--device', 'cpu', '--spec', str(root / 'spec.json'),
                    '--out', str(root / 'out'), '--script', str(root / 'script.txt'),
                    '--script-checks', str(root / 'checks.json')]
            with mock.patch.object(chat_quality.sys, 'argv', argv), \
                 mock.patch.object(chat_quality.chat, 'ChatApplication') as constructor, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(ValueError, 'registered checks'):
                chat_quality.main()
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
