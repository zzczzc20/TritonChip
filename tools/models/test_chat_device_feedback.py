#!/usr/bin/env python3
"""Exercise device-feedback ordering and EOS cache semantics without claiming board evidence."""
import types
import unittest
from contextlib import nullcontext
from unittest import mock
import torch
import torch._inductor.utils
import chat

class Tensor:
    def __init__(self, values):
        self.value = torch.tensor(values)
        self.device = types.SimpleNamespace(type='tnca')
        self.cpu_reads = 0
    def reshape(self, *_):
        return self
    def cpu(self):
        self.cpu_reads += 1
        return self.value

class Meter:
    def __enter__(self):
        self.mem = types.SimpleNamespace(synchronize=lambda: None)
        return self
    def __exit__(self, *args):
        return False
    def summary(self):
        return {'host_read_calls': 0, 'staging_download_bytes': 0, 'device_cycles': None}

class Session:
    def __init__(self, choices, prefill_compiles_decode=False):
        self.device = 'tnca'
        self.dev = torch.device('cpu')
        self.bucket = 64
        self.chunk = 0
        self.prompts = 0
        self.resets = 0
        self.cached = []
        self._cache_tokens = 0
        self.last_position = None
        self.choices = iter(choices)
        self.fed = []
        self.produced_logits = []
        self.produced_candidates = []
        self.prefill_compiles_decode = prefill_compiles_decode
        def greedy(logits):
            candidate = Tensor([[int(logits.value.argmax())]])
            self.produced_candidates.append(candidate)
            return candidate
        self.greedy = greedy
        self.model = types.SimpleNamespace(
            named_parameters=lambda: iter(()), named_buffers=lambda: iter(()),
            config=types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(num_hidden_layers=1)))
        self.cache = object()
    def prefill_prompt(self, inputs, images):
        self.cached = inputs['input_ids'][0].tolist()
        self._cache_tokens = len(self.cached)
        self.last_position = torch.tensor([len(self.cached) - 1])
        calls = ([{'kind': 'token', 'new_graphs': 1}] if self.prefill_compiles_decode else [])
        return self.next_logits(), calls, 0
    def warmup_decode(self, inputs, per_image=None, capture=None):
        capture('prefill', lambda: None)
        return capture('decode', lambda: Tensor([0, 1, 0, 0]))
    def next_logits(self):
        token = next(self.choices)
        logits = Tensor([int(i == token) for i in range(4)])
        self.produced_logits.append(logits)
        return logits
    def step_device(self, token):
        self.fed.append(int(token.cpu().item()))
        self.last_position = self.last_position + 1
        self._cache_tokens += 1
        return self.next_logits()
    def reset(self):
        self.cached = []
        self._cache_tokens = 0
        self.last_position = None
        self.resets += 1

class DeviceFeedbackTests(unittest.TestCase):
    def run_case(self, choices, forced=None, prefill_compiles_decode=False, fallback_hook=None,
                 retain_logits=True, logit_limit=None, rewind_hook=None):
        session = Session(choices, prefill_compiles_decode)
        processor = types.SimpleNamespace(batch_decode=lambda ids, **kw: [str(ids.tolist())])
        spec = types.SimpleNamespace(name='test', repo='test', revision='test')
        tnca = types.ModuleType('tnca')
        tnca.__path__ = []
        ind = types.ModuleType('tnca.inductor')
        ind.host_fallbacks = fallback_hook or (lambda code: [])
        residency = types.ModuleType('tnca.code_residency')
        residency.RepeatedCodeTrace = lambda: types.SimpleNamespace(step=nullcontext)
        capture_count = 0
        def capture(fn):
            nonlocal capture_count
            capture_count += 1
            # The decode graph compiled inside prefill is reused in generation.
            cached_decode = prefill_compiles_decode and capture_count == 4
            return fn(), ([] if cached_decode else ['test wrapper'])
        rewind_patch = (mock.patch.object(chat, 'rewind_attention_cache', side_effect=rewind_hook)
                        if rewind_hook is not None else nullcontext())
        with rewind_patch, mock.patch.dict('sys.modules', {'tnca': tnca, 'tnca.inductor': ind,
                                            'tnca.code_residency': residency}), \
             mock.patch.object(chat, 'run_with_code_capture', capture), \
             mock.patch.object(chat, 'transfer_meter', lambda enabled: Meter()), \
             mock.patch.object(chat, 'code_store_counters', lambda: None), \
             mock.patch.object(chat.L, 'generation_eos', lambda model: {3}), \
             mock.patch.object(chat.L, 'module_bytes', lambda *args: 1), \
             mock.patch.object(chat.L, 'fp8_route_in_use', lambda: {}), \
             mock.patch.object(chat.L, 'device_memory', lambda dev: {}), \
             mock.patch.object(chat.L, 'cache_bytes', lambda cache: 0):
            report = chat.run_generation(session, processor, spec, {'input_ids': torch.tensor([[8, 9]])},
                                         'prompt', new_tokens=len(choices), forced_tokens=forced,
                                         retain_logits=retain_logits, logit_limit=logit_limit)
        return session, report
    def test_feedback_and_teacher_tokens(self):
        session, report = self.run_case([1, 2, 1], [2, 0])
        self.assertEqual(session.fed, [2, 0])
        self.assertEqual(report['generated_tokens'], [2, 0, 1])
        self.assertEqual(report['candidate_tokens'], [1, 2, 1])
        self.assertEqual(session.cached, [8, 9, 2, 0])
        self.assertTrue(report['compiled_wrapper_roles_covered'])
        self.assertEqual([x.cpu_reads for x in session.produced_logits], [1, 1, 1])
        self.assertEqual([x.cpu_reads for x in session.produced_candidates], [1, 1, 1])
    def test_eos_discards_overrun_cache(self):
        session, report = self.run_case([1, 3, 2])
        self.assertEqual(session.fed, [1, 3])
        self.assertEqual(report['generated_tokens'], [1, 3])
        self.assertEqual(session.cached, [])
        self.assertEqual(session.resets, 1)
        self.assertEqual(report['executed_generation_steps'], 3)
        self.assertEqual(report['statistics']['generated_tokens'], 2)
        self.assertEqual(report['statistics']['executed_generation_steps'], 3)
        self.assertEqual(report['eos_overrun_steps'], 1)
        self.assertIs(report['eos_stop_observed'], False)
        self.assertEqual(report['session']['cache_rewind'], 'full-reset')
        self.assertEqual([x.cpu_reads for x in session.produced_logits], [1, 1, 0])
        self.assertEqual([x.cpu_reads for x in session.produced_candidates], [2, 2, 0])
    def test_decode_compiled_during_prefill_is_accounted_for(self):
        _, report = self.run_case([1, 2], prefill_compiles_decode=True)
        self.assertTrue(report['compiled_wrapper_roles_covered'])
        self.assertEqual(report['status'], 'pass')

    def test_interactive_feedback_avoids_vocabulary_downloads(self):
        session, report = self.run_case([1, 3, 2], retain_logits=False)
        self.assertEqual(report['generated_tokens'], [1, 3])
        self.assertEqual(report['logits'], [])
        self.assertIs(report['logits_retained'], False)
        self.assertEqual([x.cpu_reads for x in session.produced_logits], [0, 0, 0])
        self.assertFalse(report['steady_decode_host_reads_detected'])

    def test_diagnostics_download_only_requested_logit_steps(self):
        session, report = self.run_case([1, 2, 1], logit_limit=1)
        self.assertEqual(report['generated_tokens'], [1, 2, 1])
        self.assertEqual(len(report['logits']), 1)
        self.assertEqual(report['logit_limit'], 1)
        self.assertEqual([x.cpu_reads for x in session.produced_logits], [1, 0, 0])

    def test_eos_rewind_retains_only_the_actual_prefix(self):
        calls = []
        def rewind(cache, valid_length):
            calls.append((cache, valid_length))
            return True
        session, report = self.run_case([1, 3, 2], rewind_hook=rewind)
        self.assertEqual(report['generated_tokens'], [1, 3])
        self.assertEqual(session.cached, [8, 9, 1])
        self.assertEqual(int(session.last_position.item()), 2)
        self.assertEqual(calls, [(session.cache, 3)])
        self.assertEqual(session.resets, 0)
        self.assertEqual(session._cache_tokens, 3)
        self.assertEqual(report['session']['cache_rewind'], 'static-attention-prefix-retained')
        self.assertEqual(report['session']['retained_prefix_tokens'], 3)
        self.assertGreaterEqual(report['session']['cache_maintenance_seconds'], 0)

    def test_retrieval_timing_excludes_post_retrieval_fallback_inspection(self):
        clock = types.SimpleNamespace(now=0.0)
        clock.perf_counter = lambda: clock.now
        def inspect(_code):
            clock.now += 10.0
            return []
        with mock.patch.object(chat, 'time', clock):
            _, report = self.run_case([1, 2], fallback_hook=inspect)
        self.assertGreater(clock.now, 0.0)
        self.assertEqual(report['output_retrieval_seconds'], 0.0)


class AttentionRewindTests(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType('transformers')
        self.module.__path__ = []
        self.cache_module = types.ModuleType('transformers.cache_utils')
        class StaticLayer:
            def __init__(self, length=7, capacity=16):
                self.cumulative_length = torch.tensor(length, dtype=torch.int64)
                self.max_cache_len = capacity
                self.is_initialized = True
                self.keys = torch.arange(capacity, dtype=torch.float32).reshape(1, 1, capacity, 1)
                self.values = -self.keys.clone()
        self.static_layer = StaticLayer
        self.cache_module.StaticLayer = StaticLayer
        self.patch = mock.patch.dict('sys.modules', {'transformers': self.module,
                                                     'transformers.cache_utils': self.cache_module})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_rewinds_every_static_layer_in_place(self):
        layers = [self.static_layer(), self.static_layer()]
        lengths = [layer.cumulative_length for layer in layers]
        keys = [layer.keys.clone() for layer in layers]
        values = [layer.values.clone() for layer in layers]
        self.assertTrue(chat.rewind_attention_cache(types.SimpleNamespace(layers=layers), 3))
        self.assertEqual([int(length) for length in lengths], [3, 3])
        self.assertTrue(all(layer.cumulative_length is length for layer, length in zip(layers, lengths)))
        self.assertTrue(all(torch.equal(layer.keys, original) for layer, original in zip(layers, keys)))
        self.assertTrue(all(torch.equal(layer.values, original) for layer, original in zip(layers, values)))

    def test_mixed_or_unsupported_cache_does_not_mutate_any_layer(self):
        first = self.static_layer()
        cache = types.SimpleNamespace(layers=[first, object()])
        self.assertFalse(chat.rewind_attention_cache(cache, 3))
        self.assertEqual(int(first.cumulative_length), 7)
        self.assertFalse(chat.rewind_attention_cache(types.SimpleNamespace(layers=[]), 3))

    def test_out_of_range_and_uninitialized_layers_are_rejected(self):
        first, second = self.static_layer(), self.static_layer(capacity=4)
        cache = types.SimpleNamespace(layers=[first, second])
        self.assertFalse(chat.rewind_attention_cache(cache, 5))
        self.assertEqual(int(first.cumulative_length), 7)
        second.is_initialized = False
        self.assertFalse(chat.rewind_attention_cache(cache, 3))
        self.assertEqual(int(first.cumulative_length), 7)

    def test_length_views_and_nonstandard_dtypes_do_not_trigger_rewind(self):
        first = self.static_layer()
        backing = torch.tensor([7, 99], dtype=torch.int64)
        first.cumulative_length = backing[:1]
        cache = types.SimpleNamespace(layers=[first])
        self.assertFalse(chat.rewind_attention_cache(cache, 3))
        self.assertEqual(backing.tolist(), [7, 99])
        first.cumulative_length = torch.tensor(7, dtype=torch.int32)
        self.assertFalse(chat.rewind_attention_cache(cache, 3))
        self.assertEqual(int(first.cumulative_length), 7)

    def test_different_device_or_custom_layer_cannot_be_rewound(self):
        first = self.static_layer()
        first.keys = torch.empty((1, 1, first.max_cache_len, 1), device='meta')
        self.assertFalse(chat.rewind_attention_cache(types.SimpleNamespace(layers=[first]), 3))
        class DerivedLayer(self.static_layer):
            pass
        custom = DerivedLayer()
        self.assertFalse(chat.rewind_attention_cache(types.SimpleNamespace(layers=[custom]), 3))
        self.assertEqual(int(custom.cumulative_length), 7)


class CodeCaptureTests(unittest.TestCase):
    def test_steady_roles_reuse_inspected_code_and_reject_recompile(self):
        roles = {'decode': ['inspected decode wrapper']}
        previous = torch._dynamo.config.error_on_recompile
        def invoke():
            self.assertTrue(torch._dynamo.config.error_on_recompile)
            return 17
        with mock.patch.object(chat, 'run_with_code_capture', side_effect=AssertionError('unexpected capture')):
            self.assertEqual(chat.capture_generation_role('decode', invoke, roles), 17)
        self.assertEqual(torch._dynamo.config.error_on_recompile, previous)

    def test_prefill_is_recaptured_and_configuration_is_restored_on_failure(self):
        roles = {'prefill': ['previous wrapper'], 'selection': ['inspected selection wrapper']}
        with mock.patch.object(chat, 'run_with_code_capture', return_value=(3, ['new wrapper'])) as capture:
            self.assertEqual(chat.capture_generation_role('prefill', lambda: None, roles), 3)
        self.assertEqual(roles['prefill'], ['previous wrapper', 'new wrapper'])
        capture.assert_called_once()
        previous = torch._dynamo.config.error_on_recompile
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            chat.capture_generation_role('selection', lambda: (_ for _ in ()).throw(RuntimeError('failure')), roles)
        self.assertEqual(torch._dynamo.config.error_on_recompile, previous)

    def test_capture_reuses_the_compiled_graph(self):
        compiled = []
        def backend(graph, example_inputs):
            compiled.append(graph)
            return graph.forward
        f = torch.compile(lambda x: x + 1, backend=backend, fullgraph=True)
        for _ in range(3):
            with mock.patch('torch._dynamo.reset', side_effect=AssertionError('cache reset')):
                result, _ = chat.run_with_code_capture(lambda: f(torch.ones(4)))
            self.assertTrue(torch.equal(result, torch.full((4,), 2.0)))
        self.assertEqual(len(compiled), 1)

    def test_cached_python_wrapper_is_still_inspected(self):
        import tempfile
        from pathlib import Path
        from torch._inductor.codecache import PyCodeCache
        from tnca.inductor import host_fallbacks
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cached_wrapper.py'
            path.write_text('def possible_fallback():\n    return torch.ops.aten.add(1, 2)\n')
            def load():
                return PyCodeCache.load_by_key_path('chat_capture_test', str(path))
            for _ in range(2):
                _, code = chat.run_with_code_capture(load)
                self.assertIn('ATen op: torch.ops.aten.add', host_fallbacks(code))
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'cannot inspect cached Inductor wrapper'):
                chat.run_with_code_capture(load)


class UserReportTests(unittest.TestCase):
    def test_session_check_skips_unused_intervening_logits(self):
        import copy
        logits = [torch.tensor([1.0, -1.0])]
        first = dict(status='pass', generated_tokens=[1], candidate_tokens=[1],
                     logits=logits, steps=[], session=dict(new_graphs_this_prompt=0))
        other = dict(status='pass', generated_tokens=[2], candidate_tokens=[2],
                     logits=[], steps=[], session=dict(new_graphs_this_prompt=0))
        repeated = copy.deepcopy(first)
        app = object.__new__(chat.ChatApplication)
        app.generate = mock.Mock(side_effect=[other, repeated])
        result = app.session_check(first, 'first', (), 'other')
        self.assertTrue(result['passed'])
        self.assertIs(app.generate.call_args_list[0].kwargs['retain_logits'], False)
        self.assertEqual(app.generate.call_args_list[1].kwargs['forced'], [1])

    def test_json_flushes_text_before_finalizing_visible_time(self):
        import io
        import json
        from chat_timing import user_visible_metrics
        class Output(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushed = []
            def flush(self):
                self.flushed.append(self.getvalue())
        report = dict(status='pass', generated_text='hello', generated_tokens=[7, 2], logits=['private'],
                      user_visible_timing=user_visible_metrics({'generated_tokens': [7, 2]}, 2, [2]))
        stream = Output()
        with mock.patch.object(chat.sys, 'stdout', stream), mock.patch.object(chat.time, 'perf_counter', return_value=16):
            chat.emit_user_report(report, 10)
        output = json.loads(stream.getvalue())
        self.assertEqual(output['generated_text'], 'hello')
        self.assertNotIn('logits', output)
        self.assertIn('"generated_text": "hello"', stream.flushed[0])
        self.assertNotIn('"user_visible_timing"', stream.flushed[0])
        self.assertEqual(output['user_visible_timing']['prompt_to_return_seconds'], 6)
        self.assertEqual(output['user_visible_timing']['non_special_token_ids_per_second'], 1 / 6)

    def test_script_records_and_prints_the_flushed_token_rate(self):
        import io
        import json
        from chat_timing import user_visible_metrics
        report = dict(status='pass', generated_text='hello', generated_tokens=[7, 2],
                      user_visible_timing=user_visible_metrics({'generated_tokens': [7, 2]}, 2, [2]),
                      statistics=dict(prefill_seconds=1, decode_steps=1,
                                      mean_decode_seconds_per_token=1, decode_tokens_per_second=1),
                      steps=[dict(reused_cached_tokens=0)], session=dict(new_graphs_this_prompt=0))
        app = object.__new__(chat.ChatApplication)
        app.args = types.SimpleNamespace(image=[])
        app.generate = mock.Mock(return_value=report)
        stream, transcript = io.StringIO(), []
        with mock.patch.object(chat.sys, 'stdout', stream), \
                mock.patch.object(chat.time, 'perf_counter', side_effect=[10, 15]):
            self.assertTrue(app.repl(lines=['hi', '/quit'], transcript=transcript))
        stats = json.loads(next(line for line in stream.getvalue().splitlines() if line.startswith('{')))
        self.assertEqual(stats['non_special_token_ids_per_second'], 1 / 5)
        self.assertEqual(transcript[0]['report']['user_visible_timing']['prompt_to_return_seconds'], 5)
        self.assertIs(app.generate.call_args.kwargs['retain_logits'], False)


class PrefillSegmentationTests(unittest.TestCase):
    def test_mismatched_cache_length_discards_the_prefix_before_prefill(self):
        session = object.__new__(chat.Session)
        session.model, session.bucket, session.chunk = object(), 64, 0
        session.cached, session._cache_tokens = [1, 2], 7
        session.reset = mock.Mock()
        blocks = []
        session._call_prefill = lambda ids, positions, vision: blocks.append(ids[0].tolist()) or torch.zeros(4)
        ids = torch.tensor([[1, 2, 3]])
        positions = torch.arange(3).reshape(1, 1, 3)
        with mock.patch.object(chat.L, 'prefill_position_ids', return_value=positions):
            _, rows, reused = session.prefill_prompt({'input_ids': ids}, None)
        session.reset.assert_called_once()
        self.assertEqual(reused, 0)
        self.assertEqual(blocks, [[1, 2, 3]])
        self.assertEqual(session._cache_tokens, 3)
        self.assertEqual(rows[0]['tokens'], 3)

    def test_decode_warmup_resets_dummy_state(self):
        session = object.__new__(chat.Session)
        session.model, session.dev = object(), torch.device('cpu')
        session.cached, session.last_position = [], None
        session.prefill_prompt = mock.Mock()
        session.step_device = mock.Mock(return_value='dummy logits')
        session.reset = mock.Mock()
        self.assertEqual(session.warmup_decode({'input_ids': torch.ones((1, 4))}), 'dummy logits')
        session.prefill_prompt.assert_called_once()
        self.assertEqual(session.step_device.call_args.args[0].tolist(), [[0]])
        session.reset.assert_called_once()

    def test_short_tail_is_prefilled_as_a_block(self):
        session = object.__new__(chat.Session)
        session.model, session.bucket, session.chunk, session.cached = object(), 64, 8, []
        session.reset = mock.Mock()
        blocks = []
        session._call_prefill = lambda ids, positions, vision: blocks.append(ids[0].tolist()) or torch.zeros(4)
        session._call_decode = mock.Mock(side_effect=AssertionError('prompt tail used decode'))
        ids = torch.arange(30).reshape(1, 30)
        positions = torch.arange(30).reshape(1, 1, 30)
        with mock.patch.object(chat.L, 'prefill_position_ids', return_value=positions):
            _, rows, reused = session.prefill_prompt({'input_ids': ids}, None)
        self.assertEqual([len(block) for block in blocks], [8, 8, 8, 6])
        self.assertEqual([x for block in blocks for x in block], list(range(30)))
        self.assertEqual([row['kind'] for row in rows], ['prefill'] * 4)
        self.assertEqual(reused, 0)

if __name__ == '__main__':
    unittest.main()
