#!/usr/bin/env python3
"""Exercise device-feedback ordering and EOS cache semantics without claiming board evidence."""
import types
import unittest
from unittest import mock
import torch
import torch._inductor.utils
import chat

class Tensor:
    def __init__(self, values):
        self.value = torch.tensor(values)
        self.device = types.SimpleNamespace(type='tnca')
    def reshape(self, *_):
        return self
    def cpu(self):
        return self.value

class Meter:
    def __enter__(self):
        self.mem = types.SimpleNamespace(synchronize=lambda: None)
        return self
    def __exit__(self, *args):
        return False
    def summary(self):
        return {'host_read_calls': 0, 'staging_download_bytes': 0}

class Session:
    def __init__(self, choices, prefill_compiles_decode=False):
        self.device = 'tnca'
        self.dev = torch.device('cpu')
        self.bucket = 64
        self.chunk = 0
        self.prompts = 0
        self.resets = 0
        self.cached = []
        self.choices = iter(choices)
        self.fed = []
        self.prefill_compiles_decode = prefill_compiles_decode
        self.greedy = lambda logits: Tensor([[int(logits.value.argmax())]])
        self.model = types.SimpleNamespace(
            named_parameters=lambda: iter(()), named_buffers=lambda: iter(()),
            config=types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(num_hidden_layers=1)))
        self.cache = object()
    def prefill_prompt(self, inputs, images):
        self.cached = inputs['input_ids'][0].tolist()
        calls = ([{'kind': 'token', 'new_graphs': 1}] if self.prefill_compiles_decode else [])
        return self.next_logits(), calls, 0
    def next_logits(self):
        token = next(self.choices)
        return Tensor([int(i == token) for i in range(4)])
    def step_device(self, token):
        self.fed.append(int(token.cpu().item()))
        return self.next_logits()
    def reset(self):
        self.cached = []
        self.resets += 1

class DeviceFeedbackTests(unittest.TestCase):
    def run_case(self, choices, forced=None, prefill_compiles_decode=False):
        session = Session(choices, prefill_compiles_decode)
        processor = types.SimpleNamespace(batch_decode=lambda ids, **kw: [str(ids.tolist())])
        spec = types.SimpleNamespace(name='test', repo='test', revision='test')
        tnca = types.ModuleType('tnca')
        tnca.__path__ = []
        ind = types.ModuleType('tnca.inductor')
        ind.host_fallbacks = lambda code: []
        capture_count = 0
        def capture(fn):
            nonlocal capture_count
            capture_count += 1
            # The decode graph compiled inside prefill is reused in generation.
            cached_decode = prefill_compiles_decode and capture_count == 3
            return fn(), ([] if cached_decode else ['test wrapper'])
        with mock.patch.dict('sys.modules', {'tnca': tnca, 'tnca.inductor': ind}), \
             mock.patch.object(chat, 'run_with_code_capture', capture), \
             mock.patch.object(chat, 'transfer_meter', lambda enabled: Meter()), \
             mock.patch.object(chat, 'code_store_counters', lambda: None), \
             mock.patch.object(chat.L, 'generation_eos', lambda model: {3}), \
             mock.patch.object(chat.L, 'module_bytes', lambda *args: 1), \
             mock.patch.object(chat.L, 'fp8_route_in_use', lambda: {}), \
             mock.patch.object(chat.L, 'device_memory', lambda dev: {}), \
             mock.patch.object(chat.L, 'cache_bytes', lambda cache: 0):
            report = chat.run_generation(session, processor, spec, {'input_ids': torch.tensor([[8, 9]])},
                                         'prompt', new_tokens=len(choices), forced_tokens=forced)
        return session, report
    def test_feedback_and_teacher_tokens(self):
        session, report = self.run_case([1, 2, 1], [2, 0])
        self.assertEqual(session.fed, [2, 0])
        self.assertEqual(report['generated_tokens'], [2, 0, 1])
        self.assertEqual(report['candidate_tokens'], [1, 2, 1])
        self.assertEqual(session.cached, [8, 9, 2, 0])
        self.assertTrue(report['compiled_wrapper_roles_covered'])
    def test_eos_discards_overrun_cache(self):
        session, report = self.run_case([1, 3, 2])
        self.assertEqual(session.fed, [1, 3])
        self.assertEqual(report['generated_tokens'], [1, 3])
        self.assertEqual(session.cached, [])
        self.assertEqual(session.resets, 1)
        self.assertEqual(report['executed_generation_steps'], 3)
    def test_decode_compiled_during_prefill_is_accounted_for(self):
        _, report = self.run_case([1, 2], prefill_compiles_decode=True)
        self.assertTrue(report['compiled_wrapper_roles_covered'])
        self.assertEqual(report['status'], 'pass')


class CodeCaptureTests(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
