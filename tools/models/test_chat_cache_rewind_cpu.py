"""Bitwise CPU checks for continuing a rewound Qwen3-VL static attention cache.

This is deliberately a tiny random model, not a board performance or model
quality claim. Recurrent and sliding caches are exercised by the separate
fail-closed checks; only standard static attention is eligible for rewind.
"""
import unittest
from unittest import mock

import torch

import chat

try:
    from transformers import StaticCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
except ImportError:
    StaticCache = Qwen3VLTextConfig = Qwen3VLTextModel = None

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
except ImportError:
    Qwen3_5TextConfig = Qwen3_5TextModel = None


@unittest.skipIf(Qwen3VLTextModel is None, 'Qwen3-VL transformers implementation is unavailable')
class StaticAttentionContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.config = Qwen3VLTextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64,
            rope_parameters={'rope_type': 'default', 'rope_theta': 500000.0,
                             'mrope_section': [2, 1, 1]})
        cls.config._attn_implementation = 'eager'
        cls.model = Qwen3VLTextModel(cls.config).eval()

    def make_cache(self):
        cache = StaticCache(config=self.config, max_cache_len=32)
        cache.early_initialization(batch_size=1, num_heads=2, head_dim=8,
                                   dtype=torch.float32, device=torch.device('cpu'))
        return cache

    def call(self, cache, tokens, start):
        positions = torch.arange(start, start + len(tokens)).reshape(1, 1, -1).expand(4, 1, -1)
        return self.model(input_ids=torch.tensor([tokens]), position_ids=positions,
                          past_key_values=cache, use_cache=True).last_hidden_state

    def continuation(self, prefix, emitted, speculative, suffix):
        clean, rewound = self.make_cache(), self.make_cache()
        with torch.inference_mode():
            for cache in (clean, rewound):
                self.call(cache, prefix, 0)
                for index, token in enumerate(emitted):
                    self.call(cache, [token], len(prefix) + index)
            valid = len(prefix) + len(emitted)
            for index, token in enumerate(speculative):
                self.call(rewound, [token], valid + index)
            for layer in rewound.layers:
                layer.keys[:, :, valid:valid + len(speculative)].fill_(float('nan'))
                layer.values[:, :, valid:valid + len(speculative)].fill_(float('nan'))
            self.assertTrue(chat.rewind_attention_cache(rewound, valid,
                                                        dirty_length=valid + len(speculative)))
            expected = self.call(clean, suffix, valid)
            actual = self.call(rewound, suffix, valid)
        self.assertTrue(torch.equal(expected, actual))
        through = valid + len(suffix)
        for original, replayed in zip(clean.layers, rewound.layers, strict=True):
            self.assertTrue(torch.equal(original.keys[:, :, :through], replayed.keys[:, :, :through]))
            self.assertTrue(torch.equal(original.values[:, :, :through], replayed.values[:, :, :through]))
            self.assertEqual(int(original.cumulative_length), through)
            self.assertEqual(int(replayed.cumulative_length), through)

    def test_speculative_suffix_rewind_is_bitwise_equal_to_clean_continuation(self):
        cases = [([5, 7, 8, 9], [10], [11, 12, 13], [11, 14, 15]),
                 ([2, 4], [], [3], [3, 9]),
                 ([6, 1, 7], [3, 8], [2, 5, 4, 1], [2, 11, 12, 13])]
        for prefix, emitted, speculative, suffix in cases:
            with self.subTest(prefix=prefix, emitted=emitted, speculative=speculative):
                self.continuation(prefix, emitted, speculative, suffix)

    def test_batched_attention_clear_matches_transformers_reset_bitwise(self):
        original = StaticCache(config=self.config, max_cache_len=1024)
        optimized = StaticCache(config=self.config, max_cache_len=1024)
        for cache in (original, optimized):
            cache.early_initialization(batch_size=1, num_heads=2, head_dim=8,
                                       dtype=torch.float32, device=torch.device('cpu'))
        for first, second in zip(original.layers, optimized.layers, strict=True):
            first.keys.normal_()
            first.values.normal_()
            first.cumulative_length.fill_(17)
            second.keys.copy_(first.keys)
            second.values.copy_(first.values)
            second.cumulative_length.copy_(first.cumulative_length)
        original.reset()
        self.assertTrue(chat.clear_static_attention_cache(optimized))
        for first, second in zip(original.layers, optimized.layers, strict=True):
            self.assertTrue(torch.equal(first.keys, second.keys))
            self.assertTrue(torch.equal(first.values, second.values))
            self.assertTrue(torch.equal(first.cumulative_length, second.cumulative_length))

    def test_small_or_uninitialized_cache_uses_transformers_reset(self):
        small = self.make_cache()
        self.assertFalse(chat.clear_static_attention_cache(small))
        uninitialized = StaticCache(config=self.config, max_cache_len=1024)
        self.assertFalse(chat.clear_static_attention_cache(uninitialized))


@unittest.skipIf(Qwen3_5TextModel is None, 'Qwen3.5 transformers implementation is unavailable')
class HybridPrefillContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(77)
        cls.config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=['linear_attention', 'full_attention'], max_position_embeddings=64,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                             'partial_rotary_factor': 0.5, 'mrope_section': [1, 1, 0]})
        cls.config._attn_implementation = 'eager'
        cls.model = Qwen3_5TextModel(cls.config).eval()

    def make_cache(self):
        cache = StaticCache(config=self.config, max_cache_len=32)
        cache.early_initialization(batch_size=1, num_heads=2, head_dim=8,
                                   dtype=torch.float32, device=torch.device('cpu'))
        return cache

    def call(self, cache, tokens):
        return self.model(input_ids=torch.tensor([tokens]), past_key_values=cache,
                          use_cache=True).last_hidden_state

    def test_real_first_decode_reuse_matches_clean_hybrid_continuation(self):
        clean, warmed = self.make_cache(), self.make_cache()
        prompt = [7, 4, 1, 9]
        inputs = {'input_ids': torch.tensor([prompt])}
        session = object.__new__(chat.Session)
        session.device, session.dev, session.cache = 'tnca', torch.device('cpu'), warmed
        session.cached, session._cache_tokens, session.last_position = [], 0, None
        session.greedy = lambda logits: logits.argmax().reshape(1, 1)
        session.reset = mock.Mock()
        def prefill(data, _images):
            tokens = data['input_ids'][0].tolist()
            logits = self.call(warmed, tokens)[0, -1]
            session.cached, session._cache_tokens = tokens, len(tokens)
            session.last_position = torch.tensor([len(tokens) - 1])
            return logits, [], 0
        def decode(token):
            logits = self.call(warmed, [int(token.item())])[0, -1]
            session._cache_tokens += 1
            session.last_position = session.last_position + 1
            return logits
        session.prefill_prompt, session.step_device = prefill, decode
        import tnca.torch_device as td
        class Meter:
            def __enter__(self):
                self.mem = td.MEMORY
                return self
            def __exit__(self, *args):
                return False
            def summary(self):
                return {}
        with torch.inference_mode(), mock.patch.object(td.MEMORY, 'synchronize'), \
             mock.patch.object(chat, 'transfer_meter', lambda *args, **kwargs: Meter()):
            expected_prefill = self.call(clean, prompt)[0, -1]
            first = int(expected_prefill.argmax())
            expected_decode = self.call(clean, [first])[0, -1]
            actual_decode = session.warmup_decode(inputs, real_feedback=True)
            saved = session.consume_warm_prefill(inputs, None)
            self.assertTrue(torch.equal(saved[0][0], expected_prefill))
            self.assertTrue(torch.equal(actual_decode, expected_decode))
            self.assertEqual(int(session._warm_feedback[0].item()), first)
            self.assertTrue(torch.equal(self.call(clean, [3]), self.call(warmed, [3])))
        session.reset.assert_not_called()

    def test_dummy_decode_restore_matches_clean_hybrid_continuation_bitwise(self):
        clean, warmed = self.make_cache(), self.make_cache()
        prompt = [7, 4, 1, 9, 5]
        with torch.inference_mode():
            expected_prefill = self.call(clean, prompt)
            actual_prefill = self.call(warmed, prompt)
            self.assertTrue(torch.equal(expected_prefill, actual_prefill))
            snapshot = chat.snapshot_hybrid_prefill(warmed, len(prompt))
            self.assertIsNotNone(snapshot)
            self.call(warmed, [0])
            self.assertTrue(chat.restore_hybrid_prefill(snapshot))
            for token in (8, 2, 3):
                self.assertTrue(torch.equal(self.call(clean, [token]), self.call(warmed, [token])))
        for first, second in zip(clean.layers, warmed.layers, strict=True):
            for name in ('conv_states', 'recurrent_states'):
                if hasattr(first, name):
                    for key, state in getattr(first, name).items():
                        self.assertTrue(torch.equal(state, getattr(second, name)[key]))
            if hasattr(first, 'keys'):
                self.assertTrue(torch.equal(first.keys[:, :, :8], second.keys[:, :, :8]))
                self.assertTrue(torch.equal(first.values[:, :, :8], second.values[:, :, :8]))

    def test_deferred_hybrid_reset_matches_eager_reset_on_the_next_prompt(self):
        eager, deferred = self.make_cache(), self.make_cache()
        first, next_prompt = [7, 4, 1], [5, 8, 2]
        with torch.inference_mode():
            for cache in (eager, deferred):
                self.call(cache, first)
                self.call(cache, [9])
            eager.reset()
            expected = self.call(eager, next_prompt)
            session = object.__new__(chat.Session)
            session.device, session.model, session.cache = 'cpu', self.model, deferred
            session.cached, session._cache_tokens = first, len(first) + 1
            session.last_position, session._warm_prefill = torch.tensor([3]), None
            session._pending_reset, session.resets = False, 0
            session.bucket, session.chunk = 32, 0
            session._call_prefill = lambda ids, positions, vision: self.call(deferred, ids[0].tolist())
            session.defer_reset()
            self.assertTrue(session.schedule_idle_cleanup())
            session.wait_for_cleanup()
            with mock.patch.object(chat.L, 'prefill_position_ids', return_value=torch.tensor([[[0, 1, 2]]])):
                actual, _, reused = session.prefill_prompt({'input_ids': torch.tensor([next_prompt])}, None)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(reused, 0)
            self.assertEqual(session.resets, 1)

    def test_changed_hybrid_metadata_refuses_restore_without_mutation(self):
        cache = self.make_cache()
        with torch.inference_mode():
            self.call(cache, [7, 4, 1])
            snapshot = chat.snapshot_hybrid_prefill(cache, 3)
            self.assertIsNotNone(snapshot)
            self.call(cache, [0])
            linear = snapshot['linear'][0]
            linear.has_previous_state[0] = False
            states = [tensor.clone() for tensor in snapshot['tensors']]
            lengths = [length.clone() for length in snapshot['lengths']]
            self.assertFalse(chat.restore_hybrid_prefill(snapshot))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, snapshot['tensors'], strict=True)))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(lengths, snapshot['lengths'], strict=True)))

    def test_snapshot_allocation_failure_leaves_prefill_state_intact(self):
        cache = self.make_cache()
        with torch.inference_mode():
            self.call(cache, [7, 4, 1])
            linear = next(layer for layer in cache.layers if hasattr(layer, 'recurrent_states'))
            before = linear.recurrent_states[0].clone()
            with mock.patch.object(torch.Tensor, 'clone', side_effect=MemoryError):
                self.assertIsNone(chat.snapshot_hybrid_prefill(cache, 3))
            self.assertTrue(torch.equal(before, linear.recurrent_states[0]))


if __name__ == '__main__':
    unittest.main()
