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

    def test_changed_conversation_reuses_only_exact_static_chunks(self):
        self.exercise_static_prefix(group=False)

    def test_grouped_static_suffix_matches_original_chunks(self):
        self.exercise_static_prefix(group=True)

    def test_grouped_suffix_reuses_unedited_conversation(self):
        def make_session(cache):
            session = object.__new__(chat.Session)
            session.device, session.dev, session.cache, session.model = 'cpu', torch.device('cpu'), cache, self.model
            session.bucket, session.chunk, session.resets = 32, 4, 0
            session.cached, session._cache_tokens, session.last_position = [], 0, None
            session._warm_prefill = session._warm_feedback = session._continuation_checkpoint = None
            session._static_prefix_state = None
            session._pending_reset, session._capture_hybrid_checkpoint = False, False
            session._capture_static_prefix = True
            session._call_prefill = lambda ids, positions, vision: self.call(
                cache, ids[0].tolist(), int(positions[0, 0, 0]))[0, -1]
            return session
        baseline, optimized = make_session(self.make_cache()), make_session(self.make_cache())
        first, appended = [5, 7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20, 21]
        def positions(_model, inputs):
            return torch.arange(inputs['input_ids'].shape[1]).reshape(1, 1, -1).expand(4, 1, -1)
        with torch.inference_mode(), mock.patch.object(chat.L, 'prefill_position_ids', side_effect=positions):
            for session in (baseline, optimized):
                session.prefill_prompt({'input_ids': torch.tensor([first])}, None)
                self.call(session.cache, [22], len(first))
                session.cached.append(22)
                session._cache_tokens += 1
            followup = first + [22] + appended
            with mock.patch.dict('os.environ', {'TNCA_STATIC_SUFFIX_GROUP': '0'}):
                expected, _, _ = baseline.prefill_prompt({'input_ids': torch.tensor([followup])}, None)
            with mock.patch.dict('os.environ', {'TNCA_STATIC_SUFFIX_GROUP': '1'}):
                actual, calls, reused = optimized.prefill_prompt({'input_ids': torch.tensor([followup])}, None)
            self.assertEqual(reused, len(first) + 1)
            self.assertEqual([row['tokens'] for row in calls], [8])
            self.assertEqual(optimized._static_prefix_state['tokens'], first)
            self.assertTrue(torch.equal(expected, actual))
            for a, b in zip(baseline.cache.layers, optimized.cache.layers, strict=True):
                self.assertTrue(torch.equal(a.keys, b.keys))
                self.assertTrue(torch.equal(a.values, b.values))

    def exercise_static_prefix(self, *, group):
        def make_session(cache, capture):
            session = object.__new__(chat.Session)
            session.device, session.dev, session.cache, session.model = 'cpu', torch.device('cpu'), cache, self.model
            session.bucket, session.chunk, session.resets = 32, 4, 0
            session.cached, session._cache_tokens, session.last_position = [], 0, None
            session._warm_prefill = session._warm_feedback = session._continuation_checkpoint = None
            session._static_prefix_state = None
            session._pending_reset, session._capture_hybrid_checkpoint = False, False
            session._capture_static_prefix = capture
            session._call_prefill = lambda ids, positions, vision: self.call(
                cache, ids[0].tolist(), int(positions[0, 0, 0]))[0, -1]
            return session
        reference, optimized = make_session(self.make_cache(), False), make_session(self.make_cache(), True)
        first = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        followup = first[:10] + [20, 21, 22, 23, 24, 25]
        def positions(_model, inputs):
            return torch.arange(inputs['input_ids'].shape[1]).reshape(1, 1, -1).expand(4, 1, -1)
        with torch.inference_mode(), mock.patch.object(chat.L, 'prefill_position_ids', side_effect=positions), \
             mock.patch.dict('os.environ', {'TNCA_STATIC_SUFFIX_GROUP': '1' if group else '0'}):
            for session in (reference, optimized):
                session.prefill_prompt({'input_ids': torch.tensor([first])}, None)
                self.call(session.cache, [19], len(first))
                session.cached.append(19)
                session._cache_tokens += 1
            reference.reset()
            expected, _, _ = reference.prefill_prompt({'input_ids': torch.tensor([followup])}, None)
            actual, _, reused = optimized.prefill_prompt({'input_ids': torch.tensor([followup])}, None)
            self.assertEqual(reused, 8)
            self.assertTrue(torch.equal(expected, actual))
            for first_layer, second_layer in zip(reference.cache.layers, optimized.cache.layers, strict=True):
                self.assertTrue(torch.equal(first_layer.keys, second_layer.keys))
                self.assertTrue(torch.equal(first_layer.values, second_layer.values))
            third = followup[:14] + [26, 27, 28, 29]
            reference.reset()
            expected, _, _ = reference.prefill_prompt({'input_ids': torch.tensor([third])}, None)
            actual, _, reused = optimized.prefill_prompt({'input_ids': torch.tensor([third])}, None)
            self.assertEqual(reused, 8 if group else 12)
            self.assertTrue(torch.equal(expected, actual))
            for first_layer, second_layer in zip(reference.cache.layers, optimized.cache.layers, strict=True):
                self.assertTrue(torch.equal(first_layer.keys, second_layer.keys))
                self.assertTrue(torch.equal(first_layer.values, second_layer.values))

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

    def test_chunk_checkpoint_matches_canonical_followup_bitwise(self):
        self.exercise_chunk_checkpoint(fallback=False)

    def test_earlier_hybrid_checkpoint_handles_a_longer_template_rewrite(self):
        self.exercise_chunk_checkpoint(fallback=True)

    def exercise_chunk_checkpoint(self, *, fallback):
        def make_session(cache, capture):
            session = object.__new__(chat.Session)
            session.device, session.dev, session.cache, session.model = 'cpu', torch.device('cpu'), cache, self.model
            session.bucket, session.chunk, session.resets = 32, 4, 0
            session.cached, session._cache_tokens, session.last_position = [], 0, None
            session._warm_prefill = session._warm_feedback = session._continuation_checkpoint = None
            session._pending_reset, session._capture_hybrid_checkpoint = False, capture
            session._call_prefill = lambda ids, positions, vision: self.call(cache, ids[0].tolist())[0, -1]
            return session
        reference, optimized = make_session(self.make_cache(), False), make_session(self.make_cache(), True)
        first = [7, 4, 1, 9, 5, 6, 2, 3, 4, 1, 5, 6]
        followup = (first[:6] + [13, 14, 15, 16, 17, 18, 20, 21, 22, 23] if fallback else
                    first[:10] + [13, 14, 15, 16, 17, 18])
        def positions(_model, inputs):
            return torch.arange(inputs['input_ids'].shape[1]).reshape(1, 1, -1)
        with torch.inference_mode(), mock.patch.object(chat.L, 'prefill_position_ids', side_effect=positions), \
             mock.patch.dict('os.environ', {'TNCA_HYBRID_FALLBACK_CHECKPOINT': '1' if fallback else '0'}):
            for session in (reference, optimized):
                session.prefill_prompt({'input_ids': torch.tensor([first])}, None)
                self.call(session.cache, [19])
                session.cached.append(19)
                session._cache_tokens += 1
            self.assertEqual(optimized._continuation_checkpoint['tokens'], first[:8])
            if fallback:
                self.assertEqual(optimized._fallback_continuation_checkpoint['tokens'], first[:4])
                snapshots = (optimized._continuation_checkpoint['snapshot'],
                             optimized._fallback_continuation_checkpoint['snapshot'])
                saved_bytes = sum(t.numel() * t.element_size() for snap in snapshots for t in snap['copies'])
                self.assertLessEqual(saved_bytes, 64 * 1024 * 1024)
            reference.reset()
            expected, _, _ = reference.prefill_prompt({'input_ids': torch.tensor([followup])}, None)
            actual, _, reused = optimized.prefill_prompt({'input_ids': torch.tensor([followup])}, None)
            self.assertEqual(reused, 4 if fallback else 8)
            self.assertTrue(torch.equal(expected, actual))
            for first_layer, second_layer in zip(reference.cache.layers, optimized.cache.layers, strict=True):
                for name in ('conv_states', 'recurrent_states'):
                    if hasattr(first_layer, name):
                        for key, state in getattr(first_layer, name).items():
                            self.assertTrue(torch.equal(state, getattr(second_layer, name)[key]))
                if hasattr(first_layer, 'keys'):
                    self.assertTrue(torch.equal(first_layer.keys, second_layer.keys))
                    self.assertTrue(torch.equal(first_layer.values, second_layer.values))

    def test_guarded_eos_preserves_hybrid_state_for_the_next_turn(self):
        clean, retained = self.make_cache(), self.make_cache()
        prompt, emitted, suffix = [7, 4, 1, 9], [5, 8], [2, 3, 6]
        with torch.inference_mode():
            for cache in (clean, retained):
                self.call(cache, prompt)
                for token in emitted:
                    self.call(cache, [token])
            # A latched device stop skips all speculative writes. Retention
            # must leave recurrent tensors untouched and continue exactly.
            linear = next(layer for layer in retained.layers if hasattr(layer, 'recurrent_states'))
            old = [t.clone() for t in linear.recurrent_states.values()]
            with mock.patch.object(chat, '_write_static_lengths') as write:
                self.assertTrue(chat.retain_guarded_hybrid_cache(retained, len(prompt) + len(emitted)))
                write.assert_not_called()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, linear.recurrent_states.values(), strict=True)))
            self.assertTrue(torch.equal(self.call(clean, suffix), self.call(retained, suffix)))
            for original, actual in zip(clean.layers, retained.layers, strict=True):
                for name in ('conv_states', 'recurrent_states'):
                    if hasattr(original, name):
                        for key, state in getattr(original, name).items():
                            self.assertTrue(torch.equal(state, getattr(actual, name)[key]))
                if hasattr(original, 'keys'):
                    self.assertTrue(torch.equal(original.keys, actual.keys))
                    self.assertTrue(torch.equal(original.values, actual.values))

    def test_guarded_hybrid_rejects_unsupported_metadata_without_writes(self):
        cache = self.make_cache()
        with torch.inference_mode():
            self.call(cache, [7, 4, 1])
            linear = next(layer for layer in cache.layers if hasattr(layer, 'recurrent_states'))
            linear.has_previous_state[0] = False
            before = [layer.cumulative_length.clone() for layer in cache.layers
                      if hasattr(layer, 'cumulative_length')]
            self.assertFalse(chat.retain_guarded_hybrid_cache(cache, 3))
            after = [layer.cumulative_length for layer in cache.layers if hasattr(layer, 'cumulative_length')]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after, strict=True)))

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
