"""Bitwise CPU checks for continuing a rewound Qwen3-VL static attention cache.

This is deliberately a tiny random model, not a board performance or model
quality claim. Recurrent and sliding caches are exercised by the separate
fail-closed checks; only standard static attention is eligible for rewind.
"""
import unittest

import torch

import chat

try:
    from transformers import StaticCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
except ImportError:
    StaticCache = Qwen3VLTextConfig = Qwen3VLTextModel = None


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
            self.assertTrue(chat.rewind_attention_cache(rewound, valid))
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


if __name__ == '__main__':
    unittest.main()
