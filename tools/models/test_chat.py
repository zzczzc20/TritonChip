#!/usr/bin/env python3
import unittest

import torch

from chat import compare_logits, generation_stats, select_token, validate_sampling


class SamplingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
