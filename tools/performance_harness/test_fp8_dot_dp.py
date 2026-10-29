#!/usr/bin/env python3
"""fp8_block_reference.fp8_dot_dp against the Fraction oracle tools/model/fdp.py.

Unit test of the reference itself (the vectorised float64+TwoSum form must be
bit-identical to the exact-rational form), not of any device. Random finite
E4M3 values at mixed magnitudes, a K that is not a multiple of the group
(zero padded), and cancellation-heavy inputs.
"""
import random
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/model'), str(ROOT / 'tools/performance_harness')]
import torch  # noqa: E402
import fdp  # noqa: E402
import fp8_block_reference as ref  # noqa: E402


def e4m3_finite(rng, scale_bits):
    while True:
        x = rng.getrandbits(8)
        if (x & 0x7f) != 0x7f and ((x >> 3) & 15) <= scale_bits:
            return x


class FP8DotDP(unittest.TestCase):
    def check(self, seed, m, n, k, scale_bits):
        rng = random.Random(seed)
        a = [[e4m3_finite(rng, scale_bits) for _ in range(k)] for _ in range(m)]
        b = [[e4m3_finite(rng, scale_bits) for _ in range(k)] for _ in range(n)]
        qa = torch.tensor(a, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        wb = torch.tensor(b, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        got = ref.fp8_dot_dp(qa, wb).view(torch.int32)
        pad = (-k) % 4
        for i in range(m):
            for j in range(n):
                acc = 0
                for k0 in range(0, k + pad, 4):
                    ae = [a[i][t] if t < k else 0 for t in range(k0, k0 + 4)]
                    be = [b[j][t] if t < k else 0 for t in range(k0, k0 + 4)]
                    g = fdp.group_sum('e4m3', 'e4m3', ae, be)
                    g = ('fin', fdp.ro(g[1]), g[2])
                    acc = fdp.add_round(g, acc)
                want = acc - (1 << 32) if acc & 0x80000000 else acc
                self.assertEqual(int(got[i, j]), want, (seed, i, j))

    def test_mixed_magnitudes(self):
        self.check(1, 6, 6, 32, 15)

    def test_small_magnitudes_subnormal_results(self):
        self.check(2, 6, 6, 24, 4)

    def test_k_not_a_multiple_of_the_group(self):
        self.check(3, 5, 7, 18, 12)

    def test_long_k(self):
        self.check(4, 3, 3, 256, 15)


if __name__ == '__main__':
    unittest.main()
