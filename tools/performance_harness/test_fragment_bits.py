"""The per-K FMA reference and the bitwise K-order comparison refuse to relax.

Device-free: the reference against an explicit numpy binary32 FMA chain, its
sensitivity to K order, and compare_fragment_bits on synthetic reports.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class ReferenceTests(unittest.TestCase):
    def test_reference_is_an_ordered_binary32_fma_chain(self):
        import numpy as np
        import torch
        from fragment_case import per_k_fma_reference
        torch.manual_seed(5)
        for dtype in (torch.float16, torch.bfloat16):
            a = (torch.randn(7, 37) * .3).to(dtype)
            b = (torch.randn(37, 5) * .3).to(dtype)
            bias = (torch.randn(5) * .3).to(dtype)
            A, B = a.float().numpy(), b.float().numpy()
            acc = np.zeros((7, 5), np.float32)
            for k in range(37):
                # float64 holds the exact product; one rounding to binary32.
                acc = (A[:, k:k+1].astype(np.float64) * B[k:k+1, :] + acc).astype(np.float32)
            expect = torch.from_numpy(acc + bias.float().numpy()).to(dtype)
            got = per_k_fma_reference(a, b, bias)
            self.assertTrue(torch.equal(got.view(torch.int16), expect.view(torch.int16)), dtype)

    def test_other_k_orders_are_detectable(self):
        import torch
        from fragment_case import order_sensitivity, per_k_fma_reference
        torch.manual_seed(11)
        a = torch.randn(33, 113).to(torch.float16)
        b = torch.randn(113, 47).to(torch.float16)
        exact = per_k_fma_reference(a, b)
        found = order_sensitivity(a, b, None, exact)
        self.assertGreater(found['blocked16_mismatches'], 0)
        self.assertGreater(found['reversed_mismatches'], 0)


def _report(path, columns, out_sha, *, equal=True, tiles=None):
    row = dict(graph='mm', shape=[17, 65, 33], inputs=[dict(sha256='a'), dict(sha256='b')],
               output_bits_sha256=out_sha, reference_bits_sha256='r',
               bitwise_reference=dict(equal=equal, mismatches=0 if equal else 3, elements=1105,
                                      examples=[]),
               cold_bitwise_same=True, order_sensitivity=dict(blocked16_mismatches=4, reversed_mismatches=9),
               logical_tiles=tiles or [[16, columns, 16]])
    path.write_text(json.dumps(dict(status='functional-pass', bitwise=True, backend='systemc-tlm-grid',
                                    dtype='float16', strided=False, columns=columns,
                                    software_tree='files:x', knobs=dict(TNCA_GEMM_TILE=f'16,{columns},16'),
                                    rows=[row])))


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_compare(self, narrow, wide):
        from compare_fragment_bits import compare
        _report(self.root/'n.json', 16, **narrow)
        _report(self.root/'w.json', 32, **wide)
        return compare([self.root/'n.json', self.root/'w.json'])

    def test_identical_bits_pass(self):
        result = self.run_compare(dict(out_sha='x'), dict(out_sha='x'))
        self.assertEqual(result['status'], 'bitwise-identical')
        self.assertEqual(result['order_discrimination']['rows_where_reversed_differs'], 1)

    def test_any_difference_is_a_mismatch(self):
        self.assertEqual(self.run_compare(dict(out_sha='x'), dict(out_sha='y'))['status'], 'mismatch')
        self.assertEqual(self.run_compare(dict(out_sha='x'), dict(out_sha='x', equal=False))['status'],
                         'mismatch')
        # A "wide" run that never used the 16x32 tile proves nothing.
        self.assertEqual(self.run_compare(dict(out_sha='x'), dict(out_sha='x', tiles=[[16, 16, 16]]))['status'],
                         'mismatch')


if __name__ == '__main__':
    unittest.main()
