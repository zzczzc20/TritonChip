#!/usr/bin/env python3
"""CPU semantic/graph guard checks; native execution is a separate gate."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from archbench import registry
from archbench.ops import rope_tables
from bucket_decode import BucketDecodeStep


class BucketTests(unittest.TestCase):
    def test_tensor_positions_cache_and_future_mask(self):
        point = registry.get('model.llama_decode/tiny/fp16-b1-p16')
        model, (token, initial_k, initial_v) = point.make(0)
        model = model.float().eval()
        capacity = 19
        # Poison future slots: the attention mask must exclude these values.
        keys = torch.full((*initial_k.shape[:-2], capacity, initial_k.shape[-1]), 1000.)
        values = torch.full_like(keys, -1000.)
        keys[..., :16, :].copy_(initial_k)
        values[..., :16, :].copy_(initial_v)
        ref_k, ref_v = keys.clone(), values.clone()
        cos, sin = rope_tables(capacity, model.c.hd, 'fp32')
        step = BucketDecodeStep(copy.deepcopy(model), cos, sin, capacity)
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        compiled = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)
        position = torch.tensor([16])
        with torch.no_grad():
            for pos in range(16, capacity):
                model.cos, model.sin = cos[pos:pos + 1], sin[pos:pos + 1]
                want = model(token, ref_k[..., :pos, :], ref_v[..., :pos, :])
                ref_k[..., pos:pos + 1, :].copy_(want[1])
                ref_v[..., pos:pos + 1, :].copy_(want[2])
                got = compiled(token, keys, values, position)
                for actual, expected in zip(got[:3], want):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(keys, ref_k, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(values, ref_v, rtol=1e-5, atol=1e-5)
                self.assertEqual(got[4].item(), pos + 1)
                token, position = got[3], got[4]
        self.assertEqual(len(graphs), 1)


if __name__ == '__main__':
    unittest.main()
