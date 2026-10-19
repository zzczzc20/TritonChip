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
from bucket_decode import BucketDecodeStep, BucketGenerateStep, GenerationPrefillState


class BucketTests(unittest.TestCase):
    def test_prefill_padding_preserves_original_prompt_and_outputs(self):
        model,(tokens,) = registry.get('model.llama_prefill/tiny/fp16-b1-s16').make(0)
        model=model.float().eval()
        graphs=[]
        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward
        with torch.no_grad():
            want=GenerationPrefillState(model)(tokens)
            original_logits=model(tokens)
            torch.testing.assert_close(want[0],original_logits)
            compiled=torch.compile(GenerationPrefillState(model,19),backend=backend,fullgraph=True,dynamic=False)
            got=compiled(tokens)
            for actual,expected in zip((got[0],got[1][...,:16,:],got[2][...,:16,:],got[3]),want):
                torch.testing.assert_close(actual,expected)
            self.assertEqual(got[4].item(),16)
            self.assertEqual(got[1].shape[-2],19)
            self.assertEqual(torch.count_nonzero(got[1][...,16:,:]).item(),0)
            self.assertEqual(torch.count_nonzero(got[2][...,16:,:]).item(),0)
            with self.assertRaises(ValueError):GenerationPrefillState(model,15)(tokens)
        self.assertEqual(len(graphs),1)

    def test_generation_cache_alias_and_device_position(self):
        model, (token, initial_k, initial_v) = registry.get('model.llama_decode/tiny/fp16-b1-p16').make(0)
        model = model.float().eval()
        capacity = 19
        keys = torch.nn.functional.pad(initial_k.float(), (0,0,0,3))
        values = torch.nn.functional.pad(initial_v.float(), (0,0,0,3))
        ref_k, ref_v = keys.clone(), values.clone()
        cos,sin = rope_tables(capacity,model.c.hd,'fp32')
        module = BucketGenerateStep(copy.deepcopy(model),cos,sin,capacity)
        graphs=[]
        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward
        compiled=torch.compile(module,backend=backend,fullgraph=True,dynamic=False)
        addresses=(keys.data_ptr(),values.data_ptr())
        position=torch.tensor([16])
        with torch.no_grad():
            for i in range(16,capacity):
                model.cos,model.sin=cos[i:i+1],sin[i:i+1]
                want=model(token,ref_k[...,:i,:],ref_v[...,:i,:])
                ref_k[...,i:i+1,:].copy_(want[1]);ref_v[...,i:i+1,:].copy_(want[2])
                got=compiled(token,keys,values,position)
                torch.testing.assert_close(got[0],want[0],rtol=1e-5,atol=1e-5)
                torch.testing.assert_close(got[1],ref_k,rtol=1e-5,atol=1e-5)
                torch.testing.assert_close(got[2],ref_v,rtol=1e-5,atol=1e-5)
                self.assertEqual((got[1].data_ptr(),got[2].data_ptr()),addresses)
                self.assertEqual(got[4].item(),i+1)
                token,position=got[3],got[4]
        self.assertEqual(len(graphs),1)
        with self.assertRaisesRegex(ValueError,'cache shape'):
            module(token,keys[...,:18,:],values[...,:18,:],position)
        with self.assertRaisesRegex(ValueError,'position'):
            module(token,keys,values,torch.tensor([18],dtype=torch.int32))
        with self.assertRaisesRegex(ValueError,'RoPE'):
            BucketGenerateStep(model,cos,sin,capacity+1)

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
