#!/usr/bin/env python3
"""Standard native Torch SDPA correctness and allocation checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm', ROOT / 'tools/r40'):
    sys.path.insert(0, str(path))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5/hbm')
os.environ.setdefault('TNCA_HOST_TIMEOUT', '172800')
os.environ.setdefault('TNCA_WINDOW_REUSE', '1')
os.environ.setdefault('TNCA_PACKED_WINDOW_UNPACK', '1')
os.environ.setdefault('TNCA_SDPA', 'fused')
import torch
import triton  # noqa: F401
import torch.nn.functional as F
import tnca.torch_device as D
from tnca import artifact, device
from tnca.triton_backend.driver import launch_log
from tnca_isa import decode_fields, MAJ
from frozen_residency_tests import transfers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.out and args.out.exists():
        parser.error('--out must be new; preserve prior evidence')
    if os.environ.get('TNCA_LAUNCH') == 'skip':
        raise RuntimeError('actual native execution is required')
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    from torch._inductor import config
    config.fx_graph_cache = False
    specs = [
        ('decode_tail', 1, 1, 1, 17, 'float16', None, False),
        ('prefill_causal_tail', 1, 1, 3, 33, 'float16', None, True),
        ('bool_all_mask_row', 1, 2, 3, 19, 'float16', 'bool', False),
        ('float_all_mask_row', 1, 1, 3, 17, 'float16', 'float', False),
        ('strided_batch_heads', 2, 2, 3, 17, 'float16', None, False),
        ('bf16_tail', 1, 1, 1, 17, 'bfloat16', None, False),
        ('gqa_tail', 1, 4, 3, 19, 'float16', None, False),
        ('mqa_causal_tail', 1, 4, 3, 17, 'bfloat16', None, True),
        ('long_prefill_tail', 1, 1, 17, 129, 'float16', None, False),
        ('decode_d32_tail', 1, 1, 1, 17, 'float16', None, False, 32),
        ('decode_d64_tail', 1, 1, 1, 17, 'bfloat16', None, False, 64),
        ('bool_all_mask_d32', 1, 2, 3, 19, 'float16', 'bool', False, 32),
        ('gqa_d64_tail', 1, 4, 3, 19, 'float16', None, False, 64),
        ('mqa_d32_causal_tail', 1, 4, 3, 17, 'bfloat16', None, True, 32),
    ]
    by_name = {spec[0]: (*spec, 16) if len(spec) == 8 else spec for spec in specs}
    if args.case and (len(set(args.case)) != len(args.case) or set(args.case) - by_name.keys()):
        parser.error('--case must name distinct known cases')
    selected = [by_name[name] for name in args.case] if args.case else list(by_name.values())
    torch._dynamo.config.recompile_limit = len(specs) + 2
    results = []
    session = D.MEMORY.session()
    for name, batch, heads, length, seq, dtype, mask_kind, causal, width in selected:
        torch.manual_seed(43)
        dtype = getattr(torch, dtype)
        key_heads = 2 if name.startswith('gqa_') else 1 if name.startswith('mqa_') else heads
        q, k, v = [(torch.randn(batch, size, count, width) * .1).to(dtype).transpose(1, 2)
                   for size, count in ((length, heads), (seq, key_heads), (seq, key_heads))]
        mask = None
        if mask_kind == 'bool':
            mask = torch.ones(length, seq, dtype=torch.bool)
            mask[1, :] = False
            mask[0, 1::2] = False
        elif mask_kind == 'float':
            mask = torch.zeros(length, seq)
            mask[1, :] = -torch.inf
            mask[0, 1::2] = -.5
        dq, dk, dv = [t.to('tnca') for t in (q, k, v)]
        dm = mask.to('tnca') if mask is not None else None
        fn = lambda a, b, c, m: F.scaled_dot_product_attention(
            a, b, c, attn_mask=m, is_causal=causal, enable_gqa=key_heads != heads)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        allocated = []
        allocate = session.allocate

        def track(*a, **kw):
            buf = allocate(*a, **kw)
            if buf.name.startswith('storage@'):
                allocated.append(buf.size)
            return buf

        with torch.no_grad():
            want = fn(q.float(), k.float(), v.float(), mask)
            warm = compiled(dq, dk, dv, dm)
            torch.testing.assert_close(warm.cpu().float(), want, rtol=2e-3, atol=2e-3)
            del warm
            with patch.object(session, 'allocate', side_effect=track), \
                    transfers(session) as events, launch_log() as launches:
                out = compiled(dq, dk, dv, dm)
            torch.testing.assert_close(out.cpu().float(), want, rtol=2e-3, atol=2e-3)
        assert launches and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches)
        assert not [e for e in events if e['tensor'] or e['staging']], events
        found = []
        for entry in launches:
            manifest, segments = artifact.load(entry['artifact_path'])
            text = segments['text']
            words = [decode_fields(int.from_bytes(text[i:i + 8], 'little')) for i in range(0, len(text), 8)]
            loads = sum(f['major'] == MAJ['TXFER'] and f['func'] == 0 for f in words)
            if loads >= 2 * (width // 16):
                provenance = manifest['optional_metadata']['provenance']
                assert provenance.get('packed_window_unpack_rows', 0) > 0
                found.append(dict(artifact=entry['artifact'], descriptor_loads=loads,
                                  resources=manifest['resources']))
        assert len(found) == 1, f'{name}: online descriptor kernel not uniquely present: {found}'
        if mask is None:
            output_bytes = batch * heads * length * width * q.element_size()
            assert len(launches) == 1, 'unused auxiliary math graph was not eliminated'
            assert allocated and sum(allocated) == output_bytes, (allocated, output_bytes)
        if mask_kind:
            assert torch.equal(out.cpu()[:, :, 1], torch.zeros_like(out.cpu()[:, :, 1]))
        result = dict(case=name, head_dim=width, status='pass', tensor_allocations=allocated, online_artifacts=found,
                      kernels=len(launches), steady_tensor_transfers=0)
        results.append(result)
        print('PASS ' + json.dumps(result), flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(schema='tnca-online-sdpa-checkpoint/1',
                measured_performance=False, full_inventory=not args.case, results=results), indent=2) + '\n')
        del out
    print(f'r40-online-attention-tests PASS ({len(results)}/{len(selected)})', flush=True)


if __name__ == '__main__':
    main()
