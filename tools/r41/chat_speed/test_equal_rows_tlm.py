#!/usr/bin/env python3
"""Execute equal-address and unequal-mask cases on the DP TLM."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm'), str(ROOT / 'tools/performance_harness/systemc')]

import triton
import triton.language as tl


@triton.jit
def row_mask_kernel(A, O, N, ROW_STRIDE: tl.constexpr, ROW_LIMIT: tl.constexpr):
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 128)[None, :]
    ptr = A + row * ROW_STRIDE + (col % N)
    if ROW_LIMIT == 16:
        mask = tl.broadcast_to(col < 109, (16, 128))
    else:
        mask = (row < ROW_LIMIT) & (col < 109)
    x = tl.load(ptr, mask=mask, other=0.0)
    peak = tl.max(tl.abs(x), axis=1)
    tl.store(O + row * 128 + col, x * 2.0 + peak[:, None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    import tlm_backend
    from tnca import aot, artifact
    from tnca.runtime import Session
    tlm_backend.register()
    data = torch.arange(-1024, 1024, dtype=torch.float32).reshape(16, 128)
    cases = []
    for stride, limit in ((0, 16), (0, 7), (128, 16), (128, 7)):
        name = f'stride{stride}-rows{limit}'
        compiled = aot.compile_aot(row_mask_kernel, signature={'A': '*fp32', 'O': '*fp32', 'N': 'i32',
            'ROW_STRIDE': 'constexpr', 'ROW_LIMIT': 'constexpr'},
            constexprs={'ROW_STRIDE': stride, 'ROW_LIMIT': limit},
            profile='core-vrf-sync-v5-x8-g2c4-dp', options={'index_opt': 1},
            output=str(a.out / (name + '.tnca')))
        manifest, _ = artifact.read_header(compiled.path)
        assembly_sha = manifest['optional_metadata']['provenance']['assembly_sha256']
        expected = torch.zeros((16, 128), dtype=torch.float32)
        for row in range(limit):
            vals = data[0 if stride == 0 else row, :109]
            expected[row, :] = vals.abs().max()
            expected[row, :109] += vals * 2
        with Session(profile='core-vrf-sync-v5-x8-g2c4-dp',
                     backend='systemc-tlm-grid', host_timeout=180) as session:
            module = session.load(compiled.path)
            x = session.allocate(data.numel() * 4)
            y = session.allocate(expected.numel() * 4)
            session.upload(x, data.numpy().tobytes())
            event = session.submit(module, {'A': x, 'O': y, 'N': 112})
            session.wait(event)
            if not event.ok:
                raise RuntimeError(str(event.error))
            got = torch.frombuffer(bytearray(session.download(y)), dtype=torch.float32).reshape(16, 128)
            mismatches = int((got.view(torch.int32) != expected.view(torch.int32)).sum())
            cases.append(dict(name=name, bit_mismatches=mismatches, elements=2048,
                              device_cycles=event.device_cycles, assembly_sha256=assembly_sha))
            if mismatches:
                raise RuntimeError(f'{name}: {mismatches} bit mismatches')
    result = dict(schema='r41-equal-row-mask-tlm/1', status='pass', cases=cases,
                  equal_rows=os.environ.get('TNCA_EQUAL_ROWS', 'default'))
    (a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
