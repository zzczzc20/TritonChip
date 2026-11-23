#!/usr/bin/env python3
"""The unchanged Hub kernels, run by Triton's CPU interpreter, against the xl references (no TNCA, no device).

    xl_interpreter_check.py [--out FILE.json]

Triton's interpreter executes the kernel source of `kernels-community/finegrained-fp8` (the pinned build, archbench/
fp8_hub.py) on the CPU. Its float8 conversion is not faithful (it rounds 31.25/s to 16 where RNE gives 32; the codes of
`fp8_act_quant` differ in 2 % of elements), so the inputs here are built so that every conversion is exact: each
activation is a value of the E4M3 grid times a power-of-two block scale (the block's largest value is 448, so
scale = amax/448 is that power of two). Then the interpreter's result must equal the CPU twin (`emu`) bit for bit and the
binary64 reference to within one fp16 ulp of the largest binade (2^-10 of max|ref|, the one-ulp rounding flip a binary32
accumulation can cause). That validates the reference's structure (scale layout, per-K-block
rescaling, quantisation scales, output rounding) against the real kernel source, independently of the TNCA compiler.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['TRITON_INTERPRET'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    import torch
    from archbench import fp8_hub, xl
    k = fp8_hub.kernels()
    grid = torch.tensor([1, 1.5, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 448, 0.5, 0.25, 0.75, 24, 40, 96, 192, 320, 384])
    rows = []
    for m, n, kk in ((1, 256, 256), (16, 256, 512), (17, 384, 256), (32, 128, 384), (130, 128, 256)):
        g = xl.gen(m * 31 + n)
        w, s = xl._fp8_weight(g, n, kk)
        c = grid[torch.randint(0, len(grid), (m, kk), generator=g)] * (torch.randint(0, 2, (m, kk), generator=g) * 2 - 1)
        c.reshape(m, kk // 128, 128)[:, :, 5] = 448.0
        x = (c * 2.0 ** -6).to(torch.float16)
        assert torch.equal(x.float(), c * 2.0 ** -6)
        y = k.matmul_2d(x, w, s, [128, 128], torch.float16)
        ref64 = xl._block_matmul_cpu(x, w, s, exact=True)
        emu = xl._block_matmul_cpu(x, w, s, exact=False)
        q, sc = k.fp8_act_quant(x, 128)
        qr, scr = xl._quantise(x.float())
        rows.append(dict(m=m, n=n, k=kk,
                         matmul_2d_vs_reference_max_rel=float((y.double() - ref64).abs().max() / ref64.abs().max()),
                         matmul_2d_bit_equal_to_emu=bool((y.view(torch.int16) == emu.view(torch.int16)).all()),
                         act_quant_codes_equal=bool((q.view(torch.uint8) == qr.reshape(m, kk).view(torch.uint8)).all()),
                         act_quant_scales_equal=bool((sc == scr).all())))
        print(rows[-1], flush=True)
    ok = all(r['matmul_2d_bit_equal_to_emu'] and r['matmul_2d_vs_reference_max_rel'] <= 2 ** -10 and r['act_quant_codes_equal']
             and r['act_quant_scales_equal'] for r in rows)
    out = dict(schema='archbench-xl-interpreter-check/1', domain='Triton CPU interpreter on the unchanged Hub kernels; exact-conversion inputs',
               kernel=fp8_hub.IDENTITY, torch=torch.__version__, rows=rows, ok=ok)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + '\n')
    print('OK' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
