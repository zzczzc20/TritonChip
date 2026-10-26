#!/usr/bin/env python3
"""R40-03 v3 directed layout checks: transposed and strided standard Torch graphs.

    v3_layout_tests.py --width {16,32,64,128} --json OUT

`make r40-tiles` covers tails, masks, int64, dtypes, a strided alias and the
logical GEMM tiles on contiguous operands. This adds the transpose/stride
layouts of acceptance item 1 through the same standard native path
(`torch.compile(fullgraph=True)` on the `tnca` device, TNCA_MAX_BLOCK=width,
TNCA_GEMM_TILE=16,min(width,64),16): transposed pointwise reads, transposed
plus contiguous operands, a column-strided 2-D view, int64 beyond 32 bits read
transposed, reductions over a transposed (strided) axis, a cumsum scan (kept
at 16 lanes above a 16 cap by the delivered Inductor configuration), linear
with its [n, k] weight (B transposed) and mm with a transposed A. Every case compares
the warm and the measured call with the CPU result and requires zero steady
tensor transfers; register-block widths, logical GEMM tiles, spill plans and
resources of the executed artifacts are recorded. No performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--width', type=int, choices=(16, 32, 64, 128), required=True)
ap.add_argument('--json', type=Path, required=True)
args = ap.parse_args()
COLUMNS = min(args.width, 64)
os.environ['TNCA_MAX_BLOCK'] = str(args.width)
os.environ['TNCA_GEMM_TILE'] = f'16,{COLUMNS},16'
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
os.environ.setdefault('TNCA_HOST_TIMEOUT', '172800')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm')]

import torch  # noqa: E402
import triton  # noqa: E402,F401
import tnca.torch_device as D  # noqa: E402
from tnca import artifact, device  # noqa: E402
from tnca.triton_backend.driver import launch_log  # noqa: E402


def cases():
    g = torch.Generator().manual_seed(4003)

    def rnd(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g).to(dtype)

    big = torch.arange(65 * 33, dtype=torch.int64).reshape(65, 33)
    big = torch.where(big % 2 == 0, big + (1 << 35), -big - (1 << 37))
    return [
        ('transpose_pointwise_f32', lambda x: (x.t() * 1.5 + 1).contiguous(), (rnd(33, 127),), {}),
        ('transpose_pointwise_f16', lambda x: (x.t() * 1.5 + 1).contiguous(), (rnd(17, 65, dtype=torch.float16),),
         dict(rtol=2e-3, atol=2e-3)),
        ('transpose_plus_contiguous', lambda x, y: x.t() + y, (rnd(65, 33), rnd(33, 65)), {}),
        ('column_strided_view', lambda x: x[:, 1::2] * 2 - x[:, ::2], (rnd(31, 130),), {}),
        ('int64_transposed', lambda x: (x.t() * 3 + 7).contiguous(), (big,), {}),
        ('transpose_sum', lambda x: x.t().sum(-1), (rnd(127, 33),), dict(rtol=1e-4, atol=1e-4)),
        ('transpose_amax', lambda x: x.t().amax(-1), (rnd(65, 17),), {}),
        # A scan has no wide register-block lowering: at a cap above 16 the
        # delivered Inductor configuration keeps it at 16 lanes (retile).
        ('cumsum_scan_retile', lambda x: torch.cumsum(x, -1), (rnd(3, 127),),
         dict(rtol=1e-4, atol=1e-4)),
        ('linear_transposed_weight', lambda a, w, c: torch.nn.functional.linear(a, w, c),
         (rnd(17, 33, dtype=torch.float16) * .1, rnd(65, 33, dtype=torch.float16) * .1,
          rnd(65, dtype=torch.float16) * .1), dict(rtol=2e-3, atol=2e-3), 'gemm'),
        ('mm_transposed_a', lambda a, b: a.t() @ b,
         (rnd(33, 17, dtype=torch.float16) * .1, rnd(33, 65, dtype=torch.float16) * .1),
         dict(rtol=2e-3, atol=2e-3), 'gemm'),
    ]


def main():
    if os.environ.get('TNCA_BACKEND', 'sim-file') in ('sim-file', 'sim-pipe'):
        device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    rows, ok_all = [], True
    for spec in cases():
        name, fn, inputs, tol = spec[:4]
        gemm = len(spec) > 4
        row = dict(name=name, width=args.width, gemm_tile=[16, COLUMNS, 16] if gemm else None,
                   shapes=[list(t.shape) for t in inputs], dtypes=[str(t.dtype) for t in inputs])
        try:
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            dev = [t.to('tnca') for t in inputs]
            want = fn(*inputs)
            warm = compiled(*dev)
            torch.testing.assert_close(warm.cpu(), want, **tol)
            del warm
            with launch_log() as log:
                out = compiled(*dev)
            torch.testing.assert_close(out.cpu(), want, **tol)
            assert log and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in log), \
                'steady launches moved tensor bytes'
            kernels = []
            for e in log:
                manifest = artifact.load(e['artifact_path'])[0]
                prov = manifest['optional_metadata']['provenance']
                blocks = prov.get('register_blocks', {})
                spill = prov.get('vector_spill', {})
                assert manifest['resources']['scratch_bytes'] >= spill.get('scratch_bytes', 0)
                kernels.append(dict(
                    artifact=hashlib.sha256(Path(e['artifact_path']).read_bytes()).hexdigest(),
                    widths=blocks.get('widths', []), gemm_tiles=blocks.get('gemm_tiles', []),
                    reductions=blocks.get('reductions', []), spill=spill,
                    resources=manifest['resources']))
            if gemm and COLUMNS > 16:
                assert any([16, COLUMNS, 16] in k['gemm_tiles'] for k in kernels), \
                    f'logical tile 16x{COLUMNS}x16 not in executed artifacts: {kernels}'
            row.update(ok=True, kernels=kernels, launches=len(log))
            print(f'PASS {name}, width {args.width}, widths '
                  f'{sorted({w for k in kernels for w in k["widths"]})}, tiles '
                  f'{[k["gemm_tiles"] for k in kernels if k["gemm_tiles"]]}', flush=True)
        except Exception as error:                           # noqa: BLE001
            ok_all = False
            row.update(ok=False, error=f'{type(error).__name__}: {error}'[:2000])
            print(f'FAIL {name}, width {args.width}: {row["error"]}', flush=True)
        rows.append(row)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dict(schema='tnca-r40-v3-layout/1', width=args.width,
                                         profile=os.environ['TNCA_PROFILE'], cases=rows,
                                         ok=ok_all), indent=1) + '\n')
    passed = sum(r['ok'] for r in rows)
    print(f"r40-v3-layout {'PASS' if ok_all else 'FAIL'} ({passed}/{len(rows)}) "
          f"width {args.width}", flush=True)
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
