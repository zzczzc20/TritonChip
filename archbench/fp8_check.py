"""The FP8 points on a host with torch: each point's program, eager on the
CPU, against the reference the platforms compare with.

    python -m archbench.fp8_check [--seed 0] [--point ID ...]

For every FP8 point -- registry dtype e4m3/e5m2, or a point with an FP8
weight -- this builds the point from its seed, computes the platforms'
reference (archbench.platforms.common.reference: binary32, FP8 operands kept
FP8 for torch's own CPU _scaled_mm), runs the same module eagerly on the CPU
with the point's own low-precision inputs, and prints the max relative error
against the point's tolerance, the output and reference dtypes, and torch's
FLOP count beside the declared MACs (what selftest requires to agree for a
dense family). It exits 1 if a point fails to run, exceeds its tolerance or
declares MACs the counter does not find.

An eager CPU run is not a device result. It confirms the definitions, the
reference path and the torch build (a CPU _scaled_mm, a FLOP formula for it)
before the points go to a simulator or a board, e.g.

    python -m archbench.run --platform tnca --profile core-sim-dp \\
        --backend systemc-tlm-grid --point op.scaled_mm/tiny/e4m3-to-bf16-m16-k64-n64
"""
import argparse
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

from . import registry
from .platforms.common import compare, flatten, reference
from .registry import FP8_DTYPES


def fp8_points():
    return [p for p in registry.points()
            if p.dtype in FP8_DTYPES
            or any(v in FP8_DTYPES for v in p.params.values() if isinstance(v, str))]


def _dt(t):
    return str(t.dtype).replace('torch.', '')


def check(p, seed):
    ref = reference(p, seed)
    m, ins = p.make(seed)
    m = m.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        got = m(*ins)
    counted = fc.get_total_flops() / 2
    cmp = compare(got, ref, p.tolerance)
    macs_ok = abs(counted - p.demand.macs) <= 1e-6 * max(1.0, counted)
    return dict(ok=cmp['ok'] and macs_ok, max_rel_err=cmp['max_rel_err'], detail=cmp['detail'],
                tol=p.tolerance, out_dtype=_dt(flatten(got)[0]), ref_dtype=_dt(ref[0]),
                counted_macs=counted, declared_macs=p.demand.macs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--point', nargs='+', help='point ids (default: every FP8 point)')
    args = ap.parse_args(argv)
    pts = [registry.get(i) for i in args.point] if args.point else fp8_points()
    print(f'torch {torch.__version__}; {len(pts)} FP8 point(s), seed {args.seed}')
    fails = 0
    for p in pts:
        try:
            r = check(p, args.seed)
        except Exception as e:                               # noqa: BLE001
            fails += 1
            print(f'ERROR {p.id}: {type(e).__name__}: {str(e)[:400]}')
            continue
        fails += not r['ok']
        err = 'n/a' if r['max_rel_err'] is None else f'{r["max_rel_err"]:.3g}'
        print(f'{"PASS" if r["ok"] else "FAIL"} {p.id}: max_rel_err={err} tol={r["tol"]:.3g} '
              f'out={r["out_dtype"]} ref={r["ref_dtype"]} macs declared={r["declared_macs"]:.6g} '
              f'counted={r["counted_macs"]:.6g}' + (f'\n     {r["detail"]}' if r['detail'] else ''))
    print(f'archbench fp8_check: {"FAIL" if fails else "PASS"} ({fails} failing point(s))')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
