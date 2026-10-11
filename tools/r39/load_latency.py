#!/usr/bin/env python3
"""R39-01: the memory half of the scheduler's machine model, measured where
ArchBench runs (the board, or the device-level RTL simulation that matches it
cycle for cycle).

tools/r39/latency_probe.py measures the core on the core bench, whose memory
answers in one cycle; the board's answers through the shell and HBM. Each
kernel here is unrolled (tl.static_range) into N loads -- dependent, so each
waits for the last, or independent, so only the load/store unit limits them
-- and run at two lengths; the difference in device cycles over the
difference in loads is the per-load cost with launch and prologue cancelled.

    TNCA_BACKEND=fpga-u280 TNCA_BACKEND_ARGS='{"xclbin": ...}' \\
        python3 tools/r39/load_latency.py --json out.json

  sload chain   i = p[i] on a scalar           -> scalar load-to-use
  gather chain  idx = p[idx] on a vector       -> gather load-to-use
  vload chain   x = p[base + lane] with base from the last load's lane 0
                                               -> vector load + extract
  vload indep   N rows summed                  -> issue interval of vector loads
  sload indep   N scalars summed               -> issue interval of scalar loads
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler' / 'python')]
os.environ.setdefault('TNCA_TRITON', '1')
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v1')


def kernels(tl, triton):
    @triton.jit
    def sload_chain(p, out, N: tl.constexpr):
        i = tl.load(p)
        for _ in tl.static_range(N):
            i = tl.load(p + i)
        tl.store(out, i)

    @triton.jit
    def gather_chain(p, out, N: tl.constexpr):
        idx = tl.load(p + tl.arange(0, 16))
        for _ in tl.static_range(N):
            idx = tl.load(p + idx)
        tl.store(out + tl.arange(0, 16), idx)

    @triton.jit
    def sload_indep(p, out, N: tl.constexpr):
        acc = tl.load(p)
        for k in tl.static_range(N):
            acc += tl.load(p + 1 + k)
        tl.store(out, acc)

    @triton.jit
    def vload_indep(p, out, N: tl.constexpr):
        acc = tl.load(p + tl.arange(0, 16))
        for k in tl.static_range(N):
            acc += tl.load(p + 16 * (k + 1) + tl.arange(0, 16))
        tl.store(out + tl.arange(0, 16), acc)

    return dict(sload_chain=sload_chain, gather_chain=gather_chain,
                sload_indep=sload_indep, vload_indep=vload_indep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json')
    ap.add_argument('--lengths', nargs=2, type=int, default=[8, 40])
    args = ap.parse_args()
    import torch
    import triton
    import triton.language as tl
    from tnca.triton_backend import driver as D
    ks = kernels(tl, triton)
    lo, hi = args.lengths
    res = {}
    for name, k in ks.items():
        cyc = []
        for n in (lo, hi):
            p = torch.zeros(16 * (hi + 2), dtype=torch.int32)
            out = torch.zeros(16, dtype=torch.int32)
            k[(1,)](p, out, n)
            cyc.append(D.TNCALauncher.last.last_cycles)
        per = (cyc[1] - cyc[0]) / (hi - lo)
        res[name] = dict(cycles=cyc, lengths=[lo, hi], per_load=per)
        print(f'{name:<14s} {cyc[0]:>8d} {cyc[1]:>8d}  {per:6.2f} cycles per load')
    res['backend'] = os.environ.get('TNCA_BACKEND')
    res['profile'] = os.environ.get('TNCA_PROFILE')
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
