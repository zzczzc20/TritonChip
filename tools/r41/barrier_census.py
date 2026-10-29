#!/usr/bin/env python3
"""Barrier census of compiled TNCA assembly (R41 G2, run-2 freeze).

    barrier_census.py [--bar-n 8] [--contexts 4] [--json OUT] DIR [DIR ...]

Every `*.s` below the given directories is a compiled kernel (TNCA_KERNEL_DUMP).
A barrier is named in assembly only by the immediate of mb.init, mb.arm,
mb.arrive and mb.inval; mb.wait, mb.test and m.mma[.dp].async take a token
that mb.arm produced (the context's slice number already ORed in). So the
logical barrier indices a program uses are exactly the immediates. Codex's
guard (rtl/core/tnca_core.v d_bar_index_oob) faults an immediate >= BAR_N >> slice_k
when the launch is sliced (RSLICE), where slice_k is log2 of the number of
programs per launch (the physical-chunk width: 1, 2 or 4); unsliced launches
(one program per core) have all BAR_N. This tool reports, per kernel, the
immediates and the largest one, and checks them against the narrowest slice
the image can ever apply, BAR_N // contexts (the 4-context SPMD launch):
if every kernel is below it, no launch width can fault whatever the runtime
chooses. A kernel above it is listed with the slice it would need.
"""
import argparse
import json
import re
import sys
from pathlib import Path

OPS = re.compile(r'^\s*(mb\.(init|arm|arrive|inval))\b([^/\n]*)')
IMM = re.compile(r'\bb(\d+)\b')


def census(path):
    uses = {}
    for line in path.read_text().splitlines():
        m = OPS.match(line)
        if not m:
            continue
        for g in IMM.findall(m.group(3)):
            uses.setdefault(int(g), set()).add(m.group(2))
    return uses


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bar-n', type=int, default=8)
    ap.add_argument('--contexts', type=int, default=4)
    ap.add_argument('--json')
    ap.add_argument('dirs', nargs='+', type=Path)
    a = ap.parse_args()
    slice_min = a.bar_n // a.contexts
    rows, over = [], []
    for d in a.dirs:
        for s in sorted(d.rglob('*.s')):
            u = census(s)
            row = dict(path=str(s), barriers=sorted(u), count=len(u), max_index=max(u) if u else -1,
                       ops={str(k): sorted(v) for k, v in sorted(u.items())})
            rows.append(row)
            if row['max_index'] >= slice_min:
                over.append(row)
    kernels = len(rows)
    with_b = sum(1 for r in rows if r['count'])
    print(f'kernels {kernels}; using barriers {with_b}; '
          f'largest immediate {max((r["max_index"] for r in rows), default=-1)}; '
          f'distinct barriers per kernel max {max((r["count"] for r in rows), default=0)}; '
          f'narrowest slice {slice_min} (BAR_N {a.bar_n} / {a.contexts} contexts)')
    hist = {}
    for r in rows:
        hist[r['count']] = hist.get(r['count'], 0) + 1
    print('kernels by number of distinct barriers:', dict(sorted(hist.items())))
    for r in over:
        print(f'OVER {r["path"]}: barriers {r["barriers"]}')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(bar_n=a.bar_n, contexts=a.contexts, narrowest_slice=slice_min,
                                                kernels=rows), indent=1))
    return 1 if over else 0


if __name__ == '__main__':
    sys.exit(main())
