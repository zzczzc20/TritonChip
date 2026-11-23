#!/usr/bin/env python3
"""Write archbench/suites/xl_eta.json: the efficiencies the xl tier's device-time estimates use (archbench/xl_estimate.py).

    xl_eta.py --b41 evidence/r42/baseline/b41_board.json --r41-kernels-readme evidence/r41/fp8/kernels/README.md \
              [--out archbench/suites/xl_eta.json]

Nothing is typed in. For every point of tiers large and hbm of the 113 original points, `b41_board.json` gives the
board cycles of the R41 run-3 image; the efficiency is the strict bound of archbench.bounds (computed here from the
point's frozen demand) over those cycles. The FP8 pessimistic efficiency comes from the SystemC/TLM model cycles the
R41 FP8 kernel evidence reports for four real layer shapes (parsed from its README), over the strict bound of the same
shape. The primary FP8 efficiency, 0.2, is the planning figure of the task brief (2026-11-23), recorded as such.
Run it where torch and the registry import (thor): it needs archbench.registry.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FP8_PRIMARY = 0.2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geo(xs):
    xs = list(xs)
    return math.exp(sum(map(math.log, xs)) / len(xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b41', type=Path, required=True)
    ap.add_argument('--r41-kernels-readme', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'archbench' / 'suites' / 'xl_eta.json')
    a = ap.parse_args()
    from archbench import registry
    from archbench.bounds import tnca_bound
    from archbench.xl import fp8_gemm_demand
    from archbench.xl_estimate import PROFILE
    b41 = json.loads(a.b41.read_text())
    rows = {}
    for r in b41['rows']:
        if r['tier'] != 'large':
            continue
        p = registry.get(r['point'])
        assert p.definition_sha256() == r['point_sha256'], f'{r["point"]}: the frozen point changed'
        L, binding, _ = tnca_bound(p, PROFILE)
        rows.setdefault(r['family'], []).append(dict(point=r['point'], board_cycles=r['run3_cycles'], bound_cycles=L,
                                                     binding=binding, eta=L / r['run3_cycles']))
    families = {f: dict(eta=geo(x['eta'] for x in v), rows=v) for f, v in sorted(rows.items())}
    # FP8: the four real shapes of the R41 kernel evidence with their model cycles
    text = a.r41_kernels_readme.read_text()
    line = next(l for l in text.splitlines() if 'real layer shapes' in l)
    shapes = [tuple(int(x) for x in m.split('×')) for m in re.findall(r'\d+×\d+×\d+', line)]
    cycles = [float(x) * 1e6 for x in re.findall(r'([\d.]+) M', line.split('Model cycles:')[1])]
    assert len(shapes) == len(cycles) == 4, (shapes, cycles)
    cases = []
    for (m, n, k), c in zip(shapes, cycles):
        # the README lists M x N x K with N and K as the weight (N, K); K must be a multiple of 128 (2560 is)
        d = fp8_gemm_demand(m, n, k)
        shim = SimpleNamespace(id=f'fp8-{m}x{n}x{k}', family='fp8_linear', kind='op', demand=d, dtype='fp8')
        L, binding, _ = tnca_bound(shim, PROFILE)
        cases.append(dict(m=m, n=n, k=k, model_cycles=c, bound_cycles=L, binding=binding, eta=L / c))
    out = dict(schema='archbench-xl-eta/1', profile=PROFILE,
               sources=dict(b41=dict(path=str(a.b41), sha256=sha(a.b41), domain=b41['domain']),
                            r41_kernels_readme=dict(path=str(a.r41_kernels_readme), sha256=sha(a.r41_kernels_readme),
                                                    domain='SystemC/TLM model, uncalibrated, profile core-vrf-sync-v5 '
                                                           '(no m.mma.dp), 128-wide K blocks, bf16')),
               fp16_families=families,
               fp8=dict(primary=FP8_PRIMARY, primary_basis='the planning figure of the task brief (2026-11-23): '
                        'eta ~0.2 of the strict bound; no FP8 point has been run on a board',
                        pessimistic=dict(eta=geo(c['eta'] for c in cases), domain='SystemC/TLM model (uncalibrated)',
                                         cases=cases)))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(f'wrote {a.out}')
    for f, v in families.items():
        print(f'  {f:20s} eta {v["eta"]:.4f}  ({len(v["rows"])} large point(s))')
    print(f'  fp8 primary {FP8_PRIMARY}, pessimistic {out["fp8"]["pessimistic"]["eta"]:.4f} from', [round(c['eta'], 4) for c in cases])


if __name__ == '__main__':
    main()
