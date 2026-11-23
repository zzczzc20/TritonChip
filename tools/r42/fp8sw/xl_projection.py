#!/usr/bin/env python3
"""Projection of the FP8 compiler work onto the 32-point xl tier (domain: SystemC/TLM model prediction; not a board result).

    python3 tools/r42/fp8sw/xl_projection.py --base TLM_BASE/results.json --cand TLM_CAND/results.json \
        --inputs evidence/r42/fp8sw/xl_inputs.json [--out OUT.json]

The method of tools/r42/kfuse_speed_xl.py with two compilers instead of two hardware scenarios. The Hub FP8 kernel is timed with the
TLM (three cores, `cal` configuration) on proxy shapes that keep its per-program structure (one 16-row chunk x one 128-column block
x K/128 iterations): per M class (M = 1 and M >= 16 compile to different kernels) a wave of 4 concurrent programs per core costs
T_wave(K_it) = a + b K_it from the proxies (n = 1536, K = 512 and 1024). A launch with P programs takes ceil(P / 12) waves.
Composite points are split into the parts of archbench.xl.COMPONENTS: the FP8 projections use the wave model, every other part
keeps the xl tier's planning estimate (bound / board eta of the analogous 113-point family) unchanged -- the compiler changes of
this round can only help those parts, and that is not credited here.
"""
import argparse
import json
import math
from pathlib import Path

BLOCK = 128


def cdiv(a, b):
    return -(-a // b)


def wave_fn(results, mclass):
    m = 1 if mclass == 1 else 48
    waves = 1 if m == 1 else 3       # programs per core / 4 at n = 1536: 12 column blocks x (1 | 3 row chunks) over 3 cores x 4 contexts
    pts = []
    for p in results['points']:
        for k in (512, 1024):
            if p['key'] == f'op.fp8_linear/xl/fp8-m{m}-k{k}-n1536' and p.get('device_cycles') and p.get('status') == 'pass':
                pts.append((k // BLOCK, p['device_cycles'] / waves))
    if len(pts) < 2:
        raise SystemExit(f'missing proxies for M class {mclass}')
    (k1, c1), (k2, c2) = sorted(pts)[:2]
    b = (c2 - c1) / (k2 - k1)
    a = c1 - b * k1
    return lambda k_it: a + b * k_it, (a, b)


def fp8_cycles(m, n, k, wave):
    programs = cdiv(m, 16) * cdiv(n, BLOCK)
    return cdiv(programs, 12) * wave(k // BLOCK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--cand', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    base, cand = (json.loads(Path(x).read_text()) for x in (a.base, a.cand))
    inputs = json.loads(Path(a.inputs).read_text())
    waves = {}
    for mc in (1, 16):
        fb, ab = wave_fn(base, mc)
        fc, ac = wave_fn(cand, mc)
        waves[mc] = (fb, fc, ab, ac)
    rows = {}
    for p in inputs['points']:
        parts = {q['part']: q['cycles'] for q in p['parts']}
        fb, fc, _, _ = waves[1 if p['m_class'] == 1 else 16]
        tot0 = totc = 0.0
        for part, cyc in parts.items():
            if p['family'] == 'fp8_linear' or part == 'fp8_projections':
                tot0 += sum(fp8_cycles(m, n, k, fb) for (m, n, k) in p['fp8_shapes'])
                totc += sum(fp8_cycles(m, n, k, fc) for (m, n, k) in p['fp8_shapes'])
            else:
                tot0 += cyc
                totc += cyc
        rows[p['id']] = dict(family=p['family'], m_class=p['m_class'], fp8=bool(p['fp8_shapes']), base=tot0, cand=totc, speedup=tot0 / totc)
    geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    out = dict(
        wave_model={str(mc): dict(base_a_b=list(v[2]), cand_a_b=list(v[3])) for mc, v in waves.items()},
        points=rows,
        geomean_speedup_all=geo([r['speedup'] for r in rows.values()]),
        geomean_speedup_fp8_points=geo([r['speedup'] for r in rows.values() if r['fp8']]),
        points_n=len(rows), fp8_points_n=sum(1 for r in rows.values() if r['fp8']))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    for pid, r in sorted(rows.items()):
        print(f"{pid:62s} {r['base']:16,.0f} {r['cand']:16,.0f} {r['speedup']:6.3f}")
    print(f"geomean speedup: all {out['points_n']} points {out['geomean_speedup_all']:.4f}; {out['fp8_points_n']} FP8 points {out['geomean_speedup_fp8_points']:.4f}")


if __name__ == '__main__':
    main()
