#!/usr/bin/env python3
"""Speed prediction for the 32-point xl tier (domain: SystemC/TLM model prediction; FP8 not validated against the board).

    python3 tools/r42/kfuse_speed_xl.py --collected DIR --inputs evidence/r42/kfuse_speed/xl_inputs.json [--out OUT.json]

The Hub FP8 kernel (xl tier registration, archbench/xl.py) is timed with the TLM on proxy shapes that keep its per-program
structure (one 16-row chunk x one 128-column block x K/128 iterations) and enough programs per core to occupy the four role
contexts. Per M class (M = 1 and M >= 16 compile to different kernels) and scenario, one wave of 4 concurrent programs per
core costs T_wave(K) = a + b * K/128 (two K values). A point with P programs and K_it iterations per program takes
ceil(P / 12) waves (3 cores x 4 contexts). Composite points (layer, decoder) are split into the parts of archbench.xl.COMPONENTS:
the FP8 projections use the model above; attention, element-wise, head and the other fp16 kernels keep the planning estimate
of the xl tier (bound / board eta of the analogous 113-point family, archbench/xl_estimate.py) and the factor measured on the
analogous 113-point family (1.0 where that family has no matrix command).
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 128
H, INTER, NQ, NKV, HD = 2048, 6144, 16, 8, 128
LAYER_PROJ = [(NQ * HD, H), (NKV * HD, H), (NKV * HD, H), (H, NQ * HD), (INTER, H), (INTER, H), (H, INTER)]   # (N, K)


def cdiv(a, b):
    return -(-a // b)


def waves_for(programs):
    return cdiv(programs, 12)


def fp8_cycles(m, n, k, wave):
    """cycles of one Hub-kernel launch: programs = row chunks x column blocks; wave(k_it) -> cycles of a wave"""
    programs = cdiv(m, 16) * cdiv(n, BLOCK)
    return waves_for(programs) * wave(k // BLOCK)


def load(dir_, name):
    p = Path(dir_) / f'{name}.json'
    return json.loads(p.read_text())['points'] if p.exists() else {}


def wave_fn(tables, mclass, variant, kind='fp8_linear'):
    """T_wave(k_it) = a + b k_it from the proxies (mclass 1: m1, else m48) with K = 512 and 1024 at n = 1536"""
    m = 1 if mclass == 1 else 48
    waves = 1 if m == 1 else 3          # programs per core / 4: n=1536 -> 12 column blocks x (1 | 3 chunks) over 3 cores
    pts = []
    for tag in ('xss', 'xss2', 'xss3', 'xss4'):
        t = tables.get((tag, variant), {})
        for k in (512, 1024):
            key = f'op.{kind}/xl/fp8-m{m}-k{k}-n1536'
            if key in t and t[key].get('device_cycles'):
                pts.append((k // BLOCK, t[key]['device_cycles'] / waves))
    if len(pts) < 2:
        return None
    (k1, c1), (k2, c2) = sorted(pts)[:2]
    b = (c2 - c1) / (k2 - k1)
    a = c1 - b * k1
    return lambda k_it, a=a, b=b: a + b * k_it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collected', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--families')   # json of 113-point family factors (kfuse_speed_predict.py output)
    ap.add_argument('--out')
    a = ap.parse_args()
    inputs = json.loads(Path(a.inputs).read_text())
    tables = {}
    for p in Path(a.collected).glob('xss*_*.json'):
        stem = p.stem
        tag, variant = stem.split('_', 1)
        try:
            tables[(tag, variant)] = json.loads(p.read_text())['points']
        except Exception:
            pass
    variants = sorted({v for (_t, v) in tables})
    fam = json.loads(Path(a.families).read_text())['by_family'] if a.families else {}
    out = dict(variants=variants, points={})
    for p in inputs['points']:
        pid, family = p['id'], p['family']
        parts = {q['part']: q['cycles'] for q in p['parts']}          # planning cycles per part (bound / eta)
        row = dict(point=pid, family=family, s0_parts=parts)
        res = {}
        for v in variants:
            tot0, totv = 0.0, 0.0
            ok = True
            for part, cyc in parts.items():
                if family in ('fp8_linear',) or part == 'fp8_projections':
                    proj = p['fp8_shapes']
                    f0 = wave_fn(tables, 1 if p['m_class'] == 1 else 16, 'cal')
                    fv = wave_fn(tables, 1 if p['m_class'] == 1 else 16, v)
                    if not (f0 and fv):
                        ok = False
                        break
                    c0 = sum(fp8_cycles(m, n, k, f0) for (m, n, k) in proj)
                    cv = sum(fp8_cycles(m, n, k, fv) for (m, n, k) in proj)
                    tot0 += c0
                    totv += cv
                else:
                    tot0 += cyc
                    totv += cyc * p.get('other_factor', {}).get(v, 1.0)
            if ok:
                res[v] = dict(s0=tot0, v=totv, factor=totv / tot0)
        row['variants'] = res
        out['points'][pid] = row
    for v in variants:
        fs = [r['variants'][v]['factor'] for r in out['points'].values() if v in r['variants']]
        out.setdefault('geo_speedup', {})[v] = 1 / math.exp(sum(math.log(x) for x in fs) / len(fs)) if fs else None
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get('geo_speedup'), indent=1))


if __name__ == '__main__':
    main()
