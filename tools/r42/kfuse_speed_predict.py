#!/usr/bin/env python3
"""Speed prediction of a wider K-fused lane and of lower matrix-command instruction overhead on the 113 points.

    python3 tools/r42/kfuse_speed_predict.py --collected DIR [--out evidence/r42/kfuse_speed/predict_113.json]

Domain of every number: SystemC/TLM model prediction (profile core-vrf-sync-v5-x8-g2c4-dp, 3 cores, 100 MHz; configuration
`cal` of tools/r42/kfuse_speed_tlm_cfg.py, calibrated on single-core RTL for the K16 command time and validated against the board
cycles of evidence/r42/board/ab-int2 in evidence/r42/kfuse_speed/validation_*.txt), applied to the board as a RATIO:

    predicted board cycles (variant v) = board cycles (ab-int2) x TLM cycles (v) / TLM cycles (S0, today's lane)

so that a model bias in the absolute cycle count cancels and only the model's response to the change is used. Points the
model did not run (non-GEMM families, FP32, attention kernels without matrix commands) are held at their board cycles.
Heavy points (large, hbm tiers) use the reduced-grid proxies named in tools/r42/kfuse_speed_proxies.py: the factor of the
proxy replaces the factor of the point.

Scenarios: S0 = streams as compiled today; S1 = scalar/branch instructions of the innermost matrix loop issue for free
(issue_free_mode 1: hoisting, strength reduction, hardware loop); S2 = every instruction of the innermost matrix loop issues
for free (issue_free_mode 2: a macro-tile command sequences transfers and matrix commands, resources unchanged).
G: the lane's group size for m.mma.dp.async (today 2 for FP16/BF16, 4 for E4M3 x E4M3); 4, 8, 16 with one 64-byte B register
read per ceil(G x bytes / 4) registers per step (`mtx_dp_b_regs`).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kfuse_speed_validate as V   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAIN = ['cal', 'cal_s1', 'cal_s2', 'cal_g8', 'cal_g8_s1', 'cal_g8_s2', 'cal_g16', 'cal_g16_s1', 'cal_g16_s2']
EXTRA = ['cal_g4', 'cal_g4_s1', 'cal_g4_s2', 'cal_g8_pf', 'cal_g16_pf', 'cal_s1_pf',
         'cal_u32', 'cal_u32_s1', 'cal_u32_g8', 'cal_u32_g8_s1', 'cal_u32_g8_s2',
         'ideal', 'ideal_s1', 'ideal_s2', 'ideal_g8', 'ideal_g8_s1', 'ideal_g8_s2', 'ideal_g16', 'ideal_g16_s1',
         'ideal_g16_s2', 'ideal_g8_pf', 'ideal_g16_pf']
SENS = {'cal_u32': ['cal_u32', 'cal_u32_s1', 'cal_u32_g8', 'cal_u32_g8_s1', 'cal_u32_g8_s2'],
        'ideal': ['ideal', 'ideal_s1', 'ideal_s2', 'ideal_g8', 'ideal_g8_s1', 'ideal_g8_s2', 'ideal_g16', 'ideal_g16_s1',
                  'ideal_g16_s2', 'ideal_g8_pf', 'ideal_g16_pf']}
VARIANTS = MAIN + EXTRA


def load(dir_, prefix):
    out = {}
    for v in VARIANTS:
        p = Path(dir_) / f'{prefix}_{v}.json'
        if p.exists():
            out[v] = json.loads(p.read_text())['points']
    return out


def geo(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collected', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    board = V.board_records()
    ids = sorted(board)
    direct = load(a.collected, 'g')          # tiny/small/medium GEMM points
    heavy = load(a.collected, 'h')           # direct heavy points
    proxy = load(a.collected, 'p')           # reduced-grid proxies
    proxy_of = {}
    sys.path.insert(0, str(ROOT / 'tools/r42'))
    px = json.loads((ROOT / 'evidence/r42/kfuse_speed/proxy_of.json').read_text()) if (
        ROOT / 'evidence/r42/kfuse_speed/proxy_of.json').exists() else {}
    for proxy_id, stands in px.items():
        proxy_of[stands] = proxy_id
    rows = []
    for pid in ids:
        r = dict(point=pid, family=V.family_of(pid), tier=pid.split('/')[1], board=board[pid]['time']['device'])
        src = None
        for name, tbl in (('direct', direct), ('heavy', heavy)):
            if 'cal' in tbl and pid in tbl['cal'] and tbl['cal'][pid].get('status') == 'pass':
                src, key, t = name, pid, tbl
                break
        else:
            if pid in proxy_of and 'cal' in proxy and proxy_of[pid] in proxy['cal']:
                src, key, t = 'proxy', proxy_of[pid], proxy
        r['source'] = src
        f = {}
        if src:
            base = t['cal'][key]['device_cycles']
            r['model_s0'] = base
            for v in VARIANTS:
                if v in t and key in t[v] and t[v][key].get('status') == 'pass':
                    f[v] = t[v][key]['device_cycles'] / base
            if 'matrix_engine_commands' in (t['cal'][key].get('counters_total') or {}):
                r['mma_commands'] = t['cal'][key]['counters_total'].get('mma_commands')
        # G = 16 with S1/S2 was not run on the reduced-grid proxies: factor(g16_sX) = factor(g8_sX) x factor(g16) / factor(g8)
        # (the G = 8 -> 16 step of the same proxy; on the 30 tiny/small/medium points the two sides agree to 0.3 %)
        r['extrapolated'] = []
        if src == 'proxy':
            for sx in ('s1', 's2'):
                if f'cal_g16_{sx}' not in f and all(k in f for k in (f'cal_g8_{sx}', 'cal_g16', 'cal_g8')):
                    f[f'cal_g16_{sx}'] = f[f'cal_g8_{sx}'] * f['cal_g16'] / f['cal_g8']
                    r['extrapolated'].append(f'cal_g16_{sx}')
        r['factor'] = f
        rows.append(r)
    # geometric-mean effect on the 113 points (others held constant)
    summary = {}
    for v in VARIANTS:
        fs = [r['factor'].get(v, 1.0) for r in rows]
        have = [r for r in rows if v in r['factor']]
        summary[v] = dict(geo_speedup_113=1 / geo(fs), n_with_model=len(have),
                          geo_speedup_modelled=1 / geo([r['factor'][v] for r in have]) if have else None,
                          sum_cycles_speedup=sum(r['board'] for r in rows) / sum(r['board'] * r['factor'].get(v, 1.0) for r in rows))
    fam = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for v in VARIANTS:
            if v in r['factor']:
                fam[r['family']][v].append(1 / r['factor'][v])
    famsum = {f: {v: dict(n=len(xs), geo=geo(xs)) for v, xs in d.items()} for f, d in fam.items()}
    # sensitivity families (a different transfer path or memory): speed-up of each variant relative to ITS OWN S0 variant,
    # over the 30 tiny/small/medium GEMM points, and of that S0 variant relative to the calibrated S0
    sens = {}
    for base, vs in SENS.items():
        pts = [r for r in rows if base in r['factor']]
        if not pts:
            continue
        sens[base] = dict(n=len(pts), s0_vs_cal=1 / geo([r['factor'][base] for r in pts]))
        for v in vs:
            have = [r for r in pts if v in r['factor']]
            if have:
                sens[base][v] = 1 / geo([r['factor'][v] / r['factor'][base] for r in have])
        sens[base]['by_family'] = {f: {v: 1 / geo([r['factor'][v] / r['factor'][base] for r in pts if r['family'] == f and v in r['factor']])
                                       for v in vs if any(r['family'] == f and v in r['factor'] for r in pts)}
                                   for f in sorted({r['family'] for r in pts})}
    print(f'{"variant":<14}{"geo speed-up (113)":>20}{"modelled pts":>14}{"geo (modelled pts)":>20}{"sum-cycles":>12}')
    for v, s in summary.items():
        print(f'{v:<14}{s["geo_speedup_113"]:>20.4f}{s["n_with_model"]:>14}{(s["geo_speedup_modelled"] or float("nan")):>20.4f}{s["sum_cycles_speedup"]:>12.4f}')
    for base, s in sens.items():
        print(f'sensitivity {base}: n={s["n"]}, S0 variant vs calibrated S0 {s["s0_vs_cal"]:.4f}; relative to its own S0: ' +
              ', '.join(f'{v} {x:.4f}' for v, x in s.items() if v not in ('n', 's0_vs_cal', 'by_family')))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=summary, by_family=famsum, sensitivity=sens, rows=rows), indent=1))


if __name__ == '__main__':
    main()
