#!/usr/bin/env python3
"""Model-versus-board validation over every family of the 145 points (domain: SystemC/TLM model against board).

    python3 tools/r42/dse_validate_all.py TAG [--out evidence/r42/dse/validation_all.json]

TAG = bundle tag (val_dse_base: validation software tree 33b41987, base model). Points: direct (tiny/small/medium, heavy points that fit,
small xl points), proxy (reduced grid, scaled by the demand ratio, dse_proxies.py), wave (FP8 Hub-kernel points from the fitted wave
model). Board = round-2 records (ab-int2, xl-cand2): same software as the model runs. A family is USABLE for a verdict when its
median |error| <= 20 %, its geometric mean model/board lies in 0.8..1.25 and at least 60 % of its points are within 20 %.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dse_common as C   # noqa: E402
import dse_eval as E     # noqa: E402


def main():
    tag = sys.argv[1]
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
    B = C.board()
    bd = E.Bundle(tag)
    rows = []
    for pid, b in sorted(B.items()):
        c, kind = E.model_cycles(bd, pid, B)
        if c:
            rows.append(dict(point=pid, family=b['family'], tier=b['tier'], xl=b['xl'], board=b['r2'], model=c, ratio=c / b['r2'], kind=kind))
    fam = {}
    for r in rows:
        fam.setdefault(r['family'], []).append(r)
    res = {}
    print(f'{"family":<20}{"n":>3}{"dir/px/wave":>12}{"geo m/b":>9}{"med|e|":>8}{"max|e|":>8}{"in20":>6}  verdict')
    for f, rs in sorted(fam.items()):
        e = [abs(r['ratio'] - 1) for r in rs]
        g = C.geo([r['ratio'] for r in rs])
        kinds = [sum(r['kind'] == k for r in rs) for k in ('direct', 'proxy', 'wave')]
        ok = statistics.median(e) <= 0.20 and 0.8 <= g <= 1.25 and sum(x <= 0.2 for x in e) >= 0.6 * len(e)
        res[f] = dict(n=len(rs), kinds=kinds, geo=g, median_abs_err=statistics.median(e), max_abs_err=max(e), within20=sum(x <= .2 for x in e), usable=ok)
        print(f'{f:<20}{len(rs):>3}{"/".join(map(str, kinds)):>12}{g:>9.3f}{statistics.median(e):>8.3f}{max(e):>8.3f}{sum(x <= 0.2 for x in e):>6}  {"usable" if ok else "NOT USABLE"}')
    allr = [r['ratio'] for r in rows]
    e = [abs(x - 1) for x in allr]
    print(f'ALL modelled n={len(rows)} of {len(B)}  geo {C.geo(allr):.3f}  median|err| {statistics.median(e):.3f}  within20 {sum(x <= .2 for x in e)}')
    for t in ('tiny', 'small', 'medium', 'large', 'hbm'):
        x = [r['ratio'] for r in rows if r['tier'] == t and not r['xl']]
        if x:
            print(f'  tier {t:<7} n={len(x):>3} geo {C.geo(x):.3f} median|err| {statistics.median(abs(i - 1) for i in x):.3f}')
    x = [r['ratio'] for r in rows if r['xl']]
    if x:
        print(f'  tier xl      n={len(x):>3} geo {C.geo(x):.3f} median|err| {statistics.median(abs(i - 1) for i in x):.3f}')
    if out:
        out.write_text(json.dumps(dict(rows=rows, by_family=res), indent=1))
    # verdict per family and element type (matmul and softmax have fp32 points whose kernels differ): key `family` or `family/fp32`
    gk = {}
    for r in rows:
        k = r['family'] + ('/fp32' if r['point'].split('/')[2].startswith('fp32') else '')
        gk.setdefault(k, []).append(r)
    ver = {}
    for k, rs in gk.items():
        e = [abs(r['ratio'] - 1) for r in rs]
        g = C.geo([r['ratio'] for r in rs])
        ver[k] = bool(statistics.median(e) <= 0.20 and 0.8 <= g <= 1.25 and sum(x <= 0.2 for x in e) >= 0.6 * len(e))
    (C.ROOT / 'evidence/r42/dse/validation_families.json').write_text(json.dumps(ver, indent=1, sort_keys=True))
    print('per family and element type:', {k: v for k, v in sorted(ver.items()) if '/' in k or not v})


if __name__ == '__main__':
    main()
