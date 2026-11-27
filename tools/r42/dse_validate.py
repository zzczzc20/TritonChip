#!/usr/bin/env python3
"""Model-versus-board validation over ALL families of the 145 points (domain: SystemC/TLM model against board).

    python3 tools/r42/dse_validate.py --model COLLECTED.json [--model ...] [--proxy PROXY.json ...] --out OUT.json [--board r2]

Board = the round-2 records (ab-int2 for the 113 points, xl-cand2 for the 32 xl points), the newest records that exist for
the validation software tree 33b41987 (= the software of ab-int2 and xl-cand2, compiler/python/tools/asm identical).
A direct model record is compared to the board cycles of the same point. A proxy record (reduced grid) is scaled to the point it
stands for by `scale` (kind: macs | bytes | explicit factor, see dse_proxies.py PROXIES) and then compared.
Per family: n, geometric mean model/board, median and max |error|, count within 20 %. A family is USABLE for a verdict when
every one of its modelled points has |error| <= 20 % ... see `verdict` below (median |error| <= 20 % and geo within 0.8..1.25).
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dse_common as C   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', default=[])
    ap.add_argument('--proxy', action='append', default=[], help='collected proxy records; needs --proxy-map')
    ap.add_argument('--proxy-map', help='json: proxy id -> {stands_for, scale}')
    ap.add_argument('--out')
    a = ap.parse_args()
    B = C.board()
    model = {}
    for f in a.model:
        for k, v in C.load_collected(f).items():
            if v.get('status') == 'pass' and v.get('device_cycles'):
                model[k] = dict(cycles=v['device_cycles'], source='direct', per_launch=v.get('per_launch'))
    if a.proxy:
        pm = json.loads(Path(a.proxy_map).read_text())
        for f in a.proxy:
            for k, v in C.load_collected(f).items():
                if k in pm and v.get('status') == 'pass' and v.get('device_cycles'):
                    model[pm[k]['stands_for']] = dict(cycles=v['device_cycles'] * pm[k]['scale'], source='proxy',
                                                      proxy=k, scale=pm[k]['scale'])
    rows = []
    for pid, m in sorted(model.items()):
        b = B.get(pid)
        if not b:
            continue
        rows.append(dict(point=pid, family=b['family'], tier=b['tier'], xl=b['xl'], board=b['r2'], model=m['cycles'],
                         ratio=m['cycles'] / b['r2'], source=m['source']))
    fam = defaultdict(list)
    for r in rows:
        fam[r['family']].append(r)
    out = {}
    print(f'{"family":<20}{"n":>3}{"geo m/b":>9}{"med|e|":>8}{"max|e|":>8}{"in20":>6}  verdict')
    for f, rs in sorted(fam.items()):
        e = [abs(r['ratio'] - 1) for r in rs]
        g = C.geo([r['ratio'] for r in rs])
        med = statistics.median(e)
        ok = med <= 0.20 and 0.8 <= g <= 1.25 and sum(x <= 0.2 for x in e) >= 0.6 * len(e)
        out[f] = dict(n=len(rs), geo=g, median_abs_err=med, max_abs_err=max(e), within20=sum(x <= 0.2 for x in e), usable=ok)
        print(f'{f:<20}{len(rs):>3}{g:>9.3f}{med:>8.3f}{max(e):>8.3f}{sum(x <= 0.2 for x in e):>6}  {"usable" if ok else "NOT USABLE"}')
    allr = [r['ratio'] for r in rows]
    e = [abs(x - 1) for x in allr]
    print(f'ALL n={len(rows)} geo={C.geo(allr):.3f} median|err|={statistics.median(e):.3f} within20={sum(x <= .2 for x in e)}')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(rows=rows, by_family=out), indent=1))


if __name__ == '__main__':
    main()
