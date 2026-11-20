#!/usr/bin/env python3
"""The R42 board baseline (B41), computed from the R41 board records.

    baseline_table.py [--out evidence/r42/baseline/b41_board.json]

Reads evidence/r41/hbm_bw/g4/run3/board/original113_comparison.json (serial-grid cycle medians of the 113 original
points on the run-1 image 30e79769 and on the run-3 image 791b8af2, both measured at 100 MHz) and
evidence/r41/stage2/roofline/points.json (ArchBench demand and bound of the run-1 records). Nothing is typed in:
every value written is read from those files or computed from them.
"""
import argparse, collections, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CMP = ROOT/'evidence/r41/hbm_bw/g4/run3/board/original113_comparison.json'
ROOF = ROOT/'evidence/r41/stage2/roofline/points.json'

def geo(xs):
    xs = list(xs)
    return math.exp(sum(map(math.log, xs))/len(xs)) if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'evidence/r42/baseline/b41_board.json')
    a = ap.parse_args()
    cmp_ = json.loads(CMP.read_text())
    eta1 = {p['id']: p['eta'] for p in json.loads(ROOF.read_text())['tnca100']}
    rows = []
    for r in cmp_['rows']:
        op, tier = r['point'].split('/')[0], r['point'].split('/')[1]
        family = op.split('.', 1)[1]
        e1 = eta1.get(r['point'])
        rows.append(dict(point=r['point'], point_sha256=r['point_sha256'], family=family, tier=tier,
                         run1_cycles=r['old_cycles'], run3_cycles=r['new_cycles'],
                         run3_speedup=r['old_cycles']/r['new_cycles'],
                         run1_eta=e1, run3_eta=(e1*r['old_cycles']/r['new_cycles'] if e1 else None)))
    def group(key):
        g = collections.defaultdict(list)
        for r in rows: g[r[key]].append(r)
        return {k: dict(points=len(v), run3_over_run1_geo=geo(r['run3_speedup'] for r in v),
                        run3_eta_geo=geo(r['run3_eta'] for r in v if r['run3_eta']),
                        run3_cycles=sum(r['run3_cycles'] for r in v)) for k, v in sorted(g.items())}
    out = dict(schema='r42-board-baseline/1', domain='board (U280, 100 MHz, serial-grid device cycles, medians)',
               sources=[str(CMP.relative_to(ROOT)), str(ROOF.relative_to(ROOT))],
               run1_image='30e79769', run3_image_sha256=cmp_['new_image_sha256'], points=len(rows),
               run3_over_run1_geo=geo(r['run3_speedup'] for r in rows),
               run3_eta_geo=geo(r['run3_eta'] for r in rows if r['run3_eta']),
               by_tier=group('tier'), by_family=group('family'), rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1)+'\n')
    print(f"points {len(rows)}  run3/run1 S_geo {out['run3_over_run1_geo']:.3f}  run3 eta geo {out['run3_eta_geo']:.3f}")
    for k, v in out['by_tier'].items(): print(f"  tier {k:7s} n={v['points']:3d} S_geo {v['run3_over_run1_geo']:.3f} eta {v['run3_eta_geo']:.3f}")
    for k, v in sorted(out['by_family'].items(), key=lambda kv: kv[1]['run3_eta_geo'] or 9):
        print(f"  {k:18s} n={v['points']} S_geo {v['run3_over_run1_geo']:.3f} eta {v['run3_eta_geo'] and round(v['run3_eta_geo'],3)} cycles {v['run3_cycles']:.3g}")
if __name__ == '__main__':
    main()
