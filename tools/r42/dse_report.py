#!/usr/bin/env python3
"""Tables of the R42 dse report from the collected model runs (domain: SystemC/TLM model, applied as ratios to board cycles).

    python3 tools/r42/dse_report.py screen   > tables_screen.md     one-at-a-time effects from the tiny/small/medium runs (heavy and xl points inherit their family)
    python3 tools/r42/dse_report.py full CFG [CFG ...]            full 145-point bundles (all sets present), with the sensitivity range

hw effect of a configuration on a point  = m(head software, cfg) / m(head software, base model)   (cycles; < 1 is faster)
wall-time effect                          = hw effect x f_base / f_cfg
145-point effect                          = geometric mean of 1 / (wall-time effect)             (> 1 is faster)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dse_common as C   # noqa: E402
import dse_eval as E     # noqa: E402

COL = C.ROOT / 'evidence/r42/dse/col'
LEV = json.loads((C.ROOT / 'evidence/r42/dse/cfg/dse_levers.json').read_text()) if (C.ROOT / 'evidence/r42/dse/cfg/dse_levers.json').exists() else {}


def clock_of(name):
    return LEV.get(name, {}).get('keys', {}).get('clock_hz')


def mhz(name):
    c = clock_of(name)
    return int(c) / 1e6 if c else 100.0


def tsm(tag):
    f = COL / f'{tag}_tsm.json'
    if not f.exists():
        return None
    return {k: v['device_cycles'] for k, v in C.load_collected(f).items() if v.get('status') == 'pass' and v.get('device_cycles')}


def usable_families():
    p = C.ROOT / 'evidence/r42/dse/validation_families.json'
    return json.loads(p.read_text()) if p.exists() else {}


def screen(levers=None):
    B = C.board()
    base = tsm('head_dse_base')
    rows = {}
    names = levers or sorted(n[len('head_dse_'):-len('_tsm.json')] for n in (p.name for p in COL.glob('head_dse_*_tsm.json')))
    for n in names:
        if n == 'base':
            continue
        m = tsm(f'head_dse_{n}')
        if not m or not base:
            continue
        f = mhz(n)
        per = {p: m[p] / base[p] for p in m if p in base}
        wall = {p: r * 100.0 / f for p, r in per.items()}
        fam = {}
        for p, r in wall.items():
            fam.setdefault(B[p]['family'], []).append(r)
        tier = {}
        for p, r in wall.items():
            tier.setdefault(B[p]['tier'], []).append(r)
        # 145-point effect with family inheritance for the points that were not run
        famg = {k: C.geo(v) for k, v in fam.items()}
        allw = []
        for p, b in B.items():
            if p in wall:
                allw.append(wall[p])
            else:
                same = [r for q, r in wall.items() if B[q]['family'] == b['family']]
                allw.append(C.geo(same) if same else 1.0)
        rows[n] = dict(n=len(per), clock=f, cycles_geo=C.geo(list(per.values())), wall_geo=C.geo(list(wall.values())),
                       eff145=1 / C.geo(allw), by_family={k: 1 / v for k, v in famg.items()},
                       by_tier={k: 1 / C.geo(v) for k, v in tier.items()})
    return rows


def main():
    if sys.argv[1] == 'screen':
        rows = screen()
        print('| lever | clock MHz | cycles effect (tsm77, >1 faster) | wall effect (tsm77) | 145-point wall effect (inherit) |')
        print('|---|--:|--:|--:|--:|')
        for n, r in sorted(rows.items(), key=lambda kv: -kv[1]['eff145']):
            print(f'| {n} | {r["clock"]:.0f} | {1 / r["cycles_geo"]:.3f} | {1 / r["wall_geo"]:.3f} | {r["eff145"]:.3f} |')
        Path(C.ROOT / 'evidence/r42/dse/screen.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
