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


E_GAIN = dict(low=0.5, central=1.0, high=1.2)    # exponent on the log of a cycle gain (ratio < 1); bracket of the two lever validations that exist
E_LOSS = dict(low=1.2, central=1.0, high=0.5)    # exponent on a cycle loss (ratio > 1): the opposite side


def eff(r, e_gain, e_loss):
    return r ** (e_gain if r < 1 else e_loss)


def scenario(rows, kind, usable, with_sw=True):
    """145-point wall-time speed-up against the R41 baseline for one sensitivity scenario (kind low|central|high)"""
    out = []
    for r in rows.values():
        fam_ok = usable.get(r['family'], True)
        sw = r['sw_ratio'] if with_sw else 1.0
        hw = r['hw_ratio']
        if kind == 'low':
            hw = hw if fam_ok else 1.0                                     # hardware effects only where the model is validated
            sw = sw if fam_ok else (sw ** 0.5 if sw < 1 else sw)           # half of the log-gain of the software where it is not
        hw = eff(hw, E_GAIN[kind], E_LOSS[kind]) if hw is not None else 1.0
        cyc = r['r2'] * sw * hw
        out.append((r['b41'] / 100.0) / (cyc / r['clock']))
    return C.geo(out)


def full(tags):
    B = C.board()
    usable = usable_families()
    print('| configuration | clock | S145 low | S145 central | S145 high | central, software at round 2 | by tier (central) |')
    print('|---|--:|--:|--:|--:|--:|---|')
    res = {}
    for tag in tags:
        name = tag[len('head_dse_'):]
        cores = LEV.get(name, {}).get('cores', 3)
        rows = E.evaluate(f'head_dse_{name}', 'val_dse_base', 'head_dse_base', B, cores, mhz(name))
        for r in rows.values():
            r['clock'] = mhz(name)
        s = {k: scenario(rows, k, usable) for k in ('low', 'central', 'high')}
        s_hw = scenario(rows, 'central', usable, with_sw=False)
        tiers = {}
        for r in rows.values():
            tiers.setdefault(r['tier'], []).append((r['b41'] / 100.0) / (r['r2'] * r['sw_ratio'] * r['hw_ratio'] / r['clock']))
        res[name] = dict(S145=s, S145_sw_at_round2=s_hw, clock=mhz(name), kinds=E.summarize(rows)['kinds'],
                         by_tier={k: C.geo(v) for k, v in tiers.items()},
                         by_family={f: C.geo([(r['b41'] / 100.0) / (r['r2'] * r['sw_ratio'] * r['hw_ratio'] / r['clock']) for r in rows.values() if r['family'] == f])
                                    for f in sorted({r['family'] for r in rows.values()})},
                         rows={p: dict(kind=r['kind'], ratio=r['ratio'], sw=r['sw_ratio'], hw=r['hw_ratio']) for p, r in rows.items()})
        print(f'| {name} | {mhz(name):.0f} | {s["low"]:.3f} | {s["central"]:.3f} | {s["high"]:.3f} | {s_hw:.3f} | ' +
              ' '.join(f'{k} {v:.2f}' for k, v in sorted(res[name]['by_tier'].items())) + ' |')
    Path(C.ROOT / 'evidence/r42/dse/full.json').write_text(json.dumps(res, indent=1))


def main():
    if sys.argv[1] == 'screen':
        rows = screen()
        print('| lever | clock MHz | cycles effect (tsm77, >1 faster) | wall effect (tsm77) | 145-point wall effect (inherit) |')
        print('|---|--:|--:|--:|--:|')
        for n, r in sorted(rows.items(), key=lambda kv: -kv[1]['eff145']):
            print(f'| {n} | {r["clock"]:.0f} | {1 / r["cycles_geo"]:.3f} | {1 / r["wall_geo"]:.3f} | {r["eff145"]:.3f} |')
        Path(C.ROOT / 'evidence/r42/dse/screen.json').write_text(json.dumps(rows, indent=1))
    elif sys.argv[1] == 'full':
        full(sys.argv[2:])


if __name__ == '__main__':
    main()
