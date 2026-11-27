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


GROUP = {'tiny': 'light', 'small': 'light', 'medium': 'light', 'large': 'heavy', 'hbm': 'heavy', 'xl': 'heavy'}


def screen(levers=None):
    """one-at-a-time effect of each configuration against the head-software base model, from every set that was run for it"""
    B = C.board()
    bb = E.Bundle('head_dse_base')
    names = levers or sorted({p.name[len('head_dse_'):].rsplit('_', 1)[0] for p in COL.glob('head_dse_*_tsm.json')})
    rows = {}
    for n in names:
        if n == 'base':
            continue
        bc = E.Bundle(f'head_dse_{n}', LEV.get(n, {}).get('cores', 3))
        f = mhz(n)
        direct = {}
        for pid in B:
            c1, k1 = E.model_cycles(bc, pid, B)
            c0, k0 = E.model_cycles(bb, pid, B)
            if c1 and c0:
                direct[pid] = (c1 / c0, k1)
        by_fam_grp, by_fam = {}, {}
        for pid, (r, k) in direct.items():
            b = B[pid]
            by_fam_grp.setdefault((b['family'], GROUP[b['tier']] if not b['xl'] else 'heavy'), []).append(r)
            by_fam.setdefault(b['family'], []).append(r)
        allr, kinds = {}, {}
        for pid, b in B.items():
            if pid in direct:
                allr[pid] = direct[pid][0]; kinds[pid] = direct[pid][1]
            else:
                g = GROUP[b['tier']] if not b['xl'] else 'heavy'
                x = by_fam_grp.get((b['family'], g)) or by_fam.get(b['family'])
                allr[pid] = C.geo(x) if x else 1.0
                kinds[pid] = 'family' if x else 'held'
        wall = {p: r * 100.0 / f for p, r in allr.items()}
        fam, tier = {}, {}
        for p, r in wall.items():
            fam.setdefault(B[p]['family'], []).append(r)
            tier.setdefault(B[p]['tier'], []).append(r)
        light = [wall[p] for p in wall if B[p]['tier'] in ('tiny', 'small', 'medium') and kinds[p] != 'family']
        rows[n] = dict(n_direct=len(direct), kinds={k: sum(1 for v in kinds.values() if v == k) for k in set(kinds.values())}, clock=f,
                       cycles_geo=C.geo(list(allr.values())), wall_geo=C.geo(list(wall.values())),
                       eff145=1 / C.geo(list(wall.values())), eff_light_run=1 / C.geo(light) if light else None,
                       by_family={k: 1 / C.geo(v) for k, v in fam.items()}, by_tier={k: 1 / C.geo(v) for k, v in tier.items()})
    return rows


E_GAIN = dict(low=0.5, central=1.0, high=1.2)    # exponent on the log of a cycle gain (ratio < 1); bracket of the two lever validations that exist
E_LOSS = dict(low=1.2, central=1.0, high=0.5)    # exponent on a cycle loss (ratio > 1): the opposite side


def eff(r, e_gain, e_loss):
    return r ** (e_gain if r < 1 else e_loss)


def scenario(rows, kind, usable, with_sw=True):
    """145-point wall-time speed-up against the R41 baseline for one sensitivity scenario (kind low|central|high)"""
    out = []
    for r in rows.values():
        key = r['family'] + ('/fp32' if r['point'].split('/')[2].startswith('fp32') else '')
        fam_ok = usable.get(key, usable.get(r['family'], True))
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
        print('| lever | clock MHz | cycles effect (145, >1 faster) | wall effect (145) | wall effect, tiny/small/medium run points only | points run / inherited |')
        print('|---|--:|--:|--:|--:|--:|')
        for n, r in sorted(rows.items(), key=lambda kv: -kv[1]['eff145']):
            k = r['kinds']
            print(f'| {n} | {r["clock"]:.0f} | {1 / r["cycles_geo"]:.3f} | {r["eff145"]:.3f} | {r["eff_light_run"]:.3f} | {r["n_direct"]} / {k.get("family", 0) + k.get("held", 0)} |')
        Path(C.ROOT / 'evidence/r42/dse/screen.json').write_text(json.dumps(rows, indent=1))
    elif sys.argv[1] == 'full':
        full(sys.argv[2:])


if __name__ == '__main__':
    main()
