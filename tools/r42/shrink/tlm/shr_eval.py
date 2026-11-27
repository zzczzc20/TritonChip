#!/usr/bin/env python3
"""shr_eval.py DSEWK TAG [TAG...]  (R42-shrink; domain: SystemC/TLM model applied as ratios to board cycles)

145-point predictions of model configurations from the collected model runs, with the evaluation code of the R42-dse worker (tools/r42/dse_report.py `full`, read from
DSEWK = a git archive of origin/R42-dse plus the collected files of this branch copied into DSEWK/evidence/r42/dse/col). A TAG is the configuration name (head_dse_<TAG>_<set>.json
must exist for the sets tsm hv px xd xp). The levers file DSEWK/evidence/r42/dse/cfg/dse_levers.json must know TAG (clock and cores): levers_shrink.json of mkcfg.py is merged in.
Prints for each TAG: clock, S145 low/central/high (speed-up of the 145-point wall-time geometric mean against the R41 three-core run-3 image), and the same with the round-2 software.
"""
import json
import sys
from pathlib import Path

dsewk = Path(sys.argv[1])
sys.path.insert(0, str(dsewk / 'tools/r42'))
import dse_common as C   # noqa: E402
import dse_eval as E     # noqa: E402
import dse_report as R   # noqa: E402

for mod in (C, E, R):
    if hasattr(mod, 'ROOT'):
        mod.ROOT = dsewk
R.COL = dsewk / 'evidence/r42/dse/col'
E.COL = R.COL
lev = json.loads((dsewk / 'evidence/r42/dse/cfg/dse_levers.json').read_text())
extra = dsewk / 'evidence/r42/dse/cfg/levers_shrink.json'
if extra.exists():
    lev.update(json.loads(extra.read_text()))
R.LEV = lev
B = C.board()
usable = R.usable_families()
print('| configuration | clock | S145 low | S145 central | S145 high | central, software at round 2 | missing sets | central, corrected for the missing sets |')
print('|---|--:|--:|--:|--:|--:|---|--:|')
import shutil


def s145(name):
    cores = lev.get(name, {}).get('cores', 3)
    rows = E.evaluate(f'head_dse_{name}', 'val_dse_base', 'head_dse_base', B, cores, R.mhz(name))
    for r in rows.values():
        r['clock'] = R.mhz(name)
    return rows


def corrected(name, present):
    """central S145 of `name` times (S145 of the full base / S145 of the base run with only the same sets): the points of the missing sets inherit family ratios,
    which for a tsm-only run carry the head-software effect of the light points; the reference run inherits the same way, so the ratio cancels it."""
    if len(present) == len(E.SETS):
        return None
    ref = 'ref_' + '_'.join(present)
    for st in present:
        shutil.copy(R.COL / f'head_dse_base_{st}.json', R.COL / f'head_dse_{ref}_{st}.json')
    lev[ref] = dict(keys={}, cores=3, desc='reference')
    rows = s145(name)
    rr = s145(ref)
    full = s145('base')
    return R.scenario(rows, 'central', usable) * R.scenario(full, 'central', usable) / R.scenario(rr, 'central', usable)

for tag in sys.argv[2:]:
    name = tag
    miss = [s for s in E.SETS if not (R.COL / f'head_dse_{name}_{s}.json').exists()]
    cores = lev.get(name, {}).get('cores', 3)
    rows = E.evaluate(f'head_dse_{name}', 'val_dse_base', 'head_dse_base', B, cores, R.mhz(name))
    for r in rows.values():
        r['clock'] = R.mhz(name)
    s = {k: R.scenario(rows, k, usable) for k in ('low', 'central', 'high')}
    s_hw = R.scenario(rows, 'central', usable, with_sw=False)
    present = [x for x in E.SETS if x not in miss]
    corr = corrected(name, present) if present and name != 'base' else None
    print(f'| {name} | {R.mhz(name):.0f} | {s["low"]:.3f} | {s["central"]:.3f} | {s["high"]:.3f} | {s_hw:.3f} | {",".join(miss) or "-"} | {("%.3f" % corr) if corr else "-"} |')
