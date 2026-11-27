#!/usr/bin/env python3
"""mkcfg.py DSEWK OUTDIR NAME=lever+lever ...  (R42-shrink; domain: SystemC/TLM model what-if configurations)

Writes TLM configuration files OUTDIR/dse_NAME.cfg for the combinations given, from the lever table of the R42-dse worker
(tools/r42/dse_cfg.py, read from DSEWK = a git-archive of origin/R42-dse; nothing of that branch is modified) plus the levers of this branch:
  c5        5 cores on a 3x2 mesh (6 home slices, 5 outlets; run with CORES=5)
  hit<k>    NUCA slice hit latency + k cycles (tl_slice_hit_latency; for a hit path that gets k more registers)
  rt<k>     HBM/outlet return path + k cycles (tl_outlet_to_slice_cycles)
  lsu<k>    hybrid-LSU load completion + k cycles (lsu_complete_cycles, load_latency)
  mx<k>     matrix command latency + k cycles (mtx_complete_cycles)
  vrf<k>    vector RAW forwarding + k cycles (vector_int_latency, vector_fp_latency)
  iss<k>    taken branch / context-switch refetch + k cycles (branch_taken_cycles, park_refetch_cycles)
Also writes OUTDIR/levers_shrink.json (clock and cores per name, for the evaluation) and prints the file names.
"""
import json
import sys
from pathlib import Path

dsewk, out = Path(sys.argv[1]), Path(sys.argv[2])
cfgdir = dsewk / 'evidence/r42/dse/cfg'
b = dict(l.split('=', 1) for l in (cfgdir / 'dse_base.cfg').read_text().split())
L = json.loads((cfgdir / 'dse_levers.json').read_text())          # the lever table of the dse worker: name -> {keys, cores, desc}


def combine(levers, names):
    keys, cores, desc = {}, 3, []
    for n in names:
        keys.update(levers[n]['keys'])
        cores = max(cores, levers[n]['cores'])
        desc.append(levers[n]['desc'])
    return dict(keys=keys, cores=cores, desc=' | '.join(desc))


def add(name, kv, cores=3, desc=''):
    L[name] = dict(keys={k: str(v) for k, v in kv.items()}, cores=cores, desc=desc)


add('c5', dict(nx=3, ny=2, outlets=5, active_outlets=5), 5, '5 cores on a 3x2 mesh (6 slices), 5 outlets (run with CORES=5)')
for k in (1, 2, 3):
    add(f'hit{k}', dict(tl_slice_hit_latency=int(b['tl_slice_hit_latency']) + k), 3, f'slice hit latency +{k}')
    add(f'rt{k}', dict(tl_outlet_to_slice_cycles=int(b['tl_outlet_to_slice_cycles']) + k), 3, f'outlet return +{k}')
    add(f'lsu{k}', dict(lsu_complete_cycles=int(b['lsu_complete_cycles']) + k, load_latency=int(b['load_latency']) + k), 3, f'LSU load completion +{k}')
    add(f'mx{k}', dict(mtx_complete_cycles=int(b['mtx_complete_cycles']) + k), 3, f'matrix completion +{k}')
    add(f'vrf{k}', dict(vector_int_latency=int(b['vector_int_latency']) + k, vector_fp_latency=int(b['vector_fp_latency']) + k), 3, f'vector forwarding +{k}')
    add(f'iss{k}', dict(branch_taken_cycles=int(b['branch_taken_cycles']) + k, park_refetch_cycles=int(b['park_refetch_cycles']) + k), 3, f'branch/refetch +{k}')

summary = {}
out.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[3:]:
    name, lv = spec.split('=')
    c = combine(L, lv.split('+')) if lv != 'base' else dict(keys={}, cores=3, desc='base')
    cfg = dict(b)
    cfg.update(c['keys'])
    (out / f'dse_{name}.cfg').write_text(''.join(f'{k}={v}\n' for k, v in cfg.items()))
    summary[name] = dict(keys=c['keys'], cores=c['cores'], desc=c['desc'])
    print(f'dse_{name}.cfg cores={c["cores"]} clock={cfg.get("clock_hz")}')
(out / 'levers_shrink.json').write_text(json.dumps(summary, indent=1))
