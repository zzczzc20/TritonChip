#!/usr/bin/env python3
"""Compare completed, warmed standard-torch RTL operator runs (never board evidence)."""
import argparse
import hashlib
import json
from pathlib import Path

OPERATORS=('add','mul','silu','gelu','copy')


def read_run(path, hybrid):
    text=path.read_text()
    if 'HYBRID TORCH COUPLED RTL ALL 5 OPERATORS PASS' not in text:
        raise ValueError(f'incomplete run: {path}')
    rows={}
    for line in text.splitlines():
        if not line.startswith('{'):
            continue
        row=json.loads(line)
        name=row.get('operator')
        if name not in OPERATORS or name in rows:
            raise ValueError('unknown or duplicate operator')
        if row.get('hybrid') is not hybrid or row.get('numerics')!='pass' or not row.get('warmed'):
            raise ValueError('comparison requires matched, numerically checked warm runs')
        cycles=row.get('rtl_device_cycles')
        if row.get('launches')!=1 or not isinstance(cycles,list) or len(cycles)!=1 or type(cycles[0]) is not int or cycles[0]<=0:
            raise ValueError('comparison requires exactly one positive measured device-cycle count')
        if row.get('upload_bytes')!=[0] or row.get('download_bytes')!=[0]:
            raise ValueError('resident operator comparison requires zero tensor staging')
        if row.get('board_speedup_measured') is not False:
            raise ValueError('RTL fixture cannot supply board evidence')
        rows[name]=row
    if set(rows)!=set(OPERATORS):
        raise ValueError('operator coverage is incomplete')
    return rows


# Settings other than the hybrid memory path that change device cycles. A
# comparison that differs in any of them is not a matched control
# (doc/engineering_discipline.md 2.3.3; evidence/r41/hbm_bw/AUDIT.md F13).
CONTROL_KEYS=('max_block','sfu_lanes','cache_sets','home_background_reads')


def control(row,key):
    value=row.get(key)
    # Runs recorded before e2d5db8 omit sfu_lanes: tools/r41/hybrid_torch_rtl.py
    # then hard-coded four SFU slices for hybrid runs and one for legacy runs.
    if key=='sfu_lanes' and value is None:
        return 4 if row.get('hybrid') else 1
    return value


def compare(baseline,candidate,allow_unmatched=False):
    before,after=read_run(baseline,False),read_run(candidate,True)
    rows=[]
    deviations=set()
    for name in OPERATORS:
        a,b=before[name],after[name]
        if (a['shape'],a['dtype'])!=(b['shape'],b['dtype']):
            raise ValueError('shape/dtype mismatch')
        deviations.update(k for k in CONTROL_KEYS if control(a,k)!=control(b,k))
        old,new=a['rtl_device_cycles'][0],b['rtl_device_cycles'][0]
        rows.append(dict(operator=name,shape=a['shape'],dtype=a['dtype'],
            baseline_rtl_cycles=old,candidate_rtl_cycles=new,rtl_cycle_speedup=old/new,numerics='pass',
            baseline_max_block=a.get('max_block'),candidate_max_block=b.get('max_block'),
            baseline_grids=a.get('grids'),candidate_grids=b.get('grids')))
    if deviations and not allow_unmatched:
        raise ValueError('unmatched control settings: '+', '.join(sorted(deviations)))
    return dict(evidence='synthetic-memory-coupled-RTL-comparison',board_speedup_measured=False,
        acceptance_gate_satisfied=False,matched_control=not deviations,control_deviations=sorted(deviations),
        scope='five operators, small-cache RTL; no board qualification',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (baseline,candidate)},rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-unmatched',action='store_true',help='record a comparison whose control settings differ; it is labelled unmatched')
    a=p.parse_args()
    result=compare(a.baseline,a.candidate,a.allow_unmatched)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['rows']:
        print(('' if result['matched_control'] else 'UNMATCHED ')+f"{r['operator']}: {r['baseline_rtl_cycles']} -> {r['candidate_rtl_cycles']} RTL cycles, {r['rtl_cycle_speedup']:.3f}x")
