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


def compare(baseline,candidate):
    before,after=read_run(baseline,False),read_run(candidate,True)
    rows=[]
    for name in OPERATORS:
        a,b=before[name],after[name]
        if (a['shape'],a['dtype'])!=(b['shape'],b['dtype']):
            raise ValueError('shape/dtype mismatch')
        old,new=a['rtl_device_cycles'][0],b['rtl_device_cycles'][0]
        rows.append(dict(operator=name,shape=a['shape'],dtype=a['dtype'],
            baseline_rtl_cycles=old,candidate_rtl_cycles=new,rtl_cycle_speedup=old/new,numerics='pass'))
    return dict(evidence='synthetic-memory-coupled-RTL-comparison',board_speedup_measured=False,
        acceptance_gate_satisfied=False,scope='five float32 operators only; small cache; no image or board qualification',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (baseline,candidate)},rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=compare(a.baseline,a.candidate)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['rows']:
        print(f"{r['operator']}: {r['baseline_rtl_cycles']} -> {r['candidate_rtl_cycles']} RTL cycles, {r['rtl_cycle_speedup']:.3f}x")
