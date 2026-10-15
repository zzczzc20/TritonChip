#!/usr/bin/env python3
"""Full frozen native-device inventory; separate from staged timing controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ablate
import contract_v2
import evidence


def audit(paths):
    contract_v2.verify()
    subset,_=ablate.load_subset(contract_v2.ROOT/'spec/r40/suites/existing.json')
    records=[r for path in paths for r in ablate.load(path)]
    ids=[r['point_id'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(subset):
        raise ValueError('complete 77-point native inventory required, without duplicates or extras')
    checks=[]
    configs=set()
    for r in records:
        pid=r['point_id'];p=r.get('platform',{});k=p.get('knobs',{})
        host=r.get('host',{});prov=r.get('provenance',{})
        missing=[key for key in ablate.PROVENANCE if not prov.get(key)]
        good=(r.get('status')=='pass' and r.get('point_sha256')==subset[pid]
              and not r.get('fallbacks') and not missing
              and p.get('backend')=='fpga-u280'
              and k.get('TNCA_TORCH_DEVICE')=='native'
              and k.get('TNCA_TRANSFER_POLICY','resident')=='resident'
              and host.get('upload_bytes')==0 and host.get('download_bytes')==0
              and r.get('time',{}).get('launches',0)>0
              and bool(r.get('native_memory')))
        checks.append(dict(name=pid,ok=good))
        configs.add(ablate.digest(ablate.config_of(r,[])))
    checks.append(dict(name='one source/image/profile/clock/configuration across all points',ok=len(configs)==1))
    return checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',type=Path,required=True)
    ap.add_argument('--evidence',action='store_true')
    args=ap.parse_args()
    paths=[args.native/f'u280_{tier}.jsonl' for tier in ('tiny','small','medium')]
    try:checks=audit(paths)
    except (ValueError,KeyError,OSError,ablate.Refused) as error:
        checks=[dict(name='complete resident native inventory',ok=False,why=str(error))]
    status='capability-pass' if all(c['ok'] for c in checks) else 'fail'
    if args.evidence:
        os.environ['R40_CONTRACT']='v2'
        evidence.record('R40-01.device',status,checks,
            inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
            detail='full frozen resident inventory, independent from per-kernel staged timing controls')
    print('R40-01.device v2:',status)
    return 0 if status=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
