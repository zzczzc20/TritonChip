#!/usr/bin/env python3
"""Read-only physical HBM map audit bound to an actual native ArchBench BO.

This proves mapping, not functional PC coverage or the complete R40-04 gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tnca.backends.u50 import _xclbin_bank,_xclbin_hbm_allocation,IP_NAME
from tnca.errors import DependencyFailed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--xclbin',type=Path,required=True)
    ap.add_argument('--ip-name',default=IP_NAME)
    ap.add_argument('--json',type=Path,required=True)
    args=ap.parse_args()
    if args.json.exists():ap.error('preserve prior mapping evidence; --json must be new')
    record=None
    with args.record.open('rb') as file:
        for raw in file:
            candidate=json.loads(raw)
            if candidate.get('status')=='pass' and candidate.get('native_memory',{}).get('window'):
                record=candidate;break
    if record is None:ap.error('no passing native record with an actual BO window')
    digest=hashlib.sha256()
    with args.xclbin.open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):digest.update(block)
    image_sha=digest.hexdigest()
    if record.get('platform',{}).get('image') != 'xclbin:'+image_sha[:16]:
        ap.error('record image does not match the supplied xclbin')
    window=record['native_memory']['window'];base,size=window['base'],window['size']
    try:
        bank=_xclbin_bank(str(args.xclbin),args.ip_name)
        if not bank['base']<=base<base+size<=bank['base']+bank['size']:
            raise ValueError('BO falls outside its connected image memory group')
        physical=_xclbin_hbm_allocation(str(args.xclbin),base,size)
    except (DependencyFailed,ValueError,OSError) as error:ap.error(str(error))
    report=dict(schema='tnca-r40-physical-hbm-map/1',gate_status='not-measured',
                scope='mapping only; functional PC/port/range and full-suite/model headroom checks are separate',
                source_record=dict(path=str(args.record),line_sha256=hashlib.sha256(raw).hexdigest(),
                    point_id=record['point_id'],software_tree=record['platform'].get('software_tree')),
                image_sha256=image_sha,window=window,connected_memory=bank,**physical,
                capacity_at_least_1GiB=size>=1<<30,
                at_least_two_allocated_physical_pcs=len(physical['physical_banks'])>=2)
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('gate_status','capacity_at_least_1GiB','at_least_two_allocated_physical_pcs')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
