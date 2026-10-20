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
              and p.get('backend') in ('fpga-u280', 'fpga-u280-multicore')
              and k.get('TNCA_TORCH_DEVICE')=='native'
              and k.get('TNCA_TRANSFER_POLICY','resident')=='resident'
              and host.get('upload_bytes')==0 and host.get('download_bytes')==0
              and r.get('time',{}).get('launches',0)>0
              and bool(r.get('native_memory')))
        checks.append(dict(name=pid,ok=good))
        configs.add(ablate.digest(ablate.config_of(r,[])))
    checks.append(dict(name='one source/image/profile/clock/configuration across all points',ok=len(configs)==1))
    return checks


def bound(paths, tree, image, clock_path):
    """(identity, check): every record measured on exactly this product tree,
    full image SHA and operating clock evidence. The v3 acceptance takes this
    v2 gate only when its record names the delivery identity."""
    raw=Path(clock_path).read_bytes();clock=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if clock.get('image_sha256')!=image or type(clock.get('clock_hz')) is not int:
        raise ValueError('operating clock evidence belongs to another image')
    bad=[]
    for r in (r for path in paths for r in ablate.load(path)):
        p=r.get('platform',{});prov=r.get('provenance',{})
        if not (prov.get('software_tree')==p.get('software_tree')==tree
                and p.get('image_sha256')==image and prov.get('image')=='xclbin:'+image[:16]
                and prov.get('clock_hz')==p.get('clock_hz')==clock['clock_hz']
                and p.get('operating_clock_evidence_sha256')==digest
                and p.get('operating_clock_evidence')==clock):
            bad.append(r.get('point_id'))
    identity=dict(software_tree=tree,image_sha256=image,clock_hz=clock['clock_hz'],
                  operating_clock_evidence_sha256=digest)
    return identity,dict(name='every record on this product tree, full image SHA and operating clock evidence',
                         ok=not bad,not_bound=bad,**identity)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',type=Path,required=True)
    ap.add_argument('--evidence',action='store_true')
    ap.add_argument('--software-tree',help='with --image-sha256 and --operating-clock-evidence: require and record this identity')
    ap.add_argument('--image-sha256')
    ap.add_argument('--operating-clock-evidence',type=Path)
    args=ap.parse_args()
    given=[args.software_tree,args.image_sha256,args.operating_clock_evidence]
    if any(given) and not all(given):
        ap.error('--software-tree, --image-sha256 and --operating-clock-evidence go together')
    paths=[args.native/f'u280_{tier}.jsonl' for tier in ('tiny','small','medium')]
    identity=None
    try:
        checks=audit(paths)
        if all(given):
            identity,check=bound(paths,args.software_tree,args.image_sha256,args.operating_clock_evidence)
            checks.append(check)
    except (ValueError,KeyError,OSError,ablate.Refused) as error:
        checks=[dict(name='complete resident native inventory',ok=False,why=str(error))]
    status='capability-pass' if all(c['ok'] for c in checks) else 'fail'
    if args.evidence:
        os.environ['R40_CONTRACT']='v2'
        extra=dict(identity=identity) if identity else {}
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
        if all(given) and args.operating_clock_evidence.exists():
            inputs[str(args.operating_clock_evidence)]=hashlib.sha256(args.operating_clock_evidence.read_bytes()).hexdigest()
        evidence.record('R40-01.device',status,checks,inputs=inputs,
            detail='full frozen resident inventory, independent from per-kernel staged timing controls',**extra)
    print('R40-01.device v2:',status)
    return 0 if status=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
