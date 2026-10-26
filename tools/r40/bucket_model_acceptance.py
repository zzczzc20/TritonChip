#!/usr/bin/env python3
"""Audit full frozen native generation in one fixed KV bucket.

Only R40-08.decode is eligible for promotion. This is functional evidence;
model performance, online attention and cross-bucket migration are separate.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE_ARGUMENT=argparse.ArgumentParser(add_help=False)
SOURCE_ARGUMENT.add_argument('--source-root',type=Path,default=ROOT,
                             help='immutable product snapshot; permits external read-only auditing')
ROOT=SOURCE_ARGUMENT.parse_known_args()[0].source_root.resolve()
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'tools/r40')]
import contract_v2
from residency_acceptance import require,sha,model


def bucket_policy(report):
    require(report.get('cache_policy')=='fixed-bucket', 'a growing-shape model does not prove fixed-bucket decode')
    workload=json.loads((ROOT/'spec/r40/v2/plan.json').read_text())['workloads']['llama_generate']
    prefix=workload['prefill'];capacity=prefix+workload['decode_steps']
    require(type(report['decode_compilations']) is int and report['decode_compilations']==1,
            'exactly one actual decode compilation required')
    require(report['bucket_capacity']==capacity and len(report['steps'])==65, 'original full bucket/generation required')
    addresses=report['cache_addresses']
    require(len(addresses)==2 and addresses[0]!=addresses[1] and
            all(type(v) is int and v>0 for v in addresses), 'distinct resident cache addresses required')
    require(report['identity']['bucket_harness_sha256']==sha((ROOT/'tools/r40/bucket_decode.py').read_bytes()),
            'bucket implementation changed')
    for i,step in enumerate(report['steps']):
        require(step['index']==i and step['cache_capacity']==capacity and
                list(step['cache_addresses'])==list(addresses), 'resident cache storage changed')
        require(type(step['tensor_position']) is int and step['tensor_position']==prefix+i,
                'tensor position did not advance correctly')
        if i:
            require(type(step['decode_compilations']) is int and step['decode_compilations']==1,
                    'another decode position recompiled the bucket')
        if i>1:
            require(not step['cold_transfers'], 'unexpected cold initialization after the first decode')
    guard=report['bucket_guard']
    require(guard['capacity']==capacity and guard['next_position']==capacity and
            guard['completed_position_range']==[prefix,capacity-1] and
            guard['position_dtype']=='int64' and guard['position_location']=='device',
            'bucket boundary and guarded position metadata must be reported')
    require(guard['cross_bucket_transition']=='not-measured',
            'this frozen workload does not measure a cross-bucket transition')


def main():
    p=argparse.ArgumentParser(description=__doc__,parents=[SOURCE_ARGUMENT])
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--software-tree',required=True)
    p.add_argument('--image-sha256',required=True)
    p.add_argument('--xclbin',type=Path,required=True)
    p.add_argument('--clock-evidence',type=Path,required=True)
    p.add_argument('--hbm-bytes',type=int,default=4<<30)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence',action='store_true')
    a=p.parse_args()
    if a.out.exists():p.error('preserve previous audits; output must be fresh')
    checks=[];status='fail';contract_v2.verify()
    try:
        raw=a.report.read_bytes()
        report=json.loads(gzip.decompress(raw) if a.report.suffix=='.gz' else raw)
        bucket_policy(report)
        from tnca.operating_clock import qualified_clock
        clock=qualified_clock(a.clock_evidence,a.image_sha256)
        require(sha(a.xclbin.read_bytes())==a.image_sha256,'qualified image changed')
        config=dict(xclbin=str(a.xclbin.resolve()),image_profile='core-vrf-sync-v5',
                    core_count=3,memory_ports=3,hbm_dma=True,hbm_bytes=a.hbm_bytes,device_clock=True)
        model(report,a.software_tree,a.image_sha256,dict(backend_args=config,clock_hz=clock['clock_hz']))
        for key in ('clock_hz','clock_policy','operating_clock_evidence_sha256'):
            require(report['identity'][key]==clock[key],'operating clock evidence changed')
        if a.evidence:
            from archbench.provenance import software_tree
            require(software_tree(refresh=True)==a.software_tree,'auditor must use the final product tree')
        status='capability-pass'
        checks=[dict(name=name,ok=True) for name in (
            'full original prefill512 and 64 decode calls, independent numerics and tokens',
            'fixed resident cache addresses and device position; exactly one decode compilation',
            'zero steady tensor staging, unchanged weights, draining and host visibility',
            'same product, qualified physical image and actual operating clock')]
    except (OSError,ValueError,KeyError,TypeError) as error:
        checks.append(dict(name='complete fixed-bucket native generation',ok=False,detail=str(error)))
    result=dict(status=status,checks=checks,software_tree=a.software_tree,image_sha256=a.image_sha256,
                contract_sha256=contract_v2.LOCK_SHA256,
                auditor_sha256=sha(Path(__file__).read_bytes()),source_root=str(ROOT),
                input_sha256=sha(a.report.read_bytes()) if a.report.exists() else None,
                scope='R40-08.decode only; functional fixed-bucket generation, not performance or cross-bucket migration')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.evidence:
        import evidence
        os.environ['R40_CONTRACT']='v2'
        evidence.record('R40-08.decode',status,checks,software_tree=a.software_tree,
                        image_sha256=a.image_sha256,input_sha256=result['input_sha256'],detail=result['scope'])
    print('R40-08.decode:',status)
    return 0 if status=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
