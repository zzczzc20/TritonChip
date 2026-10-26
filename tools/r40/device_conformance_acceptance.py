#!/usr/bin/env python3
"""Audit complete, source-bound CUDA and no-CUDA host conformance together."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'tools/r40')]
import contract_v2
import evidence


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def complete_cases(log, host_class):
    tree=ast.parse((ROOT/'tools/r40/device_tests.py').read_text())
    names={node.name for node in tree.body if isinstance(node,ast.FunctionDef)
           and any(isinstance(d,ast.Name) and d.id=='case' for d in node.decorator_list)}
    rows=re.findall(r'^\s*PASS\s+(\w+)\s+(.*)$',log,re.M)
    if len(rows)!=len(names) or {n for n,_ in rows}!=names:
        raise ValueError('complete original case set required, without duplicates')
    summary=f'r40-device-tests  PASS  ({len(names)}/{len(names)})'
    if log.count(summary)!=1 or re.search(r'^\s*FAIL\s+',log,re.M):
        raise ValueError('one successful terminal summary and no failed cases required')
    mixed=dict(rows)['mixed_devices']
    expected=('2 rounds of CUDA and tnca graphs in one process, then a CPU-tensor graph'
              if host_class=='cuda-host' else 'skipped: no CUDA device on this host')
    if host_class not in ('cuda-host','no-cuda-host') or mixed!=expected:
        raise ValueError('actual CUDA coexistence/absence evidence does not match host class')
    return [dict(name=host_class+':'+name,ok=True,detail=detail) for name,detail in rows]


def audit(doc, host_class, *, software_tree, image_sha256):
    if doc['host_class']!=host_class or doc['contract_sha256']!=contract_v2.LOCK_SHA256:
        raise ValueError('host class or frozen contract mismatch')
    if doc['software_tree']!=software_tree or doc['live_source_tree']!=software_tree:
        raise ValueError('conformance source changed or belongs to another candidate')
    if doc['test_harness_sha256']!=sha((ROOT/'tools/r40/device_tests.py').read_bytes()):
        raise ValueError('original test assertions changed')
    if sha(doc['controller'].encode())!=doc['controller_sha256'] or doc['controller_sha256']!=doc['launch_controller_sha256']:
        raise ValueError('controller differs from its recorded launch')
    if sha(doc['log'].encode())!=doc['log_sha256']:
        raise ValueError('raw conformance log changed')
    state=doc['controller_state']
    if state['status']!='complete-review-required' or state['software_tree']!=software_tree:
        raise ValueError('terminal same-source controller required')
    stages=state['completed']
    if not stages or any(r['exit_code']!=0 for r in stages) or sum(r['name']==doc['stage_name'] for r in stages)!=1:
        raise ValueError('complete successful conformance stage required')
    if host_class=='cuda-host':
        if doc['backend']!='fpga-multicore-sim':raise ValueError('CUDA host must use the reviewed multicore functional backend')
    elif doc['backend']!='fpga-u280-multicore' or state['image_sha256']!=image_sha256:
        raise ValueError('no-CUDA host must use the actual final physical image')
    return complete_cases(doc['log'],host_class)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda',type=Path,required=True);p.add_argument('--no-cuda',type=Path,required=True)
    p.add_argument('--software-tree',required=True);p.add_argument('--image-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--evidence',action='store_true')
    a=p.parse_args()
    if a.out.exists():p.error('output must be fresh; preserve previous failures')
    contract_v2.verify();checks=[];inputs={};status='fail';error=None
    try:
        from archbench.provenance import software_tree
        if software_tree(refresh=True)!=a.software_tree:raise ValueError('auditor must run on the final product source')
        for path,cls in [(a.cuda,'cuda-host'),(a.no_cuda,'no-cuda-host')]:
            raw=path.read_bytes();inputs[str(path)]=sha(raw)
            checks+=audit(json.loads(raw),cls,software_tree=a.software_tree,image_sha256=a.image_sha256)
        status='capability-pass'
    except (OSError,ValueError,KeyError,TypeError) as e:
        error=str(e);checks.append(dict(name='complete matching two-host conformance',ok=False,detail=error))
    report=dict(status=status,checks=checks,error=error,inputs=inputs,software_tree=a.software_tree,
                image_sha256=a.image_sha256,contract_sha256=contract_v2.LOCK_SHA256,
                scope='Full device coexistence capability; no board performance or unrelated gate claim')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    if a.evidence:
        import os
        os.environ['R40_CONTRACT']='v2'
        evidence.record('R40-01.mixed-device',status,checks,inputs=inputs,software_tree=a.software_tree,
                        image_sha256=a.image_sha256,detail=report['scope'])
    print('R40-01.mixed-device:',status)
    return 0 if status=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
