#!/usr/bin/env python3
"""Audit the physical three-core capability gate, independently of scaling.

Requires the complete original native inventory and the original large-graph
checkpoint on the same qualified image/software/clock. Does not certify the
separate dispatch edge-case matrix, scaling or complete R40 acceptance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'),str(ROOT/'tools/r40')]
import ablate
import contract_v2
import evidence
import native_inventory
from archbench.provenance import software_tree,sha256_file,check_bundle


def checkpoint_checks(report, *, tree, image, clock):
    def require(ok, name):
        if not ok:raise ValueError(name)
        checks.append(dict(name=name,ok=True))
    checks=[]
    identity=report.get('identity',{})
    require(report.get('status')=='checkpoint-pass','original checkpoint completed successfully')
    require(identity.get('software_tree')==tree and identity.get('image_sha256')==image,
            'checkpoint matches final software and full image SHA')
    require(identity.get('contract_sha256')==contract_v2.LOCK_SHA256,
            'checkpoint binds the frozen contract')
    require(identity.get('clock_hz')==clock['clock_hz'] and
            identity.get('operating_clock_evidence_sha256')==clock['operating_clock_evidence_sha256'],
            'checkpoint uses the reviewed actual operating clock')
    require(identity.get('harness_sha256')==sha256_file(ROOT/'tools/r40/board_checkpoint.py'),
            'original checkpoint assertions match the reviewed harness')
    point=report.get('point',{})
    expected=next(p for p in json.loads((ROOT/'spec/r40/suites/R40-04.streaming.json').read_text())['points']
                  if p['id']=='op.add/hbm/fp16-n67108864')
    require(point.get('id')==expected['id'] and point.get('sha256')==expected['sha256']
            and report.get('input_bytes')==268435456,'original unscaled graph with 256 MiB inputs')
    for name in ('warm_correctness','correctness'):
        numeric=report.get(name,{})
        error=numeric.get('max_rel_err')
        require(numeric.get('ok') is True and type(error) in (int,float) and math.isfinite(error)
                and 0<=error<=point['definition']['tolerance'],name+' numerical tolerance')
    window=report.get('window',{})
    banks=window.get('allocation',{}).get('physical_banks',[])
    require(window.get('size',0)>=1<<30 and len({b['pc'] for b in banks})>=2
            and len(set(report.get('tensor_physical_pcs',[])))>=2,
            'actual >=1 GiB BO and live tensor traffic spans physical PCs')
    core=report.get('cores',{});rows=core.get('cores',[])
    require(core.get('peak_active')==3 and len(rows)==3 and {r['core'] for r in rows}=={0,1,2}
            and all(r['issued']==r['completed']>0 and r['occupied_cycles']>0 for r in rows),
            'three distinct physical cores execute concurrently and drain')
    launches=report.get('launches',[])
    require(bool(launches) and all(l['backend']=='fpga-u280-multicore'
            and l['upload_bytes']==l['download_bytes']==0 and l.get('device_timing') is not None for l in launches),
            'standard resident graph executes on physical multicore backend without staging')
    grid=launches[-1]['grid']
    require(sum(r['issued'] for r in rows)==math.prod(grid),'all grid programs accounted for by physical cores')
    ports=report.get('outlets',[])
    require(len(ports)==3 and {p['port'] for p in ports}=={0,1,2}
            and all(p['read_lines']>0 and p['write_lines']>0 for p in ports),
            'all three outlets carry actual reads and writes before validation')
    events=report.get('events',[])
    histories=[[s for s,_ in e['history']] for e in events]
    require(bool(events) and all(e['state']=='SUCCEEDED' and 'DRAINING' in h and 'SUCCEEDED' in h
            and h.index('DRAINING')<h.index('SUCCEEDED') for e,h in zip(events,histories)),
            'successful completion follows drained device visibility')
    require(any(r['op']==4 and r['status']==0 for r in report.get('completion_history',[])),
            'device completion includes successful quiesce')
    handoffs=[r for r in report.get('host_visibility_history',[]) if r['op']==18]
    require(bool(handoffs) and all(r['status']==0 for r in handoffs),
            'host numerical read follows successful ownership handoff')
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--xclbin',type=Path,required=True)
    p.add_argument('--image-sha256',required=True)
    p.add_argument('--software-tree-sha256',required=True)
    p.add_argument('--operating-clock-evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence',action='store_true')
    a=p.parse_args();checks=[];inputs={};report=dict(status='fail',gate='R40-05.multicore',
            scaling='not-measured',dispatch_edge_cases='not-measured',complete_r40='not-passed')
    try:
        contract_v2.verify()
        assert software_tree(refresh=True)==a.software_tree_sha256,'auditor software mismatch'
        assert sha256_file(a.xclbin)==a.image_sha256,'physical image mismatch'
        from tnca.operating_clock import qualified_clock
        clock=qualified_clock(a.operating_clock_evidence,a.image_sha256)
        checkpoint=json.loads(a.checkpoint.read_text())
        checks+=checkpoint_checks(checkpoint,tree=a.software_tree_sha256,image=a.image_sha256,clock=clock)
        paths=[a.native/f'u280_{tier}.jsonl' for tier in ('tiny','small','medium')]
        checks+=native_inventory.audit(paths)
        assert all(c['ok'] for c in checks),'full native inventory failed'
        for path in paths:
            for row in ablate.load(path):
                plat=row.get('platform',{});prov=row.get('provenance',{})
                assert plat.get('software_tree')==a.software_tree_sha256 and plat.get('image_sha256')==a.image_sha256
                assert plat.get('backend_args')==checkpoint['identity']['backend_args']
                assert plat.get('operating_clock_evidence_sha256')==clock['operating_clock_evidence_sha256']
                assert prov.get('clock_hz')==clock['clock_hz'] and prov.get('launch_map',{}).get('cores')==[0,1,2]
                assert plat.get('knobs',{}).get('TNCA_ACTIVE_CORES')=='3'
                bundle=prov.get('bundle') or {}
                files=bundle.get('files',{})
                assert 'graph.txt' in files and any(f.startswith('wrapper_') and f.endswith('.py') for f in files)
                assert any('ttir' in f.lower() for f in files),'missing original Triton IR'
                assert all(files.get(f'artifacts/{h}.tnca')==h for h in prov['artifacts']),'missing compiled artifact'
                bad=check_bundle(bundle,root=path.parent)
                assert not bad,f'{row["point_id"]}: compiler bundle missing/changed: {bad[:1]}'
        checks.append(dict(name='complete frozen native77 on exactly the qualified three-core image/software/clock with intact compiler bundles',ok=True))
        assert qualified_clock(a.operating_clock_evidence,a.image_sha256)==clock
        assert software_tree(refresh=True)==a.software_tree_sha256
        inputs={str(f):sha256_file(f) for f in [a.checkpoint,a.operating_clock_evidence,*paths]}
        report.update(status='capability-pass',identity=dict(software_tree=a.software_tree_sha256,
            image_sha256=a.image_sha256,contract_sha256=contract_v2.LOCK_SHA256,**clock))
    except Exception:
        report['error']=traceback.format_exc();checks.append(dict(name='complete final-image multicore capability evidence',ok=False))
    report.update(checks=checks,inputs=inputs,harness_sha256=sha256_file(Path(__file__)))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    if a.evidence:
        os.environ['R40_CONTRACT']='v2'
        evidence.record('R40-05.multicore',report['status'],checks,inputs=inputs,
            identity=report.get('identity'),harness_sha256=report['harness_sha256'],
            detail='Complete native77 plus original large graph: numerical correctness, actual three-SLR placement/qualified clock, three concurrent physical cores, all outlets and drained/host visibility. Scaling and dispatch edge matrices are separate, unpromoted gates.')
    print('R40-05.multicore:',report['status'])
    return 0 if report['status']=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
