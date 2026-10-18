#!/usr/bin/env python3
"""Run original checkpoint/inventories/model serially on one qualified image.

Fresh output, immutable identity, one board job, no reduced points, no automatic
synthesis/load/gate promotion. Stop at the first failed stage and retain output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'tools/r40')]


def frozen(name):
    return json.loads((ROOT/'spec/r40/suites'/name).read_text())['points']


def plan(python,image,image_sha,tree,clock,out):
    config=dict(xclbin=str(image),image_profile='core-vrf-sync-v5',core_count=3,
                memory_ports=3,hbm_dma=True,hbm_bytes=1<<30,device_clock=True)
    common=['--xclbin',str(image),'--image-sha256',image_sha,
            '--software-tree-sha256',tree,'--operating-clock-evidence',str(clock)]
    stages=[dict(name='original-ge64mib-checkpoint',command=[python,str(ROOT/'tools/r40/board_checkpoint.py'),
                *common,'--out',str(out/'checkpoint')])]
    for tier in ('tiny','small','medium'):
        ids=[p['id'] for p in frozen('existing.json') if p['tier']==tier]
        stages.append(dict(name='native77-'+tier,point_ids=ids,
            command=[python,'-u','-m','archbench.run','--platform','tnca',
                     '--profile','core-vrf-sync-v5','--backend','fpga-u280-multicore',
                     '--backend-args',json.dumps(config),'--point',*ids,'--jobs','1',
                     '--warmup','1','--timed','1',
                     '--bundle-dir',str(out/'bundles/native'),
                     '--out',str(out/'native'/f'u280_{tier}.jsonl')]))
    stages.append(dict(name='native77-audit',command=[python,str(ROOT/'tools/r40/native_inventory.py'),
                  '--native',str(out/'native')]))
    ids=[p['id'] for p in frozen('hbm-core.json')]
    stages.append(dict(name='original-hbm14',point_ids=ids,
        command=[python,'-u','-m','archbench.run','--platform','tnca',
                 '--profile','core-vrf-sync-v5','--backend','fpga-u280-multicore',
                 '--backend-args',json.dumps(config),'--point',*ids,'--jobs','1',
                 '--warmup','1','--timed','1',
                 '--bundle-dir',str(out/'bundles/hbm'),
                 '--out',str(out/'hbm14.jsonl')]))
    stages.append(dict(name='frozen-original-model',command=[python,'-u',str(ROOT/'tools/r40/native_model.py'),
                  *common,'--out',str(out/'model')]))
    return stages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xclbin',type=Path,required=True)
    p.add_argument('--image-sha256',required=True)
    p.add_argument('--software-tree-sha256',required=True)
    p.add_argument('--operating-clock-evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    stages=plan(sys.executable,a.xclbin.resolve(),a.image_sha256,a.software_tree_sha256,
                a.operating_clock_evidence.resolve(),out)
    record=dict(schema='tnca-conditional-delivery/1',status='planned',stages=stages,
                performance_gate='not-measured',original_100mhz_gate='not-passed',
                inventory_sampling='functional only: one untimed warmup and one steady call; cold compiler copies retained separately; not the frozen performance sample plan',
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                acceptance_basis='conditional-r40v2-20261018',completed=[])
    def save():
        temporary=out/'delivery.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out/'delivery.json')
    save()
    if a.plan_only:return 0
    from tnca.operating_clock import qualified_clock
    from archbench.provenance import software_tree
    import contract_v2
    contract_v2.verify()
    clock=qualified_clock(a.operating_clock_evidence,a.image_sha256)
    if hashlib.sha256(a.xclbin.read_bytes()).hexdigest()!=a.image_sha256:raise ValueError('image mismatch')
    if software_tree(refresh=True)!=a.software_tree_sha256:raise ValueError('software mismatch')
    from archbench import registry
    for item in frozen('existing.json')+frozen('hbm-core.json'):
        if registry.get(item['id']).definition_sha256()!=item['sha256']:
            raise ValueError('frozen workload definition changed: '+item['id'])
    record['identity']=dict(image_sha256=a.image_sha256,software_tree=a.software_tree_sha256,
                            contract_sha256=contract_v2.LOCK_SHA256,**clock)
    env=dict(os.environ,TNCA_OPERATING_CLOCK_EVIDENCE=str(a.operating_clock_evidence.resolve()),
             TNCA_TORCH_DEVICE='native',TNCA_TRANSFER_POLICY='resident',TNCA_ACTIVE_CORES='3',
             TNCA_PROFILE='core-vrf-sync-v5',R40_CONTRACT='v2',CUDA_VISIBLE_DEVICES='')
    env.setdefault('TNCA_HOST_TIMEOUT','7200');env.setdefault('TNCA_DEADLINE','4000000000')
    for stage in stages:
        record.update(status='running',current_stage=stage['name']);save()
        print('START '+stage['name'],flush=True)
        with (out/(stage['name']+'.log')).open('x') as log:
            code=subprocess.run(stage['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        if code==0 and stage['name']=='original-hbm14':
            expected={p['id']:p['sha256'] for p in frozen('hbm-core.json')}
            rows=[json.loads(line) for line in (out/'hbm14.jsonl').read_text().splitlines() if line.strip()]
            valid=(len(rows)==len(expected) and {r['point_id'] for r in rows}==set(expected))
            for row in rows:
                provenance=row.get('provenance',{})
                valid=valid and (row.get('status')=='pass' and not row.get('fallbacks')
                    and row.get('point_sha256')==expected.get(row['point_id'])
                    and provenance.get('software_tree')==a.software_tree_sha256
                    and provenance.get('clock_hz')==clock['clock_hz']
                    and row.get('platform',{}).get('image_sha256')==a.image_sha256)
            if not valid:code=1
        record['completed'].append(dict(name=stage['name'],exit_code=code));save()
        if code:
            record['status']='failed-stage-preserved';save();return 1
        if software_tree(refresh=True)!=a.software_tree_sha256:
            record['status']='software-changed';save();return 1
        qualified_clock(a.operating_clock_evidence,a.image_sha256)
    record['status']='functional-stages-complete-review-required';save()
    return 0


if __name__=='__main__':raise SystemExit(main())
