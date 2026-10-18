#!/usr/bin/env python3
"""Collect frozen runtime-only A/B experiments on one qualified physical image.

Original subsets, 5 warmups, 30 samples, three random-interleaved blocks.
No point, sample-count, clock or hardware override. Raw data and per-block
analysis are retained; complete package acceptance remains a separate audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
                str(ROOT/'tools/asm'), str(ROOT/'tools/r40')]
import ablate
import contract_v2
from archbench.provenance import software_tree

CONTROLS = {
    'R40-05.three-cores': ('TNCA_ACTIVE_CORES', '1', '3'),
    'R40-06.streaming-cores': ('TNCA_ACTIVE_CORES', '1', '3'),
    'R40-06.ports': ('TNCA_RTL_MEM_PORTS', '1', '3'),
    'R40-04.memory-service': ('TNCA_RTL_OUTSTANDING', '1', '8'),
}


def schedule(experiment, python, config, out):
    contract_v2.verify()
    plan = json.loads((ROOT/'spec/r40/v2/plan.json').read_text())
    exp = plan['experiments'][experiment]
    knob, a, b = CONTROLS[experiment]
    if exp['vary'] != [knob]:
        raise ValueError('runtime control differs from the frozen experiment')
    points = sorted(json.loads((ROOT/exp['subset']).read_text())['points'], key=lambda p:p['id'])
    sampling = plan['sample_plan']['board']
    assert (sampling['warmup'],sampling['timed'],sampling['blocks']) == (5,30,3)
    rng = random.Random(plan['sample_plan']['bootstrap']['seed'])
    entries = []
    for block in range(3):
        for index, point in enumerate(points):
            sides = ('control','candidate') if rng.randrange(2) == 0 else ('candidate','control')
            for side in sides:
                stem = f'block{block}-point{index:02d}-{side}'
                knobs = dict(TNCA_TORCH_DEVICE='native', TNCA_TRANSFER_POLICY='resident',
                             TNCA_ACTIVE_CORES='3', TNCA_RTL_MEM_PORTS='3',
                             TNCA_RTL_OUTSTANDING='8', TNCA_TUNE='0')
                knobs[knob] = a if side == 'control' else b
                raw = out/(stem+'.jsonl')
                command = [python,'-u','-m','archbench.run','--platform','tnca',
                           '--backend','fpga-u280-multicore','--backend-args',json.dumps(config),
                           '--profile','core-vrf-sync-v5','--point',point['id'],
                           '--jobs','1','--seed','0','--warmup','5','--timed','30',
                           '--timeout','7200','--bundle-dir',str(out/'bundles'/stem),
                           '--out',str(raw)]
                for key,value in sorted(knobs.items()):
                    command += ['--set',key+'='+value]
                entries.append(dict(block=block,side=side,stem=stem,point=point,
                                    knobs=knobs,raw=str(raw),command=command))
    return exp, entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=CONTROLS, required=True)
    p.add_argument('--xclbin',type=Path,required=True)
    p.add_argument('--image-sha256',required=True)
    p.add_argument('--software-tree-sha256',required=True)
    p.add_argument('--operating-clock-evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hbm-bytes',type=int,default=1<<30)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.hbm_bytes < 1<<30 or a.hbm_bytes % 64:
        p.error('--hbm-bytes must be line aligned and at least 1 GiB')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=dict(xclbin=str(a.xclbin.resolve()),image_profile='core-vrf-sync-v5',
                core_count=3,memory_ports=3,hbm_dma=True,hbm_bytes=a.hbm_bytes,device_clock=True)
    exp,entries=schedule(a.experiment,sys.executable,config,out)
    report=dict(schema='tnca-r40-board-ablation-collection/1',status='planned',
                experiment=a.experiment,frozen_experiment=exp,entries=entries,completed=[],
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                contract_sha256=contract_v2.LOCK_SHA256,required_gate_promotions=[])
    def save():
        temp=out/'collection.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'collection.json')
    save()
    if a.plan_only:return 0
    try:
        from tnca.operating_clock import qualified_clock
        assert software_tree(refresh=True)==a.software_tree_sha256,'software identity mismatch'
        assert hashlib.sha256(a.xclbin.read_bytes()).hexdigest()==a.image_sha256,'image identity mismatch'
        clock=qualified_clock(a.operating_clock_evidence,a.image_sha256)
        report['identity']=dict(software_tree=a.software_tree_sha256,image_sha256=a.image_sha256,
                                backend='fpga-u280-multicore',backend_args=config,**clock)
        env=dict(os.environ,TNCA_OPERATING_CLOCK_EVIDENCE=str(a.operating_clock_evidence.resolve()),
                 CUDA_VISIBLE_DEVICES='',R40_CONTRACT='v2')
        env['PYTHONPATH']=os.pathsep.join(str(ROOT/d) for d in ('','python','compiler/python','tools/asm'))
        env.setdefault('TNCA_HOST_TIMEOUT','7200');env.setdefault('TNCA_DEADLINE','4000000000')
        paths={(b,s):out/f'{s}{b}.jsonl' for b in range(3) for s in ('control','candidate')}
        for path in paths.values():path.touch(exist_ok=False)
        for entry in entries:
            report.update(status='running',current=entry['stem']);save()
            with (out/(entry['stem']+'.log')).open('x') as log:
                done=subprocess.run(entry['command'],cwd=ROOT,
                    env=dict(env,ARCHBENCH_BLOCK_ID=str(entry['block'])),stdout=log,stderr=subprocess.STDOUT)
            rows=ablate.load(entry['raw']);assert len(rows)==1,'expected exactly one original point'
            row=rows[0]
            # Retain even a failed record before judging it.
            with paths[entry['block'],entry['side']].open('a') as f:f.write(json.dumps(row)+'\n')
            plat=row.get('platform',{});prov=row.get('provenance',{})
            assert done.returncode==0 and row['status']=='pass' and not row.get('fallbacks'),'original point failed'
            assert row['point_id']==entry['point']['id'] and row['point_sha256']==entry['point']['sha256']
            assert plat.get('software_tree')==a.software_tree_sha256 and plat.get('image_sha256')==a.image_sha256
            assert plat.get('backend_args')==config and plat.get('backend')=='fpga-u280-multicore'
            assert prov.get('clock_hz')==clock['clock_hz']
            assert all(plat.get('knobs',{}).get(k)==v for k,v in entry['knobs'].items())
            if CONTROLS[a.experiment][0]=='TNCA_ACTIVE_CORES':
                assert prov.get('launch_map',{}).get('cores')==list(range(int(entry['knobs']['TNCA_ACTIVE_CORES'])))
            _,timing=ablate.timing_entry(row,exp['timing']);_,samples=ablate.samples_of(row,exp['timing'])
            assert timing.get('warmup')==5 and len(samples)==30 and timing.get('deterministic') is False
            assert software_tree(refresh=True)==a.software_tree_sha256,'source changed during collection'
            assert qualified_clock(a.operating_clock_evidence,a.image_sha256)==clock,'clock qualification changed'
            report['completed'].append(entry['stem']);save();print('COLLECTED '+entry['stem']+' '+row['point_id'],flush=True)
        verdicts=[]
        subset={e['point']['id']:e['point']['sha256'] for e in entries}
        for block in range(3):
            verdict=ablate.compare(paths[block,'control'],paths[block,'candidate'],subset,
                exp['vary'],timing=exp['timing'],threshold=exp['threshold'],audit_bundles=True)
            (out/f'verdict{block}.json').write_text(json.dumps(verdict,indent=2)+'\n');verdicts.append(verdict['verdict'])
        report.update(status='samples-complete-package-audit-required',block_verdicts=verdicts,
                      gate_status='not-measured',scope='Three complete ordinary sampling blocks; cross-block statistics, required traffic/scaling controls, LOO and complete package evidence remain mandatory')
    except Exception:
        report.update(status='failed-preserved',error=traceback.format_exc())
    save()
    return 0 if report['status']=='samples-complete-package-audit-required' else 1


if __name__=='__main__':raise SystemExit(main())
