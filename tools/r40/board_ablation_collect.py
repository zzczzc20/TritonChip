#!/usr/bin/env python3
"""Collect frozen runtime-only A/B experiments on one qualified physical image.

Original subsets, 5 warmups, three random-interleaved blocks. The frozen
long-task rule selects 10 versus 30 samples using a sealed qualification pass.
No point, threshold, clock or hardware override. Raw data and per-block
analysis are retained; complete package acceptance remains a separate audit.
"""
import argparse
import hashlib
import json
import math
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


def baseline_knobs(baseline):
    knobs = dict(TNCA_TORCH_DEVICE='native', TNCA_TUNE='0',
                 TNCA_MAX_BLOCK='16', TNCA_GEMM_TILE='16,16,16',
                 TNCA_REDUCTION_ROWS='1', TNCA_BANK_ALLOC='0',
                 TNCA_WINDOW_REUSE='0', TNCA_SDPA='decomposed', TNCA_SUBMIT='single')
    if baseline == 'B40-S':
        knobs.update(TNCA_TRANSFER_POLICY='per-kernel', TNCA_ACTIVE_CORES='1',
                     TNCA_RTL_MEM_PORTS='1', TNCA_RTL_OUTSTANDING='1', TNCA_PIPELINE='1')
    elif baseline == 'B40-N':
        knobs.update(TNCA_TRANSFER_POLICY='resident', TNCA_ACTIVE_CORES='3',
                     TNCA_RTL_MEM_PORTS='3', TNCA_RTL_OUTSTANDING='8', TNCA_PIPELINE='2')
    else:
        raise ValueError('unsupported frozen baseline '+baseline)
    return knobs


def sample_count(seconds, sampling):
    # Both sides qualify before any formal block; no speedup, CI or measured
    # formal sample is consulted. A failed qualification never drops a point.
    if set(seconds) != {'control', 'candidate'} or any(
            not math.isfinite(x) or x <= 0 for x in seconds.values()):
        raise ValueError('two finite positive qualification durations required')
    long = sampling['long_task']
    assert long['blocks'] == sampling['blocks'] == 3
    return long['timed'] if max(seconds.values()) > long['threshold_s'] else sampling['timed']


def sampling_command(entry, *, warmup, timed, stem, out):
    command = list(entry['command'])
    for option, value in [('--warmup', str(warmup)), ('--timed', str(timed)),
                          ('--out', str(out/(stem+'.jsonl'))),
                          ('--bundle-dir', str(out/'bundles'/stem))]:
        command[command.index(option)+1] = value
    return command


def schedule(experiment, python, config, out, *, compute_cores=3):
    contract_v2.verify()
    plan = json.loads((ROOT/'spec/r40/v2/plan.json').read_text())
    exp = dict(plan['experiments'][experiment])
    if compute_cores not in (2,3) or (compute_cores==2 and experiment!='R40-05.three-cores'):
        raise ValueError('two-core control belongs only to the frozen compute scaling experiment')
    if compute_cores==2:
        assert exp['note']=='1 -> 3 active cores; 2 cores >= 1.5'
        exp.update(threshold=1.5,required_compute_cores=2)
    knob, a, b = CONTROLS[experiment]
    if compute_cores==2:b='2'
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
                knobs = baseline_knobs(exp['baseline'])
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
    p.add_argument('--compute-cores',type=int,choices=(2,3),default=3,
                   help='Required frozen 2/3-core compute control; all other experiments use the default')
    p.add_argument('--hbm-bytes',type=int,default=1<<30)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.hbm_bytes < 1<<30 or a.hbm_bytes % 64:
        p.error('--hbm-bytes must be line aligned and at least 1 GiB')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=dict(xclbin=str(a.xclbin.resolve()),image_profile='core-vrf-sync-v5',
                core_count=3,memory_ports=3,hbm_dma=True,hbm_bytes=a.hbm_bytes,device_clock=True)
    exp,entries=schedule(a.experiment,sys.executable,config,out,compute_cores=a.compute_cores)
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
        report['transport']=dict(deadline_cycles=int(os.environ.get('TNCA_DEADLINE','24000000000')),
                                 host_timeout_s=float(os.environ.get('TNCA_HOST_TIMEOUT','7200')))
        report['identity']=dict(software_tree=a.software_tree_sha256,image_sha256=a.image_sha256,
                                backend='fpga-u280-multicore',backend_args=config,**clock)
        # Transport deadlines are declared separately; inherited tuning/legacy
        # knobs cannot contaminate the frozen baseline or stage selection.
        env={k:v for k,v in os.environ.items() if not k.startswith('TNCA_')}
        env.update(TNCA_OPERATING_CLOCK_EVIDENCE=str(a.operating_clock_evidence.resolve()),
                   TNCA_HOST_TIMEOUT=os.environ.get('TNCA_HOST_TIMEOUT','7200'),
                   TNCA_DEADLINE=os.environ.get('TNCA_DEADLINE','24000000000'),
                   CUDA_VISIBLE_DEVICES='',R40_CONTRACT='v2')
        env['PYTHONPATH']=os.pathsep.join(str(ROOT/d) for d in ('','python','compiler/python','tools/asm'))
        def collect(entry, command, stem, block, warmup, timed):
            report.update(status='running',current=stem);save()
            raw = Path(command[command.index('--out')+1])
            with (out/(stem+'.log')).open('x') as log:
                done=subprocess.run(command,cwd=ROOT,
                    env=dict(env,ARCHBENCH_BLOCK_ID=str(block)),stdout=log,stderr=subprocess.STDOUT)
            rows=ablate.load(raw);assert len(rows)==1,'expected exactly one original point'
            row=rows[0]
            plat=row.get('platform',{});prov=row.get('provenance',{})
            assert done.returncode==0 and row['status']=='pass' and not row.get('fallbacks'),'original point failed'
            assert row['point_id']==entry['point']['id'] and row['point_sha256']==entry['point']['sha256']
            assert plat.get('software_tree')==a.software_tree_sha256 and plat.get('image_sha256')==a.image_sha256
            assert plat.get('backend_args')==config and plat.get('backend')=='fpga-u280-multicore'
            assert prov.get('clock_hz')==clock['clock_hz']
            assert all(plat.get('knobs',{}).get(k)==v for k,v in entry['knobs'].items())
            assert prov.get('launch_map',{}).get('cores')==list(range(int(entry['knobs']['TNCA_ACTIVE_CORES'])))
            _,timing=ablate.timing_entry(row,exp['timing']);_,samples=ablate.samples_of(row,exp['timing'])
            assert timing.get('warmup')==warmup and len(samples)==timed and timing.get('deterministic') is False
            assert software_tree(refresh=True)==a.software_tree_sha256,'source changed during collection'
            assert qualified_clock(a.operating_clock_evidence,a.image_sha256)==clock,'clock qualification changed'
            return row, timing, samples

        sampling=json.loads((ROOT/'spec/r40/v2/plan.json').read_text())['sample_plan']['board']
        qualification={}
        # The complete count manifest is sealed before block 0. These cold/setup
        # separated calls are retained but never pooled with formal A/B samples.
        for entry in (e for e in entries if e['block']==0):
            pid=entry['point']['id']; stem='qualify-'+entry['stem']
            command=sampling_command(entry,warmup=1,timed=1,stem=stem,out=out)
            report.setdefault('qualification_commands',[]).append(command);save()
            row,timing,samples=collect(entry,command,stem,'qualification',1,1)
            assert timing.get('unit')=='cycles','qualification requires raw device clock samples'
            qualification.setdefault(pid,{})[entry['side']]=dict(
                seconds=samples[0]/clock['clock_hz'],raw=command[command.index('--out')+1],
                raw_sha256=hashlib.sha256(Path(command[command.index('--out')+1]).read_bytes()).hexdigest())
        counts={pid:sample_count({side:r['seconds'] for side,r in rows.items()},sampling)
                for pid,rows in qualification.items()}
        manifest=dict(rule=sampling,qualification=qualification,timed_per_point=counts,
                      identity=report['identity'],contract_sha256=contract_v2.LOCK_SHA256,
                      original_points={e['point']['id']:e['point']['sha256'] for e in entries},
                      scope='Prospective classification only; no qualification sample is a formal measurement')
        manifest_path=out/'sampling_manifest.json'
        manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
        with manifest_path.open('xb') as f:f.write(manifest_bytes)
        manifest_path.chmod(0o444)
        manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
        report['sampling_manifest_sha256']=manifest_sha
        for entry in entries:
            entry['command']=sampling_command(entry,warmup=5,timed=counts[entry['point']['id']],stem=entry['stem'],out=out)
        save()
        paths={(b,s):out/f'{s}{b}.jsonl' for b in range(3) for s in ('control','candidate')}
        for path in paths.values():path.touch(exist_ok=False)
        for entry in entries:
            assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==manifest_sha,'sampling manifest changed'
            row,_,_=collect(entry,entry['command'],entry['stem'],entry['block'],5,counts[entry['point']['id']])
            with paths[entry['block'],entry['side']].open('a') as f:f.write(json.dumps(row)+'\n')
            report['completed'].append(entry['stem']);save();print('COLLECTED '+entry['stem']+' '+row['point_id'],flush=True)
        verdicts=[]
        subset={e['point']['id']:e['point']['sha256'] for e in entries}
        for block in range(3):
            verdict=ablate.compare(paths[block,'control'],paths[block,'candidate'],subset,
                exp['vary'],timing=exp['timing'],threshold=exp['threshold'],audit_bundles=True)
            (out/f'verdict{block}.json').write_text(json.dumps(verdict,indent=2)+'\n');verdicts.append(verdict['verdict'])
        report.update(status='samples-complete-package-audit-required',block_verdicts=verdicts,
                      gate_status='not-measured',scope='Three complete prospectively classified frozen sampling blocks; cross-block statistics, required traffic/scaling controls, LOO and complete package evidence remain mandatory')
    except Exception:
        report.update(status='failed-preserved',error=traceback.format_exc())
    save()
    return 0 if report['status']=='samples-complete-package-audit-required' else 1


if __name__=='__main__':raise SystemExit(main())
