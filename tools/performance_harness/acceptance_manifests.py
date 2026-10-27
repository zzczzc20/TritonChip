#!/usr/bin/env python3
"""Generate bounded lead-started acceptance registrations; never run a card."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import stage2_board
ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'spec/r41/stage2/acceptance'
CONTROL='{home}/r41_accept_control_21c40aa'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def common(name,hypothesis):
    plan=ROOT/'spec/r41/stage2/plan.json'
    old=json.loads((ROOT/'spec/r41/stage1/experiments/board-gemm-tile.json').read_text())
    return dict(schema='tnca-experiment-spec/1',id=name,allowed_hosts=['acf3019'],hypothesis=hypothesis,
        competing_explanations=['changed emitted address/fragment code','host gaps or transfer boundaries','different image/clock or software tree'],
        expected_counter_changes=['actual launched artifacts, graph/device completion cycles and traffic; no synthetic measurements'],
        expected_observations=['all registered points remain correct with descriptors enabled and no CPU numerical fallback','paired completion samples and exact software/image identities retained'],
        board=stage2_board.board_block(),plan='spec/r41/stage2/plan.json',plan_sha256=sha(plan),
        measurement=json.loads(plan.read_text())['measurement'],suites=old['suites'],
        estimates=old.get('estimates',[]),sources=['tools/performance_harness','archbench','python/tnca','compiler/python','compiler/targets','spec/r39_sched_model.json','tools/asm','isa',f'spec/r41/stage2/acceptance/{name}.json'],
        env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAX_JOBS='2',
                 CUDA_VISIBLE_DEVICES='',PYTHONPATH='{root}:{root}/python:{root}/compiler/python:{root}/tools/asm'),
        registration=dict(status='registered-not-run',registered_utc='2026-10-26',authority='lead starts waiting queue; agent never opens card',budget_seconds=10800),
        budget_seconds=10800,board_batches={},steps=[])

def step(name,batch,*,source='{root}',target=None):
    out='{out}/'+(target or batch)
    return dict(id=name,kind='fpga',argv=['{python}','{root}/tools/performance_harness/board_batch.py',
        '--manifest','{root}/spec/r41/stage2/acceptance/s2-p1-gemm.json' if name.startswith('gemm') else '{root}/spec/r41/stage2/acceptance/s2-p1-policy.json',
        '--batch',batch,'--root',source,'--out',out,'--python','{python}'],
        timeout_seconds=10800,env=stage2_board.fpga_env(4*stage2_board.GIB),
        timing_boundary='qualification then formal samples ending at device completion; graph makespan and serial-grid separately',
        cache_state='private process/cache per configuration; retained failure; never tune inside a formal timing sample',
        outputs=[out+'/batch.json',out+'/formal.jsonl'])

def manifests():
    original=json.loads((ROOT/'spec/r41/stage1/experiments/board-gemm-tile.json').read_text())
    gemm=common('s2-p1-gemm','Corrected fragment index proofs remove needless K-loop scratch traffic; compare repaired N32 against legal N16 and the pre-fix compiler on the identical approved 100 MHz image.')
    configs=copy.deepcopy(original['board_batches']['gemm']['configurations'])
    for c in configs:c['knobs'].update(TNCA_INDEX_OPT='1',TNCA_SCHEDULE='1')
    old=[copy.deepcopy(c) for c in configs if c['arm']=='candidate']
    for c in old:c['id']=c['point']+'@pre-fix';c['arm']='pre-fix'
    groups=[[c['id'] for c in configs if c['point']==p] for p in dict.fromkeys(c['point'] for c in configs)]
    gemm['board_batches']={'fixed':dict(sampling='device-rule',configurations=configs,groups=groups),
                           'pre-fix':dict(sampling='device-rule',configurations=old)}
    gemm['sources'] += [CONTROL+'/python/tnca',CONTROL+'/compiler/python',CONTROL+'/spec/r39_sched_model.json']
    gemm['registration']['control_source_commit']='21c40aa'
    gemm['steps']=[step('gemm-pre-fix','pre-fix',source=CONTROL),step('gemm-fixed','fixed')]
    gemm['steps'].append(dict(id='compare',kind='compare',argv=['{python}','{root}/tools/performance_harness/acceptance_compare.py',
        '--before','{out}/pre-fix/formal.jsonl','--after','{out}/fixed/formal.jsonl','--manifest','{root}/spec/r41/stage2/acceptance/s2-p1-gemm.json','--out','{out}/comparison.json'],timeout_seconds=60,outputs=['{out}/comparison.json']))
    gemm['conclusion_source']='{out}/comparison.json'
    policy=common('s2-p1-policy','Measure primary software group A/B and leave-one-out for index/layout, scheduling and descriptor emission; use a small index-by-scheduling interaction matrix, fixed resources and identical inputs.')
    shas=stage2_board.point_shas();configs=[];groups=[]
    points=['op.matmul/medium/fp16-m128-k192-n128','op.add/medium/fp16-n49152','model.mlp/medium/fp16-t128-d128']
    arms={'full':{},'no-index':{'TNCA_INDEX_OPT':'0'},'no-schedule':{'TNCA_SCHEDULE':'0'},'no-sparse':{'TNCA_SPARSE_TENSORMAPS':'0'},'no-index-no-schedule':{'TNCA_INDEX_OPT':'0','TNCA_SCHEDULE':'0'}}
    for point in points:
        group=[]
        for arm,knobs in arms.items():
            values=dict(stage2_board.KNOBS,TNCA_INDEX_OPT='1',TNCA_SCHEDULE='1',TNCA_SPARSE_TENSORMAPS='1')
            values.update(knobs)
            id=point+'@'+arm;group.append(id)
            configs.append(dict(id=id,point=point,point_sha256=shas[point],arm=arm,knobs=values))
        groups.append(group)
    policy['board_batches']={'policy':dict(sampling='device-rule',configurations=configs,groups=groups)}
    policy['steps']=[step('policy','policy')]
    policy['registration']['interpretation']='Software group attribution only. Fixed-image physical timing fixes cannot be isolated through compiler switches and remain separately unmeasured.'
    migration=[]
    for mhz in (90,100):
        name=f's2-p1-shared-{mhz}'
        spec=common(name,'Identical software and core-vrf-sync-v5 artifacts on the two approved images: report joint hardware/frequency migration, without an iso-frequency hardware speedup claim.')
        if mhz==90:spec['board']=copy.deepcopy(original['board'])
        spec['board']['profile']='core-vrf-sync-v5'
        configs=stage2_board.configs(points)
        for c in configs:c['knobs'].update(TNCA_INDEX_OPT='1',TNCA_SCHEDULE='1')
        spec['board_batches']={'shared':dict(sampling='device-rule',configurations=configs)}
        command=step('shared','shared')
        command['argv'][command['argv'].index('--manifest')+1]='{root}/spec/r41/stage2/acceptance/'+name+'.json'
        command['env'].update(TNCA_PROFILE='core-vrf-sync-v5',TNCA_BACKEND_ARGS=json.dumps(spec['board']['backend_args'],separators=(',',':')),TNCA_OPERATING_CLOCK_EVIDENCE=spec['board']['clock_evidence'])
        spec['steps']=[command]
        spec['registration']['interpretation']='Same logical compiler profile; verify actual artifact text hashes match across images before attributing the migration. No new image build.'
        migration.append(spec)
    def direct(spec,name,script,timeout,*,output=True):
        argv=['{python}','{root}/'+script]
        if output:argv += ['--out','{out}/'+name]
        return dict(id=name,kind='fpga',argv=argv,timeout_seconds=timeout,
            env=stage2_board.fpga_env(4*stage2_board.GIB),
            timing_boundary='functional lifecycle or opt-in locality scope; raw device and host observations only, no performance acceptance claim',
            cache_state='fresh experiment-local process, Session and caches; preserved failure; no tuning',
            outputs=['{out}/'+name+'/report.json'] if output else [])
    p2=common('s2-p2-locality','Two captured standard Torch graphs execute identical native artifacts with conservative locality off/on/off; numerical results, ownership, injected-fault cleanup, traffic and latency are observed without assuming a gain.')
    p2['steps']=[direct(p2,'locality','tools/performance_harness/locality_case.py',1200)]
    p2['budget_seconds']=1200;p2['registration']['budget_seconds']=1200
    p2['registration']['contract']='doc/r41_tasks_and_acceptance.md S2-03.A; functional board evidence only, separate performance review required'
    p4=common('s2-p4-lifecycle','Explicit compound refusals, standard Torch code eviction and a rolled AOT/Session barrier loop establish lifecycle behavior on the approved image; no full-model or aggregate acceptance inference.')
    p4['sources'] += ['tools/r40/alias_negative_case.py','tools/r40/device_tests.py','tools/r40/barrier_epoch_case.py']
    p4['steps']=[direct(p4,'aliases','tools/r40/alias_negative_case.py',300),
                 direct(p4,'eviction','tools/r40/device_tests.py',1800,output=False),
                 direct(p4,'barrier','tools/r40/barrier_epoch_case.py',3600)]
    p4['steps'][1]['argv'] += ['--only','eviction']
    p4['budget_seconds']=5100;p4['registration']['budget_seconds']=5100
    p4['registration']['contract']='doc/r41_tasks_and_acceptance.md S2-04.D.3/.4/.6/.7; full-model persistent-session and sustained-admission gates remain separate'
    return [gemm,policy,*migration,p2,p4]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    SPEC.mkdir(parents=True,exist_ok=True)
    for manifest in manifests():
        path=SPEC/(manifest['id']+'.json')
        if a.check:
            if not path.is_file() or json.loads(path.read_text())!=manifest:raise ValueError('registered manifest differs: '+str(path))
        else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('acceptance manifests match' if a.check else 'acceptance manifests prepared; no execution')
if __name__=='__main__':main()
