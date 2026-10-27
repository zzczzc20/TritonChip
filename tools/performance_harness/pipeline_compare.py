#!/usr/bin/env python3
"""Compare actual three-sample pipeline records; do not infer acceptance."""
import argparse,hashlib,json,statistics
from pathlib import Path
import experiment

SHAPES={(g,(16,16,k)) for g in ('mm','addmm') for k in (32,48,64)}
IMAGE='30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
UUID='49964b17-e0b4-5bd2-dd40-7fc37b1205d3'
CLOCK='49868f45b5ced2f662e4f76bbb1bb4a2be166625915d869f20812afcee42f0d7'

def compare(paths,backend,*,experiment_report=None):
    records={name:json.loads(path.read_text()) for name,path in paths.items()}
    if len({r['software_tree'] for r in records.values()})!=1:raise ValueError('software trees differ')
    if backend=='fpga-u280-multicore':
        if experiment_report is None:raise ValueError('board identity and step probes required')
        outer=json.loads(experiment_report.read_text())
        steps={s['id']:s for s in outer['steps'] if s['id'] in paths}
        if set(steps)!=set(paths):raise ValueError('incomplete board steps')
        for step in steps.values():
            if step['status']!='completed':raise ValueError('board step did not complete')
            board=step['board']
            if (board['image_sha256'],board['image_uuid'],board['clock_hz'],board['clock_evidence_sha256'])!=(IMAGE,UUID,100000000,CLOCK):
                raise ValueError('wrong image/clock evidence')
            for probe in (board['probe_before'],board['probe_after']):
                if experiment.board_refusal(board,probe):raise ValueError('board probe refused')
            if step['identity']['software_tree']!=next(iter(records.values()))['software_tree']:
                raise ValueError('experiment and row source identities differ')
    elif backend!='systemc-tlm-grid':raise ValueError('unsupported comparison domain')
    indexed={}
    for mode,r in records.items():
        if r['status']!='functional-pass' or r['backend']!=backend or not r['bitwise'] or r['timed_samples']!=3:
            raise ValueError(f'{mode}: incomplete or wrong backend')
        k=r['knobs']
        for name,value in {'TNCA_ACTIVE_CORES':'3','TNCA_RCTX_SPMD':'1','TNCA_RCTX_SPMD_SHARED':'2',
                           'TNCA_PROFILE':'core-vrf-sync-v5-x8','TNCA_SCHEDULE':'0','TNCA_WINDOW_REUSE':'0'}.items():
            if k.get(name)!=value:raise ValueError(f'{mode}: mismatched {name}')
        rows={(x['graph'],tuple(x['shape'])):x for x in r['rows']}
        if set(rows)!=SHAPES or len(r['rows'])!=len(SHAPES):raise ValueError(f'{mode}: changed point inventory')
        for row in rows.values():
            if row['status']!='pass' or not row['bitwise_reference']['equal'] or not row['cold_bitwise_same']:
                raise ValueError(f'{mode}: numerical failure')
            if row['output_bits_sha256']!=row['reference_bits_sha256']:raise ValueError('output/reference hash mismatch')
            samples=row.get('timed_samples',[])
            if len(samples)!=3 or any(type(s['device_cycles']) is not int or s['device_cycles']<=0 or
                    s['output_bits_sha256']!=row['reference_bits_sha256'] for s in samples):
                raise ValueError(f'{mode}: timed samples invalid')
            if len(row['artifacts'])!=1 or not row['launches']:raise ValueError('missing numerical artifact')
            if any(x['grid']!=[1,1,1] or x['upload_bytes'] or x['download_bytes'] for x in row['launches']):
                raise ValueError('grid or native-residency boundary differs')
            for item in row['inputs']:
                payload=(paths[mode].parent/item['logical_contiguous_bytes']).read_bytes()
                if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('input file hash mismatch')
            output=paths[mode].parent/'artifacts'/f"{row['graph']}-{'-'.join(map(str,row['shape']))}-out.bin"
            if hashlib.sha256(output.read_bytes()).hexdigest()!=row['output_bits_sha256']:
                raise ValueError('output file hash mismatch')
            art=row['artifacts'][0];exe=art['execution'];sched=art['scheduling']
            from tnca import artifact
            manifest,segments=artifact.load(paths[mode].parent/'artifacts'/(art['code_sha256']+'.tnca'))
            if hashlib.sha256(segments['text']).hexdigest()!=art['code_sha256'] or manifest['execution']!=exe:
                raise ValueError('artifact text or execution map mismatch')
            provenance=manifest['optional_metadata']['provenance']
            if any(provenance.get(k)!=v for k,v in sched.items()) or provenance.get('execution_resources')!=art['plan']:
                raise ValueError('reported schedule/resource plan differs from actual artifact')
            if exe['role_mask']!=3 or art['plan']['resources']['vrf_limit']!=64:
                raise ValueError('role context or register budget differs')
            if mode=='paired':
                if sched.get('rctx_roles_pairs')!=1 or sched.get('rctx_roles_ring',{}).get('stages')!=2 or len(exe['program_to_role']['0'])!=2:
                    raise ValueError('paired pipeline was not applied')
            elif sched.get('rctx_spmd')!=2 or bool(sched.get('single_role_pipeline'))!=(mode=='single'):
                raise ValueError('one-role policy was not applied')
        indexed[mode]=rows
    result=[]
    for point in sorted(SHAPES):
        arms={name:rows[point] for name,rows in indexed.items()}
        if len({tuple((x['sha256'],tuple(x['stride'])) for x in row['inputs']) for row in arms.values()})!=1:
            raise ValueError('input data or layout differs across arms')
        if len({row['output_bits_sha256'] for row in arms.values()})!=1:raise ValueError('output differs across arms')
        samples={mode:[s['device_cycles'] for s in row['timed_samples']] for mode,row in arms.items()}
        medians={mode:statistics.median(values) for mode,values in samples.items()}
        result.append(dict(graph=point[0],shape=point[1],device_cycles=samples,median_cycles=medians,
            single_ratio=medians['baseline']/medians['single'],paired_ratio=medians['baseline']/medians['paired'],
            artifacts={mode:row['artifacts'][0]['code_sha256'] for mode,row in arms.items()},
            traffic={mode:[dict(before=s['counters_before'],after=s['counters_after']) for s in row['timed_samples']]
                     for mode,row in arms.items()}))
    return dict(schema='r41-p2-pipeline-comparison/1',result='measured',
        residual='Not calculated: a separate matched prediction and physical residual review is required.',
        conclusion='Three recorded samples per registered point; compare per-point ratios and retain negative results.',
        next_step='Review physical identity, sample stability, and final-source scope before rejudging the acceptance study.',
        backend=backend,software_tree=next(iter(records.values()))['software_tree'],rows=result,
        limitations=['Grid-one cases exercise one program; they do not establish three-core scaling.',
                     'Three device samples do not satisfy the separate 30-sample host-latency protocol.',
                     'SystemC results are uncalibrated and are never promoted to board evidence.'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','single','paired','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--experiment',type=Path)
    p.add_argument('--backend',choices=('fpga-u280-multicore','systemc-tlm-grid'),required=True)
    a=p.parse_args();paths={name:getattr(a,name) for name in ('baseline','single','paired')}
    result=compare(paths,a.backend,experiment_report=a.experiment)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['result'],points=len(result['rows']))))
if __name__=='__main__':main()
