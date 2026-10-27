"""Postdict measured diagnostics; refuse residuals when program/bindings differ."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from prepare_old_mapping_control import FROZEN_TREE


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def compare(model_root,board_root):
    files={};result=[]
    for family in ('memory-service','pipeline-admission'):
        path=board_root/('board-'+family)/'report.json';experiment=read(path)
        if experiment['status']!='completed-experiment' or experiment['identity']['software_tree']!=FROZEN_TREE:
            raise ValueError('complete frozen board experiment required')
        for step in experiment['steps']:
            if step['kind']=='fpga' and (step['board']['clock_hz']!=90000000 or
                step['board']['image_sha256']!='a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083'):
                raise ValueError('wrong physical domain')
        files[str(path)]=sha(path)
    def load(path,model):
        r=read(path);files[str(path)]=sha(path)
        if r['status']!='functional-pass' or r['software_tree']!=FROZEN_TREE:
            raise ValueError('functional completion on exact frozen source required')
        expected='systemc-tlm-grid' if model else 'fpga-u280-multicore'
        if r['backend']!=expected:raise ValueError('wrong execution backend')
        if model:
            identities=r.get('model_identities') or [r['model_identity']]
            for identity in identities:
                cfg=identity['config']
                if identity['dram_path']!='fast' or identity['host_handoff_policy']!='dma-maintenance':
                    raise ValueError('registered fast completed-launch domain required')
                if int(cfg['clock_hz'])!=90000000 or any(int(cfg[k]) for k in
                    ('host_launch_cycles','host_transfer_cycles','host_transfer_cycles_per_kib')):
                    raise ValueError('wrong clock or double-counted fitted host costs')
        return r
    def row(kind,name,model,physical,resources,graph=None,shape=None):
        if any(x.get('status')!='pass' for x in [model,*physical]) and kind=='pipeline':
            raise ValueError('per-graph numeric completion required')
        code=lambda r:[a['code_sha256'] for a in r['artifacts']]
        bindings=lambda r:(r['placements'],r['output_placement']) if kind=='memory' else (r['input_placements'],r['output_placement'])
        resource=lambda r:[a[resources] for a in r['artifacts']]
        checks=dict(inputs=all(x['inputs']==model['inputs'] for x in physical),
                    code=all(code(x)==code(model) for x in physical),
                    resources=all(resource(x)==resource(model) for x in physical),
                    bindings=all(bindings(x)==bindings(model) for x in physical))
        samples=[x['device_cycles'] for x in physical];predicted=model['device_cycles']
        if any(type(x) is not int or x<=0 for x in [predicted,*samples]):raise ValueError('positive completed launch cycles required')
        median=statistics.median(samples);comparable=all(checks.values())
        output_key='output_sha256' if kind=='memory' else 'output_f32_sha256'
        result.append(dict(kind=kind,case=name,graph=graph,shape=shape,identity_checks=checks,
            comparison_status='matched-postdiction' if comparable else 'not-comparable',
            refusal_reasons=[k+' differ' for k,v in checks.items() if not v],
            model_cycles=predicted,board_samples=samples,board_median_cycles=median,
            signed_residual=(predicted-median)/median if comparable else None,
            output_hashes_equal=all(x[output_key]==model[output_key] for x in physical)))
    for name,ports,depth in (('p1o1',1,1),('p1o8',1,8),('p3o8',3,8)):
        model=load(model_root/('model-'+name)/'report.json',True)
        physical=[load(board_root/'board-memory-service'/f'r{i}-{name}'/'report.json',False) for i in (1,2,3)]
        if any((x['ports'],x['outstanding'],x['elements'])!=(ports,depth,4097) for x in [model,*physical]):
            raise ValueError('registered memory cohort changed')
        row('memory',name,model,physical,'resources')
    expected={(graph,tuple(shape)) for graph in ('mm','addmm') for shape in ([16,16,48],[17,31,65])}
    for name,stage in (('baseline',None),('stage1',1),('stage2',2),('stage3',3)):
        model=load(model_root/('model-'+name)/'report.json',True)
        physical=[load(board_root/'board-pipeline-admission'/f'r{i}-{name}'/'report.json',False) for i in (1,2,3)]
        def keyed(report):
            rows={(x['graph'],tuple(x['shape'])):x for x in report['rows']}
            if len(report['rows'])!=4 or set(rows)!=expected:raise ValueError('exact pipeline cohort required')
            if any(x['ring_stages']!=stage for x in rows.values()):raise ValueError('ring stage differs')
            return rows
        m=keyed(model);ps=[keyed(x) for x in physical]
        for key,value in m.items():row('pipeline',name,value,[p[key] for p in ps],'manifest_resources',key[0],list(key[1]))
    matched=[r for r in result if r['comparison_status']=='matched-postdiction']
    return dict(schema='tnca-captured-input-postdiction/1',status='postdiction-recorded',
        software_tree=FROZEN_TREE,rows=result,matched=len(matched),not_comparable=len(result)-len(matched),
        input_report_sha256=files,domain=read(model_root/'domain.json'),comparator_sha256=sha(Path(__file__)),
        timing_qualification='uncalibrated',scope='completed-launch cycles only; captured physical inputs, code, resources and Tensor bindings; not whole-call latency',
        limitations=['postdiction after measurements; not sealed prediction or unseen holdout',
                     'no fitting; model uncertainty remains material; no default promotion',
                     'mismatched bindings retain their numerical results but have no physical residual'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-root',type=Path,required=True)
    ap.add_argument('--board-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();report=compare(a.model_root,a.board_root)
    with a.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','matched','not_comparable')},indent=2))


if __name__=='__main__':main()
