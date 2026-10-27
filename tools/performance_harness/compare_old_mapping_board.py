"""Audit all nine physical old/fixed/narrow GEMM comparisons from raw files.

Requires completed collections and actual executable artifacts. Refuses partial
or mismatched cohorts; numerical capability and performance results are separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

from prepare_old_mapping_control import FROZEN_TREE, REGISTER_BLOCKS
import stage1_acceptance as acceptance
import prediction_residuals


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def collection(directory,experiment_name,batch_name,tree,manifest,host_root,ctx):
    report=read(directory/'out'/experiment_name/'report.json') if experiment_name else read(directory/'out/report.json')
    batch_dir=directory/'out'/experiment_name/batch_name if experiment_name else directory/'out'/batch_name
    batch=read(batch_dir/'batch.json');formal=batch_dir/'formal.jsonl'
    if report['status']!='completed-experiment' or report['identity']['software_tree']!=tree:
        raise ValueError('complete experiment on the registered source required')
    if report['identity']['manifest_sha256']!=sha(manifest):raise ValueError('measured manifest identity changed')
    if (batch['status']!='complete' or batch['software_tree']!=tree or batch['failed_configurations'] or batch['missing_formal'] or
        batch['measurement']!=ctx.plan['measurement'] or batch['plan_sha256']!=sha(ctx.plan_path)):
        raise ValueError('complete batch with the original sampling rule required')
    board_steps=[s for s in report['steps'] if s['kind']=='fpga']
    if len(board_steps)!=1 or board_steps[0]['status']!='completed':raise ValueError('one completed board step required')
    for path in (batch_dir/'batch.json',formal):
        recorded=str(host_root/path.relative_to(directory))
        if board_steps[0]['outputs'].get(recorded)!=sha(path):raise ValueError('experiment output hash mismatch')
    rows=[json.loads(line) for line in formal.read_text().splitlines() if line.strip()]
    if len(rows)!=batch['configurations']:raise ValueError('formal population differs from completed batch')
    result={}
    sys.path.insert(0,str(ctx.root/'python'))
    sys.path.insert(0,str(ctx.root/'tools/r40'))
    import ablate
    from tnca import artifact
    for row in rows:
        problems=acceptance.board_record_problems(row,ctx,{})
        if problems:raise ValueError('; '.join(problems))
        if row['platform']['software_tree']!=tree:raise ValueError('wrong measured product source')
        if any(not math.isfinite(row['correctness'][k]) for k in ('max_rel_err','tol')):
            raise ValueError('finite numerical evidence required')
        tile=tuple(row['platform']['inductor']['gemm_tile']);key=(row['point_id'],tile)
        if key in result:raise ValueError('duplicate formal point/tile')
        timings={}
        for metric in ('device-cycles-serial-grid','device-graph-makespan'):
            _,samples=ablate.samples_of(row,metric)
            if not samples or any(not math.isfinite(x) or x<=0 for x in samples):raise ValueError('positive finite completed samples required')
            timings[metric]=dict(samples=samples,median=statistics.median(samples))
        bundle=row['provenance']['bundle'];bundle_dir=directory/Path(bundle['dir']).relative_to(host_root)
        if not bundle_dir.resolve().is_relative_to(directory.resolve()):raise ValueError('bundle escapes collection')
        executables=[]
        for launch in row['provenance']['launch_map']['launches']:
            if not re.fullmatch('[0-9a-f]{64}',launch['artifact']):raise ValueError('invalid artifact identity')
            rel=f"artifacts/{launch['artifact']}.tnca";path=bundle_dir/rel
            if sha(path)!=bundle['files'][rel] or sha(path)!=launch['artifact']:
                raise ValueError('actual executed artifact missing or changed')
            metadata,_=artifact.load(str(path))
            identity=prediction_residuals.executable_identity(path)
            plan=metadata.get('optional_metadata',{}).get('provenance',{}).get('execution_resources',{})
            executables.append(dict(kernel=launch['kernel'],artifact_sha256=sha(path),
                executable_sha256=identity['executable_sha256'],code_sha256=identity['code_sha256'],
                static_accumulator_round_trips=plan.get('accumulator_round_trips'),resources=metadata['resources']))
        result[key]=dict(record=row,timings=timings,executables=executables)
    from archbench.provenance import _tree_files
    product_paths={str(p.relative_to(ctx.root)) for p in _tree_files(ctx.root)}
    sources=report['identity']['sources']
    roots={Path(name[:-len(REGISTER_BLOCKS)].rstrip('/')) for name in sources if name.endswith('/'+REGISTER_BLOCKS)}
    if len(roots)!=1:raise ValueError('unambiguous recorded compiler source root required')
    source_root=next(iter(roots))
    recorded={str(Path(name).relative_to(source_root)):value for name,value in sources.items()
              if Path(name).is_relative_to(source_root) and str(Path(name).relative_to(source_root)) in product_paths}
    return result,dict(experiment_sha256=sha(directory/'out'/experiment_name/'report.json') if experiment_name else sha(directory/'out/report.json'),
                       formal_sha256=sha(formal),batch_sha256=sha(batch_dir/'batch.json'),recorded_product_sources=recorded)


def compare(frozen,index,proof,old_manifest,old_directory,fixed_directory):
    ctx=acceptance.Context(frozen,index);prep=read(proof/'preparation.json')
    if ctx.tree!=FROZEN_TREE or prep['frozen_software_tree']!=FROZEN_TREE:
        raise ValueError('frozen Stage 1 identity required')
    registration=read(old_manifest)['registration']
    if (registration['isolated_control']!=prep['control_software_tree'] or
        registration['preparation_sha256']!=sha(proof/'preparation.json') or
        registration['validation_sha256']!=sha(proof/'validation-v2.json')):
        raise ValueError('control preparation/validation does not match board registration')
    fixed_manifest=frozen/'spec/r41/stage1/experiments/board-gemm-tile.json'
    if registration['original_manifest_sha256']!=sha(fixed_manifest):raise ValueError('original comparison registration changed')
    old,old_files=collection(old_directory,'','old-mapping',prep['control_software_tree'],old_manifest,
                             Path('/home/zzhouch/r41s1_old_mapping_v1'),ctx)
    fixed,fixed_files=collection(fixed_directory,'board-gemm-tile','gemm',FROZEN_TREE,fixed_manifest,
                                Path('/home/zzhouch/r41s1_board_25094197f6e6'),ctx)
    old_sources,fixed_sources=[m.pop('recorded_product_sources') for m in (old_files,fixed_files)]
    changed=sorted(k for k in set(old_sources)|set(fixed_sources) if old_sources.get(k)!=fixed_sources.get(k))
    if changed!=[REGISTER_BLOCKS]:raise ValueError(f'measured source delta is not isolated: {changed}')
    if old_sources[REGISTER_BLOCKS]!=prep['control_file_sha256'] or fixed_sources[REGISTER_BLOCKS]!=prep['source_file_sha256']:
        raise ValueError('measured compiler source does not match the prepared ablation')
    definition=read(frozen/'spec/r40/suites/R40-03.gemm-reuse.json')
    points={r['id'] for r in definition['points']};wide=(16,32,16);narrow=(16,16,16)
    if (set(old)!={(p,wide) for p in points} or
        set(fixed)!={(p,t) for p in points for t in (narrow,wide)}):
        raise ValueError('exact nine-point old and eighteen-configuration fixed/narrow cohorts required')
    result=[]
    for point in sorted(points):
        a,b,c=old[point,wide],fixed[point,wide],fixed[point,narrow]
        if a['record']['platform']['knobs']!=b['record']['platform']['knobs']:
            raise ValueError('old/fixed knobs differ')
        for field in ('toolchain','backend_args','profile','opt_level','inductor'):
            if a['record']['platform'].get(field)!=b['record']['platform'].get(field):
                raise ValueError(f'old/fixed {field} differs')
        row=dict(point=point,point_sha256=a['record']['point_sha256'],metrics={},
                 old_executables=a['executables'],fixed_executables=b['executables'],narrow_executables=c['executables'])
        for metric in a['timings']:
            old_time,fixed_time,narrow_time=[x['timings'][metric]['median'] for x in (a,b,c)]
            row['metrics'][metric]=dict(old=old_time,fixed=fixed_time,narrow=narrow_time,
                repair_speedup_old_over_fixed=old_time/fixed_time,
                fixed_over_conservative_ratio=fixed_time/narrow_time,
                samples={label:x['timings'][metric]['samples'] for label,x in zip(('old','fixed','narrow'),(a,b,c))})
        result.append(row)
    demonstrated=[]
    for row in result:
        def loop_moves(executables):
            return sum(sum((x['static_accumulator_round_trips'] or {}).get(k,0)
                           for k in ('k_loop_reads','k_loop_writes','k_loop_zeros')) for x in executables)
        if loop_moves(row['old_executables'])>0 and loop_moves(row['fixed_executables'])==0:
            demonstrated.append(row['point'])
    if not demonstrated:raise ValueError('executed board artifacts do not demonstrate the legacy K-loop traffic and its repair')
    return dict(schema='tnca-matched-old-mapping-board-comparison/1',status='comparison-recorded',
        frozen_software_tree=FROZEN_TREE,control_software_tree=prep['control_software_tree'],
        image_sha256=ctx.board['image_sha256'],clock_hz=ctx.board['clock_hz'],
        recorded_source_audit=dict(changed_product_files=changed,unchanged_recorded_product_files=len(fixed_sources)-1),
        old_files=old_files,fixed_files=fixed_files,manifest_sha256=sha(old_manifest),
        rows=result,repair_demonstrated_in_executed_artifacts=demonstrated,comparator_sha256=sha(Path(__file__)),
        scope='single-source-method compiler ablation; original point hashes and qualified board domain; descriptive medians without noise confidence intervals',
        caveats=['source versions have different executable sizes; complete graph costs include their actual placement and host gaps',
                 'historical fixed/narrow measurements reused; no interleaved old/fixed process sampling',
                 'comparison completion is not a general wide-tile speedup or default promotion'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-root','required-index','proof','old-manifest','old-collection','fixed-collection','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();report=compare(a.frozen_root,a.required_index,a.proof,a.old_manifest,a.old_collection,a.fixed_collection)
    with a.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],points=len(report['rows'])),indent=2))


if __name__=='__main__':main()
