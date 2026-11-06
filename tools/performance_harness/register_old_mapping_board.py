"""Register the nine original GEMM points for the isolated old-mapping arm.

The existing fixed wide/narrow board records are reused for comparison. This
registration adds a diagnostic arm without changing any frozen source or point.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen-root',type=Path,required=True)
    ap.add_argument('--proof',type=Path,required=True)
    ap.add_argument('--remote-directory',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    prep=json.loads((args.proof/'preparation.json').read_text())
    validation=json.loads((args.proof/'validation-v2.json').read_text())
    if (validation['status']!='old-mapping-control-validated' or
        validation['preparation_sha256']!=sha(args.proof/'preparation.json')):
        raise ValueError('the actual isolated control must pass validation before registration')
    path=args.frozen_root/'spec/r41/stage1/experiments/board-gemm-tile.json'
    base=json.loads(path.read_text());spec=copy.deepcopy(base)
    spec['id']='board-old-mapping-control'
    spec['hypothesis']='Removing only projected SSA-user lookup restores K-loop ACC round trips; matched-source old/fixed costs are reported separately from the conservative 16x16 product baseline.'
    spec['competing_explanations']=['other compiler/runtime changes explain the difference',
        'the control changes narrow executable code','the old mapping does not actually reproduce per-K traffic',
        'unmatched inputs, placement, clock or timing boundaries are mistaken for a mapping effect']
    spec['expected_observations']=['all nine original GEMM points retain their frozen independent numerical checks',
        'the isolated control source differs in one file and one method only',
        'old/fixed and fixed/narrow ratios are reported independently; no general wide-tile promotion']
    spec['expected_counter_changes']=[
        'the old-mapping control restores ACC reads/writes/zeros inside wide-tile K loops',
        'useful MMA counts stay matched; narrow code remains unchanged',
        'launch and graph cycles are measured without assuming the repair wins every point']
    batch=copy.deepcopy(base['board_batches']['gemm'])
    configs=[c for c in batch['configurations'] if c.get('arm')=='candidate']
    if len(configs)!=9:raise ValueError('all nine registered wide configurations required')
    for cfg in configs:
        cfg['id']=cfg['id'].replace('@candidate','@old-mapping');cfg['arm']='old-mapping'
    batch['configurations']=configs
    batch['groups']=[[c['id']] for c in configs]
    batch['coverage']=dict(suite=base['registration']['subset'],sha256=base['registration']['subset_sha256'])
    spec['board_batches']={'old-mapping':batch}
    directory=Path(args.remote_directory)
    spec['registration']=dict(status='registered-not-run',frozen_product=prep['frozen_software_tree'],
        isolated_control=prep['control_software_tree'],preparation_sha256=sha(args.proof/'preparation.json'),
        validation_sha256=sha(args.proof/'validation-v2.json'),original_manifest_sha256=sha(path),
        source_delta='remove only _OpView.value_users from frozen compiler/python/tnca_cc/register_blocks.py',
        comparison='reuse completed frozen fixed-wide and conservative-narrow board cohorts; report numerical and identity checks plus descriptive ratios',
        sampling='unchanged frozen device-rule; qualification and formal samples remain separate',
        physical_access='only after 15 consecutive idle minutes; approved B40 image; no new bitstream')
    spec['steps']=[copy.deepcopy(base['steps'][0])]
    step=spec['steps'][0];step['id']='old-mapping'
    step['argv']=['{python}','tools/performance_harness/board_batch.py','--manifest',str(directory/'board-manifest.json'),
        '--batch','old-mapping','--root','{root}','--out','{out}/old-mapping','--python','{python}']
    step['outputs']=['{out}/old-mapping/batch.json','{out}/old-mapping/formal.jsonl']
    spec['sources'] += [str(directory/'board-manifest.json'),str(directory/'preparation.json'),str(directory/'validation-v2.json')]
    spec.pop('conclusion_source',None)
    # The old estimates refer to historical files absent in the isolated copy.
    spec['estimates']=[]
    spec['registration']['wall_time_budget_seconds']=step['timeout_seconds']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(spec,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='registered-not-run',configurations=len(configs),manifest_sha256=sha(args.out)),indent=2))


if __name__=='__main__':main()
