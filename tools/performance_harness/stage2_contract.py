"""Validate prospective Stage 2 inputs without rewriting historical failures."""
import hashlib
import json
from pathlib import Path

# Set to the reviewed prospective lock when numerical cross-validation completes.
LOCK_SHA256 = '5d9e4f6d41094c3492672229d0af5425425b5d9688aee276687a9be62c943f96'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def point_digest(point):
    return hashlib.sha256(json.dumps({k:v for k,v in point.items() if k!='sha256'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()

def verify(root,registry_check=True):
    lock_path=root/'spec/r41/stage2/lock.json'
    if not lock_path.is_file():raise ValueError('Stage 2 prospective lock awaits independent reference validation')
    if digest(lock_path)!=LOCK_SHA256:raise ValueError('Stage 2 prospective lock identity changed')
    lock=json.loads(lock_path.read_text())
    if lock['freeze_kind']!='prospective' or not lock['historical_failure_preserved']:
        raise ValueError('historical freeze failure must remain preserved')
    for name,expected in lock['files'].items():
        if digest(root/name)!=expected:raise ValueError(f'frozen input changed: {name}')
    plan=json.loads((root/'spec/r41/stage2/plan.json').read_text())
    if digest(root/plan['contract'])!=plan['contract_sha256']:raise ValueError('contract identity mismatch')
    def nonnull(x):
        if x is None:raise ValueError('unresolved null in prospective freeze')
        if isinstance(x,dict):
            for v in x.values():nonnull(v)
        if isinstance(x,list):
            for v in x:nonnull(v)
    nonnull(plan)
    model=json.loads((root/plan['model']['lock']).read_text())
    if model['layers']!=24 or model['vocabulary']!=248320 or model['backbone']!='bfloat16' or model['recurrent_state']!='float32':
        raise ValueError('authentic full model scope changed')
    if model['revision']!='15852e8c16360a2fea060d615a32b45270f8a8fc':raise ValueError('model revision changed')
    calibration=json.loads((root/model['numerical_cross_validation']).read_text())
    if calibration['status']!='passed' or calibration['tnca_outputs_used']:
        raise ValueError('independent numerical cross-validation missing')
    for name in ('spec/r41/stage2/model_points.json','spec/r41/stage2/xl_core.json'):
        points=json.loads((root/name).read_text())['points']
        if len({p['id'] for p in points})!=len(points) or any(point_digest(p)!=p['sha256'] for p in points):
            raise ValueError('point hash mismatch or duplicate')
        for p in points:nonnull(p)
    model_points=json.loads((root/'spec/r41/stage2/model_points.json').read_text())['points']
    if [(p['id'],p['prefill'],p['decode']) for p in model_points]!=[('Q41-core',128,16),('Q41-tail',129,4)]:
        raise ValueError('core model point scope changed')
    for p in model_points:
        if p['layers']!=24 or len(p['input_ids'])!=p['prefill']+p['decode']:raise ValueError('truncated core model')
    xl=json.loads((root/'spec/r41/stage2/xl_core.json').read_text())['points']
    if len(xl)!=8:raise ValueError('eight XL-core points required')
    if plan['budgets']['full_image_runs_total']!=5 or plan['budgets']['image_builds_started_by_acceptance']!=0:
        raise ValueError('image budget/launch authority changed')
    inventory={}
    for suite in plan['original_suites']:
        if digest(root/suite['path'])!=suite['sha256']:raise ValueError('original suite changed')
        doc=json.loads((root/suite['path']).read_text())
        if len(doc['points'])!=suite['points']:raise ValueError('original inventory incomplete')
        for p in doc['points']:inventory[p['id']]=p['sha256']
    if registry_check:
        from archbench import registry
        for name,sha in inventory.items():
            if registry.get(name).definition_sha256()!=sha:raise ValueError('original registry changed')
    return dict(schema='tnca-r41-freeze-check/1',stage=2,status='frozen-inputs-verified',
                freeze_kind='prospective',historical_failure_preserved=True,
                lock_sha256=digest(lock_path),registry_checked=registry_check,
                original_unique_points=len(inventory),required_packages=[f'S2-{i:02}' for i in range(7)],
                delivery_acceptance='not-measured')
