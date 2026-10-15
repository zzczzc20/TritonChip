#!/usr/bin/env python3
"""Judge three matched CPU/native board blocks under the immutable v2 contract.

Each --control/--native input is one complete 77-point JSONL block. Collection
must interleave A/B per point in the order fixed by the protocol bootstrap seed.
This capability runner never substitutes for native resident functional testing.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import ablate
import contract_v2
import evidence


def read_block(path, subset, exp, identity):
    records = ablate.load(path)
    ids = [r['point_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(subset):
        raise ValueError('every block must contain the complete frozen inventory exactly once')
    out = {}
    for r in records:
        pid = r['point_id']
        p = r.get('platform', {})
        knobs = p.get('knobs', {})
        if p.get('backend') != 'fpga-u280' or r.get('status') != 'pass' or r.get('fallbacks'):
            raise ValueError(f'{pid}: requires passing actual U280 execution without fallback')
        if knobs.get('TNCA_TORCH_DEVICE', 'cpu') != identity:
            raise ValueError(f'{pid}: wrong device identity')
        if knobs.get('TNCA_TRANSFER_POLICY') != 'per-kernel':
            raise ValueError(f'{pid}: both controls must explicitly select per-kernel staging')
        _, timing = ablate.timing_entry(r, exp['timing'])
        _, samples = ablate.samples_of(r, exp['timing'])
        # Long-task exception needs predeclared evidence; this runner deliberately
        # accepts the ordinary >=30-sample plan only until that path is supplied.
        if timing.get('warmup', 0) < 5 or len(samples) < 30 or timing.get('deterministic', True):
            raise ValueError(f'{pid}: requires >=5 warmup and >=30 actual board samples')
        m = r.get('measurement', {})
        if not all(m.get(k) for k in ('block_id', 'process_session', 'boot_id', 'started_ns', 'ended_ns')):
            raise ValueError(f'{pid}: lacks independent measurement identity/order')
        if not isinstance(m['started_ns'], int) or not isinstance(m['ended_ns'], int) or m['started_ns'] >= m['ended_ns']:
            raise ValueError(f'{pid}: invalid measurement interval')
        out[pid] = r
    return out


def judge(control_paths, native_paths):
    contract_v2.verify()
    plan = json.loads((contract_v2.ROOT/'spec/r40/v2/plan.json').read_text())
    exp = plan['experiments']['R40-01.device-time']
    subset, _ = ablate.load_subset(contract_v2.ROOT/exp['subset'])
    if len(control_paths) != 3 or len(native_paths) != 3:
        raise ValueError('three independent paired blocks are required')
    rng = random.Random(plan['sample_plan']['bootstrap']['seed'])
    configurations = set()
    seen_blocks = set()
    sessions = {pid:set() for pid in subset}
    pairs = []
    for ap, bp in zip(control_paths, native_paths):
        a, b = read_block(ap, subset, exp, 'cpu'), read_block(bp, subset, exp, 'native')
        block_ids = {r['measurement']['block_id'] for side in (a,b) for r in side.values()}
        if len(block_ids) != 1 or seen_blocks & block_ids:
            raise ValueError('each A/B pair needs one distinct block ID')
        seen_blocks |= block_ids
        for pid in sorted(subset):
            ma, mb = a[pid]['measurement'], b[pid]['measurement']
            if ma['boot_id'] != mb['boot_id']:
                raise ValueError(f'{pid}: order clocks are not from the same host boot')
            if ma['process_session'] == mb['process_session']:
                raise ValueError(f'{pid}: A/B must use independently configured processes')
            new_sessions = {ma['process_session'],mb['process_session']}
            if sessions[pid] & new_sessions:
                raise ValueError(f'{pid}: process/session reused across blocks')
            sessions[pid] |= new_sessions
            first, second = (ma,mb) if rng.randrange(2)==0 else (mb,ma)
            if first['ended_ns'] > second['started_ns']:
                raise ValueError(f'{pid}: A/B does not follow the preregistered random order')
        result = ablate.compare(ap,bp,subset,exp['vary'],timing=exp['timing'],threshold=exp['threshold'])
        configurations.add((result['config_a'],result['config_b']))
        pairs.append((a,b))
    if len(configurations) != 1:
        raise ValueError('configuration changed across measurement blocks')
    # Keep original blocks as input evidence; pool equally sized ordinary blocks
    # only after every block has passed the independent provenance/config audit.
    merged = []
    for side_index in (0,1):
        side = []
        for pid in subset:
            r = copy.deepcopy(pairs[0][side_index][pid])
            samples = []
            for pair in pairs:
                samples.extend(ablate.samples_of(pair[side_index][pid],exp['timing'])[1])
            if len({len(ablate.samples_of(pair[side_index][pid],exp['timing'])[1]) for pair in pairs}) != 1:
                raise ValueError(f'{pid}: blocks must have equal sample counts')
            _, timing = ablate.timing_entry(r,exp['timing'])
            timing['samples'] = samples
            side.append(r)
        merged.append(side)
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory)/name for name in ('a.jsonl','b.jsonl')]
        for path,records in zip(paths,merged):
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        result = ablate.compare(*paths,subset,exp['vary'],timing=exp['timing'],threshold=exp['threshold'])
    checks = [dict(name='complete matched independent three-block measurement',ok=True),
              dict(name='paired CI lower >= 1/1.05',ok=result['ci95'][0]>=exp['threshold']),
              dict(name='every point <=5% slowdown',ok=all(p['speedup']>=exp['threshold'] for p in result['points'])),
              dict(name='no uncorrected >2% regression',ok=not result['unexplained_regress_2pct'])]
    return result, checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control',type=Path,nargs=3,required=True)
    ap.add_argument('--native',type=Path,nargs=3,required=True)
    ap.add_argument('--json',type=Path,required=True)
    ap.add_argument('--evidence',action='store_true')
    args=ap.parse_args()
    if args.json.exists():ap.error('preserve existing reports; --json must be new')
    try:
        result,checks=judge(args.control,args.native)
    except (ValueError,KeyError,OSError,ablate.Refused) as error:
        result={};checks=[dict(name='matched measurement accepted',ok=False,why=str(error))]
    status='capability-pass' if all(c['ok'] for c in checks) else 'fail'
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.control+args.native if p.exists()}
    report=dict(status=status,checks=checks,comparison=result,inputs=inputs,contract_sha256=contract_v2.LOCK_SHA256)
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    if args.evidence:
        os.environ['R40_CONTRACT']='v2'
        evidence.record('R40-01.device-time',status,checks,comparison=result,inputs=inputs)
    print('R40-01.device-time v2:',status)
    return 0 if status=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
