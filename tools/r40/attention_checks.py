#!/usr/bin/env python3
"""Run R40-08 functional coverage; do not issue a performance verdict."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {'attention_decode', 'attention_prefill', 'llama_decode',
            'llama_prefill', 'gpt2_block'}


def inventory():
    suite = json.loads((ROOT / 'spec/r40/suites/existing.json').read_text())
    points = [p for p in suite['points'] if p['family'] in FAMILIES]
    if {p['family'] for p in points} != FAMILIES or len({p['id'] for p in points}) != len(points):
        raise ValueError('incomplete or duplicate frozen attention inventory')
    return points


def commands(out, profile):
    points = inventory()
    return {
        'inventory': [sys.executable, '-m', 'archbench.run', '--platform', 'tnca',
                      '--profile', profile, '--point', *[p['id'] for p in points],
                      '--set', 'TNCA_TORCH_DEVICE=native', '--set', 'TNCA_RCTX_ROLES=0',
                      '--warmup', '1', '--timed', '1', '--jobs', '1',
                      '--out', str(out / 'inventory.jsonl')],
        'online': [sys.executable, 'tools/r40/online_attention_tests.py',
                   '--out', str(out / 'online.json')],
        'decode': [sys.executable, 'tools/r40/decode_residency_tests.py', '--steps', '64', '--bucket'],
    }


def validate_inventory(path):
    expected = {p['id']: p['sha256'] for p in inventory()}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != len(expected) or {r['point_id'] for r in records} != expected.keys():
        raise ValueError('missing, duplicate or unexpected frozen point result')
    for record in records:
        if (record['status'] != 'pass' or record['point_sha256'] != expected[record['point_id']]
                or record.get('fallbacks') != []
                or record['platform']['knobs'].get('TNCA_TORCH_DEVICE') != 'native'
                or record['platform']['inductor'].get('tensor_descriptors') is not True
                or record['host'].get('upload_bytes') != 0
                or record['host'].get('download_bytes') != 0):
            raise ValueError(f"invalid native attention record: {record['point_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--profile', default=os.environ.get('TNCA_PROFILE', 'core-vrf-sync-v5/hbm'))
    parser.add_argument('--phases', nargs='+', choices=('inventory', 'online', 'decode'),
                        default=['inventory', 'online', 'decode'])
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if len(set(args.phases)) != len(args.phases):
        parser.error('duplicate phases')
    if not args.list and (args.out is None or args.out.exists()):
        parser.error('--out must name a new evidence directory')
    out = (args.out or ROOT / 'sim/build/r40/attention-preview').resolve()
    jobs = commands(out, args.profile)
    if args.list:
        print(json.dumps({phase: jobs[phase] for phase in args.phases}, indent=2))
        return 0
    out.mkdir(parents=True)
    env = dict(os.environ, TNCA_PROFILE=args.profile, CUDA_VISIBLE_DEVICES='')
    env['PYTHONPATH'] = ':'.join(str(ROOT / p) for p in ('python', 'compiler/python', 'tools/asm'))
    status = dict(schema='tnca-r40-attention-checks/1', measured_performance=False,
                  all_phases_run=set(args.phases) == {'inventory', 'online', 'decode'},
                  profile=args.profile, frozen_points=inventory(), phases=[])
    for phase in args.phases:
        print(f'R40 attention functional phase: {phase}', flush=True)
        with (out / f'{phase}.log').open('w') as log:
            rc = subprocess.run(jobs[phase], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        reason = None
        if rc == 0 and phase == 'inventory':
            try:
                validate_inventory(out / 'inventory.jsonl')
            except (ValueError, KeyError, TypeError, OSError) as exc:
                rc, reason = 1, str(exc)
        status['phases'].append(dict(phase=phase, command=jobs[phase], returncode=rc, reason=reason))
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        if rc:
            print(f'{phase} failed ({rc}); see {out / (phase + ".log")}', flush=True)
            return 1
    print('R40 attention functional checks finished; board/traffic/performance gates remain separate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
