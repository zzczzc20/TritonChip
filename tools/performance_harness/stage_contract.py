#!/usr/bin/env python3
"""Verify the frozen R41 stage inputs without promoting any delivery gate.

    stage_contract.py --stage 1 [--out RECORD.json] [--no-registry]

Prints, and with --out writes, one JSON record: the verification result,
the command that produced it, the product software tree
(archbench.provenance.software_tree), the git commit and dirty state, host,
interpreter and time. Nothing else is written to the record file.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGE1_LOCK_SHA256 = '58962c7e92973ad3ea4a48e94b9f159541bc863006c5bae256372d97499b1ae9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT, *, stage=1, registry_check=True):
    if stage == 2:
        from stage2_contract import verify as verify_stage2
        return verify_stage2(root, registry_check=registry_check)
    if stage != 1:
        raise ValueError('unsupported stage')
    lock_path = root/'spec/r41/stage1/lock.json'
    if sha(lock_path) != STAGE1_LOCK_SHA256:
        raise ValueError('Stage 1 lock identity changed')
    lock = json.loads(lock_path.read_text())
    for name, expected in lock['files'].items():
        if sha(root/name) != expected:
            raise ValueError(f'frozen input changed: {name}')
    plan = json.loads((root/'spec/r41/stage1/plan.json').read_text())
    if plan['contract_sha256'] != sha(root/plan['contract']):
        raise ValueError('contract identity mismatch')
    if set(plan['required']) != {f'S1-{i:02}' for i in range(6)}:
        raise ValueError('missing Stage 1 package')
    def nonnull(value):
        if value is None:
            raise ValueError('frozen plan contains an unresolved null')
        if isinstance(value, dict):
            for v in value.values():
                nonnull(v)
        elif isinstance(value, list):
            for v in value:
                nonnull(v)
    nonnull(plan)
    base = json.loads((root/plan['baseline']).read_text())
    if (base['clock_hz'] != 90000000 or base['physical_cores'] != 3 or
            base['history_status'] != 'closed-with-failures'):
        raise ValueError('B40-OBS identity/status changed')
    if registry_check:
        sys.path.insert(0, str(root))
        from archbench import registry
        inventory = {}
        for suite in plan['original_suites']:
            points = json.loads((root/suite['path']).read_text())['points']
            if len(points) != suite['points']:
                raise ValueError(f'incomplete original suite: {suite["path"]}')
            for point in points:
                if registry.get(point['id']).definition_sha256() != point['sha256']:
                    raise ValueError(f'original point changed: {point["id"]}')
                inventory[point['id']] = point['sha256']
    else:
        inventory = {}
    holdout = json.loads((root/plan['holdout']).read_text())
    seen = set()
    for point in holdout['points']:
        definition = {k: v for k, v in point.items() if k != 'sha256'}
        digest = hashlib.sha256(json.dumps(definition, sort_keys=True,
                                            separators=(',', ':')).encode()).hexdigest()
        if digest != point['sha256'] or point['id'] in seen:
            raise ValueError('holdout identity mismatch or duplicate point')
        seen.add(point['id'])
    return dict(schema='tnca-r41-freeze-check/1', stage=stage,
                status='frozen-inputs-verified', lock_sha256=sha(lock_path),
                registry_checked=registry_check, original_unique_points=len(inventory),
                required_packages=sorted(plan['required']),
                delivery_acceptance='not-measured')


def record(root=ROOT, *, stage=1, registry_check=True, command=None):
    """The verification result bound to the identity of the checked tree."""
    result = verify(root, stage=stage, registry_check=registry_check)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from archbench.provenance import git_commit, software_tree
    lock = root/f'spec/r41/stage{stage}/lock.json'
    plan = json.loads((root/f'spec/r41/stage{stage}/plan.json').read_text())
    result.update(schema='tnca-r41-freeze-check/2', command=list(command or []),
                  software_tree=software_tree(root, refresh=True), git=git_commit(root),
                  plan_sha256=sha(root/f'spec/r41/stage{stage}/plan.json'),
                  contract_sha256=plan['contract_sha256'], lock_files=len(json.loads(lock.read_text())['files']),
                  host=socket.gethostname(), python=sys.executable, python_version=sys.version.split()[0],
                  checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--out', type=Path, help='write the JSON record here (parents created)')
    ap.add_argument('--no-registry', action='store_true',
                    help='skip the point-definition registry check (needs torch); recorded as registry_checked=false')
    args = ap.parse_args()
    result = record(stage=args.stage, registry_check=not args.no_registry,
                    command=[sys.executable, *sys.argv])
    text = json.dumps(result, indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(text)
        temporary.replace(args.out)
    print(text, end='')


if __name__ == '__main__':
    main()
