"""Verify the one-time frozen R40 v2 contract before judging evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = 'spec/r40/v2/lock.json'
LOCK_SHA256 = '49fc0e02733139707ae2292eff2ef893e76e73f0723d0d002b2e7e1864af71c8'
FILES = frozenset(('doc/r40_tasks_and_acceptance_v2.md',
                   'spec/r40/v2/plan.json', 'spec/r40/v2/ablation_matrix.json',
                   'spec/r40/v2/rationale.json', 'spec/r40/v2/inherited_v1.md.gz'))


def verify(root=ROOT, *, expected=LOCK_SHA256):
    raw = (root / LOCK).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('R40 v2 lock changed; the user forbids further contract amendments')
    lock = json.loads(raw)
    if set(lock['files']) != FILES:
        raise ValueError('R40 v2 lock does not cover the complete contract')
    for path, digest in lock['files'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen R40 v2 contract changed: {path}')
    for path, digest in lock.get('dependencies', {}).items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen R40 workload/baseline changed: {path}')
    return lock


if __name__ == '__main__':
    verify()
    print('R40 v2 immutable contract: PASS')
