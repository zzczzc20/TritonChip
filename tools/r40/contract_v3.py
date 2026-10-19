"""Verify the one-time frozen R40 v3 contract before judging evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = 'doc/r40_v3/lock.json'
LOCK_SHA256 = '0963f50e5ba7b569d87ac636f7d31df748ef967609e981d0fff884f759abe211'
FILES = frozenset(('doc/r40_tasks_and_acceptance_v3.md', 'doc/r40_v3/plan.json',
                   'doc/r40_v3/R40-09.model.json'))


def verify(root=ROOT, *, expected=None):
    expected = expected or LOCK_SHA256
    raw = (root / LOCK).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('R40 v3 lock changed; the v3 relaxation is one-time and frozen')
    lock = json.loads(raw)
    if set(lock['files']) != FILES:
        raise ValueError('R40 v3 lock does not cover the complete contract')
    for path, digest in {**lock['files'], **lock['inherited']}.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen R40 v3 contract changed: {path}')
    # v3 inherits v2 whole; the v2 lock and its dependencies must still hold.
    import contract_v2
    contract_v2.verify(root)
    return lock


if __name__ == '__main__':
    verify()
    print('R40 v3 immutable contract: PASS')
