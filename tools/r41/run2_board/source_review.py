#!/usr/bin/env python3
"""Verify the staged frozen run-2 source tree against its original archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import auto

ARCHIVE_SHA = '435831da62fe43a3c0cd66c1fc29c792dc91f98390403b11111700f43057482a'
SOURCE_COMMIT = '1f0a5aff4b5d2df6ef9f3805513d37638bf5bc0d'
PATHS = ('rtl', 'tb', 'flow/fpga', 'flow/synth', 'xls', 'spec/r1_profiles.json',
         'compiler/targets', 'evidence/r41/hbm_bw/g4/run2/image.tcl')


def included(name):
    return any(name == p or name.startswith(p + '/') for p in PATHS)


def review(archive, root):
    archive, root = Path(archive), Path(root)
    if auto.sha(archive) != ARCHIVE_SHA:
        raise ValueError('frozen source archive hash mismatch')
    expected = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for item in tar:
            name = item.name.removeprefix('./').rstrip('/')
            if included(name) and item.isfile():
                if name in expected or '..' in Path(name).parts:
                    raise ValueError('ambiguous source archive member: ' + name)
                stream = tar.extractfile(item)
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                expected[name] = h.hexdigest()
    if not expected or PATHS[-1] not in expected:
        raise ValueError('frozen image sources missing from archive')
    found = {}
    for name in PATHS:
        base = root / name
        paths = [base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file())
        for path in paths:
            if path.is_symlink():
                raise ValueError('source file is a symlink: ' + str(path))
            found[str(path.relative_to(root))] = auto.sha(path)
    missing, extra = set(expected) - set(found), set(found) - set(expected)
    changed = {k for k in set(found) & set(expected) if found[k] != expected[k]}
    if missing or extra or changed:
        raise ValueError(f'frozen source mismatch: {len(missing)} missing, {len(extra)} extra, {len(changed)} changed')
    digest = hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()
    return dict(schema='tnca-run2-source-review/1', kind='source', status='pass',
                image_sha256=auto.IMAGE_SHA, source_commit=SOURCE_COMMIT,
                archive_sha256=ARCHIVE_SHA, root=str(root), paths=list(PATHS),
                files=len(found), files_sha256=digest,
                scope='extracted frozen image source only; does not identify runtime software')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(review(a.archive, a.root), indent=2)+'\n')


if __name__ == '__main__':
    main()
