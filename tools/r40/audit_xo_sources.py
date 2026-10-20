#!/usr/bin/env python3
"""Match every packaged Verilog source to an immutable Git commit.

This proves source bytes only; image parameters, routed constraints, timing,
CDC and the final xclbin identity require their own evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def audit(xo, revision):
    commit = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
    sources = git('ls-tree', '-r', '--name-only', commit, 'rtl').decode().splitlines()
    lookup = {}
    for source in sources:
        if source.endswith(('.v', '.vh')):
            lookup.setdefault(Path(source).name, []).append(source)
    rows = []
    with zipfile.ZipFile(xo) as package:
        names = [name for name in package.namelist()
                 if '/src/' in name and name.endswith(('.v', '.vh'))]
        if len(names) != len(set(names)):
            raise ValueError('duplicate packaged source entries')
        for name in sorted(names):
            data = package.read(name)
            matches = [source for source in lookup.get(Path(name).name, [])
                       if git('show', commit + ':' + source) == data]
            rows.append(dict(packaged=name, sha256=hashlib.sha256(data).hexdigest(),
                             source=matches[0] if len(matches) == 1 else None,
                             matches=matches))
    if not rows:
        raise ValueError('XO has no packaged Verilog sources')
    return dict(schema='tnca-xo-source-audit/1', source_commit=commit,
                xo_sha256=hashlib.sha256(Path(xo).read_bytes()).hexdigest(),
                source_count=len(rows), passed=all(row['source'] for row in rows),
                sources=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xo', type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be new; preserve prior evidence')
    result = audit(args.xo, args.commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"XO source audit: {result['source_count']} source(s), passed={result['passed']}")
    for row in result['sources']:
        if not row['source']:
            print('unmatched or ambiguous:', row['packaged'], row['matches'])
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
