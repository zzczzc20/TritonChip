#!/usr/bin/env python3
"""Inventory tracked evidence and likely textual references, including sparse trees.

This is a conservative discovery aid, not proof that a file is safe to remove:
dynamic paths, generated names and external readers still need manual review.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

TOKEN = re.compile(r'[A-Za-z0-9_~./*+-]+')
TEXT_LIMIT = 4_000_000


def tree():
    data = subprocess.check_output(['git', 'ls-tree', '-rlz', 'HEAD'])
    result = {}
    for record in data.split(b'\0'):
        if not record:
            continue
        metadata, name = record.split(b'\t', 1)
        mode, kind, oid, size = metadata.split()
        if kind == b'blob':
            result[name.decode('utf-8', 'surrogateescape')] = (oid.decode(), int(size))
    return result


def category(path):
    if path == 'doc/docmap.toml':
        return 'docmap'
    if path.startswith('evidence/r41/acceptance/') or path.endswith(('judgment.json', 'stage2_required_index.json', 'required_index.json')):
        return 'acceptance'
    if path in ('tools/performance_harness/stage1_acceptance.py', 'tools/performance_harness/stage2_acceptance.py'):
        return 'acceptance'
    if path.startswith('doc/r41') and path.endswith('.md'):
        return 'report'
    if re.search(r'(^|/)(test[^/]*|[^/]*_tests?\.[^/]*)$', path):
        return 'test'
    if PurePosixPath(path).name.lower() in ('makefile', 'readme.md', 'readme'):
        return 'makefile/readme'
    if path.startswith(('tools/', 'python/', 'compiler/', 'archbench/')) and path.endswith(('.py', '.sh', '.tcl')):
        return 'tool'
    return None


def resolve(source, token):
    # Markdown links and paths in JSON commonly use source-relative names.
    from posixpath import normpath
    return normpath(str(PurePosixPath(source).parent / token))


def inventory(files):
    targets = {path for path, (_, size) in files.items()
               if path.startswith(('evidence/', 'spec/r41/')) or size > 1_000_000}
    by_basename = defaultdict(set)
    for path in targets:
        by_basename[PurePosixPath(path).name].add(path)
    refs = defaultdict(lambda: defaultdict(set))
    sources = [(path, oid, category(path)) for path, (oid, size) in files.items()
               if category(path) and size <= TEXT_LIMIT]
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    try:
        for source, oid, kind in sources:
            proc.stdin.write((oid + '\n').encode())
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError(f'cannot read {source}')
            content = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
            if b'\0' in content:
                continue
            for token in set(TOKEN.findall(content.decode('utf-8', 'replace'))):
                clean = token.strip('./')
                paths = set()
                for candidate in (token, clean, resolve(source, token)):
                    if candidate in targets:
                        paths.add(candidate)
                for path in paths:
                    if path != source:
                        refs[path][kind].add(source)
                # Ambiguous basename-only mentions are reported separately.
                basename_paths = by_basename.get(PurePosixPath(token).name, ())
                for path in basename_paths:
                    if path != source and path not in paths:
                        refs[path][kind + ':basename'].add(source)
                # A specific directory reference may cover files named at runtime.
                if token.endswith('/') and token not in ('evidence/', 'spec/r41/'):
                    prefix = resolve(source, token) if token.startswith('.') else token.rstrip('/')
                    for path in targets:
                        if path != source and path.startswith(prefix.rstrip('/') + '/'):
                            refs[path][kind + ':directory'].add(source)
    finally:
        proc.stdin.close()
        proc.stdout.close()
        if proc.wait():
            raise RuntimeError('git cat-file failed')
    return [dict(path=p, bytes=files[p][1],
                 references={kind: sorted(names) for kind, names in sorted(refs[p].items())},
                 unreferenced=not bool(refs[p])) for p in sorted(targets)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='emit detailed JSON')
    args = parser.parse_args()
    rows = inventory(tree())
    if args.json:
        json.dump(rows, sys.stdout, indent=2, ensure_ascii=True)
        print()
    else:
        for row in rows:
            refs = ', '.join(f'{k}:{len(v)}' for k, v in row['references'].items()) or 'no textual references found'
            print(f"{row['bytes']:>10}  {row['path']}  {refs}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
