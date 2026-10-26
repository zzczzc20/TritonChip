#!/usr/bin/env python3
"""Compare two compile-preflight runs (tools/r40/v3_compile_preflight.py) file by file.

    v3_identity_diff.py --before RUN_A --after RUN_B --json OUT

For every (point, configuration) present in both runs: the Inductor wrapper
and every kernel dump file (TTIR, TNCA IR, assembly, artifact hex, metadata).
Run-local paths (each run's own temporary/cache directories) are normalised
before comparing; any other byte difference is reported. Used to show which
generated code a product change touches, not as a substitute for board runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def files(run):
    out = {}
    w = run/'wrapper_0.py'
    if w.exists():
        out['wrapper_0.py'] = w
    dump = run/'dump'
    if dump.is_dir():
        for f in sorted(dump.rglob('*')):
            if f.is_file():
                out[str(f.relative_to(run))] = f
    return out


def normal(path, roots):
    raw = path.read_bytes()
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        return raw
    for root in roots:
        text = text.replace(str(root), '<RUN>')
    text = re.sub(r'/tmp/[^\s\'"]+', '<TMP>', text)
    return text.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--json', type=Path, required=True)
    a = p.parse_args()
    ra, rb = a.before/'runs', a.after/'runs'
    names = sorted({d.name for d in ra.iterdir()} | {d.name for d in rb.iterdir()})
    report = dict(before=str(a.before), after=str(a.after), runs={})
    for name in names:
        da, db = ra/name, rb/name
        if not da.is_dir() or not db.is_dir():
            report['runs'][name] = dict(status='only-' + ('after' if db.is_dir() else 'before'))
            continue
        fa, fb = files(da), files(db)
        diff = sorted(k for k in set(fa) | set(fb)
                      if k not in fa or k not in fb or normal(fa[k], (da,)) != normal(fb[k], (db,)))
        ok_a = json.loads((da/'result.json').read_text()).get('status') if (da/'result.json').exists() else None
        ok_b = json.loads((db/'result.json').read_text()).get('status') if (db/'result.json').exists() else None
        report['runs'][name] = dict(status='identical' if not diff and fa else ('differs' if diff else 'empty'),
                                    before=ok_a, after=ok_b, files=len(fb), differing=diff[:50],
                                    artifacts={k: hashlib.sha256(fb[k].read_bytes()).hexdigest()
                                               for k in fb if k.endswith('.hex')})
    counts = {}
    for r in report['runs'].values():
        counts[r['status']] = counts.get(r['status'], 0) + 1
    report['summary'] = counts
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(counts))
    for name, r in report['runs'].items():
        if r['status'] != 'identical':
            print(r['status'], name, r.get('before'), '->', r.get('after'), r.get('differing', [])[:4])


if __name__ == '__main__':
    main()
