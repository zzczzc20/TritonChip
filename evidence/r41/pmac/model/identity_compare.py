#!/usr/bin/env python3
"""Default-identity check: compare two validation output trees file by file.

    identity_compare.py BASE_DIR MOD_DIR --sub OLD=NEW ... --out report.json

Every file must be byte-identical after replacing the two binaries' SHA256s
(--sub), except three named classes, each compared by presence only:
  host-time     JSON keys that carry host wall time (WALL_KEYS, the
                trace_overhead subtree of validate.py)
  binary-hash   /output_sha256/identity.json: the hash of identity.json, which
                itself must be identical after the binary-SHA substitution
  rtl-compile   iverilog *.vvp images of the RTL capture testbench (ASLR
                scope addresses); not a model output
Any other difference fails the check.
"""
import argparse
import hashlib
import json
from pathlib import Path

WALL_KEYS = {'wall_s', 'host_seconds', 'elapsed_s', 'host_s', 'host_overhead', 'wall', 'seconds',
             'process_wall_s', 'host_wall_s', 'overhead', 'overhead_range', 'traced_s', 'untraced_s',
             'host_elapsed_seconds', 'build_seconds', 'execution_seconds'}
HOST_SUBTREES = ('/results/trace_overhead/',)
BINARY_DERIVED = {'/output_sha256/identity.json'}


def diff_json(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f'{path}/{k}: only in {"mod" if k not in a else "base"}')
            else:
                out += diff_json(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_json(x, y, f'{path}[{i}]')
    elif a != b:
        out.append(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base', type=Path)
    ap.add_argument('mod', type=Path)
    ap.add_argument('--sub', action='append', default=[])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    subs = [s.split('=', 1) for s in a.sub]
    files = sorted({p.relative_to(a.base) for p in a.base.rglob('*') if p.is_file()} |
                   {p.relative_to(a.mod) for p in a.mod.rglob('*') if p.is_file()})
    rows, failures = {}, []
    for rel in files:
        pa, pb = a.base/rel, a.mod/rel
        if not pa.exists() or not pb.exists():
            rows[str(rel)] = 'missing in ' + ('base' if not pa.exists() else 'mod')
            failures.append(str(rel))
            continue
        ba, bb = pa.read_bytes(), pb.read_bytes()
        if ba == bb:
            rows[str(rel)] = 'identical'
            continue
        if rel.suffix == '.vvp':
            rows[str(rel)] = 'rtl-compile-only'
            continue
        ta, tb = ba.decode(errors='replace'), bb.decode(errors='replace')
        for old, new in subs:
            ta, tb = ta.replace(old, new), tb.replace(old, new)
        if ta == tb:
            rows[str(rel)] = 'identical-after-substitution'
            continue
        try:
            ja, jb = json.loads(ta), json.loads(tb)
        except Exception:                                    # noqa: BLE001
            rows[str(rel)] = 'DIFFERENT (non-JSON)'
            failures.append(str(rel))
            continue
        paths = diff_json(ja, jb)
        wall_only = [p for p in paths if p.rsplit('/', 1)[-1].split('[')[0] in WALL_KEYS
                     or p.startswith(HOST_SUBTREES)]
        derived = [p for p in paths if p in BINARY_DERIVED and
                   rows.get(str(rel.parent/'identity.json'), '').startswith('identical')]
        other = [p for p in paths if p not in wall_only and p not in derived]
        rows[str(rel)] = ('identical-except ' + ','.join(wall_only + derived)) if not other else \
            ('DIFFERENT ' + ', '.join(other[:20]))
        if other:
            failures.append(str(rel))
    report = dict(base=str(a.base), mod=str(a.mod), files=len(files), failures=failures,
                  counts={k: sum(1 for v in rows.values() if v.split(' ')[0] == k) for k in
                          ('identical', 'identical-after-substitution', 'identical-except', 'rtl-compile-only')},
                  status='identical' if not failures else 'DIFFERENT', rows=rows)
    a.out.write_text(json.dumps(report, indent=1) + '\n')
    print(report['status'], report['counts'], 'failures:', failures[:10])


if __name__ == '__main__':
    main()
