#!/usr/bin/env python3
"""Summarise an xl_compile_only.sh run (records, bundles, done.txt) into summary.json; nothing typed in.

    xl_compile_summary.py OUT_DIR [--out summary.json]

Per point: the record's status and problems, cold compile seconds (the record's host.wall_s: Dynamo, Inductor, Triton
and the tnca lowering of every kernel, the first call minus nothing -- it is the first call), launches in the call and in
the generated wrapper, the number of distinct compiled kernels (directories of the bundle's kernel dump), their names,
the largest launch's arguments. `verdict` is `compiles` when the status is `capacity-exceeded` for no reason other than
the CPU-tensor path's 384 KiB staging heap (every launch of a large point exceeds it by construction), the launch count
equals the wrapper's, and there is no host fallback; anything else keeps its own status.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    rows = []
    for f in sorted((a.out_dir / 'records').glob('*.jsonl')):
        lines = f.read_text().splitlines()
        if not lines:
            continue
        r = json.loads(lines[-1])
        h = r['host'] or {}
        bundle = a.out_dir / 'bundles' / r['point_id'].replace('/', '__') / 'kernels'
        kernels = sorted(d.name for d in bundle.iterdir()) if bundle.is_dir() else []
        asm, matrix, asm_sha = {}, collections.Counter(), {}
        for k in kernels:
            for s in (bundle / k).glob('*.s'):
                text = s.read_text()
                asm_sha[k] = hashlib.sha256(text.encode()).hexdigest()
                asm[k] = sum(1 for l in text.splitlines() if l.startswith('    ') and not l.strip().startswith('//'))
                # static matrix commands in the assembly (a loop body counts once): opcode, K step, A format
                matrix.update(re.findall(r'm\.mma\.(?:dp\.)?async\.k\d+\.\w+', text))
        other = [p for p in r['problems'] if not p.startswith('compile-only') and not p.startswith('the largest launch needs')]
        ok = (r['status'] in ('capacity-exceeded', 'not-measured') and not other and not r['fallbacks']
              and h.get('launches') == h.get('expected_launches') and h.get('launches', 0) > 0)
        rows.append(dict(point=r['point_id'], point_sha256=r['point_sha256'], status=r['status'],
                         verdict='compiles' if ok else r['status'], compile_s=h.get('wall_s'), process_s=h.get('process_s'),
                         launches=h.get('launches'), expected_launches=h.get('expected_launches'),
                         kernels=len(kernels), kernel_names=kernels, instructions=asm, assembly_sha256=asm_sha, static_matrix_commands=dict(matrix),
                         max_launch_bytes=h.get('max_launch_bytes'), max_launch_kernel=h.get('max_launch_kernel'),
                         fallbacks=r['fallbacks'], problems=r['problems'], profile=r['platform'].get('profile'),
                         software_tree=r['platform'].get('software_tree')))
    out = dict(schema='archbench-xl-compile-only/1', profile='core-vrf-sync-v5-x8-g2c4-dp',
               domain='compile coverage on the CPU-tensor path (TNCA_LAUNCH=skip); no kernel launched, no device time',
               points=rows, compiles=sum(r['verdict'] == 'compiles' for r in rows), total=len(rows))
    dst = a.out or a.out_dir / 'summary.json'
    dst.write_text(json.dumps(out, indent=1) + '\n')
    print(f'{out["compiles"]}/{out["total"]} compile -> {dst}')
    for r in rows:
        print(f'  {r["verdict"]:18s} {r["point"]:58s} {r["compile_s"]}s launches {r["launches"]}/{r["expected_launches"]} kernels {r["kernels"]}')


if __name__ == '__main__':
    main()
