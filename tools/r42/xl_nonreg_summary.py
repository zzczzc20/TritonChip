#!/usr/bin/env python3
"""Summarise a compile-only sweep of the tiny and small tiers (records.jsonl from archbench.run --compile-only).

    xl_nonreg_summary.py RECORDS.jsonl OUT.json

Checks the one change that touches the original points, the launch counter of archbench/platforms/tnca.py
(`_wrapper_launches` also counts user-defined Triton kernels): a compile-only record of an original point is
`not-measured` (every kernel built, none launched); `runtime-fault` with "launches for ... in the wrapper" would mean the
count now disagrees with the launches. Per point: status, launches, expected launches.
"""
import json
import sys
from pathlib import Path


def main():
    recs = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    rows = [dict(point=r['point_id'], status=r['status'], launches=(r['host'] or {}).get('launches'),
                 expected_launches=(r['host'] or {}).get('expected_launches'), problems=r['problems'][:2]) for r in recs]
    out = dict(schema='archbench-xl-nonreg/1', domain='compile-only (TNCA_LAUNCH=skip), profile core-vrf-sync-v5-x8-g2c4-dp, CPU-tensor path',
               total=len(rows), not_measured=sum(r['status'] == 'not-measured' for r in rows),
               other=sum(r['status'] != 'not-measured' for r in rows),
               launch_mismatch=sum(r['launches'] != r['expected_launches'] for r in rows), rows=rows)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
    print({k: v for k, v in out.items() if k != 'rows'})
    for r in rows:
        if r['status'] != 'not-measured' or r['launches'] != r['expected_launches']:
            print(' ', r)


if __name__ == '__main__':
    main()
