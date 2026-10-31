#!/usr/bin/env python3
"""Summarize completed MMIO grids from the optional hybrid RTL observer."""
import argparse
import json
from pathlib import Path
import re

DOORBELL = re.compile(r'MMIO DOORBELL cycle=(\d+) seq=(\d+) op=(\d+)')
WATCH = re.compile(r'HYBRID WATCH cycle=(\d+) core=(\d+) .*fast=(\d+) slow=(\d+) retired=(\d+) fast_quiet=(\d) slow_idle=(\d) cpl=(\d) ack=(\d)')


def summarize(path, cores=3):
    rows, current, prior = [], None, {c:(0,0,0) for c in range(cores)}
    def finish():
        nonlocal current, prior
        if current is None:
            return
        groups = current.pop('samples')
        complete = [cycle for cycle, samples in groups.items() if len(samples)==cores]
        if not complete:
            raise ValueError('grid has no complete per-core observation')
        cycle = max(complete)
        counts = groups[cycle]
        delta = {c:tuple(x-y for x,y in zip(counts[c], prior[c])) for c in range(cores)}
        if any(min(v)<0 or v[0]+v[1]!=v[2] for v in delta.values()):
            raise ValueError('grid did not retire every admitted request')
        current.update(observation_cycle=cycle, per_core=[dict(core=c, fast=v[0], slow=v[1], retired=v[2]) for c,v in sorted(delta.items())])
        rows.append(current)
        prior, current = counts, None
    for line in path.read_text().splitlines():
        match = DOORBELL.search(line)
        if match:
            cycle, seq, op = map(int, match.groups())
            finish()
            if op==11:
                current=dict(sequence=seq, request_cycle=cycle, samples={})
        match = WATCH.search(line)
        if match and current is not None:
            cycle, core, fast, slow, retired, quiet, idle, cpl, ack = map(int,match.groups())
            # Only a fully drained snapshot may close a grid. Periodic busy
            # observations are diagnostics, not completion evidence.
            if quiet and idle and not cpl:
                current['samples'].setdefault(cycle,{})[core]=(fast,slow,retired)
    finish()
    return dict(evidence='RTL-observation-only', operator_labels_available=False,
        board_speedup_measured=False, grids=rows,
        fast=sum(c['fast'] for r in rows for c in r['per_core']),
        slow=sum(c['slow'] for r in rows for c in r['per_core']),
        retired=sum(c['retired'] for r in rows for c in r['per_core']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.trace)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='grids'},sort_keys=True))
