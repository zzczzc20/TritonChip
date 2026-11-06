#!/usr/bin/env python3
"""Compare actual paired board records; preserve numerical/performance failures."""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import stage1_acceptance
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','manifest','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.manifest.read_text())
    definitions={point['id']:point['sha256'] for suite in spec['suites'] for point in json.loads((ROOT/suite['path']).read_text())['points']}
    def read(path):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        grouped={};ctx=SimpleNamespace(board=dict(spec['board'],clock_evidence_sha256='49868f45b5ced2f662e4f76bbb1bb4a2be166625915d869f20812afcee42f0d7'),definitions=definitions,historical_trees=set())
        for row in rows:
            assert not stage1_acceptance.board_record_problems(row,ctx,{}),row['point_id']
            tile=row['platform']['knobs']['TNCA_GEMM_TILE']
            grouped.setdefault((row['point_id'],tile),[]).append(row)
        return grouped
    before,after=read(a.before),read(a.after)
    points={p for p,tile in before};assert len(points)==9
    assert {(p,t) for p in points for t in ('16,16,16','16,32,16')}==set(after)
    def median(rows):
        return statistics.median(v for r in rows for v in r['timings']['device-graph-makespan']['samples'])
    rows=[]
    for point in sorted(points):
        old=median(before[point,'16,32,16']);narrow=median(after[point,'16,16,16']);wide=median(after[point,'16,32,16'])
        rows.append(dict(point=point,pre_fix_graph_cycles=old,narrow_graph_cycles=narrow,repaired_wide_graph_cycles=wide,
                         repaired_speedup_vs_pre_fix=old/wide,repaired_wide_over_narrow=wide/narrow,no_slower=wide<=narrow))
    passed=all(r['no_slower'] for r in rows)
    result=dict(result='pass' if passed else 'fail',residual='paired board measurements, not model predictions',
        conclusion=f"{sum(r['no_slower'] for r in rows)}/9 repaired wide configurations no slower than legal N16; all numerical records verified",
        next_step='retain conservative/default selection for losing cases; do not erase prior failed campaigns',rows=rows)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    # A negative study is complete but is not a performance pass.
if __name__=='__main__':main()
