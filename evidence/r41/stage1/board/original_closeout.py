#!/usr/bin/env python3
"""Recompute the frozen original-suite validation and B40-OBS comparison locally."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
FROZEN = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/r41s1-frozen-25094197f6e6')
sys.path.insert(0, str(FROZEN / 'tools/performance_harness'))
from stage1_acceptance import Context, board_record_problems  # noqa: E402

BOARD = Path(__file__).resolve().parent
ctx = Context(FROZEN, ROOT / 'evidence/r41/stage1/required_index.json')
source = BOARD / 'out/board-original-suite/original'
rows = [json.loads(l) for l in (source / 'formal.jsonl').read_text().splitlines()]
assert len(rows) == len({r['point_id'] for r in rows}) == 113
assert {r['point_id'] for r in rows} == set(ctx.original)
problems = [p for r in rows for p in board_record_problems(r, ctx, {})]
assert not problems, problems
batch = json.loads((source / 'batch.json').read_text())
state = json.loads((source / 'state.json').read_text())
assert batch['status'] == 'complete' and not batch['failed_configurations'] and not batch['missing_formal']
assert len(state['runs']) == 202 and all(v['ok'] for v in state['runs'].values())
baseline = {}
hashes = {}
for path in sorted((ROOT / 'evidence/r40/v3/R40-08.attention/board_native_fe1b').glob('*.jsonl.gz')):
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in gzip.open(path, 'rt'):
        r = json.loads(line)
        if r['point_id'] in ctx.original and not (path.name == 'hbm14.jsonl.gz' and r['tier'] == 'large'):
            assert r['point_id'] not in baseline
            baseline[r['point_id']] = r
assert set(baseline) == set(ctx.original)
comparisons = []
for r in sorted(rows, key=lambda r: r['point_id']):
    b = baseline[r['point_id']]
    assert b['status'] == 'pass' and b['point_sha256'] == r['point_sha256']
    for key in ['image_sha256', 'clock_hz', 'backend', 'backend_args', 'inductor', 'knobs']:
        assert b['platform'][key] == r['platform'][key], (r['point_id'], key)
    assert b['platform']['software_tree'] == 'files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75'
    assert b['time']['timing'] == r['time']['timing'] == 'serial-grid'
    bc = statistics.median(b['time']['samples']); rc = statistics.median(r['time']['samples'])
    comparisons.append(dict(point=r['point_id'], baseline_cycles=bc, stage1_cycles=rc,
                            ratio=rc/bc, baseline_samples=len(b['time']['samples']),
                            stage1_samples=len(r['time']['samples'])))
ratios = [r['ratio'] for r in comparisons]
result = dict(status='pass', validated_points=113, accepted_runs=202, failed_runs=0,
              tiers=dict(collections.Counter(r['tier'] for r in rows)),
              max_error_fraction_of_tolerance=max(r['correctness']['max_rel_err']/r['correctness']['tol'] for r in rows if r['correctness']['tol'] > 0),
              timing='median device serial-grid cycles; both at 90 MHz; cold calls excluded',
              baseline_files=hashes, comparisons=comparisons,
              geometric_mean_ratio=statistics.geometric_mean(ratios),
              sum_ratio=sum(r['stage1_cycles'] for r in comparisons)/sum(r['baseline_cycles'] for r in comparisons),
              regressions_above_5_percent=[r for r in comparisons if r['ratio'] > 1.05],
              above_10_percent=[r for r in comparisons if r['ratio'] > 1.1],
              limitation='Historical baseline has one timed sample per point; no interleaved supplementary confirmation. Graph makespan and host nonregression are not inferred from serial-grid timing.')
(BOARD / 'original-suite-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','baseline_files']}, indent=2))
