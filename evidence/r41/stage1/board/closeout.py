#!/usr/bin/env python3
"""Recompute Stage 1 board residuals locally; never connects to a device.

python3 evidence/r41/stage1/board/closeout.py --frozen-root /path/to/25094197f6e6
Writes reviewable metrics/residuals beside this script. Acceptance is run separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools/performance_harness'))
import prediction_residuals as pr
import stage1_acceptance as sa

FROZEN = 'files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a'
HOST_ROOT = Path('/home/zzhouch/r41s1_board_25094197f6e6')
BASE = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2)+'\n')

def run(frozen):
    ctx = sa.Context(frozen, ROOT/'evidence/r41/stage1/required_index.json')
    assert ctx.tree == FROZEN, ctx.tree
    index = read(BASE/'index.json')
    assert pr.sha(BASE/'finished.tgz') == index['archive_sha256']
    with tempfile.TemporaryDirectory(prefix='r41-board-') as tmp:
        tmp = Path(tmp)
        for name in ('finished.tgz', 'executables.tgz'):
            with tarfile.open(BASE/name) as t:
                t.extractall(tmp, filter='data')
        for member in index['members']:
            assert pr.sha(tmp/member['path']) == member['sha256'], member['path']
        for path in sorted((tmp/'out').glob('*/report.json')):
            report = read(path)
            assert report['status'] == 'completed-experiment', path
            assert report['identity']['software_tree'] == FROZEN, path
            manifest = frozen/'spec/r41/stage1/experiments'/f'{path.parent.name}.json'
            assert pr.sha(manifest) == report['identity']['manifest_sha256'], path
        for path in sorted((tmp/'out/board-memory-service').glob('r*/report.json')):
            case = read(path)
            assert case['status'] == 'functional-pass', path
            assert case['software_tree'] == FROZEN, path
        for path in sorted((tmp/'out/board-pipeline-admission').glob('r*/report.json')):
            case = read(path)
            assert case['status'] == 'functional-pass', path
            assert case['software_tree'] == FROZEN, path
            assert all(r['status'] == 'pass' for r in case['rows']), path
        def executable(rec):
            bundle = rec['provenance']['bundle']
            directory = tmp/Path(bundle['dir']).relative_to(HOST_ROOT)
            result = []
            for launch in rec['provenance']['launch_map']['launches']:
                rel = f"artifacts/{launch['artifact']}.tnca"
                path = directory/rel
                assert pr.sha(path) == bundle['files'][rel]
                result.append([launch['kernel'], pr.executable_identity(path)['executable_sha256']])
            return result
        formal = {}
        for path in sorted((tmp/'out').glob('**/formal.jsonl')):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                problems = sa.board_record_problems(row, ctx, {})
                assert not problems, problems
                assert row['platform']['software_tree'] == FROZEN
            formal[str(path.relative_to(tmp))] = len(rows)
        results = {'software_tree': FROZEN, 'formal_records_validated': formal, 'residual_definition': '(model - measured board graph median) / board', 'ratio_definition': 'candidate/control; ratio residual = model ratio / board ratio - 1', 'prediction_sets': {}}
        for name, prediction, batch in [('gemm-tile','gemm-subset-predictions-v2.json','gemm'), ('prospective','prospective-prediction-v2.json','prospective')]:
            source = ROOT/'evidence/r41/stage1/kickoff'/prediction
            board = tmp/f'out/board-{name}/{batch}/formal.jsonl'
            result = pr.compare([source], [board], image=ctx.board['image_sha256'], clock_hz=90000000, executables=executable)
            # Stable paths in the derived result; raw records remain byte-for-byte unchanged.
            result['predictions'][0]['path'] = str(source.relative_to(ROOT))
            result['board'][0]['path'] = str(board.relative_to(tmp))
            write(BASE/f'{name}-residuals.json', result)
            vals = sorted(abs(r['fast']['residual']) for r in result['rows'])
            ratios = []
            for point in sorted({r['point'] for r in result['rows']}):
                a,c = [next(r for r in result['rows'] if r['point']==point and r['arm']==arm) for arm in ('candidate','control')]
                br = a['board']['graph_makespan_median']/c['board']['graph_makespan_median']
                mr = a['fast']['model_graph_cycles']/c['fast']['model_graph_cycles']
                ratios.append(dict(point=point, board_candidate_control=br, model_candidate_control=mr, ratio_residual=mr/br-1, ranking_inversion=(br<1)!=(mr<1), false_gain=mr<1 and br>=1))
            sealed = read(source)['predictions']
            costs = [{k:p.get(k) for k in ('point','arm','path','status','simulator')} for p in sealed]
            results['prediction_sets'][name] = dict(summary=result['summary'], fast_p95_abs_nearest_rank=vals[math.ceil(.95*len(vals))-1], ratios=ratios, simulator_costs=costs)
        anchors = {}
        for row in [json.loads(x) for x in (tmp/'out/board-anchors/anchors/formal.jsonl').read_text().splitlines()]:
            anchors.setdefault(row['point_id'], []).append(row)
        results['anchors'] = {point:{metric:dict(median=st.median(v := [r['timings'][metric]['samples'][0] for r in rows]), range_over_median=(max(v)-min(v))/st.median(v)) for metric in ('device-cycles-serial-grid','device-graph-makespan')} for point,rows in anchors.items()}
        results['memory'] = [{k:r[k] for k in ('label','device_cycles','actual_input_values_equal','same_shader_as_domain_baseline','bindings_equal_to_domain_baseline','actual_outlet_line_deltas')} for p in sorted((tmp/'out/board-memory-service').glob('compare-r*.json')) for r in read(p)['rows']]
        results['pipeline'] = [{k:r[k] for k in ('label','graph','shape','device_cycles','reference_status','bindings_equal_to_baseline','actual_input_values_equal')} for p in sorted((tmp/'out/board-pipeline-admission').glob('compare-r*.json')) for r in read(p)['rows']]
        results['tuning'] = read(tmp/'out/board-tune-qualification/tune-check.json')
        results['deadline'] = read(tmp/'out/board-tune-qualification/deadline-expiry/report.json')
        write(BASE/'metrics.json', results)
    print('Validated formal records and executable identities; recomputed both sealed residual sets.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen-root', type=Path, required=True)
    run(ap.parse_args().frozen_root.resolve())
