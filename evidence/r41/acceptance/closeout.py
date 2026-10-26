#!/usr/bin/env python3
"""Recompute scoped run-1 judgments from archived files; no device access."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/performance_harness'))
from stage1_acceptance import board_record_problems
BASE = ROOT/'evidence/r41/acceptance'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load_archive(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}

def run():
    raw = load_archive(BASE/'board-records.tgz')
    physical = load_archive(BASE/'physical-audit.tgz')
    endurance = load_archive(ROOT/'evidence/r41/stage2/board/endurance-rerun.tgz')
    get = lambda name: json.loads(raw[name])
    original = 'out/s2-original-suite/original/'
    batch = get(original+'batch.json')
    records = [json.loads(line) for line in raw[original+'formal.jsonl'].splitlines()]
    plan = json.loads((ROOT/'spec/r41/stage1/plan.json').read_text())
    definitions = {p['id']: p['sha256'] for s in plan['original_suites']
                   for p in json.loads((ROOT/s['path']).read_text())['points']}
    ctx = SimpleNamespace(board=batch['board'], definitions=definitions, historical_trees=set())
    assert batch['status'] == 'complete' and not batch['failed_configurations'] and not batch['missing_formal']
    tree = batch['software_tree']
    points = {}
    for record in records:
        assert not board_record_problems(record, ctx, {'claimed_clock_hz':100000000}), record['point_id']
        assert record['platform']['software_tree'] == tree
        points.setdefault(record['point_id'], []).append(record)
    assert set(points) == set(definitions) and len(points) == 113
    qualified = json.loads(physical['qualified_clock.json'])
    assert sha(physical['qualified_clock.json']) == batch['board']['clock_evidence_sha256']
    for entry in qualified['evidence_files'].values():
        matches = [data for name,data in physical.items() if Path(name).name == Path(entry['path']).name]
        assert len(matches) == 1 and sha(matches[0]) == entry['sha256'], entry
    assert qualified['physical_core_slrs'] == {'0':0,'1':1,'2':2}
    assert qualified['clock_hz'] == 100000000 and qualified['cdc_review'] == 'passed'
    assert qualified['route_legal'] and qualified['timing']['tns_failing'] == qualified['timing']['ths_failing'] == qualified['timing']['tpws_failing'] == 0
    source = get('out/s2-identity/source.json')
    assert source['status'] == 'pass' and not source['missing'] and not source['changed'] and not source['extra']
    registered = json.loads((ROOT/'evidence/r41/stage2/board/image_source.json').read_text())
    assert source['files'] == len(registered['files']) and source['files_sha256'] == registered['files_sha256']
    fp8 = {}
    for manifest,sub,count in [('s2-fp8-cases','cases',100),('s2-fp8-blockwise','blockwise',16)]:
        report = get(f'out/{manifest}/{sub}/report.json')
        assert report['backend'] == 'fpga-u280-multicore' and report['software_tree'] == tree
        assert len(report['rows']) == count and all(r['status']=='pass' for r in report['rows'])
        fp8[manifest] = {'passed':count,'raw_evidence_label':report['evidence'], 'scope':'board backend and launch cycles; stale rtl-simulation label retained'}
    hbm = {}
    for manifest in ('s2-hbm8','s2-hbm8-big'):
        reports = {n:json.loads(d) for n,d in raw.items() if n.startswith('out/'+manifest+'/') and n.endswith('/report.json') and n.count('/')==3}
        assert len(reports) == (7 if manifest=='s2-hbm8' else 2)
        for report in reports.values():
            assert report['status']=='functional-pass' and report['mode']=='board' and report['software_tree']==tree
            assert all(c['ok'] for c in report['checks'])
        hbm[manifest] = {'cases':len(reports),'checks':sum(len(r['checks']) for r in reports.values())}
    end = json.loads(endurance['out/s2-endurance/mix/endurance.json'])
    assert end['hours'] >= 2 and end['sessions'] == end['passed'] == 250
    assert end['board']['image_sha256'] == batch['board']['image_sha256'] and end['board']['clock_hz']==100000000
    # Complete Stage 1 denominator; no card job is needed to replace the historical 96-point comparison.
    baseline = load_archive(ROOT/'evidence/r41/stage1/board/original-suite-rerun.tgz')
    baseline_rows = [json.loads(line) for line in baseline['out/board-original-suite/original/formal.jsonl'].splitlines()]
    bypoint = {}
    for row in baseline_rows:
        assert row['status']=='pass' and row['point_sha256']==definitions[row['point_id']]
        assert row['platform']['clock_hz']==90000000
        bypoint.setdefault(row['point_id'],[]).append(row)
    assert set(bypoint)==set(points)
    def cycles(rows):
        return statistics.median(r['time']['device'] for r in rows)
    comparison = []
    for name,rows in sorted(points.items()):
        a,b = cycles(bypoint[name]),cycles(rows)
        comparison.append({'point':name,'stage1_cycles':a,'stage2_cycles':b,'cycle_ratio':b/a,'device_time_speedup':(a/90000000)/(b/100000000)})
    result = {'scope':'run-1 measured candidate; does not bind later product trees',
              'board':batch['board'], 'software_tree':tree, 'qualified_clock':qualified,
              'source_check':{k:source[k] for k in ('commit','files_sha256','status')},
              'original':{'points':len(points),'formal_records':len(records),'large_points':sum(rows[0]['tier']=='large' for rows in points.values())},
              'fp8':fp8,'hbm':hbm,'endurance':{k:end[k] for k in ('hours','sessions','passed','software_tree','started_utc','ended_utc')},
              'comparison':{'points':113,'basis':'median serial-grid device cycles per point; joint hardware/frequency/software migration',
                 'device_time_speedup_ratio_of_sums':sum(r['stage1_cycles']/90000000 for r in comparison)/sum(r['stage2_cycles']/100000000 for r in comparison),
                 'device_time_speedup_geomean':math.exp(statistics.mean(math.log(r['device_time_speedup']) for r in comparison)),
                 'rows':comparison}}
    (BASE/'board_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    inventory = {'host':'acf3019:/home/zzhouch/r41s2_board_86a3d929cf3d', 'collection':'read-only tar stream; JSON/JSONL except bundles and experiment step caches; raw first failed endurance retained', 'archives':{}}
    for path,data in [(BASE/'board-records.tgz',raw),(BASE/'physical-audit.tgz',physical)]:
        inventory['archives'][str(path.relative_to(ROOT))]={'sha256':sha(path.read_bytes()),'members':{n:{'size':len(d),'sha256':sha(d)} for n,d in sorted(data.items())}}
    (BASE/'board_index.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('original','hbm','endurance')}))
    print(json.dumps({k:v for k,v in result['comparison'].items() if k!='rows'}))

if __name__ == '__main__':
    run()
