#!/usr/bin/env python3
"""Audit sealed runtime A/B collections without promoting a whole package gate.

Every original block, bundle, process/order identity and raw device endpoint is
checked before pooling. Qualification records are excluded from all statistics.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'tools/r40')]
import ablate
import board_ablation_collect as collect
import contract_v2


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool(rows, timing):
    """Combine independently audited blocks while preserving endpoint records."""
    result=copy.deepcopy(rows[0]); _,entry=ablate.timing_entry(result,timing)
    counts=[len(ablate.samples_of(row,timing)[1]) for row in rows]
    if len(set(counts))!=1:
        raise ValueError('blocks must use the same prospectively sealed sample count')
    if len({row['time']['launches'] for row in rows})!=1:
        raise ValueError('steady graph launch count changed between blocks')
    entry['samples']=[sample for row in rows for sample in ablate.samples_of(row,timing)[1]]
    if 'endpoint_schema' in entry:
        entry['endpoint_records']=[sample for row in rows
            for sample in ablate.timing_entry(row,timing)[1]['endpoint_records']]
    # Never retain a first block's summary as a pooled summary.
    for key in ('median','p95'):
        entry.pop(key,None)
    ablate.samples_of(result,timing)
    return result


def audit(directory):
    contract_v2.verify()
    directory=Path(directory)
    report=json.loads((directory/'collection.json').read_text())
    if report['status']!='samples-complete-package-audit-required':
        raise ValueError('complete collection required; failed/partial runs are preserved')
    if report['harness_sha256']!=digest_file(collect.__file__):
        raise ValueError('collector does not match reviewed sampling assertions')
    if report['contract_sha256']!=contract_v2.LOCK_SHA256:
        raise ValueError('contract mismatch')
    manifest_path=directory/'sampling_manifest.json'
    if digest_file(manifest_path)!=report['sampling_manifest_sha256']:
        raise ValueError('sealed sampling manifest changed')
    manifest=json.loads(manifest_path.read_text())
    plan=json.loads((ROOT/'spec/r40/v2/plan.json').read_text())
    if manifest['contract_sha256']!=contract_v2.LOCK_SHA256 or manifest['rule']!=plan['sample_plan']['board'] or manifest['identity']!=report['identity']:
        raise ValueError('sampling rule or measurement identity changed')
    cores=report['frozen_experiment'].get('required_compute_cores',3)
    exp,expected=collect.schedule(report['experiment'],'unused',report['identity']['backend_args'],directory,compute_cores=cores)
    subset={e['point']['id']:e['point']['sha256'] for e in expected}
    if manifest['original_points']!=subset or report['frozen_experiment']!=exp:
        raise ValueError('frozen point or experiment changed')
    if set(manifest['qualification'])!=set(subset) or set(manifest['timed_per_point'])!=set(subset):
        raise ValueError('incomplete qualification manifest')
    baseline_entries={(e['point']['id'],e['side']):e for e in expected if e['block']==0}
    for pid,sides in manifest['qualification'].items():
        seconds={}
        if set(sides)!={'control','candidate'}:raise ValueError('both qualification sides required')
        for side,info in sides.items():
            if digest_file(info['raw'])!=info['raw_sha256']:
                raise ValueError('qualification record changed')
            rows=ablate.load(info['raw'])
            if len(rows)!=1:raise ValueError('one qualification record required')
            row=rows[0];_,t=ablate.timing_entry(row,exp['timing']);_,samples=ablate.samples_of(row,exp['timing'])
            if row['status']!='pass' or row['point_id']!=pid or row['point_sha256']!=subset[pid] or len(samples)!=1 or t['warmup']!=1:
                raise ValueError('qualification point, correctness or sampling mismatch')
            if t['unit']!='cycles' or row['provenance']['clock_hz']!=report['identity']['clock_hz']:
                raise ValueError('qualification clock mismatch')
            for key in ('software_tree','image_sha256','backend','backend_args'):
                if row['platform'].get(key)!=report['identity'][key]:raise ValueError('qualification identity mismatch')
            if row.get('fallbacks') or any(row['platform'].get('knobs',{}).get(k)!=v for k,v in baseline_entries[pid,side]['knobs'].items()):
                raise ValueError('qualification baseline or fallback mismatch')
            seconds[side]=samples[0]/report['identity']['clock_hz']
            if seconds[side]!=info['seconds']:raise ValueError('qualification duration changed')
        if collect.sample_count(seconds,manifest['rule'])!=manifest['timed_per_point'][pid]:
            raise ValueError('sample count does not follow frozen rule')
    entries=report['entries']
    if report['completed']!=[e['stem'] for e in expected] or len(entries)!=len(expected):
        raise ValueError('incomplete or reordered original collection')
    by_pair={};sessions={pid:set() for pid in subset};boots=set()
    for wanted,entry in zip(expected,entries):
        for field in ('stem','block','side','point','knobs'):
            if entry[field]!=wanted[field]:raise ValueError('collection schedule changed: '+field)
        rows=ablate.load(entry['raw'])
        if len(rows)!=1:raise ValueError('exactly one original point per child required')
        row=rows[0];pid=entry['point']['id'];plat=row['platform'];prov=row['provenance']
        if row['point_id']!=pid or row['point_sha256']!=subset[pid] or row['status']!='pass' or row.get('fallbacks'):
            raise ValueError('failed or changed original point')
        for key in ('software_tree','image_sha256','backend','backend_args'):
            if plat.get(key)!=report['identity'][key]:raise ValueError('measurement identity changed: '+key)
        if prov.get('launch_map',{}).get('cores')!=list(range(int(entry['knobs']['TNCA_ACTIVE_CORES']))):
            raise ValueError('active physical core map differs from control')
        if prov.get('clock_hz')!=report['identity']['clock_hz']:
            raise ValueError('formal measurement clock changed')
        if any(plat.get('knobs',{}).get(k)!=v for k,v in entry['knobs'].items()):
            raise ValueError('frozen baseline/runtime control changed')
        _,t=ablate.timing_entry(row,exp['timing']);_,samples=ablate.samples_of(row,exp['timing'])
        if t['warmup']!=5 or t.get('deterministic') is not False or len(samples)!=manifest['timed_per_point'][pid]:
            raise ValueError('formal sampling differs from sealed plan')
        m=row['measurement'];session=m.get('process_session')
        if not session or session in sessions[pid] or str(m.get('block_id'))!=str(entry['block']):
            raise ValueError('independent block/process identity required')
        sessions[pid].add(session);boots.add(m.get('boot_id'))
        if not m.get('boot_id') or type(m.get('started_ns')) is not int or type(m.get('ended_ns')) is not int or m['started_ns']>=m['ended_ns']:
            raise ValueError('valid monotonic measurement interval required')
        by_pair.setdefault((entry['block'],pid),[]).append(row)
    if len(boots)!=1:raise ValueError('measurement ordering requires the same host boot')
    for pair in by_pair.values():
        if len(pair)!=2 or pair[0]['measurement']['ended_ns']>pair[1]['measurement']['started_ns']:
            raise ValueError('randomized A/B order or nonoverlap violated')
    blocks=[];cfg=set()
    for block in range(3):
        a,b=(directory/f'{s}{block}.jsonl' for s in ('control','candidate'))
        # Ensure aggregate input has exactly the records just reviewed.
        for side,path in [('control',a),('candidate',b)]:
            raw=ablate.load(path)
            expected_rows=[ablate.load(e['raw'])[0] for e in entries if e['block']==block and e['side']==side]
            if raw!=expected_rows:raise ValueError('aggregate and reviewed child records differ')
        result=ablate.compare(a,b,subset,exp['vary'],timing=exp['timing'],threshold=exp['threshold'],audit_bundles=True)
        blocks.append(result);cfg.add((result['config_a'],result['config_b']))
    if len(cfg)!=1:raise ValueError('configuration changed across blocks')
    merged=[]
    for side in ('control','candidate'):
        rows=[]
        for pid in subset:
            selected=[ablate.load(e['raw'])[0] for e in entries if e['side']==side and e['point']['id']==pid]
            rows.append(pool(selected,exp['timing']))
        merged.append(rows)
    with tempfile.TemporaryDirectory() as temp:
        paths=[Path(temp)/s for s in ('control.jsonl','candidate.jsonl')]
        for path,rows in zip(paths,merged):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        comparison=ablate.compare(*paths,subset,exp['vary'],timing=exp['timing'],threshold=exp['threshold'])
    return dict(status='measurement-audited',comparison=comparison,blocks=blocks,
                identity=report['identity'],contract_sha256=contract_v2.LOCK_SHA256,
                inputs={str(directory/n):digest_file(directory/n) for n in ('collection.json','sampling_manifest.json')},
                required_gate_promotions=[],scope='Runtime experiment only; traffic floors, LOO and other package prerequisites remain mandatory')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('output must be fresh; preserve prior results')
    try:report=audit(args.collection)
    except (OSError,ValueError,KeyError,TypeError,ablate.Refused) as error:
        report=dict(status='audit-refused',error=str(error),required_gate_promotions=[])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])
    return 0 if report['status']=='measurement-audited' else 1


if __name__=='__main__':raise SystemExit(main())
