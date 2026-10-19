#!/usr/bin/env python3
"""Audit complete frozen graph/model transfer observations, with positive controls.

Historical audits preserve their source identity. Gate promotion additionally
requires the auditor's live product tree to match both complete collections.
This tool makes no performance or fixed-KV-bucket claim.
"""
import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'tools/r40')]
import contract_v2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def graphs(doc, tree, image, launch):
    state = doc['state']
    require(state['status'] == 'complete-review-required', 'graphs incomplete')
    require(state['software_tree'] == tree and state['image_sha256'] == image,
            'graphs source/image mismatch')
    require(state['contract_sha256'] == contract_v2.LOCK_SHA256, 'graphs contract mismatch')
    require(state['clock_hz'] == 90000000, 'unexpected qualified operating clock')
    require(state['harness_sha256'] == sha((ROOT/'tools/r40/frozen_residency_tests.py').read_bytes()),
            'graph transfer assertions changed')
    files = doc['files']
    require(set(files) == set(doc['file_sha256']), 'graph file manifest incomplete')
    require(all(sha(raw.encode()) == doc['file_sha256'][name] for name, raw in files.items()),
            'graph raw file changed')
    require(sha(files['controller.py'].encode()) == launch['controller_sha256'],
            'graph controller differs from its recorded launch')
    expected = json.loads((ROOT/'spec/r40/suites/R40-02.graphs3.json').read_text())['points']
    commands = json.loads(files['commands.json'])
    require(len(state['completed']) == len(commands) == len(expected) == 24, 'all 24 graphs required')
    require(set(files) == {'controller.py', 'commands.json'} |
            {f'point{i:02d}.log' for i in range(24)}, 'all original graph logs required')
    for i, (point, command, done) in enumerate(zip(expected, commands, state['completed'])):
        require(command['point'] == point and done['point_id'] == point['id'] and
                done['point_sha256'] == point['sha256'] and done['exit_code'] == 0,
                'frozen graph/order/exit mismatch')
        args = command['command']
        require(args[args.index('--point')+1] == point['id'] and
                args[args.index('--source-tree')+1] == tree and
                args[args.index('--steps')+1] == '2', 'graph command changed')
        require(done['log'] == f'point{i:02d}.log', 'graph log association changed')
        log = files[done['log']]
        rows = re.findall(r'^PASS (\S+) (resident|per-kernel) step=([01]): '
                          r'(\d+) kernels, actual tensor bytes=(\{[^\n]+\})$', log, re.M)
        require(len(rows) == 4, 'all two-policy/two-step observations required')
        require([(p, s) for _, p, s, _, _ in rows] ==
                [('resident','0'), ('resident','1'), ('per-kernel','0'), ('per-kernel','1')],
                'duplicate/missing graph observation')
        for pid, policy, _, count, raw in rows:
            require(pid == point['id'] and int(count) >= 3, 'original multi-kernel graph required')
            counts = ast.literal_eval(raw)
            require(set(counts) == {'h2d','d2h'} and
                    all(type(v) is int and v >= 0 for v in counts.values()), 'invalid transfer count')
            require(counts == dict(h2d=0, d2h=0) if policy == 'resident' else
                    counts['h2d'] == counts['d2h'] and counts['h2d'] > 0,
                    'resident traffic or missing staging positive control')
        require(log.count('r40-frozen-residency-tests PASS (4/4)') == 1 and
                not re.search(r'Traceback|^FAIL', log, re.M), 'graph did not finish its assertions')
    return state


def model(report, tree, image, graph_state):
    identity = report['identity']
    require(identity['software_tree'] == tree and identity['image_sha256'] == image,
            'model source/image mismatch')
    require(identity['contract_sha256'] == contract_v2.LOCK_SHA256, 'model contract mismatch')
    # Preserve the reviewed pre-bucket full-model harness as an explicitly
    # historical producer. Its complete raw report remains bound to its old
    # product tree; this does not promote historical evidence for a new tree.
    historical_growing_harness = '57ac8dd246dafb3a412cd9f590ff0043a86f450a6cad92a45cf73332e6b460d8'
    allowed_harnesses = {sha((ROOT/'tools/r40/native_model.py').read_bytes())}
    if report.get('cache_policy', 'growing-shape') == 'growing-shape':
        allowed_harnesses.add(historical_growing_harness)
    require(identity['harness_sha256'] in allowed_harnesses,
            'model transfer assertions changed')
    if identity['harness_sha256'] == sha((ROOT/'tools/r40/native_model.py').read_bytes()):
        require(identity['generation_harness_sha256'] == sha((ROOT/'tools/r40/bucket_decode.py').read_bytes()),
                'generation helper changed')
    workload = json.loads((ROOT/'spec/r40/v2/plan.json').read_text())['workloads']['llama_generate']
    require(identity['workload'] == workload, 'original full frozen model required')
    require(identity['backend'] == 'fpga-u280-multicore' and
            identity['backend_args'] == graph_state['backend_args'] and
            identity['clock_hz'] == graph_state['clock_hz'], 'model/graphs physical configuration differs')
    require(report['status'] == 'functional-native-model-pass', 'model incomplete/failed')
    require(identity['weight_bytes'] == report['initial_weight_upload_bytes'] == 221267968,
            'full weights must be uploaded exactly once')
    require(report['actual_decode_calls'] == report['expected_decode_calls'] == 64 and
            report['requested_new_tokens'] == 64, 'full decode length required')
    require(report['tokens_match_independent_reference'] is True and
            report['tokens'] == report['reference_tokens'] and len(report['tokens']) == 64 and
            report['supplemental_token'] == report['supplemental_reference_token'],
            'independent reference generation mismatch')
    require(report['tolerance'] == .05 and len(report['steps']) == 65, 'frozen numerical contract changed')
    for i, step in enumerate(report['steps']):
        require(step['index'] == i and step['kind'] == ('prefill' if i == 0 else 'decode'),
                'model step missing/duplicated/reordered')
        c = step['correctness']
        require(c['ok'] is True and math.isfinite(c['max_rel_err']) and 0 <= c['max_rel_err'] <= .05,
                'model numerical failure')
        require(not step['fallbacks'] and step['wrapper_sha256'], 'model fallback/missing wrapper')
        require(not any((e['tensor'] or e['staging']) and e['bytes'] > 0 for e in step['transfers']),
                'steady tensor transfer observed')
        launches, events = step['launches'], step['events']
        require(len(launches) == len(events) and len(launches) >= 3, 'actual model launches/events required')
        for launch, event in zip(launches, events):
            require(launch['backend'] == 'fpga-u280-multicore' and
                    launch['upload_bytes'] == launch['download_bytes'] == 0,
                    'model staging/backend mismatch')
            states = [entry[0] for entry in event['history']]
            require(event['state'] == 'SUCCEEDED' and states ==
                    ['SUBMITTED','VALIDATED','QUEUED','RUNNING','DRAINING','SUCCEEDED'],
                    'model completion does not include visibility drain')
        history = step['completion_history']
        require(all(e['status'] == 0 for e in history) and
                sum(e['op'] == 4 for e in history) == len(events), 'successful per-launch quiesce required')
        require(any(e['tensor'] and e['direction'] == 'd2h' and e['bytes'] > 0
                    for e in step['validation_transfers']), 'explicit download positive control missing')
        require(any(e['op'] == 18 for e in step['host_visibility_history']) and
                all(e['status'] == 0 for e in step['host_visibility_history']),
                'successful host ownership handoff required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-manifest', type=Path, required=True)
    p.add_argument('--software-tree', required=True)
    p.add_argument('--image-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--evidence', action='store_true')
    a = p.parse_args()
    if a.out.exists():p.error('preserve prior audits; output must be fresh')
    contract_v2.verify()
    checks=[]
    try:
        compressed=a.model.read_bytes();manifest=json.loads(a.model_manifest.read_text())
        require(sha(compressed)==manifest['sha256'], 'compressed model report changed')
        raw=gzip.decompress(compressed)
        require(sha(raw)==manifest['raw_sha256'] and len(raw)==manifest['raw_bytes'], 'raw model report changed')
        state=graphs(json.loads(a.graphs.read_text()),a.software_tree,a.image_sha256,json.loads(a.launch.read_text()))
        checks.append(dict(name='all frozen graphs with actual transfer positive controls',ok=True))
        model(json.loads(raw),a.software_tree,a.image_sha256,state)
        checks.append(dict(name='full original model, zero steady transfers and completion visibility',ok=True))
        if a.evidence:
            from archbench.provenance import software_tree
            require(software_tree(refresh=True)==a.software_tree, 'historical source cannot promote current gate')
        status='capability-pass'
    except (OSError,ValueError,KeyError,TypeError) as error:
        checks.append(dict(name='complete matching residency observations',ok=False,detail=str(error)))
        status='fail'
    report=dict(status=status,checks=checks,software_tree=a.software_tree,image_sha256=a.image_sha256,
                contract_sha256=contract_v2.LOCK_SHA256,
                inputs={str(f):sha(f.read_bytes()) for f in (a.graphs,a.launch,a.model,a.model_manifest)},
                scope='Zero-transfer capability only; no performance, fixed-bucket decode or complete R40 claim')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    if a.evidence:
        import os,evidence
        os.environ['R40_CONTRACT']='v2'
        evidence.record('R40-02.zero-transfer',status,checks,software_tree=a.software_tree,
                        image_sha256=a.image_sha256,inputs=report['inputs'],detail=report['scope'])
    print('R40-02.zero-transfer:',status)
    return 0 if status=='capability-pass' else 1


if __name__=='__main__':raise SystemExit(main())
