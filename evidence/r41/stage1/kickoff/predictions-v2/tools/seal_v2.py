#!/usr/bin/env python3
"""Seal the v2 prospective predictions from fetched run_graphs.py outputs.

    seal_v2.py --runs RUNS_DIR --repo REPO [--seal-now]

RUNS_DIR holds one run_graphs.py output per (configuration group, DRAM path),
named GROUP-PATH, as written by drive_v2.py on GB10, plus groups.json and
build.json (model build identity). Every board-manifest configuration must
appear once per path: a pass with verified knobs is sealed (an inconclusive
slow fixed point is sealed as such), anything else is recorded with its
status and reason; nothing is dropped. Records are copied beside the sealed
files and bound by sha256.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

KICKOFF = 'evidence/r41/stage1/kickoff'
QUALIFIED = {'uncalibrated', 'uncalibrated-fixed-point-checked'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def munge(point):
    return point.replace('/', '__')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--only', choices=('prospective', 'gemm'), required=True)
    args = ap.parse_args()
    repo, runs = args.repo.resolve(), args.runs.resolve()
    sys.path.insert(0, str(repo/'tools/performance_harness'))
    from prediction_residuals import executable_identity
    (repo/KICKOFF/'predictions-v2/artifacts').mkdir(parents=True, exist_ok=True)
    build = json.loads((runs/'build.json').read_text())
    sched = json.loads((runs/'hbm_sched.json').read_text()) if (runs/'hbm_sched.json').exists() else {}
    not_run = {k: v['reason'] for k, v in (sched.get('not_run') or {}).items()}
    not_run_detail = {k: {f: x for f, x in v.items() if f != 'reason'} for k, v in (sched.get('not_run') or {}).items()}
    groups = json.loads((runs/'groups.json').read_text())
    pro = json.loads((repo/'spec/r41/stage1/experiments/board-prospective.json').read_text())
    gem = json.loads((repo/'spec/r41/stage1/experiments/board-gemm-tile.json').read_text())
    manifests = {}
    for c in pro['board_batches']['prospective']['configurations']:
        manifests[('prospective', c['point'], c['arm'])] = c
    for c in gem['board_batches']['gemm']['configurations']:
        manifests[('gemm', c['point'], c['arm'])] = c
    dest = repo/KICKOFF/'predictions-v2'
    dest.mkdir(parents=True, exist_ok=True)
    entries = {'prospective': [], 'gemm': []}
    trees = set()
    for (family, point, arm), cfg in sorted(manifests.items()):
        if family != args.only:
            continue
        group = (f'prospective-{arm}' if family == 'prospective' else
                 f'gemm-{"hbm" if "/hbm/" in point else "medium"}-{arm}')
        assert point in groups[group]['points'] and groups[group]['knobs'] == cfg['knobs'], (group, point)
        for path in ('fast', 'slow'):
            run = runs/f'{group}-{path}'
            single = runs/f'{group}-{path}--{munge(point)}'
            if single.exists():
                run = single
            entry = dict(id=f'{point}@{arm}/{path}', point=point, point_sha256=cfg['point_sha256'], arm=arm,
                         gemm_tile=cfg['knobs']['TNCA_GEMM_TILE'], path=path, configuration=cfg['knobs'],
                         run_graphs=run.name)
            done = json.loads((run/'DONE').read_text()) if (run/'DONE').exists() else None
            rows = {r['key']: r for r in json.loads((run/'results.json').read_text())['points']} \
                if (run/'results.json').exists() else {}
            row = rows.get(point)
            sched_key = f'{group}/{path}/{point}'
            if row is None:
                entry.update(status='not-run', reason=not_run.get(sched_key) or f'no run_graphs result (driver {done})',
                             **({'decision': not_run_detail[sched_key]} if sched_key in not_run_detail else {}))
                entries[family].append(entry)
                continue
            simulator = dict(process_wall_s=row.get('process_wall_s'), child_wall_s=row.get('child_wall_s'),
                             model_launch_host_s=row.get('model_launch_host_s'),
                             model_transfer_host_s=row.get('model_transfer_host_s'))
            entry['simulator'] = simulator
            if row.get('status') != 'pass' or not row.get('knobs_verified') or not row.get('model_graph'):
                if row.get('returncode') == 'timeout':
                    reason = (f'exceeded the 3 h budget: run_graphs killed the worker at its timeout '
                              f'({row.get("process_wall_s")} s)')
                else:
                    reason = (f'run_graphs status {row.get("status")} returncode {row.get("returncode")}: '
                              f'{str((row.get("problems") or [row.get("tail", "")])[:1])[:400]}')
                entry.update(status='not-run' if row.get('status') in ('harness-error', 'timeout') else 'failed',
                             reason=reason, returncode=row.get('returncode'))
                entries[family].append(entry)
                continue
            if row['knobs'] != cfg['knobs']:
                raise SystemExit(f'{entry["id"]}: run knobs differ from the board manifest')
            source = run/'runs'/munge(point)/'record.json'
            record = json.loads(source.read_text())
            model = record['systemc_model'][0]['identity']
            assert model['dram_path'] == path and model['address_map']['sha256'] == build['map_sha256'], entry['id']
            assert model['binary_sha256'] == build['binary_sha256'], entry['id']
            # Distinct names per family: the prospective point is also a GEMM-subset point.
            target = dest/(f'{munge(point)}@{arm}-{path}.record.json' if family == 'prospective'
                           else f'gemm-{munge(point)}@{arm}-{path}.record.json')
            shutil.copyfile(source, target)
            trees.add(record['platform']['software_tree'])
            files = {sha(f): f for f in (run/'runs'/munge(point)).rglob('*.tnca')}
            launches = []
            for launch in record['provenance']['launch_map']['launches']:
                artifact = files.get(launch['artifact'])
                if artifact is None:
                    raise SystemExit(f'{entry["id"]}: measured artifact {launch["artifact"]} not found')
                kept = dest/'artifacts'/f'{launch["artifact"]}.tnca'
                shutil.copyfile(artifact, kept)
                launches.append(dict(kernel=launch['kernel'], grid=launch.get('grid'), artifact_sha256=launch['artifact'],
                                     artifact=str(kept.relative_to(repo)), **executable_identity(kept)))
            status = row['model_timing_status']
            entry.update(
                status='sealed' if status in QUALIFIED else f'sealed-{status}',
                record=str(target.relative_to(repo)), record_sha256=sha(target),
                artifacts=sorted(record['provenance']['artifacts']),
                graph_cycles=row['model_graph']['graph_cycles'], device_cycles=row['model_graph']['device_cycles'],
                launches=launches, launch_cycles=row['model_graph']['launches'],
                measured_call=row['model_graph']['measured_call'],
                model_timing_status=status,
                model=dict(binary_sha256=model['binary_sha256'], config_sha256=model['config_sha256'],
                           config_file=Path(build['configs'][path]['path']).name,
                           config_file_sha256=build['configs'][path]['sha256'], dram_path=model['dram_path'],
                           address_map=dict(domain=model['address_map']['domain'], sha256=model['address_map']['sha256']),
                           host_handoff_policy=model.get('host_handoff_policy'),
                           untimed_cold_call=model.get('untimed_cold_call'),
                           ramulator2_commit=build['ramulator2_commit'] if path == 'slow' else None),
                dram=row.get('dram') or {},
                numerics=dict(status=row['status'], max_rel_err=row.get('max_rel_err'), tol=row.get('tol')))
            entries[family].append(entry)
    assert len(trees) == 1, trees
    tree = trees.pop()
    sealed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    common = dict(sealed_before_board_utc=sealed, software_tree=tree, source_commit=build['source_commit'],
                  evidence='model-prediction-uncalibrated',
                  model_build=dict(binary_sha256=build['binary_sha256'], ramulator2_commit=build['ramulator2_commit'],
                                   configs=build['configs'], built_from=build['source_commit']),
                  address_map=dict(domain='axi-bo-physical-pc', file='TNCA_DRAM_MAP_V1 B40 BO/PC map',
                                   sha256=build['map_sha256'],
                                   source=f'{KICKOFF}/physical-dram-map-v2-evidence.tgz ./map/map.txt '
                                          '(evidence/r41/stage1/address_map.md)'),
                  comparability=('per measured launch: kernel and executable_sha256 (artifact segments + manifest '
                                 'execution/resources); the artifact file hash also covers run-directory source paths'),
                  protocol=dict(runner='tools/performance_harness/systemc/run_graphs.py', warmup=0, timed=1,
                                measured_call='steady call after the cold call, as the board formal records',
                                cold_call='timed, except untimed for the HBM-scale points (recorded per entry)',
                                monitor='aggregate', host=build['host'], cores='taskset 16-19'),
                  history=dict(superseded=f'{KICKOFF}/prospective-prediction.json',
                               reason='v1 is bound to tree files:b3f03a89 and artifact b7cf71e0; '
                                      'the board will run c2c96d3 code: not comparable'),
                  scope=('uncalibrated predictions sealed before any board run of this tree; '
                         'they may validate but never calibrate the model'))
    if args.only == 'prospective':
        # v1 top-level fields carry the candidate on the fast path; every arm and path is in predictions.
        primary = next(e for e in entries['prospective'] if e['arm'] == 'candidate' and e['path'] == 'fast')
        record = json.loads((repo/primary['record']).read_text())
        doc = dict(schema='tnca-prospective-prediction/1', sealed_before_board_utc=sealed,
                   record_sha256=primary['record_sha256'], software_tree=tree, model_graph=record['model_graph'],
                   point=primary['point'], evidence=common['evidence'], models=record['systemc_model'],
                   primary=dict(arm='candidate', path='fast', note='v1 fields; every arm and DRAM path is in predictions'),
                   manifest='spec/r41/stage1/experiments/board-prospective.json')
        doc.update({k: v for k, v in common.items() if k not in doc})
        doc['predictions'] = entries['prospective']
        outputs = [('prospective-prediction-v2.json', doc)]
    else:
        doc = dict(schema='tnca-prediction-set/1', subset='spec/r40/suites/R40-03.gemm-reuse.json',
                   manifest='spec/r41/stage1/experiments/board-gemm-tile.json', **common, predictions=entries['gemm'])
        outputs = [('gemm-subset-predictions-v2.json', doc)]
    for name, doc in outputs:
        path = repo/KICKOFF/name
        if path.exists():
            raise SystemExit(f'{path} exists: a sealed prediction is never rewritten')
        path.write_text(json.dumps(doc, indent=2)+'\n')
        print('sealed', path.relative_to(repo), sha(path))
    print(sealed)


if __name__ == '__main__':
    main()
