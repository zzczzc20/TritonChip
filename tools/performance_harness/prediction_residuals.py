#!/usr/bin/env python3
"""Fast- and slow-path residuals of a sealed prediction set against board records.

    prediction_residuals.py --predictions evidence/r41/stage1/kickoff/gemm-subset-predictions-v2.json \
        --board BATCH/formal.jsonl [--board ...] --image-sha256 SHA --clock-hz 90000000 --out residuals.json

A prediction set (tnca-prediction-set/1, or a prospective prediction with a
`predictions` list) holds one sealed entry per (point, configuration, DRAM
path): the model record's sha256, its artifacts, its graph cycles and its
timing qualification. For every configuration this reports the board graph
makespan and the residual (model - board) / board of each path.

Board records are admitted only when they are measured board records of the
same frozen point definition and configuration, on the declared image and
clock, passing without fallback, with a completion-bounded graph makespan, and
timestamped after the set was sealed; anything else refuses the comparison.
A residual is computed only when the board executed the same programs as
the model, launch by launch (`not-comparable` otherwise). Program identity is
the artifact's executable content: its segments plus the manifest's execution
and resource sections. The artifact file hash is not used: it also covers the
source path and provenance metadata, which differ between any two run
directories even for byte-identical machine code. Board artifacts are read
from each record's bundle (board_batch.py --bundle-dir). An inconclusive slow fixed point
keeps its residual but is never a qualified one; a prediction that was not run
stays `prediction-not-run`. Nothing is fitted or adjusted.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]

CLOCK_KNOB = 'TNCA_OPERATING_CLOCK_EVIDENCE'
QUALIFIED = {'uncalibrated', 'uncalibrated-fixed-point-checked'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def when(text):
    value = datetime.datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)


def configuration(knobs):
    return {k: v for k, v in sorted((knobs or {}).items()) if k != CLOCK_KNOB and not k.startswith('TNCA_TLM_')
            and k not in ('TNCA_CACHE', 'TNCA_DRAM_PATH', 'TNCA_DRAM_MAP', 'TNCA_RAMULATOR2_CONFIG')}


def entries(path):
    doc = json.loads(Path(path).read_text())
    rows = doc.get('predictions')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path}: no sealed predictions list')
    sealed = when(doc['sealed_before_board_utc'])
    for row in rows:
        for field in ('point', 'point_sha256', 'configuration', 'path', 'status'):
            if field not in row:
                raise ValueError(f'{path}: prediction lacks {field}')
        if row['path'] not in ('fast', 'slow'):
            raise ValueError(f'{path}: unknown DRAM path {row["path"]}')
    return doc, sealed, rows


def admit(record, *, image, clock_hz, sealed):
    plat = record.get('platform') or {}
    where = f'{record.get("point_id")}'
    if record.get('evidence_level') != 'measured-board' or not str(plat.get('backend', '')).startswith('fpga-'):
        raise ValueError(f'{where}: a model or simulator record cannot stand for a board measurement')
    if plat.get('image_sha256') != image:
        raise ValueError(f'{where}: board record ran image {plat.get("image_sha256")}')
    if plat.get('clock_hz') != clock_hz:
        raise ValueError(f'{where}: board clock {plat.get("clock_hz")} Hz is not the declared {clock_hz} Hz')
    if record.get('status') != 'pass' or record.get('fallbacks'):
        raise ValueError(f'{where}: board record did not pass numerically without fallback')
    if when(record['timestamp']) <= sealed:
        raise ValueError(f'{where}: board record precedes the sealed predictions')
    boundary = ((record.get('timings') or {}).get('device-graph-makespan') or {}).get('boundaries', '')
    if 'completion' not in boundary:
        raise ValueError(f'{where}: board graph makespan does not end at device completion')


def executable_identity(path):
    """Hashes of what a launch executes, independent of the artifact's path metadata."""
    if str(ROOT/'python') not in sys.path:
        sys.path.insert(0, str(ROOT/'python'))
    from tnca import artifact
    manifest, segments = artifact.load(str(path))
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(segments.items())}
    body = dict(segments=digests, execution=manifest.get('execution'), resources=manifest.get('resources'))
    return dict(code_sha256=digests.get('text'),
                executable_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())


def board_executables(record):
    """[(kernel, executable sha256)] of the record's measured launches, from its bundle, or None."""
    prov = record.get('provenance') or {}
    bundle = (prov.get('bundle') or {}).get('dir')
    launches = (prov.get('launch_map') or {}).get('launches') or []
    if not bundle or not launches:
        return None
    out = []
    for launch in launches:
        path = Path(bundle)/'artifacts'/f'{launch.get("artifact")}.tnca'
        if not path.is_file():
            return None
        out.append([launch.get('kernel'), executable_identity(path)['executable_sha256']])
    return out


def samples(records, timing):
    return [s for r in records for s in ((r.get('timings') or {}).get(timing) or {}).get('samples') or []]


def compare(prediction_files, board_files, *, image, clock_hz, executables=board_executables):
    board = [json.loads(line) for f in board_files for line in Path(f).read_text().splitlines() if line.strip()]
    sources, configs = [], {}
    for path in prediction_files:
        doc, sealed, rows = entries(path)
        sources.append(dict(path=str(path), sha256=sha(path), sealed_before_board_utc=doc['sealed_before_board_utc'],
                            software_tree=doc.get('software_tree')))
        for row in rows:
            key = (row['point'], json.dumps(configuration(row['configuration']), sort_keys=True))
            group = configs.setdefault(key, dict(point=row['point'], point_sha256=row['point_sha256'],
                                                 arm=row.get('arm'), configuration=configuration(row['configuration']),
                                                 sealed=sealed, paths={}))
            if row['path'] in group['paths']:
                raise ValueError(f'duplicate {row["path"]} prediction for {row["point"]}')
            group['paths'][row['path']] = row
    out_rows, used = [], set()
    for (point, _), group in configs.items():
        matched = [r for r in board if r.get('point_id') == point
                   and configuration((r.get('platform') or {}).get('knobs')) == group['configuration']]
        row = dict(point=point, arm=group['arm'], configuration=group['configuration'])
        if not matched:
            row.update(status='no-board-record', fast=None, slow=None)
            out_rows.append(row)
            continue
        for record in matched:
            admit(record, image=image, clock_hz=clock_hz, sealed=group['sealed'])
            if record.get('point_sha256') != group['point_sha256']:
                raise ValueError(f'{point}: board ran another point definition')
            used.add(id(record))
        graph = samples(matched, 'device-graph-makespan')
        serial = samples(matched, 'device-cycles-serial-grid')
        programs = [executables(r) for r in matched]
        board_programs = programs[0] if programs and all(p is not None and p == programs[0] for p in programs) else None
        row['board'] = dict(records=len(matched), graph_makespan_samples=graph, serial_grid_samples=serial,
                            graph_makespan_median=statistics.median(graph) if graph else None,
                            serial_grid_median=statistics.median(serial) if serial else None,
                            programs=board_programs,
                            software_trees=sorted({(r.get('platform') or {}).get('software_tree') for r in matched}))
        for path in ('fast', 'slow'):
            p = group['paths'].get(path)
            if p is None:
                row[path] = dict(status='no-prediction')
                continue
            if not str(p['status']).startswith('sealed'):
                row[path] = dict(status=f'prediction-{p["status"]}', reason=p.get('reason'))
                continue
            entry = dict(model_graph_cycles=p['graph_cycles'], model_timing_status=p.get('model_timing_status'),
                         record_sha256=p.get('record_sha256'))
            model_programs = [[l['kernel'], l['executable_sha256']] for l in p.get('launches') or []]
            if board_programs is None:
                entry.update(status='not-comparable', residual=None,
                             reason='board program identity unavailable or inconsistent across its records')
            elif model_programs != board_programs:
                entry.update(status='not-comparable', residual=None,
                             reason='board executed different programs than the sealed model run')
            elif not row['board']['graph_makespan_median']:
                raise ValueError(f'{point}: board graph makespan unavailable')
            else:
                b = row['board']['graph_makespan_median']
                entry.update(residual=(p['graph_cycles']-b)/b,
                             qualified=p.get('model_timing_status') in QUALIFIED,
                             status=('residual-recorded' if p.get('model_timing_status') in QUALIFIED
                                     else 'residual-recorded-inconclusive-prediction'))
            row[path] = entry
        row['status'] = 'compared'
        out_rows.append(row)
    summary = {}
    for path in ('fast', 'slow'):
        values = [r[path]['residual'] for r in out_rows if (r.get(path) or {}).get('qualified')]
        summary[path] = dict(qualified_residuals=len(values),
                             median_abs=statistics.median(abs(v) for v in values) if values else None,
                             max_abs=max(abs(v) for v in values) if values else None,
                             not_comparable=sum(1 for r in out_rows if (r.get(path) or {}).get('status') == 'not-comparable'),
                             inconclusive=sum(1 for r in out_rows if (r.get(path) or {}).get('status')
                                              == 'residual-recorded-inconclusive-prediction'),
                             not_run=sum(1 for r in out_rows if (r.get(path) or {}).get('status') == 'prediction-not-run'))
    missing = sum(1 for r in out_rows if r['status'] == 'no-board-record')
    comparable = summary['fast']['qualified_residuals'] + summary['slow']['qualified_residuals']
    return dict(schema='tnca-prediction-residuals/1', predictions=sources,
                board=[dict(path=str(f), sha256=sha(f)) for f in board_files], board_records_used=len(used),
                image_sha256=image, clock_hz=clock_hz, rows=out_rows, summary=summary,
                status='residuals-recorded' if comparable and not missing else
                       'incomplete' if comparable else 'not-comparable',
                result=f'{len(out_rows)} configurations; {missing} without a board record; '
                       f'qualified residuals fast {summary["fast"]["qualified_residuals"]}, slow {summary["slow"]["qualified_residuals"]}',
                residual=dict(fast_median_abs=summary['fast']['median_abs'], slow_median_abs=summary['slow']['median_abs'],
                              definition='(model graph cycles - board graph makespan median) / board'),
                conclusion=('prospective uncalibrated residuals; they validate, never calibrate, the model, '
                            'and none of these points may enter a later fit as unseen holdout'),
                next_step=('compare fast and slow residuals against their simulator cost; '
                           'seal new predictions before any further board run of changed software'),
                performance_gate='not-measured')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--predictions', type=Path, action='append', required=True)
    ap.add_argument('--board', type=Path, action='append', required=True)
    ap.add_argument('--image-sha256', required=True)
    ap.add_argument('--clock-hz', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = compare(args.predictions, args.board, image=args.image_sha256, clock_hz=args.clock_hz)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], summary=report['summary']), indent=2))


if __name__ == '__main__':
    main()
