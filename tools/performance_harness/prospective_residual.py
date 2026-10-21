#!/usr/bin/env python3
"""Residual of a sealed prospective model prediction against a later board record.

    prospective_residual.py --prediction evidence/r41/stage1/kickoff/prospective-prediction.json \
        --model-record evidence/r41/stage1/kickoff/prospective-wide/runs/<point>/record.json \
        --board CANDIDATE/formal.jsonl [--control-board CONTROL/formal.jsonl --control-set K=V] \
        --image-sha256 SHA --clock-hz 90000000 --out residual.json

The prediction binds its model record by sha256; the record's own
platform.knobs is the predicted configuration (prospective-wide/results.json
is not a knob source, see prospective-wide-erratum.json). A board record is
admitted only when it is a measured board record of the same point
definition and configuration, on the declared image and qualified clock,
timestamped after the prediction was sealed. The residual is computed only
when the board executed the same artifacts the model executed; otherwise
the result is `not-comparable` and a new prediction must be sealed before
the next board run. Nothing here fits or adjusts the model.

The output carries result, residual, conclusion and next_step for the
experiment report (experiment.py conclusion_source).
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics

CLOCK_KNOB = 'TNCA_OPERATING_CLOCK_EVIDENCE'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_only(name):
    return name.startswith('TNCA_TLM_') or name in ('TNCA_CACHE', 'TNCA_DRAM_PATH', 'TNCA_DRAM_MAP',
                                                     'TNCA_RAMULATOR2_CONFIG')


def when(text):
    value = datetime.datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)


def configuration(knobs):
    return {k: v for k, v in sorted((knobs or {}).items()) if not model_only(k) and k != CLOCK_KNOB}


def board_rows(path, point, config, *, image, clock_hz, sealed):
    """The admissible board records, or a refusal reason."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get('point_id') == point and configuration((r.get('platform') or {}).get('knobs')) == config]
    if not rows:
        return None, 'no board record of the predicted point and configuration'
    for r in rows:
        plat = r.get('platform') or {}
        if r.get('evidence_level') != 'measured-board' or not str(plat.get('backend', '')).startswith('fpga-'):
            return None, 'a model or simulator record cannot stand for a board measurement'
        if plat.get('image_sha256') != image:
            return None, f'board record ran image {plat.get("image_sha256")}'
        if plat.get('clock_hz') != clock_hz:
            return None, f'board record clock {plat.get("clock_hz")} Hz is not the declared {clock_hz} Hz'
        if r.get('status') != 'pass' or r.get('fallbacks'):
            return None, 'board record did not pass numerically without fallback'
        if when(r['timestamp']) <= sealed:
            return None, 'board record precedes the sealed prediction'
        if ((r.get('timings') or {}).get('device-graph-makespan') or {}).get('boundaries', '').find('completion') < 0:
            return None, 'board graph makespan does not end at device completion'
    return rows, None


def median(rows, timing):
    samples = [s for r in rows for s in ((r.get('timings') or {}).get(timing) or {}).get('samples') or []]
    return (statistics.median(samples) if samples else None), samples


def compare(prediction_path, model_record, board, *, image, clock_hz, control=None, control_set=None):
    prediction = json.loads(Path(prediction_path).read_text())
    model = json.loads(Path(model_record).read_text())
    out = dict(schema='tnca-prospective-residual/1', prediction=str(prediction_path),
               prediction_sha256=sha(prediction_path), model_record=str(model_record),
               model_record_sha256=sha(model_record), point=prediction['point'],
               model_evidence=prediction.get('evidence'), performance_gate='not-measured')
    if prediction.get('record_sha256') != out['model_record_sha256']:
        raise ValueError('the sealed prediction does not bind this model record')
    if model.get('point_id') != prediction['point']:
        raise ValueError('model record is another point')
    config = configuration((model.get('platform') or {}).get('knobs'))
    sealed = when(prediction['sealed_before_board_utc'])
    out.update(configuration=config, sealed_before_board_utc=prediction['sealed_before_board_utc'])
    rows, refusal = board_rows(board, prediction['point'], config, image=image, clock_hz=clock_hz, sealed=sealed)
    if refusal:
        raise ValueError(refusal)
    if any(r.get('point_sha256') != model.get('point_sha256') for r in rows):
        raise ValueError('board and model ran different point definitions')
    out['board_records'] = dict(path=str(board), sha256=sha(board), records=len(rows),
                                software_trees=sorted({(r.get('platform') or {}).get('software_tree') for r in rows}),
                                timestamps=[r['timestamp'] for r in rows])
    model_artifacts = sorted((model.get('provenance') or {}).get('artifacts') or [])
    board_artifacts = sorted({a for r in rows for a in (r.get('provenance') or {}).get('artifacts') or []})
    out.update(model_artifacts=model_artifacts, board_artifacts=board_artifacts,
               model_software_tree=(model.get('platform') or {}).get('software_tree'))
    graph, graph_samples = median(rows, 'device-graph-makespan')
    serial, serial_samples = median(rows, 'device-cycles-serial-grid')
    out['board'] = dict(graph_makespan_median=graph, graph_makespan_samples=graph_samples,
                        serial_grid_median=serial, serial_grid_samples=serial_samples)
    predicted = prediction['model_graph']
    out['model'] = dict(graph_cycles=predicted['graph_cycles'], device_cycles=predicted['device_cycles'])
    if control:
        if not control_set:
            raise ValueError('a board control needs the knobs it changes')
        control_config = dict(config, **control_set)
        crow, crefusal = board_rows(control, prediction['point'], control_config, image=image,
                                    clock_hz=clock_hz, sealed=sealed)
        if crefusal is None:
            cgraph, _ = median(crow, 'device-graph-makespan')
            out['board_control'] = dict(configuration=control_config, graph_makespan_median=cgraph,
                                        candidate_over_control=(graph/cgraph if graph and cgraph else None),
                                        scope='board A/B context only; no control prediction was sealed')
        else:
            out['board_control'] = dict(configuration=control_config, refusal=crefusal)
    if model_artifacts != board_artifacts:
        out.update(status='not-comparable', residual=None,
                   result='board executed different artifacts than the sealed model run',
                   conclusion='the sealed prediction does not describe the measured machine code; no residual is claimed',
                   next_step='seal a new prediction on the board software tree before the next board run')
        return out
    if not graph:
        raise ValueError('board graph makespan unavailable')
    residual = dict(graph_makespan_relative=(predicted['graph_cycles']-graph)/graph,
                    serial_grid_relative=((predicted['device_cycles']-serial)/serial if serial else None),
                    definition='(model - board) / board, board median of formal samples')
    out.update(status='residual-recorded', residual=residual,
               result=f'board graph makespan median {graph} cycles; model {predicted["graph_cycles"]}',
               conclusion=('prospective uncalibrated prediction residual recorded; it does not calibrate the model '
                           'and was not used for fitting'),
               next_step='retain as validation evidence; any refit must exclude this point from its holdout claims')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prediction', type=Path, required=True)
    ap.add_argument('--model-record', type=Path, required=True)
    ap.add_argument('--board', type=Path, required=True)
    ap.add_argument('--control-board', type=Path)
    ap.add_argument('--control-set', action='append', default=[], help='KEY=VALUE the control changes')
    ap.add_argument('--image-sha256', required=True)
    ap.add_argument('--clock-hz', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = compare(args.prediction, args.model_record, args.board, image=args.image_sha256,
                     clock_hz=args.clock_hz, control=args.control_board,
                     control_set=dict(kv.split('=', 1) for kv in args.control_set))
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'point', 'residual')}, indent=2))


if __name__ == '__main__':
    main()
