#!/usr/bin/env python3
"""One real frozen-graph candidate for tnca.autotune.run_measurement.

The supervisor supplies a fresh --events path and enforces the shared graph
budget. This worker saves the complete ArchBench record before judging it.
It neither selects a winner nor certifies a performance gate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = [str(ROOT/p) for p in ('', 'python', 'compiler/python', 'tools/asm', 'tools/r40')]
sys.path[:0] = SOURCE_PATHS
import ablate
import contract_v2
from archbench.provenance import software_tree, sha256_file, check_bundle

SUITES = {'holdout':'spec/r40/suites/R40-09.holdout.json',
          'hbm':'spec/r40/suites/hbm-core.json',
          'native':'spec/r40/suites/existing.json'}
WIDTHS = (16, 32)  # Same predeclared candidates as compiled_candidates evidence.


def validate_result(record, request, *, functional=False, samples=1):
    """Accept complete resident numeric/device-event data, never host duration."""
    p = record.get('platform', {})
    backend = 'fpga-multicore-sim' if functional else 'fpga-u280-multicore'
    if (record.get('status') != 'pass' or record.get('fallbacks')
            or record.get('point_id') != request['point_id']
            or record.get('point_sha256') != request['point_sha256']
            or p.get('backend') != backend
            or p.get('software_tree') != request['software_tree']
            or p.get('backend_args') != request['backend_args']
            or p.get('inductor', {}).get('max_block') != request['max_block']
            or p.get('knobs', {}).get('TNCA_TUNE') not in (None, '0')):
        raise ValueError('candidate numerical, source, point or configuration mismatch')
    if (record.get('host', {}).get('upload_bytes') != 0
            or record.get('host', {}).get('download_bytes') != 0
            or not record.get('provenance', {}).get('artifacts')):
        raise ValueError('candidate requires actual resident execution without tensor staging')
    provenance = record['provenance']
    bundle = provenance.get('bundle', {})
    if check_bundle(bundle):
        raise ValueError('candidate compiler bundle is missing or changed')
    if any(bundle.get('files', {}).get(f'artifacts/{h}.tnca') != h
           for h in provenance['artifacts']):
        raise ValueError('candidate bundle lacks a matching compiled artifact')
    window = record.get('native_memory', {}).get('window')
    if not isinstance(window, dict):
        raise ValueError('candidate lacks its actual native memory map')
    entry = record.get('timings', {}).get('device-graph-makespan', {})
    if (entry.get('endpoint_schema') != 'tnca-launch-endpoints/1'
            or entry.get('warmup') != 1 or entry.get('deterministic') is not False
            or len(entry.get('samples', [])) != samples):
        raise ValueError('candidate needs one warmup and complete steady device endpoints')
    ablate.samples_of(record, 'graph-makespan')  # Recompute every raw span/count.
    if functional:
        if request.get('image_sha256') is not None or entry.get('clock_hz') is not None:
            raise ValueError('functional RTL cannot claim a physical image or measured clock')
    else:
        image = request['image_sha256']
        banks = window.get('allocation', {}).get('physical_banks', [])
        if (p.get('image_sha256') != image or p.get('image') != 'xclbin:'+image[:16]
                or p.get('clock_policy') != '100MHz-undivided'
                or entry.get('clock_hz') != 100000000
                or window.get('size', 0) < 1 << 30
                or len({b['pc'] for b in banks}) < 2):
            raise ValueError('candidate physical image, clock or actual multi-PC window mismatch')
    return entry['endpoint_records'][0]


def source_origins():
    """Refuse stale imported modules or data outside this immutable checkout."""
    # Parent shells may export a different checkout in PYTHONPATH. The
    # compiler and its children must resolve this same immutable source.
    os.environ['PYTHONPATH'] = os.pathsep.join(SOURCE_PATHS)
    import importlib
    origins = {}
    for name in ('tnca', 'tnca_cc', 'archbench'):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(ROOT):
            raise ValueError(f'{name} imported outside the immutable source: {origin}')
        origins[name] = dict(path=str(origin), sha256=sha256_file(origin))
    from tnca_cc._datapath import find
    model = Path(find('spec', 'r39_sched_model.json')).resolve()
    if model != ROOT/'spec/r39_sched_model.json':
        raise ValueError(f'scheduler model is outside the immutable source: {model}')
    origins['scheduler_model'] = dict(path=str(model), sha256=sha256_file(model))
    return origins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--events', type=Path, required=True)
    ap.add_argument('--functional-rtl', action='store_true',
                    help='functional worker validation only; never board/holdout evidence')
    args = ap.parse_args()
    events = args.events.resolve()
    raw, info, error = (events.with_suffix(s) for s in ('.raw.json','.info.json','.error.json'))
    bundle = events.with_suffix('.bundle')
    if any(p.exists() for p in (events, raw, info, error, bundle)):
        ap.error('preserve previous candidate files; use a fresh events path')
    events.parent.mkdir(parents=True, exist_ok=True)
    try:
        contract_v2.verify()
        request = json.loads(args.request.read_text())
        suite = request.get('suite', 'holdout')
        if suite not in SUITES or (suite == 'native' and not args.functional_rtl):
            raise ValueError('board candidates require the frozen holdout or HBM suite')
        points = json.loads((ROOT/SUITES[suite]).read_text())['points']
        point = next(p for p in points if p['id'] == request['point_id'])
        if (point['sha256'] != request['point_sha256']
                or type(request['max_block']) is not int or request['max_block'] not in WIDTHS):
            raise ValueError('point definition or predeclared candidate changed')
        if software_tree(refresh=True) != request['software_tree']:
            raise ValueError('worker source differs from the immutable request')
        config = request['backend_args']
        if config.get('core_count') != 3 or config.get('memory_ports') != 3 or not config.get('hbm_dma'):
            raise ValueError('candidate requires three cores/outlets and ownership-capable DMA')
        if args.functional_rtl:
            if request.get('image_sha256') is not None or 'xclbin' in config:
                raise ValueError('functional requests cannot name a board image')
        elif (config.get('hbm_bytes', 0) < 1 << 30 or config.get('device_clock') is not True
              or sha256_file(config['xclbin']) != request['image_sha256']):
            raise ValueError('candidate image hash, clock or allocation request mismatch')
        base = request.get('base_knobs', {})
        reserved = {'TNCA_BACKEND','TNCA_BACKEND_ARGS','TNCA_PROFILE','TNCA_OPT_LEVEL',
                    'TNCA_TRITON','TNCA_TORCH_DEVICE','TNCA_TRANSFER_POLICY','TNCA_ACTIVE_CORES',
                    'TNCA_MAX_BLOCK','TNCA_TENSOR_DESCRIPTORS','TNCA_LAUNCH',
                    'TNCA_HOST_TIMEOUT','TNCA_DEADLINE','TNCA_COUNTERS','TNCA_TUNE','TNCA_DATA_DIR'}
        if any(not k.startswith('TNCA_') or not isinstance(v,str) or k in reserved for k,v in base.items()):
            raise ValueError('base knobs must be explicit strings and cannot replace worker invariants')
        for key in list(os.environ):
            if key.startswith('TNCA_'):
                del os.environ[key]
        os.environ.update(base)
        os.environ.update(TNCA_BACKEND='fpga-multicore-sim' if args.functional_rtl else 'fpga-u280-multicore',
                          TNCA_BACKEND_ARGS=json.dumps(config), TNCA_PROFILE='core-vrf-sync-v5',
                          TNCA_OPT_LEVEL='2', TNCA_TRITON='1', TNCA_TORCH_DEVICE='native',
                          TNCA_TRANSFER_POLICY='resident', TNCA_ACTIVE_CORES='3',
                          TNCA_MAX_BLOCK=str(request['max_block']), TNCA_TENSOR_DESCRIPTORS='1',
                          TNCA_COUNTERS='0', TNCA_TUNE='0',
                          TNCA_HOST_TIMEOUT='7200', TNCA_DEADLINE='4000000000', CUDA_VISIBLE_DEVICES='')
        origins = source_origins()
        info.write_text(json.dumps(dict(request=request, module_origins=origins,
            request_sha256=sha256_file(args.request),worker_sha256=sha256_file(Path(__file__)),
            contract_sha256=contract_v2.LOCK_SHA256,
            timing_policy=dict(fixed_candidate=True,recursive_tuning=False,
                               optional_counter_reads=False,host_time_ranking=False),
            scope='functional-RTL-only' if args.functional_rtl else 'board-candidate-only',
            gate_status='not-measured'),indent=2)+'\n')
        from archbench import registry
        from archbench.platforms import tnca
        actual = registry.get(point['id'])
        if actual.definition_sha256() != point['sha256']:
            raise ValueError('registry no longer matches the frozen point')
        record = tnca.run(actual,profile='core-vrf-sync-v5',opt_level=2,seed=0,
                          warmup=1,timed=1,bundle=str(bundle))
        raw.write_text(json.dumps(record,indent=2)+'\n')  # Keep failed compiler/numeric results too.
        records = validate_result(record,request,functional=args.functional_rtl)
        if software_tree(refresh=True) != request['software_tree']:
            raise ValueError('worker source changed during the experiment')
        with events.open('x') as stream:
            json.dump(records,stream)
            stream.write('\n')
        return 0
    except Exception:
        error.write_text(json.dumps(dict(status='candidate-refused',gate_status='not-measured',
                                         error=traceback.format_exc()),indent=2)+'\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
