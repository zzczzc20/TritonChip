#!/usr/bin/env python3
"""Validate archived lifecycle preflight reports and derive a scoped summary."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aliases', type=Path, required=True)
    parser.add_argument('--barrier', type=Path, required=True)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--model-log', type=Path, required=True)
    parser.add_argument('--rtl-log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    aliases = load(args.aliases)
    assert aliases['schema'] == 'tnca-r41-alias-negative/1'
    assert aliases['status'] == 'pass' and aliases['board_acceptance'] == 'not-measured'
    expected_aliases = {
        'undeclared-overlapping-storage-arguments',
        'out-of-bounds-view',
        'forged-high-bit-storage-owner',
        'distinct-storage-handles-overlap',
    }
    assert {row['name'] for row in aliases['checks']} == expected_aliases
    assert all(row['status'] == 'refused-before-admission' for row in aliases['checks'])

    barrier = load(args.barrier)
    assert barrier['schema'] == 'tnca-r41-barrier-epoch/1'
    assert barrier['status'] == 'pass' and barrier['board_acceptance'] == 'not-measured'
    assert barrier['event']['ok'] and barrier['event']['state'] == 'SUCCEEDED'
    assert barrier['requested_arms'] > 65536
    assert barrier['observed']['completed_arm_wait_iterations'] == barrier['requested_arms']

    grid = load(args.grid)
    assert grid['schema'] == 'tnca-r41-grid-admission/1'
    assert grid['status'] == 'pass' and grid['board_acceptance'] == 'not-measured'
    assert grid['requested_programs'] > 4096
    checks = {row['name']: row for row in grid['checks']}
    assert set(checks) == {
        'opt-in-required', 'bounded-queue-overflow',
        'exact-grid-and-drain', 'advertised-capacity-refusal',
    }
    assert all(row['status'] == 'pass' for row in checks.values())
    assert checks['exact-grid-and-drain']['unique_program_ids'] == grid['requested_programs']
    assert checks['exact-grid-and-drain']['completion_atomics'] == (
        grid['requested_programs'] + checks['bounded-queue-overflow']['admitted_dependent'])
    assert checks['advertised-capacity-refusal']['requested'] == (
        checks['advertised-capacity-refusal']['advertised_limit'] + 1)
    job_ids = [event['job_id'] for event in grid['events']]
    assert len(job_ids) == len(set(job_ids))
    assert all(event['state'] == 'SUCCEEDED' for event in grid['events'])

    model_log = args.model_log.read_text()
    eviction = re.findall(r'PASS\s+eviction\s+([^\n]+)', model_log)
    assert len(eviction) == 1, eviction
    assert 'r40-device-tests  PASS  (1/1)' in model_log
    cache = re.search(r'CODE CACHE EPOCH PASS (\d+) RTL launches, one-module region, live buffer preserved', model_log)
    assert cache
    assert 'Traceback' not in model_log

    rtl_log = args.rtl_log.read_text()
    dispatch = re.search(r'r40-grid-dispatch-tests PASS \((\d+) configurations, registered_ranges=(\d+)\)', rtl_log)
    assert dispatch and 'GRID DISPATCH PASS' in rtl_log
    assert 'PASS barrier generations 0..65537: no early release at 16-bit boundary' in rtl_log
    assert 'PASS barrier exhaustion faults before wrap; init recovers' in rtl_log
    assert 'FAIL' not in rtl_log and 'Traceback' not in rtl_log

    result = {
        'schema': 'tnca-r41-lifecycle-closeout/1',
        'status': 'model-and-rtl-preflight-pass',
        'board_acceptance': 'not-measured',
        'software_trees': sorted({aliases['software_tree'], barrier['software_tree'], grid['software_tree']}),
        'aliases': {
            'backend': aliases['backend'],
            'refusals': len(aliases['checks']),
            'report_sha256': hashlib.sha256(args.aliases.read_bytes()).hexdigest(),
        },
        'barrier': {
            'backend': barrier['backend'],
            'completed_arm_wait_iterations': barrier['observed']['completed_arm_wait_iterations'],
            'device_cycles': barrier['event']['device_cycles'],
            'report_sha256': hashlib.sha256(args.barrier.read_bytes()).hexdigest(),
        },
        'grid': {
            'backend': grid['backend'],
            'programs': grid['requested_programs'],
            'completed_events': len(grid['events']),
            'completion_atomics': checks['exact-grid-and-drain']['completion_atomics'],
            'queue_limit': checks['bounded-queue-overflow']['queue_limit'],
            'report_sha256': hashlib.sha256(args.grid.read_bytes()).hexdigest(),
        },
        'standard_torch_eviction': eviction[0],
        'code_cache': {'rtl_launches': int(cache.group(1)),
                       'live_buffer_preserved': True},
        'rtl_grid_configurations': int(dispatch.group(1)),
        'rtl_barrier_wrap_and_exhaustion': 'pass',
        'interpretation': 'Functional model/RTL preflight only. The AOT grid and barrier cases are not standard-Torch or physical-board results. No full-model persistent-session, performance, or board acceptance inference.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'aliases': len(aliases['checks']),
                      'barrier_iterations': barrier['requested_arms'],
                      'grid_programs': grid['requested_programs']}))


if __name__ == '__main__':
    main()
