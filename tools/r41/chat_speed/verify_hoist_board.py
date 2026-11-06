#!/usr/bin/env python3
"""Validate TNCA placement and decode-read evidence after the hoist board run."""
import json
import os
from pathlib import Path
import select

WORK = Path.home() / 'r41run3'
OUT = WORK / 'chat_hoist_precise_20261031'
REPORT = OUT / 'session/r41-q35-hoist-session-20261031/q35-session/report.json'
RESULT = OUT / 'placement-check.json'


def check(report):
    errors = []
    if report.get('model_scope') != 'full-checkpoint':
        errors.append('not a full-checkpoint result')
    if report.get('fp8_route_in_use', {}).get('kind') != 'experimental-tnca-hoist':
        errors.append('hoist route was not selected')
    rounds = [report]
    session = report.get('persistent_session_check', {})
    rounds.extend(session.get(k, {}) for k in ('other', 'repeat'))
    for i, row in enumerate(rounds):
        if row.get('status') != 'pass' or row.get('cpu_numerical_fallback') is not False:
            errors.append(f'round {i}: status or CPU fallback')
        fallbacks = row.get('compiled_wrapper_fallbacks')
        if (row.get('compiled_wrapper_roles_covered') is not True or
                not isinstance(fallbacks, dict) or
                not {'prefill', 'decode', 'selection'}.issubset(fallbacks) or any(fallbacks.values())):
            errors.append(f'round {i}: compiled wrapper evidence')
        for kind in ('parameters', 'buffers'):
            placements = row.get('model_tensor_placements', {}).get(kind)
            if (not isinstance(placements, dict) or not placements or
                    any(not name.startswith('tnca:') or type(count) is not int or count <= 0
                        for name, count in placements.items())):
                errors.append(f'round {i}: {kind} placement')
        decode = [s for s in row.get('steps', []) if s.get('kind') == 'decode']
        if not decode or any(s['transfers_inside_step']['host_read_calls'] != 0 or
                             s['transfers_inside_step']['staging_download_bytes'] != 0 for s in decode):
            errors.append(f'round {i}: decode host reads')
    prefill = [s for s in report.get('steps', []) if s.get('kind') == 'prefill']
    launches = sum(s.get('transfers_inside_step', {}).get('by_kernel', {})
                   .get('_precise_block_quant', {}).get('launches', 0) for s in prefill)
    if launches < 1:
        errors.append('hoisted quantisation did not run')
    return errors, launches


def main():
    try:
        fd = os.pidfd_open(1205212)
    except ProcessLookupError:
        pass
    else:
        try:
            select.select([fd], [], [])
        finally:
            os.close(fd)
    stage = OUT / 'session/stage.json'
    if not stage.is_file() or json.loads(stage.read_text()).get('status') != 'pass' or not REPORT.is_file():
        result = dict(status='not-run', reason='registered hoist board stage did not pass')
    else:
        errors, launches = check(json.loads(REPORT.read_text()))
        result = dict(status='failed' if errors else 'pass', errors=errors,
                      hoisted_prefill_launches=launches, report=str(REPORT))
    with RESULT.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
