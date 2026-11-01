#!/usr/bin/env python3
"""Compare two complete argument-upload TLM results, retaining both inputs."""
import argparse
import json
from pathlib import Path


def compare(control, candidate):
    problems = []
    for name, row, enabled in (('control', control, False), ('candidate', candidate, True)):
        if row.get('status') != 'pass' or row.get('cache_enabled') != enabled:
            problems.append(f'{name} did not pass with the requested setting')
        if row.get('bit_mismatches') != 0 or row.get('host_read_calls') != 0 or row.get('host_fallbacks'):
            problems.append(f'{name} failed bitwise or no-host checks')
    for key in ('launches', 'instructions'):
        if control.get(key) != candidate.get(key):
            problems.append(f'{key} changed')
    if candidate['argument_uploads']['uploads_attempted'] >= control['argument_uploads']['uploads_attempted']:
        problems.append('argument upload attempts did not decrease')
    return dict(status='fail' if problems else 'pass', problems=problems,
                scope='uncalibrated DP TLM pointwise microbenchmark, not board or model throughput',
                control=control, candidate=candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.control.read_text()), json.loads(args.candidate.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result['status'] != 'pass'


if __name__ == '__main__':
    raise SystemExit(main())
