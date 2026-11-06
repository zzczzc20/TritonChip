#!/usr/bin/env python3
"""Cycle-level capacity model for the independent per-PC AXI stream engines."""
import argparse
from collections import deque
import json
from pathlib import Path
import random
import re


def stream_lane(mode, lines, burst=16, credits=128, burst_out=8,
                read_latency=25, write_latency=19, stalls=False, seed=4141):
    rng = random.Random(seed)
    read_left = 0 if mode == 'fill' else lines
    write_left = 0 if mode == 'reduce' else lines
    reads = deque()
    writes = deque()
    responses = deque()
    reserved = 0
    buffered = 0
    write_bursts = 0
    completed_writes = 0
    cycle = 0
    peak_buffered = 0
    credit_wait = 0
    metadata_wait = 0
    data_wait = 0

    def ready(percent):
        return not stalls or rng.randrange(100) >= percent

    while cycle < 100000000:
        cycle += 1
        old_read = reads[0] if reads else None
        old_write = writes[0] if writes else None
        r_fire = False
        w_fire = False

        if responses and responses[0] <= cycle:
            responses.popleft()
            write_bursts -= 1
            completed_writes += 1

        if old_read is not None and old_read['due'] <= cycle and ready(20):
            if mode != 'copy' or buffered < credits:
                r_fire = True
                old_read['left'] -= 1
                if mode == 'copy':
                    buffered += 1
                if old_read['left'] == 0:
                    reads.popleft()

        if old_write is not None and ready(45) and (mode == 'fill' or buffered > 0):
            w_fire = True
            old_write['left'] -= 1
            if mode == 'copy':
                buffered -= 1
                reserved -= 1
            if old_write['left'] == 0:
                writes.popleft()
                responses.append(cycle + write_latency)

        if mode != 'copy' and r_fire:
            reserved -= 1

        if read_left and ready(35):
            length = min(burst, read_left, 64 - ((lines - read_left) % 64))
            if len(reads) < burst_out and reserved + length <= credits:
                reads.append({'left': length, 'due': cycle + read_latency})
                read_left -= length
                reserved += length
            elif reserved + length > credits:
                credit_wait += 1
            else:
                metadata_wait += 1

        if write_left and ready(40):
            length = min(burst, write_left, 64 - ((lines - write_left) % 64))
            if len(writes) < burst_out and write_bursts < burst_out:
                writes.append({'left': length})
                write_left -= length
                write_bursts += 1
            else:
                metadata_wait += 1

        if mode == 'copy' and writes and buffered == 0:
            data_wait += 1
        peak_buffered = max(peak_buffered, buffered)
        if (read_left == 0 and write_left == 0 and not reads and not writes
                and not responses and reserved == 0 and buffered == 0):
            return {'cycles': cycle, 'lines': lines, 'peak_buffered': peak_buffered,
                    'credit_wait_cycles': credit_wait, 'metadata_wait_cycles': metadata_wait,
                    'copy_data_wait_cycles': data_wait}
    raise RuntimeError('stream model did not drain')


def result_for(mode, lanes, lines, args):
    per_lane = [stream_lane(mode, lines, burst=args.burst, credits=args.credits,
                burst_out=args.burst_out, read_latency=args.read_latency,
                write_latency=args.write_latency, stalls=args.stalls,
                seed=args.seed + pc * 7919) for pc in range(lanes)]
    cycles = max(row['cycles'] for row in per_lane) + lanes + 2
    directions = (0 if mode == 'fill' else 1) + (0 if mode == 'reduce' else 1)
    transferred = lanes * lines * 64 * directions
    bandwidth = transferred / cycles * 0.1
    return {'mode': mode, 'lanes': lanes, 'lines_per_lane': lines,
            'cycles': cycles, 'transferred_bytes': transferred,
            'gb_per_s_at_100mhz': bandwidth,
            'conservative_derated_gb_per_s': bandwidth * 0.9,
            'meets_50gbps_derated': bandwidth * 0.9 >= 50,
            'per_lane': per_lane, 'board_measured': False}


def validate_rtl(path):
    text = path.read_text()
    rows = []
    pattern = re.compile(r'STREAM COMMAND PASS mode=(\d+) lines=(\d+) status=0 cycles=(\d+)')
    names = {0: 'copy', 1: 'fill', 2: 'reduce'}
    for match in pattern.finditer(text):
        mode, lines, cycles = map(int, match.groups())
        if lines < 128:
            continue
        predicted = stream_lane(names[mode], lines, stalls='stall=1' in text)
        rows.append({'mode': names[mode], 'lines': lines, 'rtl_cycles': cycles,
                     'model_cycles': predicted['cycles'],
                     'model_over_rtl': predicted['cycles'] / cycles})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lines', type=int, default=16384)
    parser.add_argument('--burst', type=int, default=16)
    parser.add_argument('--credits', type=int, default=128)
    parser.add_argument('--burst-out', type=int, default=8)
    parser.add_argument('--read-latency', type=int, default=25)
    parser.add_argument('--write-latency', type=int, default=19)
    parser.add_argument('--seed', type=int, default=4141)
    parser.add_argument('--stalls', action='store_true')
    parser.add_argument('--rtl-log', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [result_for(mode, lanes, args.lines, args)
            for lanes in (8, 16, 32) for mode in ('copy', 'fill', 'reduce')]
    result = {'model': 'cycle-level-independent-per-PC-stream-lanes',
              'configuration': {'burst': args.burst, 'credits': args.credits,
                  'burst_out': args.burst_out, 'read_latency': args.read_latency,
                  'write_latency': args.write_latency, 'stalls': args.stalls,
                  'pc_stripe_bytes': 4096, 'clock_hz': 100000000},
              'scope': 'native stream commands only; not core or torch throughput',
              'rows': rows,
              'rtl_validation': [row for path in args.rtl_log for row in validate_rtl(path)]}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != 'per_lane'}, sort_keys=True))


if __name__ == '__main__':
    main()
