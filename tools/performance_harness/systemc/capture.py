#!/usr/bin/env python3
"""Capture the real randomized NUCA test and export lossless word transactions.

Runs on GB10. The monitor is another simulation top; product RTL is untouched.
RTL numerical responses are checked against a second byte-memory oracle before
they become expected values for the TLM replay.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run import HERE, ROOT, sha


def convert(raw, output):
    requests, pending, memory = [], {}, {}
    header = terminal = False
    last_cycle = 0
    for line in raw.read_text().splitlines():
        f = line.split()
        if f == ['TNCA_NUCA_CAPTURE', '1']:
            if header:
                raise ValueError('duplicate header')
            header = True
            continue
        if not header or terminal:
            raise ValueError('bad capture boundary')
        if f[0] == 'Q' and len(f) == 9:
            seq, cycle, src, tag, write = map(int, f[1:6])
            address, data, be = (int(v, 16) for v in f[6:9])
            if seq != len(requests)+1 or (src, tag) in pending or cycle < last_cycle or write not in (0, 1):
                raise ValueError('bad request sequence/tag/time')
            if address % 8 or not 0 <= be <= 255:
                raise ValueError('invalid address/byte enables')
            old = memory.get(address, 0)
            request = dict(id=seq, cycle=cycle, source=src, tag=tag, op=write,
                           address=address, data=data, be=be, expected=old)
            requests.append(request)
            pending[src, tag] = request
            if write:
                mask = sum(255 << (b*8) for b in range(8) if be & (1 << b))
                memory[address] = (old & ~mask) | (data & mask)
            last_cycle = cycle
        elif f[0] == 'P' and len(f) == 5:
            cycle, src, tag = map(int, f[1:4])
            value = int(f[4], 16)
            request = pending.pop((src, tag))
            if cycle < request['cycle'] or (not request['op'] and value != request['expected']):
                raise ValueError('RTL response disagrees with independent issue-order memory oracle')
            request['response_cycle'] = cycle
        elif f[0] == 'END' and len(f) == 3:
            if int(f[1]) != len(requests) or pending:
                raise ValueError('partial capture')
            terminal = True
        else:
            raise ValueError('invalid capture record')
    if not header or not terminal or not requests:
        raise ValueError('incomplete capture')
    with output.open('x') as out:
        out.write('# tnca-word-trace/1; accepted standalone home-0 requests, source remapped to core 0\n')
        for r in requests:
            out.write(f"{r['id']} {r['cycle']} 0 {r['op']} {r['address']:#x} {r['data']:#x} "
                      f"{r['be']:#x} 0 {int(not r['op'])} {r['expected']:#x} T\n")
    return dict(requests=len(requests), reads=sum(not r['op'] for r in requests),
                rtl_completion_cycle=max(r['response_cycle'] for r in requests),
                scope='standalone NUCA accepted requests; source remapping removes NoC latency; response values independently checked',
                timing_calibration='not-measured')


def capture(out, count=8000, seed=7, latency=40):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out/'capture.vvp'
    files = [ROOT/'tb/nuca/tb_nuca_slice.v', ROOT/'rtl/nuca/tnca_nuca_slice.v',
             ROOT/'tb/common/tnca_dram_model.v', HERE/'rtl_capture.v']
    sources = files + sorted((ROOT/'rtl/common').glob('*.v*'))
    identity = dict(source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    count=count, seed=seed, latency=latency,
                    iverilog=subprocess.check_output(['iverilog', '-V'], stderr=subprocess.DEVNULL, text=True).splitlines()[0])
    command = ['iverilog', '-g2012', '-I', str(ROOT/'rtl/common'), '-y', str(ROOT/'rtl/common'),
               '-s', 'tb_nuca_slice', '-s', 'tnca_tlm_capture',
               '-P', f'tb_nuca_slice.DRAM_LAT={latency}', '-o', str(binary), *map(str, files)]
    build_start=time.monotonic()
    with (out/'build.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    build_seconds=time.monotonic()-build_start
    command = ['vvp', str(binary), f'+nreq={count}', f'+seed={seed}', f'+tlm_capture={out}/rtl.capture']
    execution_start=time.monotonic()
    with (out/'rtl.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=600)
    execution_seconds=time.monotonic()-execution_start
    log = (out/'rtl.log').read_text()
    if result.returncode or 'FAIL' in log or 'FATAL' in log or '\nPASS\n' not in log:
        raise ValueError('RTL run failed; preserve capture and logs')
    report = convert(out/'rtl.capture', out/'requests.trace')
    report.update(build_seconds=build_seconds,execution_seconds=execution_seconds,
                  host_time_scope='vvp process startup, complete testbench and capture output; compilation and trace conversion excluded',
                  identity=identity, capture_sha256=sha(out/'rtl.capture'), trace_sha256=sha(out/'requests.trace'))
    (out/'capture.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=8000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--latency', type=int, default=40)
    a = p.parse_args()
    if a.count <= 0 or a.latency <= 0:
        p.error('positive request count and latency required')
    print(json.dumps(capture(a.out, a.count, a.seed, a.latency)))
