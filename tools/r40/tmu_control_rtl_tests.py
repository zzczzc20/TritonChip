#!/usr/bin/env python3
"""Probe existing TMU access through public Session and real three-core RTL.

This isolates the runtime-control transport. It does not implement or certify
standard Torch DCO lifetime analysis or frozen workload performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))
import triton  # Load the compiler plugin before its runtime adapters.
from tnca import aot
from tnca.runtime import Session

TMU = 0x7FFF0000


def _policy(b, clear):
    args = b.special('arg_base')
    out, value = b.sload(args, 0), b.sload(args, 8)
    tmu = b.const(TMU)
    if clear:
        b.sstore(tmu, b.const(0), offset=0x18)
        b.fence()
    b.sstore(tmu, value)
    b.fence()
    b.sstore(out, b.sload(tmu), size='d')
    b.sstore(out, b.sload(tmu, 0x20), offset=8, size='d')
    b.sstore(out, b.sload(tmu, 0x28), offset=16, size='d')
    b.fence()
    b.exit()


def reset_policy(b, target):
    _policy(b, True)


def set_policy(b, target):
    _policy(b, False)


def register_range(b, target):
    args = b.special('arg_base')
    ranges, out = b.sload(args, 0), b.sload(args, 8)
    offset = b.sbini('mul', b.program_id(0), 64)
    tmu = b.const(TMU)
    b.sstore(tmu, b.sbin('add', ranges, offset), offset=0x100)
    b.sstore(tmu, b.const(64), offset=0x108)
    # Keep every tile live and disable bypass: probe transport without
    # asserting a last-use fact about any live allocation.
    b.sstore(tmu, b.const((6 << 40) | 0xFFFFFFFF), offset=0x110)
    b.fence()
    dest = b.sbin('add', out, offset)
    b.sstore(dest, b.sload(tmu, 0x20), size='d')
    b.sstore(dest, b.sload(b.sbin('add', ranges, offset), 56), offset=8, size='d')
    b.fence()
    b.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope='functional TMU transport only', gate_status='not-measured',
                  physical_image_sha256=None, measured_performance=False, results=[],
                  contract_sha256=hashlib.sha256((ROOT/'spec/r40/v2/lock.json').read_bytes()).hexdigest())
    paths = {}
    for name, callback, signature in (
        ('reset', reset_policy, {'out':'*i64', 'policy':'i64'}),
        ('policy', set_policy, {'out':'*i64', 'policy':'i64'}),
        ('register', register_range, {'ranges':'*i8', 'out':'*i64'})):
        paths[name] = aot.compile_aot(callback, kernel_name='tmu_'+name,
            source_kind='tnca-ir', profile='core-vrf-sync-v5', signature=signature,
            output=str(out/(name+'.tnca'))).path
    report['artifact_sha256'] = {name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name,path in paths.items()}
    with Session(profile='core-vrf-sync-v5', backend='fpga-multicore-sim',
                 core_count=3, memory_ports=3, maintenance=True, hbm_dma=True,
                 board_top=True, memory_controls=True, host_timeout=1800) as session:
        modules = {name:session.load(path) for name,path in paths.items()}
        result, ranges, records = session.allocate(24), session.allocate(192), session.allocate(192)
        session.upload(ranges, bytes(192))
        session.upload(records, bytes(192))
        def policy(value, reset=False):
            event = session.wait(session.submit(modules['reset' if reset else 'policy'],
                                                {'out':result, 'policy':value}))
            assert event.ok, event.error
            got = struct.unpack('<QQQ', session.download(result))
            assert got[0] == value, got
            report['results'].append(dict(operation='policy', value=value, registered=got[1], tile_notices=got[2]))
            return got[1], got[2]
        before, notices_before = policy(4, reset=True)
        for expected in (before+3, before+6):
            event = session.wait(session.submit(modules['register'], {'ranges':ranges, 'out':records}, grid=(3,1,1)))
            assert event.ok, event.error
            # These counters reset on each launch: sample before the next
            # policy/readback helper, which is itself a one-program grid.
            metrics = session.dev.physical_grid_metrics()
            assert metrics['peak_active'] == 3, metrics
            assert all(row['issued'] == row['completed'] == 1 for row in metrics['cores']), metrics
            actual, notices = policy(4)
            assert actual == expected, (actual, expected)
            assert notices == notices_before + expected - before, (notices, expected)
            values = struct.unpack('<24Q', session.download(records))
            assert all(before < values[i*8] <= expected for i in range(3)), values
            assert all(values[i*8+1] == 0 for i in range(3)), values
            report['results'].append(dict(operation='register', total=actual,
                tile_notices=notices, per_program=[values[i*8] for i in range(3)], physical_grid=metrics))
        for value in (1, 2, 0x22, 0):
            assert policy(value)[0] == before+6
        policy(0, reset=True)
        report['final_control_grid'] = session.dev.physical_grid_metrics()
        report['simulator_directory'] = session.dev.link.dir
        report['status'] = 'functional-pass'
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('TMU CONTROL TRANSPORT PASS; standard Torch DCO and board gates not-measured', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
