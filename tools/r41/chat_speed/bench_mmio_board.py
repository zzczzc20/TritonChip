#!/usr/bin/env python3
"""Small real-board launch A/B with the existing image; never programs it.

Compile the same one-word store before opening the device, then time repeated
resident launches with descriptor reuse disabled and enabled in one exclusive
session. The comparison is kernel-launch latency, not model tokens/s.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import struct
import sys
import time

PROFILE = 'core-vrf-sync-v5-x8-g2c4-dp'
IMAGE_UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'


def store_value(builder, target):
    base = builder.special('arg_base')
    output = builder.sload(base, 0)
    value = builder.sload(base, 8)
    builder.sstore(output, value, size='d')
    builder.fence()
    builder.exit()


def compile_probe(path):
    from tnca import aot
    aot.compile_aot(store_value, kernel_name='mmio_store_value',
                    signature={'output': '*i64', 'value': 'i64'},
                    source_kind='tnca-ir', profile=PROFILE,
                    options={'rctx_spmd': 0}, output=str(path))


def board_idle(auto, monitor, sysfs):
    try:
        row = auto.probe(sysfs)
        data = json.loads(monitor.read_text())
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        stamp = time.monotonic()
        return (auto.idle(row) and row['uuid'] == IMAGE_UUID
                and auto.clock_mhz(sysfs) == 100
                and data.get('boot_id') == boot
                and data.get('identity') == [row['uuid'], row['total execs']]
                and data.get('start') is not None and data.get('last') is not None
                and 0 <= stamp - data['last'] <= 15
                and data['last'] - data['start'] >= 900)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 1000 or args.out.exists() or args.artifact.exists():
        raise ValueError('invalid repetition count or existing output')
    compile_probe(args.artifact)  # No FPGA context is acquired here.
    sys.path.insert(0, str(args.source_root / 'tools/r41/run2_board'))
    import auto
    from tnca.runtime import Session
    work = Path.home() / 'r41run3'
    sysfs = Path('/sys/bus/pci/devices')
    lock_path = work / 'candidate_queue.lock'
    print('probe compiled; waiting for idle FPGA and queue lock', flush=True)
    with lock_path.open('a') as lock:
        while True:
            if board_idle(auto, work / 'idle_monitor.json', sysfs):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if board_idle(auto, work / 'idle_monitor.json', sysfs):
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(60)
        rows = []
        with Session(profile=PROFILE, backend='fpga-u280-multicore',
                     xclbin=str(args.image), image_profile=PROFILE, core_count=3,
                     memory_ports=3, hbm_dma=True, hbm_bytes=8 * (1 << 30),
                     host_timeout=180) as session:
            module = session.load(str(args.artifact))
            output = session.allocate(8)
            session.upload(output, bytes(8))
            link = session.dev.link
            if getattr(link, '_request_register_shadow', None) is None:
                raise RuntimeError('optimized exclusive XRT link was not loaded')
            for name, enabled in (('baseline', False), ('optimized', True)):
                link._request_register_shadow = {} if enabled else None
                first_write, first_reuse = link.request_register_writes, link.request_register_reuses
                cycles = 0
                started = time.perf_counter()
                for value in range(args.repetitions):
                    event = session.submit(module, {'output': output, 'value': value}, grid=(1, 1, 1))
                    session.wait(event)
                    if not event.ok:
                        raise RuntimeError(f'{name}: launch failed: {event.error}')
                    cycles += event.device_cycles or 0
                elapsed = time.perf_counter() - started
                writes = link.request_register_writes - first_write
                reused = link.request_register_reuses - first_reuse
                actual = struct.unpack('<q', session.download(output))[0]
                if actual != args.repetitions - 1:
                    raise RuntimeError(f'{name}: result {actual} differs from reference')
                rows.append(dict(mode=name, launches=args.repetitions, seconds=elapsed,
                                 seconds_per_launch=elapsed / args.repetitions,
                                 logical_register_writes=writes, descriptor_writes_avoided=reused,
                                 device_cycles=cycles, last_value=actual))
                print(json.dumps(rows[-1]), flush=True)
        result = dict(status='pass', domain='FPGA board, synthetic resident store launches',
                      image_uuid=IMAGE_UUID, clock_hz=100000000, source='c14b4317:python/tnca/backends/u50.py over frozen runtime 02d704f4',
                      recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      rows=rows, speedup=rows[0]['seconds'] / rows[1]['seconds'],
                      limitations=['not a Qwen model or end-user token measurement',
                                   'output downloaded once after each timed launch sequence'])
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
