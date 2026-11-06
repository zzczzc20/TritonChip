#!/usr/bin/env python3
"""Small queued FPGA proof of device-side EOS suppression; never programs an image."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import struct
import sys
import time

from bench_mmio_board import board_idle, IMAGE_UUID, PROFILE


def work(builder, target):
    args = builder.special('arg_base')
    output = builder.sload(args, 0)
    value = builder.sload(args, 8)
    with builder.loop(builder.const(0), 100000, 'work_loop'):
        builder.comment('deterministic work that must disappear after EOS')
    builder.sstore(output, value)
    builder.fence()
    builder.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from tnca import aot
    from tnca.torch_device.eos import _latch_eos
    work_path, latch_path = args.out / 'work.tnca', args.out / 'latch.tnca'
    aot.compile_aot(work, kernel_name='guarded_work', signature={'OUT': '*i64', 'VALUE': 'i64'},
                    source_kind='tnca-ir', profile=PROFILE, options={'rctx_spmd': 0}, output=str(work_path))
    aot.compile_aot(_latch_eos, kernel_name='_latch_eos',
                    signature={'TOKEN': '*i64', 'FLAG': '*i64', 'EOS': 'constexpr'},
                    constexprs={'EOS': (2,)}, profile=PROFILE,
                    options={'rctx_spmd': 0}, output=str(latch_path))
    sys.path.insert(0, str(args.source_root / 'tools/r41/run2_board'))
    import auto
    from tnca.runtime import Session
    run = Path.home() / 'r41run3'
    sysfs = Path('/sys/bus/pci/devices')
    print('EOS probe compiled; waiting for 15-minute idle interval and queue lock', flush=True)
    with (run / 'candidate_queue.lock').open('a') as lock:
        while True:
            if board_idle(auto, run / 'idle_monitor.json', sysfs):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if board_idle(auto, run / 'idle_monitor.json', sysfs):
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(60)
        current = auto.probe(sysfs)
        if not auto.idle(current) or current['uuid'] != IMAGE_UUID or auto.clock_mhz(sysfs) != 100:
            raise RuntimeError('board became busy or changed before the EOS probe')
        with Session(profile=PROFILE, backend='fpga-u280-multicore',
                     xclbin=str(args.image), image_profile=PROFILE, core_count=3,
                     memory_ports=3, hbm_dma=True, hbm_bytes=8 * (1 << 30), host_timeout=180) as session:
            marker, worker = session.load(str(latch_path)), session.load(str(work_path))
            tokens, flag, output = session.allocate(24), session.allocate(8), session.allocate(8)
            session.upload(tokens, struct.pack('<3q', 1, 2, 3))
            session.upload(flag, bytes(8))
            session.upload(output, bytes(8))
            events, previous = [], None
            started = time.perf_counter()
            with session.device_stop_guard(flag):
                for index, value in enumerate((11, 22, 33)):
                    marker_event = session.submit(marker, {'TOKEN': (tokens, index * 8), 'FLAG': flag},
                                                  depends_on=[previous] if previous else [])
                    work_event = session.submit(worker, {'OUT': output, 'VALUE': value},
                                                depends_on=[marker_event])
                    events.append((marker_event, work_event))
                    previous = work_event
                session.wait(previous)
            elapsed = time.perf_counter() - started
            if any(not event.ok for pair in events for event in pair):
                raise RuntimeError('device EOS probe had a failed launch')
            result_value = struct.unpack('<q', session.download(output))[0]
            flag_value = struct.unpack('<q', session.download(flag))[0]
            rows = [dict(token=token, marker_cycles=marker_event.device_cycles,
                         work_cycles=work_event.device_cycles)
                    for token, (marker_event, work_event) in zip((1, 2, 3), events)]
            valid = result_value == 11 and flag_value == 1 and rows[1]['work_cycles'] < rows[0]['work_cycles']
            result = dict(status='pass' if valid else 'fail', domain='FPGA synthetic device EOS guard',
                          image_uuid=IMAGE_UUID, clock_hz=100000000, elapsed_seconds=elapsed,
                          output=result_value, flag=flag_value, rows=rows,
                          scope='one non-EOS token followed by EOS and another token; tensor reads only after completion',
                          limitation='not a model-quality or end-user token-rate measurement')
            (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            if not valid:
                raise RuntimeError('device EOS guard produced incorrect output or did not skip work')


if __name__ == '__main__':
    main()
