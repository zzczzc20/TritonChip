#!/usr/bin/env python3
"""Queue an interactive full-model chat on the already loaded run-2 image."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bench_mmio_board import PROFILE, board_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('vl', 'q35'), required=True)
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--bucket', type=int, default=256)
    parser.add_argument('--prefill-chunk', type=int, default=0)
    args = parser.parse_args()
    if args.new_tokens < 1 or args.bucket <= args.new_tokens or args.prefill_chunk < 0:
        parser.error('use a positive token count, a larger bucket and a nonnegative prefill chunk')

    root = Path(__file__).resolve().parents[3]
    work = Path.home() / 'r41run3'
    sys.path.insert(0, str(root / 'tools/r41/run2_board'))
    import auto
    sysfs, monitor = Path('/sys/bus/pci/devices'), work / 'idle_monitor.json'
    print('Waiting for 15 minutes of board idleness and the shared queue lock.', flush=True)
    with (work / 'candidate_queue.lock').open('a') as lock:
        while True:
            if board_idle(auto, monitor, sysfs):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if board_idle(auto, monitor, sysfs):
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(60)
        env = os.environ.copy()
        env.pop('TNCA_LAUNCH', None)
        env.update(TNCA_BOARD='u280', TNCA_TORCH_DEVICE='native',
                   TNCA_BACKEND='fpga-u280-multicore', TNCA_PROFILE=PROFILE,
                   TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='6000',
                   TNCA_OPERATING_CLOCK_EVIDENCE=str(work / 'qualified_clock.json'),
                   HF_HOME=str(Path.home() / 'r41s2_fp8_inputs/hf'), HF_HUB_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES='')
        env.setdefault('TNCA_COMPILE_CACHE_DIR', str(work / 'interactive_compile_cache'))
        env['TNCA_BACKEND_ARGS'] = json.dumps(dict(xclbin=str(work / 'tnca_u280.xclbin'),
            image_profile=PROFILE, core_count=3, memory_ports=3, hbm_dma=True,
            hbm_bytes=8 * (1 << 30), device_clock=True))
        paths = (root / 'python', root / 'compiler/python', root / 'tools/asm',
                 root / 'tools/performance_harness/systemc')
        env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
        command = [sys.executable, str(root / 'tools/models/chat.py'), '--model', args.model,
                   '--device', 'tnca', '--tnca-tree', str(root), '--interactive',
                   '--threads', '2', '--new-tokens', str(args.new_tokens), '--bucket', str(args.bucket),
                   '--prefill-chunk', str(args.prefill_chunk)]
        print('Board ready. Loading the full model; first compilation can take several minutes.', flush=True)
        return subprocess.call(command, cwd=root, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
