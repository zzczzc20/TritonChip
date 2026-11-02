#!/usr/bin/env python3
"""Queue one small, real-model cold-to-warm generation comparison on the current image.

The registered idle monitor and queue lock protect other FPGA users. The test
never changes the image and labels its one-layer results as reduced depth.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bench_mmio_board import board_idle, IMAGE_UUID, PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--overlay-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--prompt', default='What is the capital of France? Answer with one word.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_root / 'tools/r41/run2_board'))
    import auto
    work = Path.home() / 'r41run3'
    sysfs = Path('/sys/bus/pci/devices')
    print('waiting for 15-minute idle interval and the shared queue lock', flush=True)
    with (work / 'candidate_queue.lock').open('a') as lock:
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
        env = os.environ.copy()
        env.update(TNCA_BOARD='u280', TNCA_TORCH_DEVICE='native', TNCA_BACKEND='fpga-u280-multicore',
                   TNCA_PROFILE=PROFILE, TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='6000',
                   TNCA_OPERATING_CLOCK_EVIDENCE=str(work / 'qualified_clock.json'),
                   HF_HOME=str(Path.home() / 'r41s2_fp8_inputs/hf'), HF_HUB_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES='', TNCA_COMPILE_CACHE_DIR=str(args.out / 'compile-cache'))
        env['TNCA_BACKEND_ARGS'] = json.dumps(dict(xclbin=str(work / 'tnca_u280.xclbin'),
            image_profile=PROFILE, core_count=3, memory_ports=3, hbm_dma=True,
            hbm_bytes=8 * (1 << 30), device_clock=True))
        reports = []
        for mode in ('baseline-cold', 'optimized-warm'):
            # Another client may not use this cooperative queue lock. Refuse
            # an occupied or changed device before starting either process;
            # XRT's exclusive context arbitrates the remaining open race.
            current = auto.probe(sysfs)
            if (not auto.idle(current) or current['uuid'] != IMAGE_UUID
                    or auto.clock_mhz(sysfs) != 100):
                raise RuntimeError(f'board became busy or changed before {mode}')
            run_env = dict(env)
            if mode == 'baseline-cold':
                run_env.update(TNCA_REUSE_WARM_PREFILL='0', TNCA_REUSE_HYBRID_PREFILL='0',
                               TNCA_MMIO_DESCRIPTOR_CACHE='0')
            else:
                for name in ('TNCA_REUSE_WARM_PREFILL', 'TNCA_REUSE_HYBRID_PREFILL',
                             'TNCA_MMIO_DESCRIPTOR_CACHE'):
                    run_env.pop(name, None)
            out = args.out / mode
            command = [str(args.python), '-u', str(args.overlay_root / 'tools/models/chat.py'),
                       '--model', 'vl', '--device', 'tnca', '--tnca-tree', str(args.overlay_root),
                       '--layers', '1', '--prefill-chunk', '8', '--new-tokens', '4',
                       '--bucket', '256', '--threads', '2', '--prompt', args.prompt, '--out', str(out)]
            print('starting', mode, datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
            with (args.out / f'{mode}.log').open('x') as log:
                rc = subprocess.run(command, cwd=args.overlay_root, env=run_env,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            print('finished', mode, 'returncode', rc, flush=True)
            if rc != 0:
                raise RuntimeError(f'{mode} failed; preserved its log and any failure report')
            reports.append(json.loads((out / 'report.json').read_text()))
        from safetensors.torch import load_file
        import torch
        a = load_file(str(args.out / 'baseline-cold/logits.safetensors'))
        b = load_file(str(args.out / 'optimized-warm/logits.safetensors'))
        exact = a.keys() == b.keys() and all(torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8)) for k in a)
        tokens_equal = reports[0]['generated_tokens'] == reports[1]['generated_tokens']
        rows = [dict(mode=mode, status=report.get('status'), tokens=report.get('generated_tokens'),
                     timing=report.get('user_visible_timing'),
                     cpu_numerical_fallback=report.get('cpu_numerical_fallback'),
                     decode_host_reads=[(s.get('transfers_inside_step') or {}).get('host_read_calls')
                                        for s in report.get('steps', []) if s.get('kind') == 'decode'])
                for mode, report in zip(('baseline-cold', 'optimized-warm'), reports)]
        status = 'pass' if exact and tokens_equal and all(row['status'] == 'pass' for row in rows) else 'fail'
        result = dict(status=status, domain='FPGA board; reduced one-layer VL checkpoint', image_uuid=IMAGE_UUID,
                      clock_hz=100000000, scope='cold process then warm-cache process; not a matched warm A/B',
                      source='speed worktree overlay b13cbc5f with frozen compiler 02d704f4',
                      prompt=args.prompt, logits_bitwise_equal=exact, tokens_equal=tokens_equal, rows=rows)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if status != 'pass':
            raise RuntimeError('reduced-model comparison failed')


if __name__ == '__main__':
    main()
