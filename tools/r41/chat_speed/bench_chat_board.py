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
    parser.add_argument('--baseline-overlay-root', type=Path)
    parser.add_argument('--warm-cache-dir', type=Path)
    parser.add_argument('--device-eos-probe', action='store_true',
                        help='compare warm reduced-model turns with a diagnostic EOS')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', choices=('vl', 'q35'), default='vl')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--new-tokens', type=int, default=4)
    parser.add_argument('--prompt', default='What is the capital of France? Answer with one word.')
    args = parser.parse_args()
    if args.device_eos_probe and (args.baseline_overlay_root or args.warm_cache_dir):
        parser.error('the device EOS probe uses one source overlay and its own cache')
    if bool(args.baseline_overlay_root) != bool(args.warm_cache_dir):
        parser.error('warm A/B requires both a baseline overlay and an existing warm cache')
    if args.warm_cache_dir and not args.warm_cache_dir.is_dir():
        parser.error('warm cache directory does not exist')
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
                   CUDA_VISIBLE_DEVICES='', TNCA_COMPILE_CACHE_DIR=str(
                       args.warm_cache_dir or args.out / 'compile-cache'))
        env['TNCA_BACKEND_ARGS'] = json.dumps(dict(xclbin=str(work / 'tnca_u280.xclbin'),
            image_profile=PROFILE, core_count=3, memory_ports=3, hbm_dma=True,
            hbm_bytes=8 * (1 << 30), device_clock=True))
        reports = []
        modes = (('baseline', 'device-eos-guard') if args.device_eos_probe else
                 ('baseline-warm', 'optimized-warm') if args.warm_cache_dir else
                 ('baseline-cold', 'optimized-warm'))
        for mode in modes:
            # Another client may not use this cooperative queue lock. Refuse
            # an occupied or changed device before starting either process;
            # XRT's exclusive context arbitrates the remaining open race.
            current = auto.probe(sysfs)
            if (not auto.idle(current) or current['uuid'] != IMAGE_UUID
                    or auto.clock_mhz(sysfs) != 100):
                raise RuntimeError(f'board became busy or changed before {mode}')
            run_env = dict(env)
            for name in ('TNCA_REUSE_WARM_PREFILL', 'TNCA_REUSE_HYBRID_PREFILL',
                         'TNCA_MMIO_DESCRIPTOR_CACHE', 'TNCA_CODE_REUSE',
                         'TNCA_DEFER_CACHE_RESET', 'TNCA_REUSE_FIRST_DECODE', 'TNCA_DEVICE_EOS_GUARD'):
                run_env.pop(name, None)
            if mode == 'baseline-cold':
                run_env.update(TNCA_REUSE_WARM_PREFILL='0', TNCA_REUSE_HYBRID_PREFILL='0',
                               TNCA_MMIO_DESCRIPTOR_CACHE='0', TNCA_CODE_REUSE='0', TNCA_DEFER_CACHE_RESET='0')
            if args.device_eos_probe:
                run_env['TNCA_DEVICE_EOS_GUARD'] = '1' if mode == 'device-eos-guard' else '0'
            root = args.baseline_overlay_root if mode == 'baseline-warm' else args.overlay_root
            out = args.out / mode
            if args.device_eos_probe:
                command = [str(args.python), '-u', str(root / 'tools/r41/chat_speed/bench_eos_model_entry.py'),
                           '--tree', str(root), '--model', args.model, '--layers', str(args.layers),
                           '--new-tokens', str(args.new_tokens), '--prompt', args.prompt, '--out', str(out)]
            else:
                command = [str(args.python), '-u', str(root / 'tools/models/chat.py'),
                           '--model', args.model, '--device', 'tnca', '--tnca-tree', str(root),
                           '--layers', str(args.layers), '--prefill-chunk', '8', '--new-tokens', str(args.new_tokens),
                           '--bucket', '256', '--threads', '2', '--prompt', args.prompt, '--out', str(out)]
            print('starting', mode, datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
            with (args.out / f'{mode}.log').open('x') as log:
                rc = subprocess.run(command, cwd=root, env=run_env,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            print('finished', mode, 'returncode', rc, flush=True)
            if rc != 0:
                raise RuntimeError(f'{mode} failed; preserved its log and any failure report')
            reports.append(json.loads((out / 'report.json').read_text()))
        from safetensors.torch import load_file
        import torch
        a = load_file(str(args.out / modes[0] / 'logits.safetensors'))
        b = load_file(str(args.out / modes[1] / 'logits.safetensors'))
        exact = a.keys() == b.keys() and all(torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8)) for k in a)
        tokens_equal = reports[0]['generated_tokens'] == reports[1]['generated_tokens']
        rows = [dict(mode=mode, status=report.get('status'), tokens=report.get('generated_tokens'),
                     timing=report.get('user_visible_timing'),
                     cpu_numerical_fallback=report.get('cpu_numerical_fallback'),
                     decode_host_reads=[(s.get('transfers_inside_step') or {}).get('host_read_calls')
                                        for s in report.get('steps', []) if s.get('kind') == 'decode'],
                     warmup_decode_transfers=[{key: s['warmup_decode_transfers'].get(key)
                                               for key in ('host_read_calls', 'host_read_bytes',
                                                           'staging_download_bytes', 'launches', 'device_cycles')}
                                              for s in report.get('steps', []) if 'warmup_decode_transfers' in s],
                     warmup_strategy=(report.get('decode_warmup') or {}).get('prefill_strategy'),
                     diagnostic_eos_token=report.get('diagnostic_eos_token'),
                     device_eos_guard_enabled=(report.get('device_eos_guard') or {}).get('enabled'))
                for mode, report in zip(modes, reports)]
        same_eos = not args.device_eos_probe or rows[0]['diagnostic_eos_token'] == rows[1]['diagnostic_eos_token']
        status = 'pass' if exact and tokens_equal and same_eos and all(row['status'] == 'pass' for row in rows) else 'fail'
        warm_hits = all((row['timing'].get('compiler') or {}).get('fx_graph_cache_misses') == 0 for row in rows)
        if args.device_eos_probe:
            scope = 'matched in-process warm reduced-model turns with artificial EOS; not model quality'
        elif args.warm_cache_dir:
            scope = ('matched warm-cache processes, baseline then candidate' if warm_hits else
                     'nominally warm-cache processes; at least one compile miss')
        else:
            scope = 'cold process then warm-cache process; not a matched warm A/B'
        result = dict(status=status, domain=f'FPGA board; reduced {args.layers}-layer {args.model} checkpoint', image_uuid=IMAGE_UUID,
                      clock_hz=100000000, scope=scope,
                      source='supplied overlays with frozen compiler 02d704f4',
                      prompt=args.prompt, logits_bitwise_equal=exact, tokens_equal=tokens_equal, rows=rows)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if status != 'pass':
            raise RuntimeError('reduced-model comparison failed')


if __name__ == '__main__':
    main()
