#!/usr/bin/env python3
"""Queue one small, real-model cold-to-warm generation comparison on the current image.

The registered idle monitor and queue lock protect other FPGA users. The test
never changes the image and labels its one-layer results as reduced depth.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from bench_mmio_board import board_idle, IMAGE_UUID, PROFILE


def compilation_free(rows):
    """Require observed compilation counters from both measured turns."""
    compiler = [(row.get('timing') or {}).get('compiler') or {} for row in rows]
    observed = len(compiler) == 2 and all(item.get('available') is True for item in compiler)
    no_misses = observed and all(item.get('fx_graph_cache_misses') == 0 for item in compiler)
    return no_misses and all(item.get('dynamo_compile_calls') == 0 for item in compiler)


def comparison_scope(rows, *, device_eos_probe, warm_cache):
    """Do not call timed compilation an equivalent warm-model comparison."""
    no_compile = compilation_free(rows)
    if device_eos_probe:
        if no_compile:
            return 'matched in-process warm reduced-model turns with artificial EOS; not model quality'
        return ('in-process warm reduced-model turns with artificial EOS; timed compilation occurred '
                'or was not measured, so this is not a matched warm speed comparison')
    if warm_cache:
        return ('matched warm-cache processes, baseline then candidate' if no_compile else
                'nominally warm-cache processes; timed compilation occurred or was not measured')
    return 'cold process then warm-cache process; not a matched warm A/B'


def matched_warm_wall_speedup(rows, *, device_eos_probe, warm_cache):
    """Only expose an A/B wall ratio when both measured turns were warm."""
    if not (device_eos_probe or warm_cache) or not compilation_free(rows):
        return None
    seconds = [(row.get('timing') or {}).get('prompt_to_return_seconds') for row in rows]
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or
           not math.isfinite(s) or s <= 0 for s in seconds):
        return None
    return seconds[0] / seconds[1]


def model_command(args, root, out):
    """Build the selected model probe after its overlay and output are known."""
    if args.static_prefix_probe:
        command = [str(args.python), '-u', str(root / 'tools/r41/chat_speed/bench_static_continue_entry.py'),
                   '--tree', str(root), '--layers', str(args.layers), '--out', str(out)]
        return command + (['--group-suffix'] if args.static_group_suffix else []) + \
            (['--normal-chat-eos'] if args.static_normal_chat else [])
    if args.hybrid_eos_continuation:
        command = [str(args.python), '-u', str(root / 'tools/r41/chat_speed/bench_hybrid_continue_entry.py'),
                   '--tree', str(root), '--layers', str(args.layers), '--prompt', args.prompt, '--out', str(out)]
        words = getattr(args, 'hybrid_context_words', 0)
        if words:
            command.extend(['--context-words', str(words)])
        if getattr(args, 'hybrid_warm_followup', False):
            command.append('--warm-followup')
        return command
    if args.device_eos_probe or args.early_eos_probe:
        return [str(args.python), '-u', str(root / 'tools/r41/chat_speed/bench_eos_model_entry.py'),
                '--tree', str(root), '--model', args.model, '--layers', str(args.layers),
                '--new-tokens', str(args.new_tokens), '--prompt', args.prompt, '--out', str(out)]
    return [str(args.python), '-u', str(root / 'tools/models/chat.py'),
            '--model', args.model, '--device', 'tnca', '--tnca-tree', str(root),
            '--layers', str(args.layers), '--prefill-chunk', '8', '--new-tokens', str(args.new_tokens),
            '--bucket', '256', '--threads', '2', '--prompt', args.prompt, '--out', str(out)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--overlay-root', type=Path, required=True)
    parser.add_argument('--baseline-overlay-root', type=Path)
    parser.add_argument('--warm-cache-dir', type=Path)
    parser.add_argument('--device-eos-probe', action='store_true',
                        help='compare warm reduced-model turns with a diagnostic EOS')
    parser.add_argument('--early-eos-probe', action='store_true',
                        help='compare guarded submissions with the device completion stop signal')
    parser.add_argument('--hybrid-eos-continuation', action='store_true',
                        help='compare hybrid cache reset and retention in one reduced-model session')
    parser.add_argument('--hybrid-context-words', type=int, default=0,
                        help='add repeated context to the reduced Qwen3.5 continuation')
    parser.add_argument('--hybrid-warm-followup', action='store_true',
                        help='warm both conversation shapes before the measured turns')
    parser.add_argument('--static-prefix-probe', action='store_true',
                        help='compare static-prefix reuse in a reduced edited conversation')
    parser.add_argument('--static-group-suffix', action='store_true',
                        help='also test a larger suffix prefill segment')
    parser.add_argument('--static-normal-chat', action='store_true',
                        help='measure ordinary follow-up after diagnostic EOS')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', choices=('vl', 'q35'), default='vl')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--new-tokens', type=int, default=4)
    parser.add_argument('--prompt', default='What is the capital of France? Answer with one word.')
    args = parser.parse_args()
    if args.model == 'q35' and args.layers < 4:
        parser.error('Qwen3.5 needs at least four layers to include a full-attention cache')
    if args.static_group_suffix and not args.static_prefix_probe:
        parser.error('grouped suffix requires the static prefix probe')
    if args.static_normal_chat and not args.static_group_suffix:
        parser.error('normal-chat comparison requires grouped suffix and static prefix probe')
    if args.static_prefix_probe and (args.model != 'vl' or args.hybrid_eos_continuation or
                                     args.device_eos_probe or args.early_eos_probe or args.baseline_overlay_root):
        parser.error('static prefix probe needs VL, one overlay and no other probe')
    if args.hybrid_eos_continuation and (args.model != 'q35' or args.device_eos_probe or
                                         args.early_eos_probe or args.baseline_overlay_root):
        parser.error('hybrid continuation needs Qwen3.5, one overlay and no other probe')
    if (args.hybrid_context_words or args.hybrid_warm_followup) and not args.hybrid_eos_continuation:
        parser.error('hybrid warmup and context require the hybrid continuation probe')
    if args.device_eos_probe and args.early_eos_probe:
        parser.error('select only one EOS comparison')
    if (args.device_eos_probe or args.early_eos_probe) and args.baseline_overlay_root:
        parser.error('the device EOS probe uses one source overlay')
    if not (args.device_eos_probe or args.early_eos_probe or args.hybrid_eos_continuation or args.static_prefix_probe) and bool(args.baseline_overlay_root) != bool(args.warm_cache_dir):
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
        modes = (('static-prefix',) if args.static_prefix_probe else
                 ('hybrid-eos-continuation',) if args.hybrid_eos_continuation else
                 ('guarded-submissions', 'early-stop') if args.early_eos_probe else
                 ('baseline', 'device-eos-guard') if args.device_eos_probe else
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
                         'TNCA_DEFER_CACHE_RESET', 'TNCA_REUSE_FIRST_DECODE', 'TNCA_DEVICE_EOS_GUARD',
                         'TNCA_DEVICE_EOS_EARLY_STOP', 'TNCA_REUSE_GUARDED_HYBRID',
                         'TNCA_HYBRID_PREFIX_CHECKPOINT', 'TNCA_HYBRID_FALLBACK_CHECKPOINT',
                         'TNCA_STATIC_PREFIX_REWIND',
                         'TNCA_STATIC_SUFFIX_GROUP'):
                run_env.pop(name, None)
            if mode == 'baseline-cold':
                run_env.update(TNCA_REUSE_WARM_PREFILL='0', TNCA_REUSE_HYBRID_PREFILL='0',
                               TNCA_MMIO_DESCRIPTOR_CACHE='0', TNCA_CODE_REUSE='0', TNCA_DEFER_CACHE_RESET='0')
            if args.device_eos_probe:
                run_env['TNCA_DEVICE_EOS_GUARD'] = '1' if mode == 'device-eos-guard' else '0'
            if args.early_eos_probe:
                run_env['TNCA_DEVICE_EOS_GUARD'] = '1'
                run_env['TNCA_DEVICE_EOS_EARLY_STOP'] = '1' if mode == 'early-stop' else '0'
            root = args.baseline_overlay_root if mode == 'baseline-warm' else args.overlay_root
            out = args.out / mode
            command = model_command(args, root, out)
            print('starting', mode, datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
            with (args.out / f'{mode}.log').open('x') as log:
                rc = subprocess.run(command, cwd=root, env=run_env,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            print('finished', mode, 'returncode', rc, flush=True)
            if rc != 0:
                raise RuntimeError(f'{mode} failed; preserved its log and any failure report')
            if args.hybrid_eos_continuation or args.static_prefix_probe:
                print((out / 'result.json').read_text(), flush=True)
                return
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
                     executed_generation_steps=report.get('executed_generation_steps'),
                     device_eos_guard_enabled=(report.get('device_eos_guard') or {}).get('enabled'),
                     completion_probe=(report.get('device_eos_guard') or {}).get('completion_probe'))
                for mode, report in zip(modes, reports)]
        eos_comparison = args.device_eos_probe or args.early_eos_probe
        same_eos = not eos_comparison or rows[0]['diagnostic_eos_token'] == rows[1]['diagnostic_eos_token']
        status = 'pass' if exact and tokens_equal and same_eos and all(row['status'] == 'pass' for row in rows) else 'fail'
        if args.early_eos_probe:
            signal = rows[1]['completion_probe'] or {}
            if (not signal.get('enabled') or not signal.get('signalled') or
                    rows[1]['executed_generation_steps'] != len(rows[1]['tokens']) or
                    rows[0]['executed_generation_steps'] <= len(rows[0]['tokens'])):
                status = 'fail'
        scope = comparison_scope(rows, device_eos_probe=eos_comparison,
                                 warm_cache=bool(args.warm_cache_dir))
        result = dict(status=status, domain=f'FPGA board; reduced {args.layers}-layer {args.model} checkpoint', image_uuid=IMAGE_UUID,
                      clock_hz=100000000, scope=scope,
                      matched_warm_wall_speedup=(matched_warm_wall_speedup(
                          rows, device_eos_probe=eos_comparison, warm_cache=bool(args.warm_cache_dir))
                          if status == 'pass' else None),
                      source='supplied overlays with frozen compiler 02d704f4',
                      prompt=args.prompt, logits_bitwise_equal=exact, tokens_equal=tokens_equal, rows=rows)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if status != 'pass':
            raise RuntimeError('reduced-model comparison failed')


if __name__ == '__main__':
    main()
