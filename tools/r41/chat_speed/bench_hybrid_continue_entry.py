#!/usr/bin/env python3
"""Compare guarded hybrid-cache continuation on a reduced Qwen3.5 board model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--followup', default='What did I just ask?')
    args = parser.parse_args()
    if args.layers < 4:
        parser.error('Qwen3.5 needs a full-attention layer')
    sys.path.insert(0, str(args.tree / 'tools/models'))
    import chat
    import qwen_fp8_lib as library
    from tnca.torch_device import MEMORY
    from tnca.torch_device.eos import latch_eos

    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['TNCA_DEVICE_EOS_GUARD'] = '0'
    chat_args = chat.parser().parse_args([
        '--model', 'q35', '--device', 'tnca', '--tnca-tree', str(args.tree),
        '--layers', str(args.layers), '--prefill-chunk', '8', '--new-tokens', '2',
        '--bucket', '256', '--threads', '2', '--prompt', args.prompt])
    app = chat.ChatApplication(chat_args)
    app.device_session()
    MEMORY.synchronize()
    MEMORY.session().prepare_device_stop_guard()
    probe, calibration = MEMORY.session().prepare_device_stop_probe()
    if probe is None:
        raise RuntimeError(f'device completion probe could not be calibrated: {calibration}')
    warm = app.generate(args.prompt, retain_logits=False)
    if warm['status'] != 'pass' or not warm['generated_tokens']:
        raise RuntimeError('warmup failed')
    eos = warm['generated_tokens'][-1]
    library.generation_eos = lambda model: {eos}
    # Compile the flag marker before either measured conversation.
    flag = torch.zeros((1,), dtype=torch.long).to(app.session.dev)
    token = torch.tensor([[0 if eos != 0 else 1]], dtype=torch.long).to(app.session.dev)
    latch_eos(token, flag, {eos})
    MEMORY.synchronize()
    os.environ['TNCA_DEVICE_EOS_GUARD'] = '1'
    reports = []
    for mode in ('reset', 'retain'):
        app.reset_conversation()
        app.session.reset()
        os.environ['TNCA_REUSE_GUARDED_HYBRID'] = '1' if mode == 'retain' else '0'
        os.environ['TNCA_HYBRID_PREFIX_CHECKPOINT'] = '1' if mode == 'retain' else '0'
        os.environ['TNCA_DEVICE_EOS_EARLY_STOP'] = '1' if mode == 'retain' else '0'
        chat_args.new_tokens = 4
        first = app.generate(args.prompt, keep_history=True, retain_logits=True)
        if (first['status'] != 'pass' or not first['stopped_at_eos']
                or not first['device_eos_guard']['enabled']):
            raise RuntimeError(f'{mode}: the guarded first turn did not reach diagnostic EOS')
        chat.save_report(first, args.out / f'{mode}-first')
        chat_args.new_tokens = 2
        started = time.perf_counter()
        second = app.generate(args.followup, keep_history=True, prompt_started=started, retain_logits=True)
        chat.finalize_user_visible_metrics(second['user_visible_timing'], time.perf_counter() - started,
                                          'follow-up through reduced-model text')
        chat.save_report(second, args.out / f'{mode}-second')
        if second['status'] != 'pass':
            raise RuntimeError(f'{mode}: follow-up failed')
        reports.append((first, second))
        print(json.dumps({'mode': mode, 'first_cache': first['session']['cache_rewind'],
                          'followup_reused': second['steps'][0]['reused_cached_tokens'],
                          'followup_seconds': second['user_visible_timing']['prompt_to_return_seconds']}), flush=True)

    exact = all(len(a['logits']) == len(b['logits']) and
                all(torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                    for x, y in zip(a['logits'], b['logits'], strict=True))
                for a, b in zip(reports[0], reports[1], strict=True))
    tokens_equal = all(a['generated_tokens'] == b['generated_tokens']
                       for a, b in zip(reports[0], reports[1], strict=True))
    reads_zero = all(not turn['steady_decode_host_reads_detected'] and
                     turn['cpu_numerical_fallback'] is False for pair in reports for turn in pair)
    reused = reports[1][1]['steps'][0]['reused_cached_tokens']
    expected_modes = (reports[0][0]['session']['cache_rewind'] == 'full-reset-deferred' and
                      reports[1][0]['session']['cache_rewind'] == 'hybrid-prefix-retained-after-device-stop' and
                      reports[0][1]['steps'][0]['reused_cached_tokens'] == 0 and reused > 0)
    probe_info = reports[1][0]['device_eos_guard']['completion_probe']
    stopped_submitting = (probe_info['enabled'] and probe_info['signalled'] and
                          reports[1][0]['executed_generation_steps'] == len(reports[1][0]['generated_tokens']) and
                          reports[0][0]['executed_generation_steps'] > len(reports[0][0]['generated_tokens']))
    rows = []
    for mode, (first, second) in zip(('reset', 'retain'), reports, strict=True):
        rows.append(dict(mode=mode, first_tokens=first['generated_tokens'],
                         first_cache=first['session']['cache_rewind'],
                         first_executed_steps=first['executed_generation_steps'],
                         first_completion_probe=first['device_eos_guard']['completion_probe'],
                         followup_tokens=second['generated_tokens'],
                         followup_reused_tokens=second['steps'][0]['reused_cached_tokens'],
                         followup_timing=second['user_visible_timing']))
    passed = exact and tokens_equal and reads_zero and expected_modes and stopped_submitting
    result = dict(status='pass' if passed else 'fail',
                  scope='reduced four-layer Qwen3.5; artificial EOS, two conversational turns; not model quality',
                  logits_bitwise_equal=exact, tokens_equal=tokens_equal, no_fallback_and_zero_decode_reads=reads_zero,
                  diagnostic_eos_token=eos, expected_cache_paths=expected_modes,
                  stopped_submitting_after_eos=stopped_submitting, rows=rows)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('guarded hybrid continuation comparison failed')


if __name__ == '__main__':
    main()
