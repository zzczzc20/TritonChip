#!/usr/bin/env python3
"""Compare exact static-prefix reuse on a small edited-conversation workload."""
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
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--group-suffix', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree / 'tools/models'))
    import chat

    args.out.mkdir(parents=True, exist_ok=False)
    prefix = 'context ' * (221 if args.group_suffix else 200)
    first_prompt = prefix + ('France' if args.group_suffix else 'What is the capital of France?')
    edited_prompt = prefix + ('Germany' if args.group_suffix else 'What is the capital of Germany?')
    followup = '?' + ' please' * 4 if args.group_suffix else 'What did I ask?'
    chat_args = chat.parser().parse_args([
        '--model', 'vl', '--device', 'tnca', '--tnca-tree', str(args.tree),
        '--layers', str(args.layers), '--prefill-chunk', '8', '--new-tokens', '2',
        '--bucket', '256', '--threads', '2', '--prompt', first_prompt])
    app = chat.ChatApplication(chat_args)
    warm = app.generate(first_prompt, retain_logits=False)
    if warm['status'] != 'pass':
        raise RuntimeError('reduced-model warmup failed')
    if args.group_suffix:
        # Warm the 16-token graph before either measured conversation.
        app.session.reset()
        app.session.chunk = 16
        chat_args.new_tokens = 1
        wider = app.generate(first_prompt, retain_logits=False)
        app.session.chunk = 8
        if wider['status'] != 'pass':
            raise RuntimeError('grouped prefill warmup failed')
    reports = []
    for mode in ('reset', 'reuse'):
        app.reset_conversation()
        app.session.reset()
        os.environ['TNCA_STATIC_PREFIX_REWIND'] = '1' if mode == 'reuse' else '0'
        os.environ['TNCA_STATIC_SUFFIX_GROUP'] = '1' if args.group_suffix and mode == 'reuse' else '0'
        chat_args.new_tokens = 1
        first = app.generate(first_prompt, keep_history=True, retain_logits=True)
        if first['status'] != 'pass':
            raise RuntimeError(f'{mode}: first turn failed')
        chat.save_report(first, args.out / f'{mode}-first')
        # Editing an old user message invalidates its changed suffix. Both
        # paths see the same canonical conversation; only the shared prefix
        # is retained by the candidate.
        app.history[0]['content'][0]['text'] = edited_prompt
        chat_args.new_tokens = 2
        started = time.perf_counter()
        second = app.generate(followup, keep_history=True, prompt_started=started, retain_logits=True)
        print('assistant>', second['generated_text'], flush=True)
        chat.finalize_user_visible_metrics(second['user_visible_timing'], time.perf_counter() - started,
                                          'edited follow-up prompt through assistant stdout flush')
        chat.save_report(second, args.out / f'{mode}-second')
        if second['status'] != 'pass':
            raise RuntimeError(f'{mode}: edited follow-up failed')
        reports.append((first, second))
        print(json.dumps({'mode': mode, 'reused': second['steps'][0]['reused_cached_tokens'],
                          'seconds': second['user_visible_timing']['prompt_to_return_seconds']}), flush=True)

    exact = all(len(a['logits']) == len(b['logits']) and
                all(torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                    for x, y in zip(a['logits'], b['logits'], strict=True))
                for a, b in zip(reports[0], reports[1], strict=True))
    tokens_equal = all(a['generated_tokens'] == b['generated_tokens']
                       for a, b in zip(reports[0], reports[1], strict=True))
    no_fallback = all(not turn['steady_decode_host_reads_detected'] and
                      turn['cpu_numerical_fallback'] is False for pair in reports for turn in pair)
    reused = [pair[1]['steps'][0]['reused_cached_tokens'] for pair in reports]
    correct_path = reused[0] == 0 and reused[1] > 0
    grouped = any(call['tokens'] > 8 for call in reports[1][1]['steps'][0]['calls'])
    if args.group_suffix:
        correct_path = correct_path and grouped
    rows = [dict(mode=mode, first_tokens=first['generated_tokens'],
                 followup_tokens=second['generated_tokens'], followup_reused_tokens=second['steps'][0]['reused_cached_tokens'],
                 followup_segments=[call['tokens'] for call in second['steps'][0]['calls']],
                 followup_timing=second['user_visible_timing'])
            for mode, (first, second) in zip(('reset', 'reuse'), reports, strict=True)]
    compilers = [row['followup_timing'].get('compiler') or {} for row in rows]
    matched = all(c.get('available') is True and c.get('dynamo_compile_calls') == 0 and
                  c.get('fx_graph_cache_misses') == 0 for c in compilers)
    passed = exact and tokens_equal and no_fallback and correct_path
    result = dict(status='pass' if passed else 'fail',
                  scope=('reduced VL, edited prior prompt with 221 repeated context words and grouped suffix; not full-model quality'
                         if args.group_suffix else
                         'one-layer VL, edited prior prompt with 200 repeated context words; not full-model quality'),
                  logits_bitwise_equal=exact, tokens_equal=tokens_equal, no_fallback_and_zero_decode_reads=no_fallback,
                  expected_cache_paths=correct_path, matched_warm=matched,
                  matched_warm_wall_speedup=(rows[0]['followup_timing']['prompt_to_return_seconds'] /
                                            rows[1]['followup_timing']['prompt_to_return_seconds']
                                            if matched and passed else None), rows=rows)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('static prefix continuation comparison failed')


if __name__ == '__main__':
    main()
