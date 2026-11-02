#!/usr/bin/env python3
"""Run one warm reduced-model turn with a deterministic, diagnostic EOS."""
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
    parser.add_argument('--model', choices=('q35', 'vl'), required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--new-tokens', type=int, default=12)
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()
    if args.model == 'q35' and args.layers < 4:
        parser.error('Qwen3.5 needs at least four layers to include a full-attention cache')
    sys.path.insert(0, str(args.tree / 'tools/models'))
    import chat
    import qwen_fp8_lib as library
    from tnca.torch_device import MEMORY
    from tnca.torch_device.eos import latch_eos

    mode = os.environ.get('TNCA_DEVICE_EOS_GUARD', '0')
    os.environ['TNCA_DEVICE_EOS_GUARD'] = '0'
    chat_args = chat.parser().parse_args([
        '--model', args.model, '--device', 'tnca', '--tnca-tree', str(args.tree),
        '--layers', str(args.layers), '--prefill-chunk', '8', '--new-tokens', '2',
        '--bucket', '256', '--threads', '2', '--prompt', args.prompt])
    app = chat.ChatApplication(chat_args)
    warm = app.generate(args.prompt, retain_logits=False)
    if warm['status'] != 'pass' or not warm['generated_tokens']:
        raise RuntimeError('model warmup failed')
    # Pick an actually generated token as the diagnostic EOS. This is a
    # reduced-model suppression proof, not a model-quality conversation.
    eos = warm['generated_tokens'][-1]
    library.generation_eos = lambda model: {eos}
    if mode == '1':
        # Compile this marker variant before timing either measured token.
        sentinel = 0 if eos != 0 else 1
        flag = torch.zeros((1,), dtype=torch.long).to(app.session.dev)
        token = torch.tensor([[sentinel]], dtype=torch.long).to(app.session.dev)
        latch_eos(token, flag, {eos})
        MEMORY.synchronize()
    os.environ['TNCA_DEVICE_EOS_GUARD'] = mode
    chat_args.new_tokens = args.new_tokens
    started = time.perf_counter()
    result = app.generate(args.prompt, prompt_started=started, retain_logits=True)
    print('assistant>', result['generated_text'], flush=True)
    chat.finalize_user_visible_metrics(result['user_visible_timing'], time.perf_counter() - started,
                                      'warm prompt through flushed reduced-model text')
    result['diagnostic_eos_token'] = eos
    result['diagnostic_scope'] = 'reduced model with artificially selected EOS; not model quality'
    chat.save_report(result, args.out)
    if result['status'] != 'pass' or not result['stopped_at_eos']:
        raise RuntimeError('measured EOS turn failed or did not reach the diagnostic EOS')
    print(json.dumps({'status': result['status'], 'tokens': result['generated_tokens'],
                      'timing': result['user_visible_timing']}), flush=True)


if __name__ == '__main__':
    main()
