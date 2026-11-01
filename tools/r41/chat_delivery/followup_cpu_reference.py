#!/usr/bin/env python3
"""Reconstruct a recorded two-turn conversation and score its follow-up on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/models'))
import torch  # noqa: E402
import chat  # noqa: E402
from chat_quality import first_token_reference, score_author_turn  # noqa: E402
from rescore_deadline_quality import checkpoint_identity  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('device', 'reference', 'spec', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    device, reference, spec = (json.loads(path.read_text()) for path in
                               (args.device, args.reference, args.spec))
    identity = checkpoint_identity(device, reference)
    if device.get('device') != 'tnca' or device.get('model_scope') != 'full-checkpoint' or \
            reference.get('device') != 'cpu' or reference.get('status') != 'reference-recorded' or \
            device.get('model') != reference.get('model'):
        raise ValueError('recorded board and CPU full-checkpoint evidence is required')
    item = next(p for p in device['prompts'] if p['id'] == 'multiturn')
    prompt = next(p for p in spec['prompts'] if p['id'] == 'multiturn')
    turns = item['turns']
    if len(turns) != 2 or [turn['user'] for turn in turns] != prompt['turns'] or not turns[0]['tokens']:
        raise ValueError('recorded two-turn prompt or first response is incomplete')
    model = {'qwen3.5-4b-fp8': 'q35', 'qwen3vl-2b-fp8': 'vl'}[device['model']]
    cli = chat.parser().parse_args(['--model', model, '--device', 'cpu', '--threads', '4',
                                    '--new-tokens', '1', '--bucket', str(spec['bucket']),
                                    '--prefill-chunk', str(device.get('prefill_chunk') or 0)])
    app = chat.ChatApplication(cli)
    if (app.spec.repo, app.spec.revision) != (reference['repo'], reference['revision']):
        raise ValueError('loaded model does not match the registered CPU reference')
    loaded_index = (app.load_report.get('snapshot_view') or {}).get('original_index_sha256')
    reference_index = (reference.get('load', {}).get('snapshot_view') or {}).get('original_index_sha256')
    if not loaded_index or loaded_index != reference_index:
        raise ValueError('loaded CPU checkpoint index differs from the recorded reference')
    first = chat.user_message(turns[0]['user'])
    inputs, text = chat.make_inputs(app.spec, app.processor, [first])
    generated = app.processor.batch_decode(torch.tensor(turns[0]['tokens'])[None], skip_special_tokens=False)[0]
    app.history = [first, dict(role='assistant', content=[dict(type='text', text=turns[0]['text'])])]
    app.transcript = (text + generated, inputs['input_ids'][0].tolist() + turns[0]['tokens'])
    messages = app.history + [chat.user_message(turns[1]['user'])]
    candidate, rendered = chat.make_inputs(app.spec, app.processor, messages)
    actual = app.continuation(messages, candidate, rendered) or candidate
    ids = actual['input_ids'][0].tolist()
    if len(ids) != turns[1].get('prompt_tokens'):
        raise ValueError('reconstructed follow-up token count differs from the board record')
    cpu = app.generate(turns[1]['user'], keep_history=True, check=False)
    if cpu['prompt_tokens'] != len(ids) or not cpu.get('logits'):
        raise ValueError('CPU follow-up did not use the reconstructed prompt')
    cpu_turn = dict(text=cpu['generated_text'], tokens=cpu['generated_tokens'],
                    first_token_reference=first_token_reference(cpu['logits'][0]))
    score = score_author_turn(turns[1], cpu_turn, prompt['content_checks'][1], prompt['keywords'])
    result = dict(schema='r41-followup-cpu-reference/1', model=device['model'], checkpoint_identity=identity,
                  loaded_checkpoint_index_sha256=loaded_index,
                  source_sha256={str(path): sha(path) for path in (args.device, args.reference, args.spec)},
                  board_prompt_tokens=turns[1]['prompt_tokens'], reconstructed_prompt_tokens=len(ids),
                  reconstructed_input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                  input_scope='reconstructed from recorded first-response tokens; the old board input IDs were not saved',
                  cpu_reference=cpu_turn, score=score, original_score=item['scores'][1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'first_token_pass': score['first_token_pass'], 'content_pass': score['content_pass']}))


if __name__ == '__main__':
    main()
