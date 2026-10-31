#!/usr/bin/env python3
"""Compare full and chunked CPU prefill on the same checkpoint, inputs and cache length."""
import argparse
import json
from pathlib import Path
import torch
import chat


def difference(a, b):
    x, y = a.float(), b.float()
    return dict(shape=list(x.shape), equal=bool(torch.equal(a, b)),
                max_abs=float((x-y).abs().max()) if x.numel() else 0.0,
                rel_l2=float((x-y).norm()/x.norm().clamp_min(1e-30)) if x.numel() else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('vl', 'q35'), required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--bucket', type=int, default=256)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    args = chat.parser().parse_args(['--model', a.model, '--device', 'cpu', '--threads', str(a.threads),
                                     '--bucket', str(a.bucket), '--new-tokens', '1'])
    app = chat.ChatApplication(args)
    chat.L.install_fp8_route('reference', dot='exact', scale_div='ieee')
    inputs, text = chat.make_inputs(app.spec, app.processor, [chat.user_message(a.prompt)])
    sessions, results = [], []
    for chunk in (0, a.chunk):
        session = chat.Session(app.model, app.spec, 'cpu', a.bucket, chunk)
        logits, calls, reused = session.prefill_prompt(inputs, None)
        state, meta = chat.L.cache_state(session.cache)
        sessions.append((session, state, meta))
        results.append(dict(chunk=chunk, argmax=int(logits.float().argmax()), calls=calls, reused=reused,
                            logits=logits.detach().clone()))
    first, second = sessions[0][1], sessions[1][1]
    if set(first) != set(second):
        raise RuntimeError('cache state keys differ')
    cache = {name: difference(first[name], second[name]) for name in sorted(first)}
    for name in cache:
        if first[name].ndim >= 3 and first[name].shape[2] == second[name].shape[2]:
            dims = tuple(i for i in range(first[name].ndim) if i != 2)
            per_position = (first[name].float()-second[name].float()).abs().amax(dim=dims)
            indices = torch.nonzero(per_position != 0).flatten()
            cache[name]['first_differing_position'] = int(indices[0]) if indices.numel() else None
    out = dict(schema='r41-cpu-prefill-comparison/1', scope='CPU diagnostic, not board acceptance',
               model=app.spec.name, prompt=a.prompt, prompt_text=text,
               prompt_tokens=int(inputs['input_ids'].shape[1]), bucket=a.bucket, chunk=a.chunk,
               full_argmax=results[0]['argmax'], chunked_argmax=results[1]['argmax'],
               logits=difference(results[0]['logits'], results[1]['logits']),
               cache_metadata=[sessions[0][2],sessions[1][2]], cache=cache,
               calls=[results[0]['calls'],results[1]['calls']], versions=chat.L.versions(), load=app.load_report)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('model','prompt_tokens','full_argmax','chunked_argmax','logits')}),flush=True)

if __name__ == '__main__':
    main()
