#!/usr/bin/env python3
"""Register full quality, session and script stages for grouped-tail device chat."""
import copy
import hashlib
import json
from pathlib import Path

root = Path.home()/'r41run3/runtime_nohost_v4_20261031'
board = root/'spec/r41/run3/board'
stages = root/'spec/r41/run3/stages'
for model in ('vl', 'q35'):
    base = json.loads((board/f'r3-s2-qwen-chat-v3-{model}.json').read_text())
    original = json.loads((stages/f'chat-{model}.json').read_text())
    base['registration']['status'] = 'registered-not-run'
    base['registration']['scope'] = ('Run-2 full-checkpoint chat using device-side feedback, graph-preserving '
                                    'evidence capture, and grouped short prefill tails; quality prompts, '
                                    'references and scoring remain unchanged. Each step has its own budget.')
    base.setdefault('chat_quality', {})['prefill'] = ('Text is prefilled in blocks of up to eight tokens; '
                                                       'the final short block stays together. Image turns are whole.')
    base['budget_seconds'] = 10800
    base['budget_note'] = ('The quality, session and scripted-chat steps are separate experiments, each capped '
                           'at 10,800 seconds; original prompt and numerical scoring rules are unchanged.')
    rows = []
    for step in base['steps']:
        kind = 'quality' if step['id'].endswith('-quality') else 'session'
        spec = copy.deepcopy(base)
        spec['id'] = f'r3-s2-qwen-chat-v3-{model}-nohost-v4-{kind}-20261031'
        selected = copy.deepcopy(step)
        if kind == 'quality':
            selected['timeout_seconds'] = 10800
        spec['steps'] = [selected]
        target = board/f'{spec["id"]}.json'
        target.write_text(json.dumps(spec, indent=2)+'\n')
        checks = [c for c in original['experiments'][0]['checks'] if c['path'].startswith(f'{model}-{kind}/')]
        if not checks:
            raise ValueError('missing registered checks for '+spec['id'])
        rows.append(dict(id=spec['id'], manifest=str(target.relative_to(root)),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         checks=checks, budget_seconds=10800))
    if model == 'vl':
        session = next(s for s in base['steps'] if s['id'] == 'vl-session')
        spec = copy.deepcopy(base)
        spec['id'] = 'r3-s2-qwen-chat-v3-vl-nohost-v4-script-20261031'
        script = copy.deepcopy(session)
        script['id'] = 'vl-script'
        script['timeout_seconds'] = 10800
        script['argv'] = [str(Path.home()/'r41s2_fp8env/bin/python'), '-I', '-u', '{root}/tools/models/chat.py',
                          '--model', 'vl', '--device', 'tnca', '--tnca-tree', '{root}', '--threads', '4',
                          '--script', '{root}/spec/r41/chat/script-text.txt', '--new-tokens', '32',
                          '--bucket', '256', '--prefill-chunk', '8', '--out', '{out}/vl-script']
        script['outputs'] = ['{out}/vl-script/transcript.json']
        spec['steps'] = [script]
        target = board/f'{spec["id"]}.json'
        target.write_text(json.dumps(spec, indent=2)+'\n')
        rows.append(dict(id=spec['id'], manifest=str(target.relative_to(root)),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(), budget_seconds=10800,
                         checks=[dict(path='vl-script/transcript.json', field=['turns'], length_equals=3)]))
    original['id'] = f'chat-{model}-nohost-v4-20261031'
    original['experiments'] = rows
    (stages/f'chat-{model}-nohost-v4-20261031.json').write_text(json.dumps(original, indent=2)+'\n')
