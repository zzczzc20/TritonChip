#!/usr/bin/env python3
"""Register a candidate no-host-feedback chat retry in a separate staged tree."""
import hashlib
import json
from pathlib import Path
import copy

root = Path.home() / 'r41run3/runtime_nohost_v3_20261031'
board = root / 'spec/r41/run3/board'
stages = root / 'spec/r41/run3/stages'
for model in ('vl', 'q35'):
    base = board / f'r3-s2-qwen-chat-v3-{model}.json'
    spec = json.loads(base.read_text())
    spec['id'] = f'r3-s2-qwen-chat-v3-{model}-nohost-v3-20261031'
    spec['registration']['status'] = 'registered-not-run'
    spec['registration']['scope'] = ('run-2 retry using device-side greedy token feedback and graph-preserving evidence capture; same registered quality '
                                     'and session checks, plus the registered scripted conversation for VL')
    if model == 'vl':
        session = next(s for s in spec['steps'] if s['id'] == 'vl-session')
        script = copy.deepcopy(session)
        script['id'] = 'vl-script'
        script['argv'] = [str(Path.home()/'r41s2_fp8env/bin/python'), '-I', '-u', '{root}/tools/models/chat.py',
                          '--model', 'vl', '--device', 'tnca', '--tnca-tree', '{root}', '--threads', '4',
                          '--script', '{root}/spec/r41/chat/script-text.txt', '--new-tokens', '32',
                          '--bucket', '256', '--prefill-chunk', '8', '--out', '{out}/vl-script']
        script['outputs'] = ['{out}/vl-script/transcript.json']
    spec['budget_seconds'] = 10800
    target = board / f'r3-s2-qwen-chat-v3-{model}-nohost-v3-20261031.json'
    target.write_text(json.dumps(spec, indent=2) + '\n')
    reg = json.loads((stages / f'chat-{model}.json').read_text())
    reg['id'] = f'chat-{model}-nohost-v3-20261031'
    experiment = reg['experiments'][0]
    experiment['id'] = spec['id']
    experiment['manifest'] = str(target.relative_to(root))
    experiment['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    experiment['budget_seconds'] = 10800
    if model == 'vl':
        script_spec = copy.deepcopy(spec)
        script_spec['id'] = 'r3-s2-qwen-chat-v3-vl-nohost-v3-script-20261031'
        script_spec['steps'] = [script]
        script_target = board/'r3-s2-qwen-chat-v3-vl-nohost-v3-script-20261031.json'
        script_target.write_text(json.dumps(script_spec, indent=2) + '\n')
        reg['experiments'].append(dict(
            id=script_spec['id'], manifest=str(script_target.relative_to(root)),
            sha256=hashlib.sha256(script_target.read_bytes()).hexdigest(), budget_seconds=10800,
            checks=[dict(path='vl-script/transcript.json', field=['turns'], length_equals=3)]))
    (stages / f'chat-{model}-nohost-v3-20261031.json').write_text(json.dumps(reg, indent=2) + '\n')
