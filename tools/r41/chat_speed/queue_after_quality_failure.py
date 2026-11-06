#!/usr/bin/env python3
"""Run gated speed sessions after the completed, honestly recorded chat failure."""
import json
import sys
from pathlib import Path

WORK = Path.home() / 'r41run3'
ROOT = WORK / 'runtime_nohost_v6_resume_20261031'
OUT = WORK / 'chat_speed_after_quality_failure_retry_20261031'
QUALITY = (WORK / 'chat_nohost_v6_resume_20261031/multiturn/'
           'r41-q35-segment-multiturn-20261031/q35-quality/quality.json')
sys.path.insert(0, str(WORK / 'runtime_activation_equal_20261031/tools/r41/run2_board'))
import q35_timeout_recovery as recovery


def main():
    OUT.mkdir(exist_ok=False)
    recovery.OUT = OUT
    data = json.loads(QUALITY.read_text())
    prompts = data.get('prompts', [])
    if (data.get('status') != 'failed-chat-quality' or data.get('reduced_proof') is not True
            or data.get('model_scope') != 'full-checkpoint' or len(prompts) != 1
            or prompts[0].get('id') != 'multiturn' or prompts[0].get('passed') is not False
            or len(prompts[0].get('turns', [])) != 2):
        raise ValueError('expected recorded segmented quality failure is missing')
    for turn in prompts[0]['turns']:
        ev = turn['device_evidence']
        if ev.get('cpu_numerical_fallback') is not False or ev.get('compiled_wrapper_roles_covered') is not True:
            raise ValueError('no-fallback evidence is missing')
        if any(ev['compiled_wrapper_fallbacks'].values()):
            raise ValueError('a compiled wrapper fell back')
        for kind in ('parameters', 'buffers'):
            placements = ev['model_tensor_placements'].get(kind)
            if not placements or any(not k.startswith('tnca:') or type(v) is not int or v <= 0
                                     for k, v in placements.items()):
                raise ValueError('model placement is not all TNCA')
        decode = [s for s in turn['steps'] if s.get('kind') == 'decode']
        if not decode or any(s['transfers_inside_step']['host_read_calls'] != 0 for s in decode):
            raise ValueError('decode host-read evidence is missing or nonzero')
    recovery.record('quality-failure-preserved', path=str(QUALITY), sha256=recovery.digest(QUALITY),
                    reason='first turn did not match registered CPU-reference rule')
    for name in ('event_index', 'equalrow_event', 'pipeline_equalrow', 'activation_equal'):
        root = WORK / ('runtime_' + name + '_20261031')
        if name == 'activation_equal' and recovery.digest(root / 'compiler/python/tnca_cc/frontend.py') != \
                'a8ff866ea37cd70551d137b59261424a7a4b7df2f0b929cc11c123f79e993bfb':
            raise ValueError('activation candidate changed')
        registration = json.loads((root / 'spec/r41/run3/stages/chat-q35-nohost-v5-20261031.json').read_text())
        registration['experiments'] = registration['experiments'][:1]
        reg = OUT / (name + '-registration.json')
        with reg.open('x') as f:
            json.dump(registration, f, indent=2); f.write('\n')
        recovery.stage(root, OUT / ('speed-' + name), reg, OUT / ('speed-' + name + '.log'))
    recovery.record('finished')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            recovery.record('stopped', error=f'{type(exc).__name__}: {exc}')
        raise
