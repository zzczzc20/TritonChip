#!/usr/bin/env python3
"""Queue the gated activation candidate after the preceding board stages."""
import json
import sys
from pathlib import Path

WORK = Path.home() / 'r41run3'
ROOT = WORK / 'runtime_activation_equal_20261031'
OUT = WORK / 'chat_activation_equal_20261031'
PREVIOUS = WORK / 'chat_nohost_v6_resume_20261031'
SHA256 = 'a8ff866ea37cd70551d137b59261424a7a4b7df2f0b929cc11c123f79e993bfb'
sys.path.insert(0, str(ROOT / 'tools/r41/run2_board'))
import q35_timeout_recovery as recovery


def main():
    OUT.mkdir(exist_ok=False)
    recovery.OUT = OUT
    recovery.record('waiting', preceding_watcher_pid=1197968)
    recovery.wait_pid(1197968)
    if not (PREVIOUS / 'segmented-coverage.json').is_file():
        raise RuntimeError('quality coverage is missing')
    for name in ('event_index', 'equalrow_event', 'pipeline_equalrow'):
        path = PREVIOUS / ('speed-' + name) / 'stage.json'
        if not path.is_file() or json.loads(path.read_text()).get('status') != 'pass':
            raise RuntimeError('preceding acceleration stage did not pass: ' + name)
    if recovery.digest(ROOT / 'compiler/python/tnca_cc/frontend.py') != SHA256:
        raise RuntimeError('candidate compiler source changed')
    reg = json.loads((ROOT / 'spec/r41/run3/stages/chat-q35-nohost-v5-20261031.json').read_text())
    reg['experiments'] = reg['experiments'][:1]
    path = OUT / 'session-registration.json'
    path.write_text(json.dumps(reg, indent=2) + '\n')
    recovery.stage(ROOT, OUT / 'session', path, OUT / 'stage.log')
    recovery.record('finished')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            recovery.record('stopped', error=f'{type(exc).__name__}: {exc}')
        raise
