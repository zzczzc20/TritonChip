#!/usr/bin/env python3
"""Run the event-index board comparison after the priority chat stage passes."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--priority-pid', type=int, required=True)
    p.add_argument('--priority-stage', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    def record(event, **fields):
        with (a.out / 'queue.jsonl').open('a') as f:
            f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **fields)) + '\n')
    record('waiting', priority_pid=a.priority_pid)
    try:
        fd = os.pidfd_open(a.priority_pid)
    except ProcessLookupError:
        pass
    else:
        try:
            select.select([fd], [], [])
        finally:
            os.close(fd)
    if not a.priority_stage.is_file() or json.loads(a.priority_stage.read_text()).get('status') != 'pass':
        record('not-started', reason='priority chat stage did not pass')
        return
    registration = json.loads(a.registration.read_text())
    registration['experiments'] = registration['experiments'][:1]
    selected = a.out / 'session-registration.json'
    selected.write_text(json.dumps(registration, indent=2) + '\n')
    record('starting', scope='Qwen3.5 session event-index comparison')
    cmd = [str(a.python), str(a.root / 'tools/r41/run2_board/stage.py'),
           '--root', str(a.root), '--out', str(a.out / 'session'),
           '--python', str(a.python), '--registration', str(selected)]
    with (a.out / 'stage.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    record('finished', returncode=result.returncode)


if __name__ == '__main__':
    main()
