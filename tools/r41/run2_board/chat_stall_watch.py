#!/usr/bin/env python3
"""Preserve and stop only a demonstrably idle priority chat so the queued diagnostic can run."""
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

home = Path.home()
work = home/'r41run3'
sys.path.insert(0, str(work/'runtime_prepared/tools/r41/run2_board'))
import auto

pid = int(sys.argv[1]) if len(sys.argv) > 1 else 1150557
marker = sys.argv[2] if len(sys.argv) > 2 else 'runtime_prepared/tools/models/chat.py'
model = sys.argv[3] if len(sys.argv) > 3 else 'q35'
output = work/(sys.argv[4] if len(sys.argv) > 4 else 'chat_priority_20261031/chat-q35')
log_path = work/(sys.argv[5] if len(sys.argv) > 5 else f'chat_priority_20261031/q35-liveness-watch-{pid}.jsonl')
idle_start = None
prior = None
last_notice = 0.0
with log_path.open('x', buffering=1) as log:
    def record(kind, **data):
        log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **data))+'\n')
    while True:
        proc = Path(f'/proc/{pid}')
        try:
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            raw = (proc/'stat').read_text()
        except (FileNotFoundError, ProcessLookupError):
            record('process-exited')
            break
        if marker not in cmd or f'--model {model}' not in cmd:
            record('identity-changed')
            break
        fields = raw[raw.rfind(')')+2:].split()
        ticks = int(fields[11]) + int(fields[12])
        try:
            children = (proc/'task'/str(pid)/'children').read_text().strip()
            board = auto.probe(Path('/sys/bus/pci/devices'))
            logs = sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in output.rglob('*.log') if p.is_file())
        except (FileNotFoundError, ProcessLookupError, OSError, ValueError) as exc:
            idle_start = prior = None
            record('observation-error', error=str(exc))
            time.sleep(10)
            continue
        now = time.monotonic()
        snapshot = (ticks, logs, board.get('total execs'))
        owned = (board.get('uuid') == 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3' and
                 ((str(pid) in str(board.get('client pid', '')) and board.get('contexts') == 1) or
                  (board.get('contexts') == 0 and not board.get('client pid'))))
        quiet = owned and board.get('outstanding execs') == 0 and not children
        if not quiet or snapshot != prior:
            idle_start = now
        elif idle_start is None:
            idle_start = now
        idle_seconds = now - idle_start
        if now - last_notice >= 60:
            record('sample', cpu_ticks=ticks, idle_seconds=round(idle_seconds, 1),
                   children=children, board=board)
            last_notice = now
        if quiet and idle_seconds >= 600:
            # Revalidate identity immediately before signalling the specific process.
            cmd2 = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if marker not in cmd2 or f'--model {model}' not in cmd2:
                record('identity-changed-before-signal')
                break
            record('terminate-stalled-process', cpu_ticks=ticks, idle_seconds=idle_seconds,
                   board=board, log_files=logs, reason='no CPU, log, or device progress for 600 seconds')
            os.kill(pid, signal.SIGTERM)
            break
        prior = snapshot
        time.sleep(10)
