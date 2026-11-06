#!/usr/bin/env python3
"""Advance from the traced VL diagnostic only after ten minutes of no measurable work."""
import datetime
import os
from pathlib import Path
import signal
import sys
import time
import json

home = Path.home()
work = home/'r41run3'
sys.path.insert(0, str(work/'runtime_prepared/tools/r41/run2_board'))
import auto

marker = str(work/'chat_trace_inputs/trace_wrapper.py')
watcher = Path('/proc/1151183')
clock_ticks = os.sysconf('SC_CLK_TCK')
log_path = work/'chat_trace_inputs/stall-watch.jsonl'
with log_path.open('x', buffering=1) as log:
    def record(kind, **data):
        log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **data))+'\n')
    record('waiting-traced-vl')
    pid = None
    while watcher.exists():
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if marker in cmd and '--model vl' in cmd and '--device tnca' in cmd:
                pid = int(proc.name)
                break
        if pid is not None:
            break
        time.sleep(10)
    if pid is None:
        record('trace-waiter-exited-before-diagnostic')
        sys.exit(0)
    record('monitoring-traced-vl', pid=pid)
    prior_ticks = None
    prior_execs = None
    progress_at = time.monotonic()
    last_notice = 0.0
    proc = Path('/proc')/str(pid)
    while True:
        try:
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            ticks = int(fields[11]) + int(fields[12])
            children = (proc/'task'/str(pid)/'children').read_text().strip()
            board = auto.probe(Path('/sys/bus/pci/devices'))
        except (FileNotFoundError, ProcessLookupError):
            record('trace-process-exited', pid=pid)
            break
        except (OSError, ValueError) as exc:
            prior_ticks = prior_execs = None
            progress_at = time.monotonic()
            record('observation-error', error=str(exc))
            time.sleep(10)
            continue
        if marker not in cmd or '--model vl' not in cmd:
            record('identity-changed', pid=pid)
            break
        now = time.monotonic()
        owned = (board.get('uuid') == 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3' and
                 str(pid) in str(board.get('client pid', '')) and board.get('contexts') == 1)
        quiet = owned and board.get('outstanding execs') == 0 and not children
        if (not quiet or prior_ticks is None or ticks - prior_ticks >= clock_ticks or
                board.get('total execs') != prior_execs):
            progress_at, prior_ticks, prior_execs = now, ticks, board.get('total execs')
        idle = now - progress_at
        if now - last_notice >= 60:
            record('sample', pid=pid, cpu_ticks=ticks, idle_seconds=round(idle, 1),
                   children=children, board=board)
            last_notice = now
        if quiet and idle >= 600:
            cmd2 = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if marker not in cmd2 or '--model vl' not in cmd2:
                record('identity-changed-before-signal', pid=pid)
                break
            record('terminate-stalled-trace', pid=pid, idle_seconds=idle, cpu_ticks=ticks, board=board,
                   reason='under one CPU second and no device progress for 600 seconds; timed stacks preserved')
            os.kill(pid, signal.SIGTERM)
            break
        time.sleep(10)
