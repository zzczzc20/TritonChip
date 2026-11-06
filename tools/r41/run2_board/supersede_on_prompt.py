#!/usr/bin/env python3
"""Switch a known failing VL quality run after a completed, flushed prompt."""
import ctypes
import datetime
import json
import os
from pathlib import Path
import select
import shutil
import signal
import struct
import time

work = Path.home()/'r41run3'
run = work/'chat_nohost_v3_20261031'
root = run/'chat-vl/r3-s2-qwen-chat-v3-vl-nohost-v3-20261031'
worker, successor = 1168615, 1177837
attempt = next((root/'steps/vl-quality').glob('*/attempt-0001'))
stdout = attempt/'stdout.log'

def identity(pid, marker):
    try:
        return marker in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False

assert identity(worker, 'runtime_nohost_v3_20261031/tools/models/chat_quality.py')
assert identity(successor, 'chat_fix_inputs/v4/nohost_queue_v4.py')
size = stdout.stat().st_size
libc = ctypes.CDLL(None, use_errno=True)
libc.inotify_init1.argtypes = [ctypes.c_int]
libc.inotify_init1.restype = ctypes.c_int
libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
libc.inotify_add_watch.restype = ctypes.c_int
fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(stdout), 0x00000002) < 0:
    raise OSError(ctypes.get_errno(), 'inotify setup failed')
print('waiting for next flushed prompt', flush=True)
while True:
    if not identity(worker, 'runtime_nohost_v3_20261031/tools/models/chat_quality.py'):
        print('worker exited; no signal sent', flush=True)
        break
    if not identity(successor, 'chat_fix_inputs/v4/nohost_queue_v4.py'):
        raise RuntimeError('replacement queue is no longer running')
    ready, _, _ = select.select([fd], [], [], 1)
    if not ready:
        continue
    try:
        os.read(fd, 65536)
        data = stdout.read_bytes()
    except (BlockingIOError, FileNotFoundError):
        continue
    if len(data) <= size or not data.endswith(b'\n'):
        continue
    new_lines = data[size:].decode(errors='replace').splitlines()
    answers = []
    for line in new_lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and 'prompt' in row and 'assistant' in row:
            answers.append(row)
    if not answers:
        size = len(data)
        continue
    # A prompt line is printed only after generation, output retrieval and any
    # cache reset have returned. Pause immediately before the next prompt starts.
    os.kill(worker, signal.SIGSTOP)
    try:
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = Path(f'/proc/{worker}/status').read_text()
            if '\nState:\tT' in state or '\nState:\tt' in state:
                break
            time.sleep(.02)
        else:
            raise RuntimeError('worker did not enter stopped state')
        out = run/'superseded_after_prompt'
        out.mkdir(exist_ok=False)
        for path in attempt.glob('*.log'):
            shutil.copyfile(path, out/path.name)
        quality = root/'vl-quality/quality.json'
        if quality.exists():
            shutil.copyfile(quality, out/'quality.json')
        for name in ('status', 'stat', 'wchan'):
            shutil.copyfile(Path(f'/proc/{worker}')/name, out/('proc_'+name))
        shutil.copyfile('/sys/bus/pci/devices/0000:04:00.1/kdsstat', out/'kdsstat')
        record = dict(kind='superseded-after-prompt', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      worker_pid=worker, replacement_queue_pid=successor, completed_answers=answers,
                      reason='registered quality already failed; grouped-tail fix passed both complete CPU quality suites',
                      signal='SIGTERM at flushed prompt boundary')
        (out/'interruption.json').write_text(json.dumps(record, indent=2)+'\n')
        with (run/'queue.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        os.kill(worker, signal.SIGTERM)
        print(json.dumps(record), flush=True)
    finally:
        # SIGCONT also ensures a failed archival cannot leave our process frozen.
        try:
            os.kill(worker, signal.SIGCONT)
        except ProcessLookupError:
            pass
    break
os.close(fd)
