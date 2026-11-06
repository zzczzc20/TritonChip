#!/usr/bin/env python3
"""Switch the known graph-reset retry only while its next wrapper is compiling."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

work = Path.home()/'r41run3'
run = work/'chat_nohost_v2_20261031'
root = run/'chat-vl/r3-s2-qwen-chat-v3-vl-nohost-20261031'
cache = root/'caches/inductor'
worker, queue, successor = 1160892, 1152880, 1167438

def identity(pid, marker):
    try:
        return marker in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False

def latest():
    return max(((p.stat().st_mtime_ns, p) for p in cache.rglob('*.py') if p.is_file()), default=(0, None))

before, _ = latest()
print('waiting for next compiled wrapper', flush=True)
while True:
    if not (identity(worker, 'runtime_nohost_20261031/tools/models/chat_quality.py') and
            identity(queue, 'chat_fix_inputs/v2/nohost_queue.py') and
            identity(successor, 'chat_fix_inputs/v3/nohost_queue_v3.py')):
        print('process identity changed; no signal sent', flush=True)
        break
    events = [json.loads(s) for s in (run/'queue.jsonl').read_text().splitlines()]
    if events[-1].get('kind') != 'stage-start' or events[-1].get('model') != 'vl':
        raise RuntimeError('queue no longer executing the expected stage')
    timestamp, wrapper = latest()
    if timestamp > before:
        out = run/'superseded_compile_reset'
        out.mkdir(exist_ok=False)
        attempt = next((root/'steps/vl-quality').glob('*/attempt-0001'))
        for path in attempt.glob('*.log'):
            shutil.copyfile(path, out/path.name)
        quality = root/'vl-quality/quality.json'
        if quality.exists():
            shutil.copyfile(quality, out/'quality.json')
        for name in ('status', 'stat', 'wchan'):
            shutil.copyfile(Path(f'/proc/{worker}')/name, out/('proc_'+name))
        shutil.copyfile('/sys/bus/pci/devices/0000:04:00.1/kdsstat', out/'kdsstat')
        record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      worker_pid=worker, queue_pid=queue, new_wrapper=str(wrapper),
                      wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                      reason='run_and_get_code resets Dynamo on every token; the tested graph-preserving retry is queued',
                      signal='SIGTERM')
        (out/'interruption.json').write_text(json.dumps(record, indent=2)+'\n')
        with (run/'queue.jsonl').open('a') as log:
            log.write(json.dumps(dict(kind='superseded-for-graph-reuse', **record))+'\n')
        os.kill(queue, signal.SIGTERM)
        os.kill(worker, signal.SIGTERM)
        print(json.dumps(record), flush=True)
        break
    time.sleep(1)
