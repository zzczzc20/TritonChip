#!/usr/bin/env python3
"""Run the two registered chat retries after the live chats and VL diagnostic release the card."""
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

home = Path.home()
work = home/'r41run3'
root = work/'runtime_nohost_20261031'
out = work/'chat_nohost_20261031'
sys.path.insert(0, str(root/'tools/r41/run2_board'))
import auto

uuid = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
image_hash = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
expected = json.loads((work/'chat_fix_inputs/sha256.json').read_text())
for name, digest in expected.items():
    if auto.sha(root/name) != digest:
        raise ValueError('changed retry input: ' + name)
out.mkdir(exist_ok=False)
with (work/'chat_nohost_waiter.lock').open('a') as only, (out/'queue.jsonl').open('x', buffering=1) as log:
    fcntl.flock(only, fcntl.LOCK_EX|fcntl.LOCK_NB)
    def record(kind, **kw):
        log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kw))+'\n')
    def live(pid, marker):
        try:
            return marker in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, ProcessLookupError):
            return False
    record('waiting-existing-chat-and-trace', inputs=expected)
    while live(1149434, 'chat_priority_20261031.py') or live(1151183, 'chat_trace_inputs/waiter.py'):
        time.sleep(5)
    for model in ('vl', 'q35'):
        reg = root/'spec/r41/run3/stages'/f'chat-{model}-nohost-20261031.json'
        while True:
            try:
                row = auto.probe(Path('/sys/bus/pci/devices'))
                if row['uuid'] != uuid:
                    raise RuntimeError('candidate image changed; no programming attempted')
                ready = auto.idle(row) and auto.clock_mhz(Path('/sys/bus/pci/devices')) == 100
            except OSError as exc:
                record('probe-error', error=str(exc))
                time.sleep(5)
                continue
            if ready:
                with (work/'candidate_queue.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        row = auto.probe(Path('/sys/bus/pci/devices'))
                        if auto.idle(row) and row['uuid'] == uuid and auto.clock_mhz(Path('/sys/bus/pci/devices')) == 100:
                            if auto.sha(work/'tnca_u280.xclbin') != image_hash:
                                raise ValueError('image file changed')
                            for name, digest in expected.items():
                                if auto.sha(root/name) != digest:
                                    raise ValueError('changed retry input: ' + name)
                            record('stage-start', model=model, board=row, registration_sha256=auto.sha(reg))
                            target = out/f'chat-{model}'
                            cmd = [str(home/'tc_env/torchenv/bin/python'), str(root/'tools/r41/run2_board/stage.py'),
                                   '--root', str(root), '--out', str(target), '--python',
                                   str(home/'tc_env/torchenv/bin/python'), '--registration', str(reg)]
                            with (out/f'chat-{model}.log').open('x') as stream:
                                try:
                                    run = subprocess.run(cmd, cwd=root, env=auto.board_environment(), stdout=stream,
                                                         stderr=subprocess.STDOUT, timeout=22000)
                                    rc = run.returncode
                                except subprocess.TimeoutExpired:
                                    rc = 124
                            summary = target/'stage.json'
                            record('stage-end', model=model, returncode=rc,
                                   status=json.loads(summary.read_text()).get('status') if summary.exists() else None,
                                   summary_sha256=auto.sha(summary) if summary.exists() else None)
                            break
            time.sleep(5)
    record('completed', scope='both chat retries attempted; inspect actual results')
