#!/usr/bin/env python3
"""Queue one recorded quality baseline per model without loading an image."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HOME = Path.home()
WORK = HOME / 'r41run3'
ORIGINAL = WORK / 'runtime_activation_equal_20261031'
ROOT = WORK / 'runtime_deadline_baseline_20261101'
OUT = WORK / 'chat_deadline_baseline_20261101'
PAYLOAD = WORK / 'deadline_baseline_payload_20261101'
PYTHON = HOME / 'tc_env/torchenv/bin/python'
UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
IMAGE = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
STOP = datetime.datetime(2026, 11, 1, 12, tzinfo=datetime.timezone.utc)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(kind, **fields):
    with (OUT / 'queue.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     kind=kind, **fields)) + '\n')


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError('baseline source or output already exists')
    OUT.mkdir()
    subprocess.run(['cp', '-a', '--reflink=auto', str(ORIGINAL), str(ROOT)], check=True)
    for name in ('tools/models/chat.py', 'tools/models/chat_timing.py', 'tools/models/chat_quality.py',
                 'spec/r41/chat/quality-author-20261101.json'):
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAYLOAD / name, target)
    payload_sha = {name: sha(PAYLOAD / name) for name in
                   ('tools/models/chat.py', 'tools/models/chat_timing.py', 'tools/models/chat_quality.py',
                    'spec/r41/chat/quality-author-20261101.json', 'DEADLINE_24H.md')}
    for model in ('vl', 'q35'):
        reg0 = json.loads((ROOT / 'spec/r41/run3/stages' /
                           f'chat-{model}-nohost-v5-20261031.json').read_text())
        row0 = next(row for row in reg0['experiments'] if '-quality-' in row['id'])
        source = ROOT / row0['manifest']
        if sha(source) != row0['sha256']:
            raise ValueError(f'{model}: original quality manifest changed')
        spec = json.loads(source.read_text())
        name = f'r41-deadline-{model}-baseline-20261101'
        spec['id'] = name
        spec['registration']['scope'] = ('One full-checkpoint baseline on the existing software, with '
            'wall timing and author-rule scoring instrumentation; no performance-package runtime changes. '
            'The original image, prompt, token limit and counters are retained. One-hour budget.')
        spec['registration']['deadline_directive_sha256'] = payload_sha['DEADLINE_24H.md']
        spec['sources'].append('spec/r41/chat/quality-author-20261101.json')
        step = spec['steps'][0]
        argv = step['argv']
        index = argv.index('--spec') + 1
        if argv[index] != '{root}/spec/r41/chat/quality-v1.json':
            raise ValueError(f'{model}: unexpected registered quality spec')
        argv[index] = '{root}/spec/r41/chat/quality-author-20261101.json'
        step['timing_boundary'] = 'prompt entry through flushed assistant output; per-call warmup, prefill, decode and retrieval recorded'
        manifest = ROOT / 'spec/r41/run3/board' / (name + '.json')
        manifest.write_text(json.dumps(spec, indent=2) + '\n')
        registration = dict(schema=reg0['schema'], id=name, image_identity=reg0['image_identity'],
            experiments=[dict(id=name, manifest=str(manifest.relative_to(ROOT)), sha256=sha(manifest),
                              budget_seconds=3600, checks=[
                                  dict(path=f'{model}-quality/quality.json', field=['status'], equals='pass'),
                                  dict(path=f'{model}-quality/quality.json', field=['model_scope'], equals='full-checkpoint'),
                                  dict(path=f'{model}-quality/quality.json', field=['reduced_proof'], equals=False)])])
        path = OUT / (name + '-registration.json')
        path.write_text(json.dumps(registration, indent=2) + '\n')
        record('registered', model=model, manifest_sha256=sha(manifest), registration_sha256=sha(path),
               original_manifest_sha256=row0['sha256'], payload_sha256=payload_sha)


def wait_for_handoff(auto, sysfs, lock, model):
    """Release our queue lock if someone else uses the card; require a new idle interval."""
    current = auto.probe(sysfs)
    if current['uuid'] == UUID and auto.idle(current) and auto.clock_mhz(sysfs) == 100:
        return current
    record('waiting-card', model=model, board=current)
    fcntl.flock(lock, fcntl.LOCK_UN)
    tracker = auto.IdleTracker(max_gap=15)
    while datetime.datetime.now(datetime.timezone.utc) < STOP:
        time.sleep(10)
        try:
            row = auto.probe(sysfs)
            ready = tracker.observe(time.monotonic(), row)
            if not ready or row['uuid'] != UUID or auto.clock_mhz(sysfs) != 100:
                continue
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            current = auto.probe(sysfs)
            if current == row and auto.idle(current) and auto.clock_mhz(sysfs) == 100:
                return current
            fcntl.flock(lock, fcntl.LOCK_UN)
        except (OSError, ValueError):
            tracker = auto.IdleTracker(max_gap=15)
    return None


def main():
    prepare()
    sys.path.insert(0, str(ROOT / 'tools/r41/run2_board'))
    import auto
    sysfs = Path('/sys/bus/pci/devices')
    tracker = auto.IdleTracker(max_gap=15)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    record('waiting', image_uuid=UUID, image_sha256=IMAGE, stop_utc=STOP.isoformat())
    while datetime.datetime.now(datetime.timezone.utc) < STOP:
        try:
            row = auto.probe(sysfs)
            now = time.monotonic()
            ready = tracker.observe(now, row)
            try:
                monitor = json.loads((WORK / 'idle_monitor.json').read_text())
                ready |= (monitor['boot_id'] == boot and monitor['identity'] == [row['uuid'], row['total execs']]
                          and monitor['start'] is not None and monitor['last'] is not None
                          and 0 <= now - monitor['last'] <= 15 and now - monitor['start'] >= 900
                          and auto.idle(row))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if row['uuid'] != UUID or auto.clock_mhz(sysfs) != 100:
                record('waiting-image-or-clock', board=row)
                time.sleep(10)
                continue
            if ready:
                with (WORK / 'candidate_queue.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(10)
                        continue
                    current = auto.probe(sysfs)
                    if current != row or not auto.idle(current) or auto.clock_mhz(sysfs) != 100:
                        continue
                    if sha(WORK / 'tnca_u280.xclbin') != IMAGE:
                        raise ValueError('loaded image file hash differs')
                    for model in ('vl', 'q35'):
                        remaining = (STOP - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                        if remaining <= 0:
                            record('deadline-before-stage', model=model)
                            break
                        current = wait_for_handoff(auto, sysfs, lock, model)
                        if current is None:
                            record('deadline-while-waiting-card', model=model)
                            break
                        name = f'r41-deadline-{model}-baseline-20261101'
                        reg = OUT / (name + '-registration.json')
                        remaining = (STOP - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                        if remaining <= 1:
                            record('deadline-before-stage', model=model)
                            break
                        if remaining < 3600:
                            bounded = json.loads(reg.read_text())
                            bounded['experiments'][0]['budget_seconds'] = int(remaining)
                            reg = OUT / (name + '-deadline-registration.json')
                            with reg.open('x') as stream:
                                json.dump(bounded, stream, indent=2)
                                stream.write('\n')
                            record('deadline-budget', model=model, seconds=int(remaining),
                                   registration_sha256=sha(reg))
                        record('stage-start', model=model, board=current, registration_sha256=sha(reg))
                        command = [str(PYTHON), str(ROOT / 'tools/r41/run2_board/stage.py'),
                                   '--root', str(ROOT), '--out', str(OUT / model), '--python', str(PYTHON),
                                   '--registration', str(reg)]
                        with (OUT / f'{model}.log').open('x') as stream:
                            result = subprocess.run(command, cwd=ROOT, env=auto.board_environment(),
                                                    stdout=stream, stderr=subprocess.STDOUT)
                        summary = OUT / model / 'stage.json'
                        record('stage-end', model=model, returncode=result.returncode,
                               stage_sha256=sha(summary) if summary.is_file() else None)
                    record('finished', scope='one baseline attempt per model; inspect retained results')
                    return
        except (OSError, ValueError) as error:
            tracker = auto.IdleTracker(max_gap=15)
            record('probe-or-setup-error', error=str(error))
        time.sleep(10)
    record('deadline-before-queue', scope='no further baseline stages started')


if __name__ == '__main__':
    main()
