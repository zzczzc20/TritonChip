#!/usr/bin/env python3
"""Resume missing Qwen3.5 quality prompts only after the original stage times out.

A segmented result is explicitly a reduced proof; it does not rewrite the
original stage or claim the registered single-session run passed. No image is
programmed. Separate pre-existing acceleration A/B stages follow only after
all five independently evidenced prompts pass.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from datetime import datetime, timezone

HOME = Path.home()
WORK = HOME / 'r41run3'
ORIG = WORK / 'runtime_nohost_v6_20261031'
ROOT = WORK / 'runtime_nohost_v6_resume_20261031'
OUT = WORK / 'chat_nohost_v6_resume_20261031'
ORIGINAL_OUT = WORK / 'chat_nohost_v6_20261031/chat-q35'
QUALITY_ID = 'r3-s2-qwen-chat-v3-q35-nohost-v5-quality-20261031'
PIDS = [1191357, 1191640, 1193463, 1194129]
PYTHON = HOME / 'tc_env/torchenv/bin/python'
UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
IDS = ('fact', 'arith', 'reason', 'instruct', 'multiturn')


def record(event, **fields):
    with (OUT / 'queue.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **fields)) + '\n')


def wait_pid(pid):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([fd], [], [])
    finally:
        os.close(fd)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_prompt(p):
    if p.get('id') not in IDS or p.get('passed') is not True or not p.get('turns'):
        raise ValueError('quality prompt is missing, unrecognised or failed')
    for turn in p['turns']:
        ev = turn['device_evidence']
        if ev.get('cpu_numerical_fallback') is not False or ev.get('compiled_wrapper_roles_covered') is not True:
            raise ValueError('missing no-CPU-fallback evidence')
        if any(ev['compiled_wrapper_fallbacks'].values()):
            raise ValueError('compiled wrapper reported a CPU fallback')
        placements = ev['model_tensor_placements']
        for kind in ('parameters', 'buffers'):
            placement = placements.get(kind)
            if (not isinstance(placement, dict) or not placement
                    or any(not name.startswith('tnca:') or type(count) is not int or count <= 0
                           for name, count in placement.items())):
                raise ValueError(f'{kind} placement is missing or not fully on TNCA')
        decode = [s for s in turn['steps'] if s.get('kind') == 'decode']
        if not decode or any(s['transfers_inside_step']['host_read_calls'] != 0 for s in decode):
            raise ValueError('decode host-read evidence is missing or nonzero')


def prepare():
    # This independent copy is prepared before waiting; no file in ORIG is
    # changed. All per-prompt manifests are frozen before any resumed run.
    if ROOT.exists() and OUT.exists():
        # Reattachment to our already-prepared queue is safe only before it
        # starts an experiment and only if every registered manifest is intact.
        if (OUT / 'segmented-coverage.json').exists() or any((OUT / x).exists() for x in IDS):
            raise FileExistsError('recovery already ran; do not attach')
        for prompt in IDS:
            name = 'r41-q35-segment-' + prompt + '-20261031'
            reg = json.loads((OUT / (name + '-registration.json')).read_text())
            row = reg['experiments'][0]
            if row['id'] != name or digest(ROOT / row['manifest']) != row['sha256']:
                raise ValueError('prepared recovery manifest changed')
        record('reattached', stage_pid=PIDS[0])
        return
    if ROOT.exists() or OUT.exists():
        raise FileExistsError('incomplete recovery tree or output')
    subprocess.run(['cp', '-a', '--reflink=auto', str(ORIG), str(ROOT)], check=True)
    OUT.mkdir()
    reg0 = json.loads((ROOT / 'spec/r41/run3/stages/chat-q35-nohost-v5-20261031.json').read_text())
    row0 = next(r for r in reg0['experiments'] if r['id'] == QUALITY_ID)
    source = ROOT / row0['manifest']
    if digest(source) != row0['sha256']:
        raise ValueError('original registered manifest changed')
    spec0 = json.loads(source.read_text())
    for prompt in IDS:
        spec = json.loads(json.dumps(spec0))
        name = 'r41-q35-segment-' + prompt + '-20261031'
        spec['id'] = name
        spec['registration']['scope'] = ('Timeout recovery: one unchanged registered quality prompt, '
            'separate session and explicitly reduced proof. Original timeout remains recorded.')
        spec['steps'][0]['argv'].extend(['--prompts', prompt])
        path = ROOT / 'spec/r41/run3/board' / (name + '.json')
        with path.open('x') as f:
            json.dump(spec, f, indent=2); f.write('\n')
        reg = dict(schema=reg0['schema'], id=name, image_identity=reg0['image_identity'],
            experiments=[dict(id=name, manifest=str(path.relative_to(ROOT)), sha256=digest(path),
                budget_seconds=10800, checks=[
                    dict(path='q35-quality/quality.json', field=['status'], equals='pass'),
                    dict(path='q35-quality/quality.json', field=['model_scope'], equals='full-checkpoint'),
                    dict(path='q35-quality/quality.json', field=['reduced_proof'], equals=True),
                    dict(path='q35-quality/quality.json', field=['prompts'], length_equals=1)])])
        with (OUT / (name + '-registration.json')).open('x') as f:
            json.dump(reg, f, indent=2); f.write('\n')
    record('waiting-for-original', stage_pid=PIDS[0], scope='timeout-only segmented recovery')


def stage(root, target, reg, log):
    sys.path.insert(0, str(ROOT / 'tools/r41/run2_board'))
    import auto
    sysfs = Path('/sys/bus/pci/devices')
    while True:
        with (WORK / 'candidate_queue.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                row = auto.probe(sysfs)
                if row['uuid'] != UUID or auto.clock_mhz(sysfs) != 100:
                    raise RuntimeError('board image or clock changed; no image programming attempted')
                if auto.idle(row):
                    if digest(WORK / 'tnca_u280.xclbin') != IMAGE_SHA:
                        raise RuntimeError('image file hash changed')
                    command = [str(PYTHON), str(root / 'tools/r41/run2_board/stage.py'),
                        '--root', str(root), '--out', str(target), '--python', str(PYTHON),
                        '--registration', str(reg)]
                    record('stage-start', root=str(root), output=str(target), board=row)
                    with log.open('x') as f:
                        result = subprocess.run(command, cwd=root, env=auto.board_environment(),
                                                stdout=f, stderr=subprocess.STDOUT)
                    summary = target / 'stage.json'
                    if result.returncode or not summary.is_file() or json.loads(summary.read_text()).get('status') != 'pass':
                        raise RuntimeError('stage failed: ' + str(target))
                    record('stage-pass', output=str(target), stage_sha256=digest(summary))
                    return
        time.sleep(600)


def main():
    prepare()
    wait_pid(PIDS[0])
    if (ORIGINAL_OUT / 'stage.json').exists():
        record('not-needed', reason='original stage completed')
        return
    report_path = ORIGINAL_OUT / QUALITY_ID / 'report.json'
    report = json.loads(report_path.read_text())
    if not any(row.get('id') == 'q35-quality' and row.get('status') == 'timeout'
               for row in report.get('steps', [])):
        record('not-started', reason='original stage failed for a reason other than quality timeout')
        return
    for pid in PIDS[1:]:
        wait_pid(pid)
    original = ORIGINAL_OUT / QUALITY_ID / 'q35-quality/quality.json'
    quality = json.loads(original.read_text())
    seen = {}
    for row in quality.get('prompts', []):
        check_prompt(row)
        if row['id'] in seen:
            raise ValueError('duplicate original prompt')
        seen[row['id']] = dict(path=str(original), sha256=digest(original))
    missing = [x for x in IDS if x not in seen]
    record('timeout-recovery', original_report=str(report_path), original_quality=str(original), missing=missing)
    for prompt in missing:
        name = 'r41-q35-segment-' + prompt + '-20261031'
        target = OUT / prompt
        stage(ROOT, target, OUT / (name + '-registration.json'), OUT / (prompt + '.log'))
        path = target / name / 'q35-quality/quality.json'
        data = json.loads(path.read_text())
        if data.get('reduced_proof') is not True or len(data.get('prompts', [])) != 1 or data['prompts'][0].get('id') != prompt:
            raise ValueError('segmented output does not match registered prompt')
        check_prompt(data['prompts'][0])
        seen[prompt] = dict(path=str(path), sha256=digest(path))
    if set(seen) != set(IDS):
        raise ValueError('missing quality coverage')
    combined = dict(scope='segmented timeout recovery; original single-session stage timed out',
                    original_report=str(report_path), prompts=seen)
    with (OUT / 'segmented-coverage.json').open('x') as f:
        json.dump(combined, f, indent=2); f.write('\n')
    record('segmented-quality-covered', path=str(OUT / 'segmented-coverage.json'))
    for name in ('event_index', 'equalrow_event', 'pipeline_equalrow'):
        root = WORK / ('runtime_' + name + '_20261031')
        registration = json.loads((root / 'spec/r41/run3/stages/chat-q35-nohost-v5-20261031.json').read_text())
        registration['experiments'] = registration['experiments'][:1]
        reg = OUT / (name + '-registration.json')
        with reg.open('x') as f:
            json.dump(registration, f, indent=2); f.write('\n')
        stage(root, OUT / ('speed-' + name), reg, OUT / ('speed-' + name + '.log'))
    record('finished')


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        if OUT.exists():
            record('stopped', error=f'{type(e).__name__}: {e}')
        raise
