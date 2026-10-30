#!/usr/bin/env python3
"""Run the author's approved run-2 board plan after a guarded idle interval.

The plan is prepared separately and must describe every required stage. Missing
inputs, malformed sysfs, a changed image, or a failed stage stop the queue.
The queue never resets a device, kills a process, or retries a failed load.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

IMAGE_SHA = '1bf769fdbfbeb93586be86561664d43e785af3a01a739d19a3882647132d2d13'
IMAGE_UUID = '01490bbd-d831-23bf-d783-56ec51bd71ce'
REQUIRED = ('identity', 'hbm8', 'original113', 'membound-default', 'membound-tiled2',
            'fp8-cases', 'fp8-blockwise', 'fp8-hub', 'qwen-layers', 'endurance',
            'chat-vl', 'chat-q35')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def probe(sysfs):
    root = Path(sysfs) / '0000:04:00.1'
    fields = {}
    lines = (root / 'kdsstat').read_text().splitlines()
    for i, line in enumerate(lines):
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key, value = key.strip(), value.strip()
        if key in ('contexts', 'outstanding execs', 'total execs'):
            if key in fields or not value.isdecimal():
                raise ValueError('missing or ambiguous kds counter ' + key)
            fields[key] = int(value)
        if key == 'client pid':
            if key in fields:
                raise ValueError('ambiguous client list')
            fields[key] = value or ' '.join(x.strip() for x in lines[i + 1:] if x.strip() and ':' not in x)
    if set(fields) != {'contexts', 'outstanding execs', 'total execs', 'client pid'}:
        raise ValueError('incomplete kdsstat')
    fields['uuid'] = (root / 'xclbinuuid').read_text().strip().lower()
    if len(fields['uuid']) != 36:
        raise ValueError('missing loaded image UUID')
    return fields


def clock_mhz(sysfs):
    counters = sorted((Path(sysfs) / '0000:04:00.0').glob('clock_freq_counter*/clock_counter_freqs'))
    if not counters:
        raise ValueError('no board clock counter')
    return int(counters[0].read_text().split()[0])


def idle(row):
    return row['contexts'] == row['outstanding execs'] == 0 and not row['client pid']


class IdleTracker:
    def __init__(self, seconds=900, max_gap=15):
        self.seconds, self.max_gap = seconds, max_gap
        self.start = self.last = self.identity = None

    def observe(self, now, row):
        identity = (row['uuid'], row['total execs'])
        if not idle(row):
            self.start = self.last = self.identity = None
            return False
        if self.last is None or now < self.last or now - self.last > self.max_gap or identity != self.identity:
            self.start = now
        self.last, self.identity = now, identity
        return now - self.start >= self.seconds


def validate_plan(plan, image):
    if sha(image) != IMAGE_SHA:
        raise ValueError('image hash is not the approved run-2 image')
    stages = plan.get('stages')
    if plan.get('schema') != 'tnca-r41-run2-board-plan/1' or not isinstance(stages, list):
        raise ValueError('missing registered run-2 plan')
    if [s.get('id') for s in stages] != list(REQUIRED):
        raise ValueError('run-2 plan must include all required stages in order')
    for stage in stages:
        argv = stage.get('argv')
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
            raise ValueError('each stage requires a complete argv')
        if not isinstance(stage.get('timeout_seconds'), int) or stage['timeout_seconds'] <= 0:
            raise ValueError('each stage requires a positive timeout')
        checks = stage.get('checks')
        if not isinstance(checks, list) or not checks or any(
            not isinstance(c, dict) or not isinstance(c.get('path'), str) or not c['path']
            or not isinstance(c.get('field'), list) or not c['field'] or 'equals' not in c for c in checks
        ):
            raise ValueError('each stage must declare result-file checks')
    evidence = plan.get('preflight_evidence', [])
    if not isinstance(evidence, list) or {item.get('kind') for item in evidence} != {'source', 'clock-review', 'cdc-review', 'hbm-map'}:
        raise ValueError('source, clock, CDC and HBM map evidence is required')
    for item in evidence:
        p = Path(item['path'])
        if not p.is_file() or sha(p) != item['sha256']:
            raise ValueError('preflight evidence missing or changed: ' + str(p))
        result = json.loads(p.read_text())
        if result.get('status') != 'pass' or result.get('image_sha256') != IMAGE_SHA or result.get('kind') != item['kind']:
            raise ValueError('preflight evidence is not a passing image-bound review: ' + str(p))
    if not Path(plan.get('root', '')).is_dir():
        raise ValueError('plan source root is missing')


def board_environment():
    # The documented loader and later tests need the Vitis and XRT setup even
    # when the watcher was launched from a non-interactive login session.
    result = subprocess.run(['bash', '-c',
        'set -e; source /tools/Xilinx/Vitis/2023.2/settings64.sh >/dev/null; '
        'source /opt/xilinx/xrt/setup.sh >/dev/null; export TNCA_BOARD=u280; env -0'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    env = {}
    for entry in result.stdout.split(b'\0'):
        if entry:
            key, value = entry.split(b'=', 1)
            env[os.fsdecode(key)] = os.fsdecode(value)
    if env.get('TNCA_BOARD') != 'u280':
        raise ValueError('board environment setup failed')
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--program', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--poll', type=float, default=5)
    a = p.parse_args()
    if not 0 < a.poll <= 10:
        raise ValueError('poll must be in (0,10] seconds')
    plan = json.loads(a.plan.read_text())
    validate_plan(plan, a.image)
    if not a.program.is_file():
        raise ValueError('program.sh missing')
    env = board_environment()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'queue.lock').open('w') as lock, (a.out / 'queue.jsonl').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kw):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kw))+'\n')
        tracker = IdleTracker(max_gap=max(15, a.poll * 2.5))
        while True:
            try:
                row = probe(a.sysfs)
                ready = tracker.observe(time.monotonic(), row)
                record('idle-probe', **row)
                if ready:
                    break
            except (OSError, ValueError) as e:
                tracker = IdleTracker(max_gap=max(15, a.poll * 2.5))
                record('probe-error', error=str(e))
            time.sleep(a.poll)
        # program.sh performs its own device and idle checks; repeat the probe
        # immediately before each operation and refuse a changed idle identity.
        for mode in ('--check', '--go'):
            # The image and review files may have changed during the wait or
            # between --check and --go. Never rely on the initial digest alone.
            validate_plan(plan, a.image)
            now = probe(a.sysfs)
            if not idle(now) or (now['uuid'], now['total execs']) != tracker.identity:
                raise RuntimeError('card changed or became busy before programming')
            record('program', mode=mode, previous=now['uuid'])
            with (a.out / 'program.log').open('a') as f:
                subprocess.run(['bash', str(a.program), mode, str(a.image)], stdout=f, stderr=subprocess.STDOUT,
                               env=env, check=True)
        row = probe(a.sysfs)
        if row['uuid'] != IMAGE_UUID or clock_mhz(a.sysfs) != 100:
            raise RuntimeError('loaded image UUID or clock differs; stopping')
        for stage in plan['stages']:
            row = probe(a.sysfs)
            if not idle(row) or row['uuid'] != IMAGE_UUID or clock_mhz(a.sysfs) != 100:
                raise RuntimeError('card busy or image/clock changed before ' + stage['id'])
            if any(Path(check['path']).exists() for check in stage['checks']):
                raise RuntimeError('stage result already exists; use a fresh output: ' + stage['id'])
            record('stage-start', stage=stage['id'])
            with (a.out / (stage['id'] + '.log')).open('a') as f:
                result = subprocess.run(stage['argv'], cwd=plan['root'], env=env, stdout=f,
                                        stderr=subprocess.STDOUT, timeout=stage['timeout_seconds'])
            record('stage-end', stage=stage['id'], returncode=result.returncode)
            if result.returncode:
                raise RuntimeError('stage failed: ' + stage['id'])
            for check in stage['checks']:
                value = json.loads(Path(check['path']).read_text())
                for key in check['field']:
                    value = value[key]
                if value != check['equals']:
                    raise RuntimeError('stage result check failed: ' + stage['id'])
            row = probe(a.sysfs)
            if row['uuid'] != IMAGE_UUID or not idle(row) or clock_mhz(a.sysfs) != 100:
                raise RuntimeError('card busy or image/clock changed after ' + stage['id'])
        record('completed')


if __name__ == '__main__':
    main()
