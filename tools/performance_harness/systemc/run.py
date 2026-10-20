#!/usr/bin/env python3
"""Run a sealed TLM experiment. Model time is a prediction, never FPGA evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = ROOT / 'doc/r40_tasks_and_acceptance_v2.md'
CONTRACT_LOCK = ROOT / 'spec/r40/v2/lock.json'
CONTRACT_LOCK_SHA256 = '49fc0e02733139707ae2292eff2ef893e76e73f0723d0d002b2e7e1864af71c8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sources():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.cpp', '.hpp', '.sh', '.cfg', '.txt', '.v', '.vh')
            and '__pycache__' not in p.parts and not any(s.startswith('build') for s in p.relative_to(HERE).parts)}


def systemc_library(binary):
    """Bind the loaded shared library as well as the executable that uses it."""
    text = subprocess.check_output(['ldd', str(binary)], text=True)
    libraries = re.findall(r'libsystemc[^\s]*\s+=>\s+(/\S+)', text)
    if len(libraries) != 1:
        raise ValueError('exactly one resolved shared SystemC library required')
    path = Path(libraries[0]).resolve()
    return dict(path=str(path), sha256=sha(path))


def run(binary, config, trace, out, timeout=1800, source_identity=None, mode='memory', monitor='aggregate'):
    sys.path.insert(0, str(ROOT/'python'))
    from tnca._child_exec import child_argv
    if monitor not in ('off', 'aggregate', 'trace'):
        raise ValueError('unknown socket monitor mode')
    binary, config, trace, out = (Path(p).resolve() for p in (binary, config, trace, out))
    if timeout <= 0:
        raise ValueError('positive host execution timeout required')
    if sha(CONTRACT_LOCK) != CONTRACT_LOCK_SHA256:
        raise ValueError('frozen contract lock identity mismatch')
    lock = json.loads(CONTRACT_LOCK.read_text())
    if lock['files']['doc/r40_tasks_and_acceptance_v2.md'] != sha(CONTRACT):
        raise ValueError('frozen contract document mismatch')
    dependencies = {}
    if mode == 'chip':
        for line in trace.read_text().splitlines():
            fields = shlex.split(line, comments=True)
            if not fields:
                continue
            arity = {'code': 2, 'data': 3, 'expect': 3, 'launch': 7, 'launch_roles': 9, 'launch_mask': 8, 'fence': 1}
            if fields[0] not in arity or len(fields) != arity[fields[0]]:
                raise ValueError('unsupported chip command or trailing fields')
            if fields[0] in ('code', 'data', 'expect'):
                path = Path(fields[-1])
                if not path.is_absolute():
                    raise ValueError('chip script data/code/expect paths must be absolute')
                dependencies[str(path)] = sha(path)
    elif mode != 'memory':
        raise ValueError('unknown model mode')
    identity = dict(monitor=monitor, schema='tnca-systemc-experiment/1', backend='systemc-tlm-prediction', mode=mode,
                    binary_sha256=sha(binary), config_sha256=sha(config), trace_sha256=sha(trace),
                    sources=sources(), host=dict(machine=platform.machine(), node=platform.node()),
                    image_sha256=None, measured_clock_hz=None, calibration='not-measured',
                    contract_sha256=CONTRACT_LOCK_SHA256, contract_document_sha256=sha(CONTRACT), source_identity=source_identity,
                    systemc_library=systemc_library(binary), dependencies_sha256=dependencies)
    out.mkdir(parents=True, exist_ok=False)
    (out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    command = [str(binary), str(config), str(trace), str(out)]
    start = time.monotonic()
    code, timeout_hit = None, False
    with (out/'stdout.log').open('x') as stdout, (out/'stderr.log').open('x') as stderr:
        process = subprocess.Popen(child_argv(command), stdout=stdout, stderr=stderr, start_new_session=True,
                                   env=dict(os.environ, TNCA_TLM_MONITOR=monitor))
        def expire():
            nonlocal timeout_hit
            if process.poll() is None:
                timeout_hit = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            code = process.wait()  # blocking waitpid, no 50-ms timeout polling quantization
        finally:
            timer.cancel()
            timer.join()
    elapsed = time.monotonic()-start
    unchanged = (identity['sources'] == sources() and identity['binary_sha256'] == sha(binary)
                 and identity['config_sha256'] == sha(config) and identity['trace_sha256'] == sha(trace)
                 and identity['contract_document_sha256'] == sha(CONTRACT)
                 and identity['contract_sha256'] == sha(CONTRACT_LOCK)
                 and all(sha(p) == digest for p, digest in dependencies.items())
                 and identity['systemc_library'] == systemc_library(binary))
    summary, summary_error = None, None
    if (out/'summary.json').exists():
        try:
            summary = json.loads((out/'summary.json').read_text())
        except (ValueError, OSError) as e:
            summary_error = str(e)
    passed = code == 0 and unchanged and summary is not None and summary['status'] == 'completed'
    report = dict(schema='tnca-systemc-run/1', status='completed-prediction' if passed else 'failed',
                  command=command, identity=identity, returncode=code, timeout=timeout_hit,
                  inputs_unchanged=unchanged, summary=summary, summary_error=summary_error, host_elapsed_seconds=elapsed,
                  host_clock_resolution_seconds=time.get_clock_info('monotonic').resolution,
                  host_time_scope='process startup, execution, output writes and blocking wait; excludes pre/post hashing',
                  timing_error_bound_cycles=None, performance_gate='not-measured',
                  output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE/'configs/r40-three-core.cfg')
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--source-identity', type=Path)
    p.add_argument('--monitor', choices=('off', 'aggregate', 'trace'), default='aggregate')
    p.add_argument('--mode', choices=('memory', 'chip'), default='memory')
    a = p.parse_args()
    identity = json.loads(a.source_identity.read_text()) if a.source_identity else None
    report = run(a.binary, a.config, a.trace, a.out, a.timeout, identity, mode=a.mode, monitor=a.monitor)
    print(json.dumps({k: report[k] for k in ('status', 'host_elapsed_seconds', 'timing_error_bound_cycles', 'performance_gate')}))
    return 0 if report['status'] == 'completed-prediction' else 1


if __name__ == '__main__':
    raise SystemExit(main())
