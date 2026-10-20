#!/usr/bin/env python3
"""Run/resume a bounded experiment with explicit inputs, outputs and identities.

The manifest supplies the hypothesis, competing explanations, counter changes,
commands and budgets. Each command can compile, screen analytically, execute
SystemC/RTL/FPGA, or compare reports. Exit zero means those commands finished;
it never promotes their evidence to a performance or delivery acceptance gate.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

KINDS = {'compile', 'analytical', 'systemc', 'rtl', 'fpga', 'compare', 'report'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def files(paths):
    result = {}
    for path in paths:
        if not path.exists():
            raise ValueError(f'missing declared input/output: {path}')
        children = [path] if path.is_file() else sorted(path.rglob('*'))
        for child in children:
            if child.is_file() and not any(p in ('.git', '__pycache__', '.pytest_cache')
                                           for p in child.parts):
                result[str(child)] = digest(child)
    return result


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def run(manifest, root, out, *, python, retry_failed=False):
    spec = json.loads(manifest.read_text())
    if socket.gethostname() not in spec['allowed_hosts']:
        raise ValueError('this experiment may run only on its declared remote hosts')
    if not all(spec.get(k) for k in ('id', 'hypothesis', 'competing_explanations',
                                    'expected_observations', 'sources', 'steps')):
        raise ValueError('experiment lacks its hypothesis, evidence or commands')
    root, out = root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def expand(value):
        return (value.replace('{root}', str(root)).replace('{out}', str(out))
                .replace('{python}', str(python)).replace('{home}', str(Path.home())))
    def paths(names):
        return [Path(expand(p)) if Path(expand(p)).is_absolute() else root/expand(p)
                for p in names]
    source_files = files(paths(spec['sources']))
    common = dict(manifest_sha256=digest(manifest), sources=source_files,
                  host=socket.gethostname(), python=str(python), python_sha256=digest(Path(python).resolve()))
    report = dict(schema='tnca-experiment/1', experiment=spec['id'], identity=common,
                  hypothesis=spec['hypothesis'], competing_explanations=spec['competing_explanations'],
                  expected_observations=spec['expected_observations'], status='running', steps=[],
                  delivery_acceptance='not-measured')
    report_path = out/'report.json'
    ids = [step['id'] for step in spec['steps']]
    if len(set(ids)) != len(ids) or any(not name.replace('-', '').replace('_', '').isalnum() for name in ids):
        raise ValueError('step ids must be unique simple names')
    for step in spec['steps']:
        if step['kind'] not in KINDS or not isinstance(step['argv'], list) or not step['argv']:
            raise ValueError('each step requires an explicit command and evidence kind')
        timeout = step['timeout_seconds']
        if type(timeout) not in (int, float) or not 0 < timeout <= 259200:
            raise ValueError('each command requires a finite positive budget')
        argv = [expand(arg) for arg in step['argv']]
        # Product knobs are explicit experiment inputs. Ambient TNCA knobs
        # must not silently change an otherwise identical measurement.
        env = {k: v for k, v in os.environ.items() if not k.startswith('TNCA_')}
        env.update({k: expand(str(v)) for k, v in spec.get('env', {}).items()})
        env.update({k: expand(str(v)) for k, v in step.get('env', {}).items()})
        cwd = paths([step.get('cwd', '.')])[0]
        explicit_env = set(spec.get('env', {})) | set(step.get('env', {}))
        public_env = {k: v for k, v in env.items()
                      if k in explicit_env or k in
                      ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CC', 'CXX',
                       'CMAKE_PREFIX_PATH', 'CPATH', 'LIBRARY_PATH',
                       'OMP_NUM_THREADS', 'MKL_NUM_THREADS') or
                      k.startswith(('TNCA_', 'TORCH_', 'TORCHINDUCTOR_', 'TRITON_'))}
        identity = dict(common, command=argv, cwd=str(cwd), timeout_seconds=timeout,
                        environment=public_env, inputs=files(paths(step.get('inputs', []))), kind=step['kind'])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = out/'steps'/step['id']/key
        directory.mkdir(parents=True, exist_ok=True)
        previous = sorted(directory.glob('attempt-*/result.json'))
        if previous:
            old = json.loads(previous[-1].read_text())
            if old['status'] == 'completed':
                current_outputs = files(paths(step.get('outputs', [])))
                if current_outputs != old['outputs'] or files(paths(spec['sources'])) != source_files:
                    raise ValueError('completed evidence changed; preserve it and choose a new experiment identity')
                report['steps'].append(dict(old, reused=True))
                publish(report_path, report)
                continue
            if not retry_failed:
                report.update(status='stopped-on-preserved-failure')
                report['steps'].append(old)
                publish(report_path, report)
                return report
        attempt = directory/f'attempt-{len(previous) + 1:04}'
        attempt.mkdir(exist_ok=False)
        row = dict(id=step['id'], kind=step['kind'], fingerprint=key, identity=identity,
                   started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   status='running', attempt=str(attempt), reused=False)
        report['steps'].append(row)
        publish(report_path, report)
        started = time.monotonic()
        proc = None
        try:
            with (attempt/'stdout.log').open('xb') as stdout, (attempt/'stderr.log').open('xb') as stderr:
                proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout,
                                        stderr=stderr, start_new_session=True)
                row['pid'] = proc.pid
                publish(report_path, report)
                try:
                    row['returncode'] = proc.wait(timeout=timeout)
                    row['status'] = 'completed' if proc.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    row['status'] = 'timeout'
                    os.killpg(proc.pid, signal.SIGKILL)
                    row['returncode'] = proc.wait()
            # Kill descendants even when a supervising command exits first.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            row['outputs'] = files(paths(step.get('outputs', []))) if row['status'] == 'completed' else {}
            if files(paths(spec['sources'])) != source_files:
                row['status'] = 'source-changed'
        except BaseException as error:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            row.update(status='error', error=f'{type(error).__name__}: {error}')
        finally:
            row['host_wall_seconds'] = time.monotonic() - started
            row['child_max_rss_kib_cumulative'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            row['logs'] = files([attempt/'stdout.log', attempt/'stderr.log'])
            publish(attempt/'result.json', row)
            publish(report_path, report)
        if row['status'] != 'completed':
            report['status'] = 'stopped-on-failure'
            publish(report_path, report)
            return report
    report['status'] = 'completed-experiment'
    publish(report_path, report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--retry-failed', action='store_true')
    args = ap.parse_args()
    result = run(args.manifest, args.root, args.out, python=args.python,
                 retry_failed=args.retry_failed)
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'completed-experiment' else 1)
