#!/usr/bin/env python3
"""Run/resume a bounded experiment with explicit inputs, outputs and identities.

The manifest supplies the hypothesis, competing explanations, expected counter
changes, commands and budgets. Each command can compile, screen analytically,
execute SystemC/RTL/FPGA, or compare reports. Exit zero means those commands
finished; it never promotes their evidence to a performance or delivery
acceptance gate.

Reuse is exact. A step's key covers the manifest, sources, product software
tree, command, environment, inputs and evidence kind, and, when declared, its
timing boundary and cache state. An `fpga` step additionally requires a board
identity (manifest or step `board` block) whose image sha256, image UUID,
qualified clock evidence hash, clock and memory-map identity enter the key:
records of a different image, clock qualification or memory map never dedup
together (R41 S1-00 rule 4). Before running a board step the loaded image,
measured clock counter and card use are probed read-only; a wrong image, a
different clock or another user's context refuses the step without running
it. The probe is repeated after the step and a changed image fails it.

The report carries the contract's explicit fields (R41 S1-01 A): hypothesis,
competing explanations, expected counter changes, result, residual,
conclusion and next step. The last four come from a declared
`conclusion_source` produced by a compare/report step, or from a reviewed
`--conclusion FILE`; otherwise they stay `not-recorded`.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

KINDS = {'compile', 'analytical', 'systemc', 'rtl', 'fpga', 'compare', 'report'}
BOARD_KINDS = {'fpga'}
BOARD_FIELDS = ('image', 'image_sha256', 'image_uuid', 'clock_evidence', 'clock_hz',
                'memory_map', 'device')
CONCLUSION_FIELDS = ('result', 'residual', 'conclusion', 'next_step')
XCLBIN_MAGIC = b'xclbin2'
# The uuid sits in the axlf header at a layout-dependent offset (see
# python/tnca/backends/u50.py _file_carries_uuid); test membership instead.
XCLBIN_UUID_SEARCH = 4096


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def software_tree(root):
    """archbench.provenance.software_tree of the experiment root, or None.

    Loaded from the file itself: importing the archbench package could pull
    in heavier modules, and the digest must be the product's own definition.
    """
    path = root/'archbench/provenance.py'
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('_experiment_provenance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.software_tree(root, refresh=True)


def canonical_uuid(text):
    raw = str(text).strip().lower().replace('-', '')
    if len(raw) != 32 or any(c not in '0123456789abcdef' for c in raw):
        raise ValueError(f'not a uuid: {text!r}')
    return f'{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}'


def xclbin_carries_uuid(path, uuid):
    with path.open('rb') as stream:
        head = stream.read(XCLBIN_UUID_SEARCH)
    if not head.startswith(XCLBIN_MAGIC):
        raise ValueError(f'{path} is not an .xclbin')
    return bytes.fromhex(canonical_uuid(uuid).replace('-', '')) in head


def board_identity(board, resolve):
    """The declared board identity, verified against the files it names.

    Nothing here touches the card: the image and clock evidence are files,
    the memory map is a declaration (optionally backed by a hashed source).
    """
    missing = [k for k in BOARD_FIELDS if board.get(k) in (None, '', {})]
    if missing:
        raise ValueError(f'fpga step lacks board identity fields: {missing}')
    if type(board['clock_hz']) is not int or board['clock_hz'] <= 0:
        raise ValueError('board clock_hz must be a positive integer')
    image = resolve(board['image'])
    sha = digest(image)
    if sha != board['image_sha256']:
        raise ValueError(f'declared image sha256 {board["image_sha256"]} does not match {image} ({sha})')
    uuid = canonical_uuid(board['image_uuid'])
    if not xclbin_carries_uuid(image, uuid):
        raise ValueError(f'image {image} does not carry the declared UUID {uuid}')
    clock_path = resolve(board['clock_evidence'])
    clock_sha = digest(clock_path)
    if board.get('clock_evidence_sha256') not in (None, clock_sha):
        raise ValueError('qualified clock evidence changed since it was declared')
    clock = json.loads(clock_path.read_text())
    if clock.get('image_sha256') != sha:
        raise ValueError('qualified clock evidence is bound to another image')
    if clock.get('clock_hz') != board['clock_hz']:
        raise ValueError(f'declared clock {board["clock_hz"]} Hz differs from the qualified '
                         f'clock evidence ({clock.get("clock_hz")} Hz)')
    memory_map = board['memory_map']
    if not isinstance(memory_map, dict):
        raise ValueError('memory_map must be a declared mapping')
    source = None
    if memory_map.get('source'):
        source_path = resolve(memory_map['source'])
        source = digest(source_path)
        if memory_map.get('source_sha256') not in (None, source):
            raise ValueError('memory-map source changed since it was declared')
    return dict(image=str(image), image_sha256=sha, image_uuid=uuid,
                clock_evidence=str(clock_path), clock_evidence_sha256=clock_sha,
                clock_hz=board['clock_hz'], clock_qualification=clock.get('qualification'),
                memory_map=memory_map, memory_map_sha256=canonical_sha(memory_map),
                memory_map_source_sha256=source, device=board['device'],
                backend=board.get('backend'), backend_args=board.get('backend_args'))


def sysfs_probe(board):
    """Read-only observation of the card: loaded UUID, clock counter, use."""
    base = Path('/sys/bus/pci/devices')/board['device']
    mgmt = base.parent/board.get('mgmt_device', board['device'][:-1]+'0')
    row = dict(host=socket.gethostname(), observed_utc=utc(), device=board['device'])
    try:
        row['loaded_uuid'] = canonical_uuid((base/'xclbinuuid').read_text())
    except (OSError, ValueError) as error:
        row['loaded_uuid'] = None
        row['loaded_uuid_error'] = f'{type(error).__name__}: {error}'
    counters = sorted(mgmt.glob('clock_freq_counter*/clock_counter_freqs'))
    try:
        row['clock_counter_mhz'] = [int(v) for v in counters[0].read_text().split()] if counters else None
    except (OSError, ValueError):
        row['clock_counter_mhz'] = None
    fields = {}
    try:
        client_lines = []
        reading_clients = False
        for line in (base/'kdsstat').read_text().splitlines():
            if ':' in line:
                name, _, value = line.partition(':')
                fields[name.strip()] = value.strip()
                reading_clients = name.strip() == 'client pid'
                if reading_clients and value.strip():
                    client_lines.append(value.strip())
            elif reading_clients and line.strip():
                client_lines.append(line.strip())
        fields['client pid'] = ' '.join(client_lines)
    except OSError:
        pass
    row['contexts'] = int(fields['contexts']) if fields.get('contexts', '').isdigit() else None
    row['client_pid'] = fields.get('client pid') or None
    value = fields.get('outstanding execs', '')
    row['outstanding_execs'] = int(value) if value.isdigit() else None
    return row


def board_refusal(identity, probe):
    """Why the card's observed state is not the declared board, or None."""
    if probe.get('loaded_uuid') != identity['image_uuid']:
        return (f'loaded image {probe.get("loaded_uuid")} is not the declared image '
                f'{identity["image_uuid"]} ({identity["image_sha256"]})')
    mhz = probe.get('clock_counter_mhz')
    if not mhz or mhz[0] != round(identity['clock_hz']/1e6):
        return f'measured clock counter {mhz} MHz differs from the declared {identity["clock_hz"]} Hz'
    if probe.get('contexts') is None:
        return 'card use (kdsstat contexts) is unreadable'
    if 'outstanding_execs' in probe and probe['outstanding_execs'] is None:
        return 'card use (kdsstat outstanding execs) is unreadable'
    if probe.get('outstanding_execs', 0) != 0:
        return f'card in use: outstanding execs={probe["outstanding_execs"]}'
    if probe['contexts'] != 0 or probe.get('client_pid'):
        return f'card in use: contexts={probe["contexts"]} client={probe.get("client_pid")}'
    return None


def conclusion_fields(record, source):
    missing = [k for k in CONCLUSION_FIELDS if k not in record]
    if missing:
        raise ValueError(f'conclusion record {source} lacks {missing}')
    return {k: record[k] for k in CONCLUSION_FIELDS}


def run(manifest, root, out, *, python, retry_failed=False, conclusion=None, board_probe=sysfs_probe, budget_seconds=None):
    spec = json.loads(manifest.read_text())
    if socket.gethostname() not in spec['allowed_hosts']:
        raise ValueError('this experiment may run only on its declared remote hosts')
    if not all(spec.get(k) for k in ('id', 'hypothesis', 'competing_explanations',
                                    'expected_observations', 'sources', 'steps')):
        raise ValueError('experiment lacks its hypothesis, evidence or commands')
    board_steps = [s for s in spec['steps'] if s.get('kind') in BOARD_KINDS]
    if board_steps and not spec.get('expected_counter_changes'):
        raise ValueError('a board experiment must declare its expected counter changes')
    root, out = root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def expand(value):
        return (value.replace('{root}', str(root)).replace('{out}', str(out))
                .replace('{python}', str(python)).replace('{home}', str(Path.home())))
    def paths(names):
        return [Path(expand(p)) if Path(expand(p)).is_absolute() else root/expand(p)
                for p in names]
    def resolve(name):
        return paths([name])[0]
    source_files = files(paths(spec['sources']))
    tree = software_tree(root)
    common = dict(manifest_sha256=digest(manifest), sources=source_files,
                  host=socket.gethostname(), python=str(python), python_sha256=digest(Path(python).resolve()))
    if tree is not None:
        common['software_tree'] = tree
    counters = spec.get('expected_counter_changes') or spec['expected_observations']
    report = dict(schema='tnca-experiment/2', experiment=spec['id'], identity=common,
                  hypothesis=spec['hypothesis'], competing_explanations=spec['competing_explanations'],
                  expected_counter_changes=counters,
                  expected_counter_changes_source=('expected_counter_changes' if spec.get('expected_counter_changes')
                                                   else 'expected_observations'),
                  expected_observations=spec['expected_observations'],
                  result='not-recorded', residual='not-recorded', conclusion='not-recorded',
                  next_step='not-recorded', status='running', steps=[],
                  delivery_acceptance='not-measured')
    if spec.get('registration'):
        report['registration'] = spec['registration']
    report_path = out/'report.json'
    ids = [step['id'] for step in spec['steps']]
    if len(set(ids)) != len(ids) or any(not name.replace('-', '').replace('_', '').isalnum() for name in ids):
        raise ValueError('step ids must be unique simple names')
    experiment_started = time.monotonic()
    total_budget = spec.get('budget_seconds')
    if budget_seconds is not None:
        if type(budget_seconds) not in (int, float) or not 0 < budget_seconds <= 259200:
            raise ValueError('caller budget must be finite and positive')
        total_budget = min(total_budget, budget_seconds) if total_budget is not None else budget_seconds
    if total_budget is not None and (type(total_budget) not in (int, float) or not 0 < total_budget <= 259200):
        raise ValueError('experiment total budget must be finite and positive')
    for step in spec['steps']:
        if step['kind'] not in KINDS or not isinstance(step['argv'], list) or not step['argv']:
            raise ValueError('each step requires an explicit command and evidence kind')
        timeout = step['timeout_seconds']
        if type(timeout) not in (int, float) or not 0 < timeout <= 259200:
            raise ValueError('each command requires a finite positive budget')
        if total_budget is not None:
            remaining = total_budget - (time.monotonic() - experiment_started)
            if remaining <= 0:
                report.update(status='stopped-on-budget', total_budget_seconds=total_budget)
                publish(report_path, report)
                return report
            timeout = min(timeout, remaining)
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
        # A different timing boundary or cache state is a different
        # measurement even when every command byte is equal.
        for name in ('timing_boundary', 'cache_state'):
            if step.get(name) is not None:
                identity[name] = step[name]
        board = None
        if step['kind'] in BOARD_KINDS:
            if not step.get('timing_boundary') or not step.get('cache_state'):
                raise ValueError(f'fpga step {step["id"]} must declare its timing boundary and cache state')
            if tree is None:
                raise ValueError('an fpga step requires the product software tree identity')
            declared = dict(spec.get('board') or {}, **(step.get('board') or {}))
            board = board_identity(declared, resolve)
            identity['board'] = board
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
        probe_before = None
        if board is not None:
            # Read-only: nothing is loaded and no attempt is created when the
            # card is not the declared board.
            probe_before = board_probe(dict(declared, **board))
            reason = board_refusal(board, probe_before)
            if reason:
                report.update(status='refused-board-identity', refusal=dict(
                    step=step['id'], fingerprint=key, reason=reason, probe=probe_before))
                publish(report_path, report)
                return report
        attempt = directory/f'attempt-{len(previous) + 1:04}'
        attempt.mkdir(exist_ok=False)
        row = dict(id=step['id'], kind=step['kind'], fingerprint=key, identity=identity,
                   started_utc=utc(), status='running', attempt=str(attempt), reused=False)
        if board is not None:
            row['board'] = dict(host=socket.gethostname(), image_uuid=board['image_uuid'],
                                image_sha256=board['image_sha256'], clock_hz=board['clock_hz'],
                                clock_evidence_sha256=board['clock_evidence_sha256'],
                                memory_map_sha256=board['memory_map_sha256'], probe_before=probe_before)
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
            # A board record must belong to one product tree; other steps keep
            # the declared-source check (their build outputs may live in-tree).
            if files(paths(spec['sources'])) != source_files or (board is not None and software_tree(root) != tree):
                row['status'] = 'source-changed'
            if board is not None:
                after = board_probe(dict(declared, **board))
                row['board']['probe_after'] = after
                if (after.get('loaded_uuid'), (after.get('clock_counter_mhz') or [None])[0]) != \
                        (probe_before.get('loaded_uuid'), (probe_before.get('clock_counter_mhz') or [None])[0]):
                    row['status'] = 'board-identity-changed'
        except BaseException as error:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            row.update(status='error', error=f'{type(error).__name__}: {error}')
        finally:
            row['ended_utc'] = utc()
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
    # Result, residual, conclusion and next step are recorded only from a
    # declared comparison output or a reviewed conclusion file, never guessed.
    if conclusion is not None:
        report.update(conclusion_fields(json.loads(conclusion.read_text()), conclusion))
        report['conclusion_record'] = dict(path=str(conclusion), sha256=digest(conclusion))
    elif spec.get('conclusion_source'):
        source = resolve(spec['conclusion_source'])
        produced = {p for s in spec['steps'] for p in map(str, paths(s.get('outputs', [])))}
        if str(source) not in produced:
            raise ValueError('conclusion_source must be a declared step output')
        record = json.loads(source.read_text())
        if all(k in record for k in CONCLUSION_FIELDS):
            report.update(conclusion_fields(record, source))
            report['conclusion_record'] = dict(path=str(source), sha256=digest(source))
    publish(report_path, report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--budget-seconds', type=float, help='additional cap; cannot increase the registered budget')
    ap.add_argument('--conclusion', type=Path,
                    help='reviewed JSON with result/residual/conclusion/next_step for a completed experiment')
    args = ap.parse_args()
    result = run(args.manifest, args.root, args.out, python=args.python,
                 retry_failed=args.retry_failed, conclusion=args.conclusion, budget_seconds=args.budget_seconds)
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'completed-experiment' else 1)
