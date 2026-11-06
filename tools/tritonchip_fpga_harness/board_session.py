"""One card session on the shared U280 for an agent's candidate image.

Run on the card host from the platform's own checkout (this file's tree):

    python -m tritonchip_fpga_harness.board_session --xclbin X --sha256 S
        --bdf 0000:04:00.1 --out DIR [--socket P] [--stop FILE]
        [--idle-seconds 900] [--deadline-seconds N] [--python PY]
        [--device NAME|PATH] [--base-map PATH] [--relay-idle-seconds S]

No candidate-written code runs here. The candidate's host side (its compiler
and python/tnca) runs on another machine and reaches the card through
python/tnca/backends/board_relay.py, which this session starts from this
checkout's python and which executes only link-level operations. In order:

 1. admission: the image's sha256 is S; xclbinutil reads its UUID and clocks
    and DATA_CLK is the device description's kernel clock; its static HBM map
    equals the registered base map when the R41 map tooling and a registered
    map exist (else recorded `not-checked`). CDC and core placement need the
    routed checkpoint, which the card host does not have: recorded
    `not-checked`, never passed.
 2. idle: the card continuously idle for --idle-seconds (no context, no
    outstanding exec, no client of any account, the same loaded UUID and exec
    count throughout), then the shared lease taken non-blocking and the card
    probed again under it (acceptance_queue's rule: a busy lease keeps the
    interval, a change under the lease restarts it).
 3. program, only when the loaded UUID is not the image's: flow/fpga/program.sh
    --check, then --go. Judged by the loaded UUID and the clock counter, never
    by program.sh's exit status (reloading the image already on the card is a
    silent no-op, which is why an already-loaded image is not reloaded).
 4. three probes 2 s apart: the image's UUID and the kernel clock (100 MHz on
    R41) on the board counter; DIR/clock_evidence.json from them.
 5. the relay on DIR/relay.sock (or --socket), observer logs in DIR/observe/,
    then DIR/ready.json: everything a client needs (socket, image, clock,
    clock evidence inline, the exact backend_args for `relay(...)`).
 6. wait until the relay exits: --stop FILE (default DIR/stop) appears, or no
    connection for --relay-idle-seconds, or the deadline. Past the deadline
    the session makes the stop file itself, and only then terminates its own
    relay process.
 7. the lease released, then DIR/session.json (written once, at the end):
    status ok | refused-admission | busy-timeout | program-failed |
    relay-failed | timeout, with timings and probes before and after.

The image stays loaded afterwards; the next session overwrites it. The session
never resets the card, never kills a process it did not start, and touches the
card only through program.sh and the relay. --deadline-seconds bounds the whole
session, idle wait included. A card wedged badly enough that sysfs reads hang
hangs the probe too; nothing here can time out a read the kernel will not
return.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
import pwd
import re
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tritonchip_fpga_harness.board_lease import board_lease, lease_path
else:
    from .board_lease import board_lease, lease_path

ROOT = Path(__file__).resolve().parents[2]
VITIS_SETTINGS = '/tools/Xilinx/Vitis/2023.2/settings64.sh'
XRT_SETUP = '/opt/xilinx/xrt/setup.sh'
POLL_SECONDS = 5.0
RELAY_UP_SECONDS = 120.0
RELAY_GRACE_SECONDS = 60.0
CDC_NOT_CHECKED = 'not-checked: needs the routed checkpoint, which the card host does not have'


class _End(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    os.replace(tmp, path)


def _on_path(directory):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def _experiment():
    """tools/performance_harness/experiment.py: the queues' probe and UUID helpers."""
    _on_path(ROOT / 'tools' / 'performance_harness')
    import experiment
    return experiment


def board_env(bdf):
    """Vitis and XRT for program.sh, xclbinutil and the relay (auto.board_environment)."""
    out = subprocess.run(['bash', '-c', f'set -e; source {VITIS_SETTINGS} >/dev/null; '
                          f'source {XRT_SETUP} >/dev/null; env -0'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    env = {}
    for entry in out.split(b'\0'):
        if entry:
            k, v = entry.split(b'=', 1)
            env[os.fsdecode(k)] = os.fsdecode(v)
    env.update(TNCA_BOARD='u280', BDF=bdf)
    return env


def xclbin_info(image, env):
    """UUID and clocks as xclbinutil reports them (the harness's board_pilot)."""
    info = subprocess.run(['xclbinutil', '--info', '--input', str(image)], capture_output=True,
                          text=True, check=True, env=env).stdout
    u = re.search(r'UUID\s*\(xclbin\):\s*(\S+)', info, re.I)
    clocks = {}
    for name in ('hbm_aclk', 'DATA_CLK'):
        m = re.search(r'Name:\s*' + name + r'\s*\n\s*Index:.*\n\s*Type:.*\n\s*Frequency:\s*(\d+) MHz', info)
        clocks[name] = int(m.group(1)) if m else None
    return dict(uuid=u.group(1).lower() if u else None, clocks=clocks)


def probe(bdf):
    """Read-only sysfs: loaded UUID, kernel clock counter, kdsstat use."""
    return _experiment().sysfs_probe(dict(device=bdf))


def idle(row):
    return (row.get('contexts') == 0 and row.get('outstanding_execs') == 0
            and not row.get('client_pid') and isinstance(row.get('total_execs'), int))


def client_owners(row):
    """The accounts behind kdsstat's client pids, for the record of a busy card."""
    owners = []
    for pid in re.findall(r'\d+', row.get('client_pid') or ''):
        try:
            status = Path(f'/proc/{pid}/status').read_text()
            uid = int(re.search(r'^Uid:\s*(\d+)', status, re.M).group(1))
            owners.append(dict(pid=int(pid), user=pwd.getpwuid(uid).pw_name))
        except (OSError, AttributeError, KeyError):
            owners.append(dict(pid=int(pid), user=None))
    return owners


def platform_commit():
    def git(*args):
        try:
            r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        except OSError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    status = git('status', '--porcelain', '--untracked-files=no')
    return dict(root=str(ROOT), commit=git('rev-parse', 'HEAD'),
                dirty=None if status is None else bool(status))


# ---------------------------------------------------------------------------
# 1. admission
# ---------------------------------------------------------------------------
def admit(a, image, env, record):
    adm = record['admission']
    adm.update(cdc=CDC_NOT_CHECKED, core_placement=CDC_NOT_CHECKED)
    if not image.is_file():
        raise _End('refused-admission', f'no image at {image}')
    adm['sha256'] = sha = sha256(image)
    if sha != a.sha256.lower():
        raise _End('refused-admission', f'the image sha256 is {sha}, not the declared {a.sha256}')
    try:
        info = xclbin_info(image, env)
    except (OSError, subprocess.CalledProcessError) as e:
        raise _End('refused-admission', f'xclbinutil cannot read the image: {e}')
    adm.update(uuid=info['uuid'], clocks_metadata=info['clocks'])
    experiment = _experiment()
    try:
        uuid = experiment.canonical_uuid(info['uuid'])
        carried = experiment.xclbin_carries_uuid(image, uuid)
    except ValueError as e:
        raise _End('refused-admission', f'no usable image UUID: {e}')
    if not carried:
        raise _End('refused-admission', f'the image header does not carry its reported UUID {uuid}')
    _on_path(ROOT / 'python')
    from tnca import devices
    try:
        desc = devices.load(a.device)
    except devices.DeviceError as e:
        raise _End('refused-admission', f'device description {a.device}: {e}')
    mhz = desc['platform']['kernel_clock_mhz']
    adm['device'] = dict(name=desc['name'], path=str(devices.path(a.device)),
                         fingerprint=devices.fingerprint(desc), kernel_clock_mhz=mhz)
    if info['clocks'].get('DATA_CLK') not in (None, mhz):
        raise _End('refused-admission', f'the image metadata clocks DATA_CLK at '
                                        f'{info["clocks"]["DATA_CLK"]} MHz, not {mhz}')
    try:
        adm['hbm_map'] = hbm_map(image, Path(a.base_map) if a.base_map else
                                 ROOT / 'spec' / 'r41' / 'run3' / 'xclbin_hbm_map.json')
    except Exception as e:                                 # noqa: BLE001
        raise _End('refused-admission', f'the R41 map tooling cannot read the image: '
                                        f'{type(e).__name__}: {e}')
    if adm['hbm_map']['status'] == 'differs':
        raise _End('refused-admission', 'the image\'s HBM map differs from the registered base map')
    return dict(sha256=sha, uuid=uuid, clocks=info['clocks'], mhz=mhz,
                backend_args=devices.backend_args(desc))


def hbm_map(image, base):
    """The R41 map tooling's static parse of the image, against the registered
    base map; the image's own sha256 is the one field allowed to differ."""
    tool = ROOT / 'tools' / 'r41' / 'run2_board' / 'candidate_map.py'
    if not tool.is_file():
        return dict(status='not-checked', reason=f'no R41 map tooling at {tool}')
    if not base.is_file():
        return dict(status='not-checked', reason=f'no registered base map at {base}')
    _on_path(tool.parent)                  # candidate_map imports its sibling `auto`
    spec = importlib.util.spec_from_file_location('_board_session_candidate_map', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mine = json.loads(json.dumps(module.read_map(image)))
    registered = json.loads(base.read_text())
    keys = ('groups', 'allocation_8gib')
    same = all(k in registered and mine.get(k) == registered[k] for k in keys)
    return dict(status='equal' if same else 'differs', compared=list(keys), base_map=str(base),
                base_map_sha256=sha256(base), image_map=mine)


# ---------------------------------------------------------------------------
# 2. idle, then the lease
# ---------------------------------------------------------------------------
def wait_and_claim(bdf, seconds, deadline, stop, lease, record):
    """The first probe of a continuous idle interval long enough, under the lease.

    The stop file ends the wait too: a harness that has given up on the card
    should not have it programmed for nobody.
    """
    idle_since = identity = last = None
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise _End('busy-timeout', f'the card was not continuously idle for {seconds:g}s '
                                       'with the lease free before the deadline')
        if stop.exists():
            raise _End('busy-timeout', f'{stop} appeared before the card was claimed')
        row = probe(bdf)
        if idle(row):
            current = (row.get('loaded_uuid'), row['total_execs'], tuple(row.get('clock_counter_mhz') or ()))
            if (idle_since is None or current != identity or last is None
                    or now - last > POLL_SECONDS * 3):
                idle_since, identity = now, current
            if now - idle_since >= seconds:
                attempt = contextlib.ExitStack()
                try:
                    attempt.enter_context(board_lease(bdf, blocking=False))
                except BlockingIOError:
                    attempt = None           # a cooperative user holds it; keep the interval
                if attempt is not None:
                    again = probe(bdf)
                    if (idle(again) and again['total_execs'] == row['total_execs']
                            and again.get('loaded_uuid') == row.get('loaded_uuid')):
                        lease.enter_context(attempt.pop_all())
                        return again
                    attempt.close()
                    idle_since = identity = None
        else:
            idle_since = identity = None
            record['probes']['last_busy'] = dict(row, owners=client_owners(row))
        last = now
        time.sleep(max(0.0, min(POLL_SECONDS, deadline - time.monotonic())))


# ---------------------------------------------------------------------------
# 3, 4. program if needed, then verify
# ---------------------------------------------------------------------------
def program(a, image, admitted, env, before, out, record):
    rec = record['program']
    rec['previous_uuid'] = before.get('loaded_uuid')
    record['programmed'] = False
    if before.get('loaded_uuid') == admitted['uuid']:
        return
    for mode in ('--check', '--go'):
        if sha256(image) != admitted['sha256']:
            raise _End('refused-admission', 'the image changed before it was programmed')
        now = probe(a.bdf)
        if not idle(now) or now['total_execs'] != before['total_execs']:
            raise _End('program-failed', f'the card changed or became busy before program.sh {mode}')
        log = out / f'program{mode}.log'
        record['programmed'] = mode == '--go'
        with log.open('w') as fh:
            try:
                rc = subprocess.run(['bash', str(ROOT / 'flow' / 'fpga' / 'program.sh'), mode, str(image)],
                                    env=env, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=900).returncode
            except subprocess.TimeoutExpired:
                rc = None
        rec[f'{mode[2:]}_exit'] = rc
        rec[f'{mode[2:]}_log'] = str(log)
        if mode == '--check' and rc != 0:
            raise _End('program-failed', f'program.sh --check refused (exit {rc}); see {log}')
    # Whatever --go said, the card's own answer decides (verify()).


def verify(a, admitted, out, record):
    """Three probes 2 s apart: the image's UUID and the kernel clock, on the card."""
    samples = []
    for i in range(3):
        samples.append(probe(a.bdf))
        if i < 2:
            time.sleep(2)
    record['clock'] = dict(samples=samples)
    loaded = {s.get('loaded_uuid') for s in samples}
    clocks = [(s.get('clock_counter_mhz') or [None])[0] for s in samples]
    record['probes']['after_program'] = samples[-1]
    if loaded != {admitted['uuid']}:
        raise _End('program-failed', f'the card reports {sorted(map(str, loaded))}, not the image '
                                     f'{admitted["uuid"]}')
    if set(clocks) != {admitted['mhz']}:
        raise _End('program-failed', f'the kernel clock counter reads {clocks} MHz, not {admitted["mhz"]}')
    if not all(idle(s) for s in samples):
        raise _End('busy-timeout', 'another client opened the card after the lease was taken')
    record['clock']['mhz'] = admitted['mhz']
    evidence = dict(schema='tnca-agent-harness-board-clock/1', image_sha256=admitted['sha256'],
                    image_uuid=admitted['uuid'], clock_hz=admitted['mhz'] * 1_000_000,
                    kernel_clock_mhz=admitted['mhz'], hbm_aclk_metadata_mhz=admitted['clocks'].get('hbm_aclk'),
                    counter_samples=[s.get('clock_counter_mhz') for s in samples], bdf=a.bdf,
                    captured_utc=utc(), programmed=record['programmed'],
                    note='kernel clock from the board counter, three samples 2 s apart after the '
                         'loaded UUID was verified; hbm_aclk from xclbin metadata only')
    write_json(out / 'clock_evidence.json', evidence)
    return evidence


# ---------------------------------------------------------------------------
# 5, 6. the relay
# ---------------------------------------------------------------------------
def start_relay(a, image, env, paths, deadline, record):
    remaining = max(1.0, deadline - time.monotonic())
    cmd = [a.python, '-m', 'tnca.backends.board_relay', '--xclbin', str(image), '--bdf', a.bdf,
           '--unix', str(paths['socket']), '--observe-dir', str(paths['observe']), '--stop', str(paths['stop']),
           '--idle-timeout', str(a.relay_idle_seconds), '--deadline-seconds', f'{remaining:.0f}']
    relay_env = {k: v for k, v in env.items() if not k.startswith('TNCA_')}
    relay_env['PYTHONPATH'] = str(ROOT / 'python')
    log = paths['out'] / 'relay.log'
    proc = subprocess.Popen(cmd, cwd=str(ROOT / 'python'), env=relay_env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=log.open('a'), start_new_session=True)
    record['relay'].update(pid=proc.pid, command=cmd, log=str(log))
    ready, _, _ = select.select([proc.stdout], [], [], RELAY_UP_SECONDS)
    line = proc.stdout.readline().decode(errors='replace').strip() if ready else ''
    if not line.startswith('listening '):
        stop_relay(proc, paths, record)
        raise _End('relay-failed', f'the relay did not come up ({line or "no output"}); see {log}')
    return proc


def stop_relay(proc, paths, record, *, grace=RELAY_GRACE_SECONDS):
    """Ask through the stop file; only then terminate the relay this session started."""
    if proc.poll() is None:
        paths['stop'].touch()
        record['relay']['stopped_by_session'] = True
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    rest = proc.stdout.read().decode(errors='replace').splitlines()
    for line in rest:
        if line.startswith('stopped '):
            try:
                record['relay']['summary'] = json.loads(line[len('stopped '):])
            except ValueError:
                record['relay']['summary'] = line
    record['relay']['exit_code'] = proc.returncode


def run_relay(proc, paths, deadline, record):
    """Until the relay ends itself, or the deadline (plus its grace) passes."""
    while proc.poll() is None and time.monotonic() < deadline + RELAY_GRACE_SECONDS:
        time.sleep(1.0)
    timed_out = proc.poll() is None
    stop_relay(proc, paths, record)
    summary = record['relay'].get('summary')
    reason = summary.get('reason') if isinstance(summary, dict) else None
    if timed_out or reason == 'deadline':
        raise _End('timeout', 'the session deadline ended the relay')
    if proc.returncode != 0 or reason not in ('stop-file', 'idle-timeout'):
        raise _End('relay-failed', f'the relay exited {proc.returncode} ({reason}); see {record["relay"]["log"]}')


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xclbin', required=True)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--bdf', required=True)
    ap.add_argument('--out', required=True, help='a fresh directory for this session')
    ap.add_argument('--socket', help='the relay socket (default DIR/relay.sock)')
    ap.add_argument('--stop', help='the relay stops when this appears (default DIR/stop)')
    ap.add_argument('--idle-seconds', type=float, default=900.0)
    ap.add_argument('--deadline-seconds', type=float, default=10800.0,
                    help='bounds the whole session, idle wait included')
    ap.add_argument('--relay-idle-seconds', type=float, default=1800.0,
                    help='the relay stops after this long with no connection')
    ap.add_argument('--python', default=sys.executable, help='the interpreter that runs the relay')
    ap.add_argument('--device', default='u280-3core-r41',
                    help='device description (spec/devices name or path) the backend args come from')
    ap.add_argument('--base-map', help='registered HBM map (default spec/r41/run3/xclbin_hbm_map.json)')
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    paths = dict(out=out, socket=Path(a.socket).resolve() if a.socket else out / 'relay.sock',
                 stop=Path(a.stop).resolve() if a.stop else out / 'stop', observe=out / 'observe')
    # Refusals that leave nothing behind: there is no session to record yet.
    problems = []
    try:
        lease = lease_path(a.bdf)
    except ValueError as e:
        problems.append(str(e))
    if (out / 'ready.json').exists() or (out / 'session.json').exists():
        problems.append(f'{out} already holds a session; use a fresh --out')
    if paths['stop'].exists():
        problems.append(f'{paths["stop"]} exists already and would stop the relay at once')
    if len(os.fsencode(str(paths['socket']))) >= 108:
        problems.append(f'{paths["socket"]} is too long for a Unix socket path')
    if problems:
        print('board_session: ' + '; '.join(problems), file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    image = Path(a.xclbin).resolve()
    t0 = time.monotonic()
    deadline = t0 + a.deadline_seconds
    record = dict(schema='tnca-board-session/1', status=None, reason=None, phase='admission',
                  host=socket.gethostname(), user=pwd.getpwuid(os.getuid()).pw_name,
                  session_pid=os.getpid(), started_utc=utc(), platform=platform_commit(),
                  image=dict(path=str(image), sha256_declared=a.sha256.lower()), bdf=a.bdf,
                  out=str(out), socket=str(paths['socket']), stop=str(paths['stop']),
                  observe_dir=str(paths['observe']), admission={}, programmed=None, program={},
                  clock={}, probes={}, timings={}, lease=dict(path=str(lease)), relay={})
    timings = record['timings']

    def _signalled(signum, _frame):
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        raise _End('timeout', f'the session received signal {signum}')
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _signalled)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)     # an ssh that drops is not a stop

    proc = None
    try:
        try:
            env = board_env(a.bdf)
        except (OSError, subprocess.CalledProcessError) as e:
            raise _End('refused-admission', f'no Vitis/XRT environment on this host: {e}')
        admitted = admit(a, image, env, record)
        record['image'].update(sha256=admitted['sha256'], uuid=admitted['uuid'],
                               clocks_metadata=admitted['clocks'])
        with contextlib.ExitStack() as held:
            try:
                record['phase'] = 'idle'
                t = time.monotonic()
                try:
                    before = wait_and_claim(a.bdf, a.idle_seconds, deadline, paths['stop'], held, record)
                finally:
                    timings['idle_wait_seconds'] = round(time.monotonic() - t, 1)
                lease_taken = time.monotonic()
                record['lease']['taken_utc'] = utc()
                record['probes']['before'] = before
                record['phase'] = 'program'
                t = time.monotonic()
                try:
                    program(a, image, admitted, env, before, out, record)
                    evidence = verify(a, admitted, out, record)
                finally:
                    timings['program_seconds'] = round(time.monotonic() - t, 1)
                paths['observe'].mkdir(exist_ok=True)
                record['phase'] = 'relay'
                t = time.monotonic()
                try:
                    proc = start_relay(a, image, env, paths, deadline, record)
                    write_json(out / 'ready.json', dict(
                        schema='tnca-board-session-ready/1', ready_utc=utc(), host=record['host'],
                        session_pid=os.getpid(), relay_pid=proc.pid, bdf=a.bdf,
                        socket=str(paths['socket']), stop=str(paths['stop']),
                        observe_dir=str(paths['observe']), observe_log='<observe_dir>/<label>.log',
                        image=dict(path=str(image), sha256=admitted['sha256'], uuid=admitted['uuid']),
                        programmed=record['programmed'], clock_mhz=admitted['mhz'],
                        clock_evidence=evidence, clock_evidence_path=str(out / 'clock_evidence.json'),
                        backend='fpga-u280-multicore-relay', backend_args=admitted['backend_args'],
                        device=record['admission']['device'], platform=record['platform'],
                        relay_idle_seconds=a.relay_idle_seconds,
                        deadline_utc=(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                            seconds=deadline - time.monotonic())).strftime('%Y-%m-%dT%H:%M:%SZ')))
                    run_relay(proc, paths, deadline, record)
                finally:
                    if proc is not None and proc.returncode is None:
                        stop_relay(proc, paths, record, grace=10)
                    timings['relay_seconds'] = round(time.monotonic() - t, 1)
            finally:
                if 'before' in record['probes']:
                    record['probes']['after'] = probe(a.bdf)
                    record['lease']['held_seconds'] = round(time.monotonic() - lease_taken, 1)
        record['status'] = 'ok'
    except _End as e:
        record['status'], record['reason'] = e.status, e.reason
    except Exception as e:                                 # noqa: BLE001
        # Nothing expected this; the phase it happened in names the status.
        record['status'] = {'admission': 'refused-admission', 'idle': 'busy-timeout',
                            'program': 'program-failed'}.get(record['phase'], 'relay-failed')
        record['reason'] = f'unexpected {type(e).__name__}: {e}'
    finally:
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        if 'held_seconds' in record['lease']:
            record['lease']['released_utc'] = utc()
        timings['total_seconds'] = round(time.monotonic() - t0, 1)
        record['ended_utc'] = utc()
        write_json(out / 'session.json', record)
    print(json.dumps(dict(status=record['status'], reason=record['reason'], out=str(out))))
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
