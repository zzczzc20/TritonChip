#!/usr/bin/env python3
"""R40-10 item 4: run the GPU comparison campaign one point at a time on a
shared GPU, only while no other user's process holds the device.

    python tools/r40/v3_gpu_campaign.py --raw RAW \
        --stage archbench:inductor-cudagraph:points_existing.txt:archbench_inductor-cudagraph_existing.jsonl \
        --stage boundary:inductor-cudagraph:points_unique.txt:boundary_inductor-cudagraph.jsonl ...

Stage kinds: `archbench` runs `python -m archbench.run --platform gpu
--variant V --point P` (the unchanged ArchBench GPU record); `boundary` runs
tools/r40/v3_gpu_boundaries.py for P. Every point runs in its own process,
like archbench.run does it.

Gate. Before each attempt the driver waits until nvidia-smi has listed no
compute process owned by another user for --quiet seconds. During the
attempt it polls every --poll seconds; if another user's compute process
appears, the attempt is stopped at once (the device is theirs: a GPU
timing taken beside another job is not a measurement, and ours would
disturb theirs) and the point is retried after the next quiet window. The
quiet window applies when another user was seen within the last --quiet
seconds; otherwise one clean check suffices. A point still contended after
--max-attempts is left unmeasured (a gate line says so), never run beside
another job. Clean records go to the stage's OUT; records of attempts that
saw another process (if any finished) to OUT.contended.jsonl; every attempt
gets a line in OUT.gate.jsonl with the processes seen. A rerun resumes:
points that already have a clean record in OUT are skipped.

--fresh-cache gives every attempt its own empty TORCHINDUCTOR_CACHE_DIR and
TRITON_CACHE_DIR (removed afterwards). Inductor's FX graph cache key does
not include torch._inductor.utils.is_big_gpu, so without it a graph compiled
by the inductor-triton-bigsm variant is served to the frozen inductor-triton
variant (observed 2026-10-19: Triton GEMM templates in records of a
configuration that cannot generate them on this GPU).
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
ME = pwd.getpwuid(os.getuid()).pw_name
LAST_FOREIGN = [None]          # monotonic time another user's process was last seen
COSEEN = {}                    # same-account compute processes seen during the current attempt


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def _ppid(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _mine(pid):
    """A descendant of this driver: the point being measured."""
    me = os.getpid()
    while pid and pid > 1:
        if pid == me:
            return True
        pid = _ppid(pid)
    return False


def compute_apps():
    """Compute processes on the GPU other than this driver's own children:
    [(pid, user, mem, kind)], kind `other-user` or `same-account` (another
    job of this account, e.g. a conformance run; recorded, not waited for)."""
    try:
        q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20).stdout
    except Exception as e:                                   # noqa: BLE001
        return [('?', f'nvidia-smi failed: {type(e).__name__}', None, 'other-user')]
    out = []
    for line in q.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if _mine(pid):
            continue
        try:
            user = pwd.getpwuid(os.stat(f'/proc/{pid}').st_uid).pw_name
        except (FileNotFoundError, KeyError):
            user = '?'
        out.append((pid, user, parts[1] if len(parts) > 1 else None,
                    'same-account' if user == ME else 'other-user'))
    return out


def foreign():
    """Other users' compute processes (the gate); records every co-resident."""
    apps = compute_apps()
    out = [a for a in apps if a[3] == 'other-user']
    if out:
        LAST_FOREIGN[0] = time.monotonic()
    COSEEN.update({str(a[0]): dict(user=a[1], mem=a[2], kind=a[3]) for a in apps if a[3] == 'same-account'})
    return [a[:3] for a in out]


def wait_quiet(quiet, poll):
    """Until no other user's compute process is listed: for `quiet` seconds
    if one was seen within the last `quiet` seconds, else for one check."""
    t0 = time.monotonic()
    seen = {}
    announced = False
    while True:
        f = foreign()
        for pid, user, mem in f:
            seen[str(pid)] = dict(user=user, mem=mem)
        if f and not announced:
            print(f'  [{now()}] waiting: other users on the GPU {f}', flush=True)
            announced = True
        last = LAST_FOREIGN[0]
        if not f and (last is None or time.monotonic() - last >= quiet):
            break
        time.sleep(poll)
    return dict(waited_s=round(time.monotonic() - t0, 1), seen_while_waiting=seen)


def run_attempt(cmd, cwd, env, poll, abort_on_contention, log_path, timeout):
    seen = {}
    stop = threading.Event()
    errf = open(log_path, 'w')
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=errf,
                            text=True, start_new_session=True)
    aborted = []

    def watch():
        while not stop.is_set():
            f = foreign()
            if f:
                for pid, user, mem in f:
                    seen[str(pid)] = dict(user=user, mem=mem, t=now())
                if abort_on_contention and not aborted:
                    aborted.append(now())
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            stop.wait(poll)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        aborted.append('timeout')
    stop.set()
    th.join()
    errf.close()
    return proc.returncode, out, seen, aborted


def oom(tmp):
    """A CUDA out-of-memory in the attempt's point log (another process can
    hold unified memory between two polls)."""
    logs = tmp.parent / (tmp.stem + '_logs')
    return any('out of memory' in f.read_text(errors='replace').lower() for f in logs.glob('*.log')) \
        if logs.exists() else False


def done_points(out):
    if not out.exists():
        return set()
    return {json.loads(line)['point_id'] for line in out.read_text().splitlines() if line.strip()}


def stage(kind, variant, points_file, out, args):
    pts = [x.strip() for x in Path(points_file).read_text().splitlines() if x.strip()]
    out = Path(args.raw) / out
    work = out.parent / (out.stem + '_work')
    work.mkdir(parents=True, exist_ok=True)
    have = done_points(out)
    gate = out.with_suffix('.gate.jsonl')
    contended = out.with_suffix('.contended.jsonl')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'python'), env.get('PYTHONPATH', '')])
    print(f'=== {now()} stage {kind} {variant} {points_file}: {len(pts)} points, {len(have)} already clean',
          flush=True)
    n_pass = 0
    for pid in pts:
        n_oom = 0
        if pid in have:
            continue
        safe = pid.replace('/', '__')
        for attempt in range(1, args.max_attempts + 1):
            w = wait_quiet(args.quiet, args.poll)
            tmp = work / f'{safe}.attempt{attempt}.jsonl'
            if tmp.exists():
                tmp.unlink()
            if kind == 'archbench':
                cmd = [sys.executable, '-m', 'archbench.run', '--platform', 'gpu', '--variant', variant,
                       '--point', pid, '--out', str(tmp)]
            else:
                cmd = [sys.executable, str(ROOT / 'tools/r40/v3_gpu_boundaries.py'), '--variant', variant,
                       '--point', pid, '--out', str(tmp)]
            aenv = env
            cache = None
            if args.fresh_cache:
                cache = work / f'{safe}.attempt{attempt}_cache'
                shutil.rmtree(cache, ignore_errors=True)
                cache.mkdir(parents=True)
                aenv = dict(env, TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                            TRITON_CACHE_DIR=str(cache / 'triton'))
            start = now()
            t = time.monotonic()
            COSEEN.clear()
            foreign()
            rc, so, seen, aborted = run_attempt(cmd, ROOT, aenv, args.poll, True,
                                                work / f'{safe}.attempt{attempt}.driver.log', args.timeout)
            if cache is not None:
                shutil.rmtree(cache, ignore_errors=True)
            recs = [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()] if tmp.exists() else []
            rec = recs[-1] if recs else None
            clean = not seen and not aborted and rec is not None
            oom_retry = clean and rec.get('status') == 'runtime-fault' and oom(tmp) and n_oom < 3
            if oom_retry:
                n_oom += 1
                clean = False
            dest = out if clean else contended
            if rec is not None:
                with open(dest, 'a') as fh:
                    fh.write(json.dumps(rec, default=str) + '\n')
            g = dict(point_id=pid, kind=kind, variant=variant, attempt=attempt, start_utc=start, end_utc=now(),
                     run_s=round(time.monotonic() - t, 1), returncode=rc, status=rec and rec.get('status'),
                     waited_s=w['waited_s'], seen_while_waiting=w['seen_while_waiting'],
                     seen_during_run=seen, aborted=aborted or None, clean=clean,
                     kept_in=dest.name if rec is not None else None, poll_s=args.poll, quiet_s=args.quiet,
                     oom_retry=oom_retry or None, owner=ME,
                     same_account_coresident=dict(COSEEN) or None,
                     fresh_compile_cache=bool(cache))
            with open(gate, 'a') as fh:
                fh.write(json.dumps(g) + '\n')
            print(f'{(rec or {}).get("status", "no-record"):16s} {kind:9s} {variant:22s} {pid:58s} '
                  f'attempt {attempt} clean={clean} waited={w["waited_s"]}s run={g["run_s"]}s'
                  + (f' seen={list(seen)}' if seen else ''), flush=True)
            if clean:
                n_pass += bool(rec.get('status') == 'pass')
                break
        else:
            with open(gate, 'a') as fh:
                fh.write(json.dumps(dict(point_id=pid, kind=kind, variant=variant, clean=False,
                                         outcome='not-measured: contended in every attempt',
                                         attempts=args.max_attempts, end_utc=now())) + '\n')
    print(f'=== {now()} stage {kind} {variant} done: {n_pass} new pass -> {out}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--stage', action='append', required=True, help='KIND:VARIANT:POINTS_FILE:OUT')
    ap.add_argument('--quiet', type=float, default=180)
    ap.add_argument('--poll', type=float, default=5)
    ap.add_argument('--max-attempts', type=int, default=30)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--fresh-cache', action='store_true',
                    help='an empty Inductor and Triton cache directory for every attempt')
    args = ap.parse_args(argv)
    for s in args.stage:
        kind, variant, pf, out = s.split(':')
        stage(kind, variant, pf, out, args)
    print(f'=== {now()} campaign done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
