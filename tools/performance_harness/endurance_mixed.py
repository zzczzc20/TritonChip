#!/usr/bin/env python3
"""Two-hour continuous mixed board run with session and resource-growth records (R41 S2-01 item 8).

    endurance_mixed.py --manifest M --batch mix --root R --out O --python P [--duration 7200] [--fp8-every 3]
    endurance_mixed.py wrap --rusage FILE -- CMD...      # internal: run CMD, record its wait4 rusage

Rounds until --duration seconds have passed (a round started before the end
completes). Each round runs every configuration of the registered board batch
once, in a seeded order that changes per round, through board_batch.Batch
(one ArchBench process = one runtime session, the batch's own numeric,
identity and knob checks, the read-only card probe before every run). Every
--fp8-every rounds one more session runs fp8_cases.py --quick --group
scaled_mm on the same board (FP8 MMA + X8 transposed load).

Per session: start/end, wall, pass/problems, device cycles, and the child's
own peak RSS from wait4 (the `wrap` shim around the session command). Every
--sample-seconds, a host row: MemAvailable, Committed_AS, this process's RSS
and open fds, and the read-only card probe (UUID, clock counter, contexts).

The run stops at once on any card change (another image, another clock, or
a busy card before a session): board_batch refuses, and the record says so.

endurance.json: sessions, pass count, hours, per-configuration peak-RSS and
cycle slopes (least squares over the run, per hour, relative to the first
round), host MemAvailable slope, and result/residual/conclusion/next_step.
Growth above GROWTH_LIMIT (10 % over the run) in a configuration's peak RSS
or the MemAvailable drop is flagged as a residual, not hidden.
"""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GROWTH_LIMIT = 0.10


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def wrap(rusage, cmd):
    proc = subprocess.Popen(cmd)
    _, status, ru = os.wait4(proc.pid, 0)
    code = os.waitstatus_to_exitcode(status)
    Path(rusage).write_text(json.dumps(dict(returncode=code, max_rss_kib=ru.ru_maxrss, user_s=ru.ru_utime,
                                            sys_s=ru.ru_stime, minflt=ru.ru_minflt, majflt=ru.ru_majflt)))
    return code


def meminfo(path='/proc/meminfo'):
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            k, _, v = line.partition(':')
            if k in ('MemAvailable', 'Committed_AS', 'MemFree'):
                out[k + '_kib'] = int(v.split()[0])
    except OSError:
        pass
    return out


def self_usage():
    row = dict(rss_kib=None, fds=None)
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                row['rss_kib'] = int(line.split()[1])
        row['fds'] = len(os.listdir('/proc/self/fd'))
    except OSError:
        pass
    return row


def slope(points):
    """Least-squares slope of (x, y) pairs, or None."""
    if len(points) < 3:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((x - mx) ** 2 for x, _ in points)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in points) / den


def growth(series, hours):
    """Relative change over the run implied by the slope, against the first value."""
    pts = [(t, v) for t, v in series if v is not None]
    s = slope(pts)
    if s is None or not pts or not pts[0][1]:
        return None
    return s * hours / pts[0][1]


class Sampler(threading.Thread):
    def __init__(self, path, probe, board, every):
        super().__init__(daemon=True)
        self.path, self.probe, self.board, self.every = path, probe, board, every
        self.stop_event = threading.Event()
        self.rows = []

    def run(self):
        while not self.stop_event.is_set():
            row = dict(utc=utc(), t=time.monotonic(), **meminfo(), **self_usage())
            if self.probe:
                try:
                    p = self.probe(self.board)
                    row.update(loaded_uuid=p.get('loaded_uuid'), clock_counter_mhz=p.get('clock_counter_mhz'),
                               contexts=p.get('contexts'))
                except Exception as error:      # observation only
                    row['probe_error'] = repr(error)
            self.rows.append(row)
            with self.path.open('a') as fh:
                fh.write(json.dumps(row) + '\n')
            self.stop_event.wait(self.every)


def run(args, probe=None, batch_cls=None, fp8_runner=None):
    import board_batch
    import experiment
    root, out = args.root.resolve(), args.out.resolve()
    reg = board_batch.load(args.manifest, args.batch, root, out, args.python)
    reg['batch'].setdefault('name', args.batch)
    out.mkdir(parents=True, exist_ok=True)
    tree = experiment.software_tree(root)
    if tree is None and probe is experiment.sysfs_probe:
        raise board_batch.Refused('board runs require the product software tree identity')
    shim = [str(args.python), '-u', str(HERE / 'endurance_mixed.py'), 'wrap', '--rusage']
    batch = (batch_cls or board_batch.Batch)(reg, root=root, out=out, python=args.python, tree=tree, probe=probe)
    batch.bind()
    board = reg['board']
    sessions_path = out / 'sessions.jsonl'
    sampler = Sampler(out / 'host.jsonl', probe, board, args.sample_seconds)
    sampler.start()
    rng = random.Random(args.seed)
    started, t0 = utc(), time.monotonic()
    sessions, stop_reason, rnd = [], None, 0
    try:
        while time.monotonic() - t0 < args.duration and stop_reason is None:
            rnd += 1
            order = list(reg['order'])
            rng.shuffle(order)
            plan = [('point', cid) for cid in order]
            if args.fp8_every and rnd % args.fp8_every == 0:
                plan.insert(rng.randrange(len(plan) + 1), ('fp8', 'fp8_cases/quick/scaled_mm'))
            for kind, cid in plan:
                tag = f'r{rnd:03d}'
                rusage = out / 'rusage' / f"{cid.replace('/', '__')}-{tag}.json"
                rusage.parent.mkdir(parents=True, exist_ok=True)
                s0 = time.monotonic()
                row = dict(round=rnd, kind=kind, config=cid, tag=tag, started_utc=utc(),
                           t_start_s=round(s0 - t0, 3))
                try:
                    if kind == 'point':
                        batch.runner = shim + [str(rusage), '--'] + [str(args.python), '-u', '-m', 'archbench.run']
                        entry = batch.run_one(reg['configs'][cid], tag, 0, 1)
                        row.update(ok=entry['ok'], problems=entry['problems'], raw=entry.get('raw'),
                                   device_seconds=entry.get('seconds'),
                                   serial_grid_cycles=(entry.get('serial_grid_samples') or [None])[0])
                    else:
                        batch.check_card()
                        row.update((fp8_runner or run_fp8)(args, root, out / 'fp8' / tag, shim + [str(rusage), '--']))
                except board_batch.Refused as error:
                    row.update(ok=False, problems=[f'refused: {error}'])
                    stop_reason = str(error)
                if rusage.is_file():
                    ru = json.loads(rusage.read_text())
                    row.update(max_rss_kib=ru['max_rss_kib'], returncode=ru['returncode'])
                row.update(ended_utc=utc(), wall_s=round(time.monotonic() - s0, 3), host=meminfo())
                sessions.append(row)
                with sessions_path.open('a') as fh:
                    fh.write(json.dumps(row) + '\n')
                print(f"{row['ended_utc']} {'OK  ' if row['ok'] else 'FAIL'} {tag} {cid} "
                      f"{row.get('max_rss_kib')} KiB {row['problems'] or ''}", flush=True)
                if stop_reason:
                    break
    finally:
        sampler.stop_event.set()
        sampler.join(timeout=5)
    hours = (time.monotonic() - t0) / 3600
    per = {}
    for s in sessions:
        per.setdefault(s['config'], []).append(s)
    configs = {}
    for cid, rows in per.items():
        configs[cid] = dict(
            sessions=len(rows), passed=sum(1 for r in rows if r['ok']),
            max_rss_kib=[r.get('max_rss_kib') for r in rows],
            rss_growth_over_run=growth([(r['t_start_s'] / 3600, r.get('max_rss_kib')) for r in rows], hours),
            cycles=[r.get('serial_grid_cycles') for r in rows],
            cycles_growth_over_run=growth([(r['t_start_s'] / 3600, r.get('serial_grid_cycles')) for r in rows],
                                          hours))
    host = sampler.rows
    avail = [(r['t'] - t0, r.get('MemAvailable_kib')) for r in host]
    mem_drop = growth([(t / 3600, v) for t, v in avail], hours)
    residual = []
    failed = [s for s in sessions if not s['ok']]
    if failed:
        residual.append(f'{len(failed)} of {len(sessions)} sessions failed')
    if stop_reason:
        residual.append(f'stopped: {stop_reason}')
    if hours * 3600 < args.duration:
        residual.append(f'ran {hours:.2f} h, less than {args.duration / 3600:.2f} h')
    for cid, c in configs.items():
        if c['rss_growth_over_run'] is not None and c['rss_growth_over_run'] > GROWTH_LIMIT:
            residual.append(f'{cid}: peak RSS grows {c["rss_growth_over_run"]:.1%} over the run')
    if mem_drop is not None and mem_drop < -GROWTH_LIMIT:
        residual.append(f'host MemAvailable falls {-mem_drop:.1%} over the run (shared host; check host.jsonl)')
    ok = not failed and not stop_reason and hours * 3600 >= args.duration
    record = dict(schema='tnca-r41s2-endurance/1', started_utc=started, ended_utc=utc(), hours=round(hours, 4),
                  duration_target_s=args.duration, rounds=rnd, sessions=len(sessions),
                  passed=sum(1 for s in sessions if s['ok']), seed=args.seed, fp8_every=args.fp8_every,
                  board=dict(image_sha256=board['image_sha256'], image_uuid=board['image_uuid'],
                             clock_hz=board['clock_hz'], backend_args=board['backend_args']),
                  software_tree=tree, configurations=configs, host_samples=len(host),
                  host_mem_available_change_over_run=mem_drop, growth_limit=GROWTH_LIMIT,
                  sessions_record=str(sessions_path), host_record=str(out / 'host.jsonl'),
                  result='pass' if ok and not residual else ('pass-with-residual' if ok else 'fail'),
                  residual=residual or 'none',
                  conclusion=(f'{len(sessions)} sessions over {hours:.2f} h on the '
                              f'{getattr(args, "image_label", "run-1")} image, '
                              f'{sum(1 for s in sessions if s["ok"])} passed'),
                  next_step='fold into the S2-01 delivery record' if ok else 'triage from sessions.jsonl')
    experiment.publish(out / 'endurance.json', record)
    return record


def run_fp8(args, root, out, shim):
    """One fp8_cases.py --quick --group scaled_mm session on the board (env from the step).

    fp8_cases.py creates its output directory itself and refuses an existing one, so
    the log lives beside it, not inside it.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = shim + [str(args.python), '-u', str(root / 'tools/performance_harness/fp8_cases.py'),
                  '--out', str(out), '--quick', '--group', 'scaled_mm']
    with out.with_name(out.name + '.log').open('w') as fh:
        code = subprocess.call(cmd, cwd=root, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
    rep = out / 'report.json'
    problems = [] if code == 0 else [f'exit {code}']
    cases = []
    if rep.is_file():
        data = json.loads(rep.read_text())
        cases = data.get('rows') or data.get('cases') or []     # fp8_cases writes 'rows'
        bad = [c.get('case') for c in cases if c.get('status') != 'pass']
        if bad or not cases:
            problems.append(f'{len(bad)} of {len(cases)} cases not pass')
    else:
        problems.append('no report')
    return dict(ok=not problems, problems=problems, raw=str(rep), cases=len(cases))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'wrap':
        argv = sys.argv[2:]
        if len(argv) < 4 or argv[0] != '--rusage' or argv[2] != '--':
            raise SystemExit('usage: endurance_mixed.py wrap --rusage FILE -- CMD...')
        return wrap(argv[1], argv[3:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--batch', default='mix')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--python', type=Path, default=Path(sys.executable))
    ap.add_argument('--duration', type=float, default=7200)
    ap.add_argument('--fp8-every', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--sample-seconds', type=float, default=60)
    ap.add_argument('--image-label', choices=('run-1', 'run-2'), default='run-1')
    args = ap.parse_args()
    import experiment
    record = run(args, probe=experiment.sysfs_probe)
    print(json.dumps({k: record[k] for k in ('result', 'hours', 'sessions', 'passed', 'residual')}, indent=1))
    return 0 if record['result'].startswith('pass') else 1


if __name__ == '__main__':
    raise SystemExit(main())
