#!/usr/bin/env python3
"""Run a registered board batch with the frozen Stage 1 sampling rule.

    board_batch.py --manifest spec/r41/stage1/experiments/board-anchors.json --batch anchors \
        --root SRC --out DIR --python PY [--var NAME=VALUE ...] [--dry-run] [--retry-failed]

The batch lives in an experiment manifest (tools/performance_harness/
experiment.py runs this command as an `fpga` step). The manifest registers,
before any measurement:

- `measurement`: a verbatim copy of spec/r41/stage1/plan.json `measurement`.
  A copy that differs from the frozen plan refuses the batch;
- `board`: image, sha256, UUID, qualified clock evidence, clock, backend and
  backend arguments, memory map and transport limits;
- `board_batches[NAME]`: the configurations (point id, frozen definition
  sha256, complete knob set), groups and the sampling mode.

Sampling modes (plan.json `measurement.long_device_tasks`):

- `device-rule`: each configuration of a group first runs a qualification
  record (no warm-up, `qualification_samples` timed calls). When every
  configuration of the group qualifies at <= `short_threshold_seconds`, each
  runs again with `short_timed_samples` timed calls and those are the formal
  records; otherwise the qualification record is (`long_timed_samples`).
- `anchor-stability`: `anchor_stability_samples` independent processes per
  configuration, interleaved round by round, each one timed call.
- `fixed`: one process per configuration with the registered warm-up/timed.

Every run is its own ArchBench process with only its configuration's knobs
(ambient TNCA_* variables are removed). Each record must pass numerically
without fallback and must carry the declared point definition, image,
qualified clock, backend arguments, memory window, product software tree,
knobs and sample counts; otherwise that run fails and is preserved. Runs
resume exactly; a failed run is rerun only with --retry-failed, its raw
output kept. `formal.jsonl` holds the formal records; `batch.json` the
summary. Device seconds are serial-grid cycles over the qualified clock.
--dry-run lists the runs and estimates board time from reference records.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment  # noqa: E402

MODES = ('device-rule', 'anchor-stability', 'fixed')
CLOCK_KNOB = 'TNCA_OPERATING_CLOCK_EVIDENCE'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    return experiment.digest(Path(path))


class Refused(ValueError):
    """The registration itself is not runnable."""


def expander(root, out, python, variables=None):
    def expand(value):
        if isinstance(value, str):
            for name, text in (variables or {}).items():
                value = value.replace('{'+name+'}', text)
            return (value.replace('{root}', str(root)).replace('{out}', str(out))
                    .replace('{python}', str(python)).replace('{home}', str(Path.home())))
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    return expand


def resolve(root, name):
    path = Path(name)
    return path if path.is_absolute() else root/path


def load(manifest, name, root, out, python, variables=None):
    """The verified registration: plan sampling, board, configurations."""
    spec = json.loads(Path(manifest).read_text())
    expand = expander(root, out, python, variables)
    plan_path = resolve(root, spec.get('plan', 'spec/r41/stage1/plan.json'))
    if spec.get('plan_sha256') != sha_file(plan_path):
        raise Refused('manifest is not registered against the frozen plan identity')
    plan = json.loads(plan_path.read_text())
    if spec.get('measurement') != plan['measurement']:
        raise Refused('registered sampling rules differ from the frozen plan measurement section')
    batches = spec.get('board_batches') or {}
    if name not in batches:
        raise Refused(f'no registered board batch {name!r}')
    batch = batches[name]
    if batch.get('sampling') not in MODES:
        raise Refused(f'sampling must be one of {MODES}')
    if batch['sampling'] == 'fixed':
        fixed = batch.get('fixed') or {}
        if type(fixed.get('warmup')) is not int or type(fixed.get('timed')) is not int or fixed['timed'] < 1:
            raise Refused('fixed sampling needs integer warmup and positive timed counts')
    board = expand(spec.get('board') or {})
    for field in ('image', 'image_sha256', 'image_uuid', 'clock_evidence', 'clock_hz', 'backend',
                  'backend_args', 'profile', 'transport', 'memory_map', 'device'):
        if board.get(field) in (None, '', {}):
            raise Refused(f'board registration lacks {field}')
    if board['backend_args'].get('xclbin') != board['image']:
        raise Refused('backend arguments name another image file')
    definitions = {}
    for suite in spec.get('suites') or []:
        path = resolve(root, suite['path'])
        if sha_file(path) != suite['sha256']:
            raise Refused(f'frozen suite changed: {suite["path"]}')
        for point in json.loads(path.read_text())['points']:
            if definitions.setdefault(point['id'], point['sha256']) != point['sha256']:
                raise Refused(f'conflicting frozen definitions for {point["id"]}')
    configs = []
    for expansion in batch.get('expand') or []:
        path = resolve(root, expansion['suite'])
        if sha_file(path) != expansion['sha256']:
            raise Refused(f'frozen suite changed: {expansion["suite"]}')
        for point in json.loads(path.read_text())['points']:
            configs.append(dict(id=f'{expansion.get("prefix", "")}{point["id"]}', point=point['id'],
                                point_sha256=point['sha256'], knobs=dict(expansion['knobs'])))
    configs += [dict(c) for c in batch.get('configurations') or []]
    if not configs:
        raise Refused('a board batch needs configurations')
    ids = [c['id'] for c in configs]
    if len(set(ids)) != len(ids):
        raise Refused('configuration ids must be unique')
    for c in configs:
        c['knobs'] = expand(c['knobs'])
        if definitions.get(c['point']) != c['point_sha256']:
            raise Refused(f'{c["point"]}: definition sha256 is not the frozen suite definition')
        if not isinstance(c['knobs'], dict) or any(not k.startswith('TNCA_') or not isinstance(v, str)
                                                   for k, v in c['knobs'].items()):
            raise Refused(f'{c["id"]}: knobs must be explicit TNCA_* strings')
        if CLOCK_KNOB in c['knobs']:
            raise Refused('the clock evidence comes from the board registration, not a knob')
        if any('{' in v for v in c['knobs'].values()):
            raise Refused(f'{c["id"]}: unexpanded registration variable in its knobs (pass --var)')
    coverage = batch.get('coverage')
    if coverage:
        path = resolve(root, coverage['suite'])
        if sha_file(path) != coverage['sha256']:
            raise Refused('coverage suite changed')
        required = {p['id'] for p in json.loads(path.read_text())['points']}
        present = {c['point'] for c in configs}
        if present != required:
            raise Refused(f'batch does not cover its registered suite exactly: missing {sorted(required-present)[:5]}, '
                          f'extra {sorted(present-required)[:5]}')
    groups = batch.get('groups') or [[c['id']] for c in configs]
    flat = [i for g in groups for i in g]
    if sorted(flat) != sorted(ids):
        raise Refused('groups must partition the configurations exactly')
    return dict(spec=spec, plan=plan, plan_path=plan_path, board=board, batch=batch,
                configs={c['id']: c for c in configs}, order=ids, groups=groups)


def schedule(reg):
    """The first-stage runs: (config id, tag, warmup, timed)."""
    m = reg['plan']['measurement']['long_device_tasks']
    mode = reg['batch']['sampling']
    if mode == 'anchor-stability':
        rounds = m['anchor_stability_samples']
        order = reg['order']
        # Interleave and rotate the order each round, so no configuration is
        # always measured first after another one.
        return [(order[(i+r) % len(order)], f's{r+1}', 0, 1) for r in range(rounds) for i in range(len(order))]
    if mode == 'fixed':
        fixed = reg['batch']['fixed']
        return [(cid, 'fixed', fixed['warmup'], fixed['timed']) for cid in reg['order']]
    return [(cid, 'q', 0, m['qualification_samples']) for group in reg['groups'] for cid in group]


class Batch:
    def __init__(self, reg, *, root, out, python, retry_failed=False, runner=None, tree=None, probe=None):
        self.reg, self.root, self.out, self.python = reg, root, out, python
        self.retry_failed = retry_failed
        self.runner = runner or [str(python), '-u', '-m', 'archbench.run']
        self.tree = tree
        self.probe = probe
        self.state_path = out/'state.json'
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else dict(schema='tnca-board-batch-state/1', runs={}))

    def save(self):
        self.state['observed_utc'] = utc()
        experiment.publish(self.state_path, self.state)

    def env(self):
        board = self.reg['board']
        env = {k: v for k, v in os.environ.items() if not k.startswith('TNCA_')}
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONFAULTHANDLER='1',
                   PYTHONPATH=os.pathsep.join(str(self.root/d) for d in ('', 'python', 'compiler/python', 'tools/asm')),
                   TNCA_OPERATING_CLOCK_EVIDENCE=str(board['clock_evidence']),
                   TNCA_DEADLINE=str(board['transport']['deadline_cycles']),
                   TNCA_HOST_TIMEOUT=str(board['transport']['host_timeout_seconds']))
        return env

    def command(self, cfg, stem, warmup, timed):
        board = self.reg['board']
        raw = self.out/'runs'/(stem+'.jsonl')
        cmd = [*self.runner, '--platform', 'tnca', '--backend', board['backend'],
               '--backend-args', json.dumps(board['backend_args']), '--profile', board['profile'],
               '--point', cfg['point'], '--jobs', '1', '--seed', '0', '--warmup', str(warmup),
               '--timed', str(timed), '--timeout', str(board['transport']['host_timeout_seconds']),
               '--bundle-dir', str(self.out/'bundles'/stem), '--out', str(raw)]
        for k, v in sorted(cfg['knobs'].items()):
            cmd += ['--set', f'{k}={v}']
        return raw, cmd

    def check(self, cfg, row, warmup, timed):
        board = self.reg['board']
        plat, prov = row.get('platform') or {}, row.get('provenance') or {}
        problems = []
        corr = row.get('correctness') or {}
        if row.get('status') != 'pass' or row.get('fallbacks'):
            problems.append(f"status {row.get('status')} fallbacks {row.get('fallbacks')} {row.get('problems')}")
        if corr.get('max_rel_err') is None or corr.get('tol') is None or corr['max_rel_err'] > corr['tol']:
            problems.append('numerical comparison missing or out of tolerance')
        if row.get('point_id') != cfg['point'] or row.get('point_sha256') != cfg['point_sha256']:
            problems.append('point identity')
        if row.get('evidence_level') != 'measured-board':
            problems.append(f"evidence level {row.get('evidence_level')}")
        if plat.get('backend') != board['backend'] or plat.get('backend_args') != board['backend_args']:
            problems.append('backend or backend arguments')
        if plat.get('image_sha256') != board['image_sha256']:
            problems.append(f"image {plat.get('image_sha256')}")
        if plat.get('clock_hz') != board['clock_hz'] or prov.get('clock_hz') != board['clock_hz']:
            problems.append(f"clock {plat.get('clock_hz')}/{prov.get('clock_hz')}")
        if plat.get('operating_clock_evidence_sha256') != self.state.get('clock_evidence_sha256'):
            problems.append('qualified clock evidence identity')
        if self.tree and plat.get('software_tree') != self.tree:
            problems.append(f"software tree {plat.get('software_tree')}")
        expected = dict(cfg['knobs'], **{CLOCK_KNOB: str(board['clock_evidence'])})
        if plat.get('knobs') != expected:
            problems.append('knobs differ from the registered configuration')
        if (plat.get('inductor') or {}).get('tensor_descriptors') is not True:
            problems.append('tensor descriptors disabled')
        cores = int(cfg['knobs'].get('TNCA_ACTIVE_CORES', '1'))
        if (prov.get('launch_map') or {}).get('cores') != list(range(cores)):
            problems.append(f"launch map cores {(prov.get('launch_map') or {}).get('cores')}")
        window = (row.get('native_memory') or {}).get('window') or {}
        memory_map = board.get('memory_map') or {}
        for key, field in (('window_base', 'base'), ('window_bytes', 'size'), ('local_limit', 'local_limit')):
            if key in memory_map and window.get(field) != memory_map[key]:
                problems.append(f'memory window {field} {window.get(field)} != {memory_map[key]}')
        t = (row.get('timings') or {}).get('device-cycles-serial-grid') or {}
        if t.get('warmup') != warmup or len(t.get('samples') or []) != timed:
            problems.append('sampling')
        return problems

    def run_one(self, cfg, tag, warmup, timed):
        stem = f"{cfg['id'].replace('/', '__')}-{tag}"
        raw, cmd = self.command(cfg, stem, warmup, timed)
        done = self.state['runs'].get(stem)
        if done and done['ok']:
            if sha_file(raw) != done['raw_sha256']:
                raise Refused(f'completed board record changed: {raw}')
            return done
        if done and not self.retry_failed:
            return done
        self.check_card()
        (self.out/'runs').mkdir(parents=True, exist_ok=True)
        (self.out/'logs').mkdir(exist_ok=True)
        if raw.exists():
            # A failed earlier attempt stays in its preserved copy.
            raw.rename(raw.with_suffix(f'.failed{len(list(raw.parent.glob(raw.stem+".failed*")))}'))
        log = self.out/'logs'/(stem+'.log')
        self.state.update(status='running', current=stem)
        self.save()
        started, clock = utc(), time.monotonic()
        timeout = self.reg['plan']['budgets']['board_point_timeout_seconds']
        problems = []
        with log.open('w') as fh:
            proc = subprocess.Popen(cmd, cwd=self.root, env=self.env(), stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                problems.append(f'board point budget {timeout}s exhausted')
                code = None
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()] if raw.exists() else []
        if code:
            problems.append(f'exit {code}')
        if len(rows) != 1:
            problems.append(f'{len(rows)} records')
        else:
            problems += self.check(cfg, rows[0], warmup, timed)
        if self.tree and experiment.software_tree(self.root) != self.tree:
            problems.append('product software tree changed during the batch')
        entry = dict(ok=not problems, problems=problems, config=cfg['id'], point=cfg['point'], tag=tag,
                     raw=str(raw), log=str(log), command=cmd, started=started, ended=utc(),
                     wall_seconds=round(time.monotonic()-clock, 3), warmup=warmup, timed=timed)
        if raw.exists():
            entry['raw_sha256'] = sha_file(raw)
        if rows and not problems:
            t = rows[0]['timings']
            entry['seconds'] = t['device-cycles-serial-grid']['median']/self.reg['board']['clock_hz']
            entry['serial_grid_samples'] = t['device-cycles-serial-grid']['samples']
            entry['graph_makespan_samples'] = (t.get('device-graph-makespan') or {}).get('samples')
        self.state['runs'][stem] = entry
        self.save()
        print(f"{utc()} {'OK  ' if entry['ok'] else 'FAIL'} {stem} {entry.get('seconds', '')} {problems}", flush=True)
        if any('software tree changed' in p for p in problems):
            raise Refused('product software tree changed during the batch')
        return entry

    def check_card(self):
        """Read-only: the loaded image, clock counter and card use before each run."""
        if self.probe is None:
            return
        board = self.reg['board']
        observed = self.probe(board)
        identity = dict(image_uuid=experiment.canonical_uuid(board['image_uuid']), clock_hz=board['clock_hz'],
                        image_sha256=board['image_sha256'])
        reason = experiment.board_refusal(identity, observed)
        self.state.setdefault('probes', []).append(dict(observed, refusal=reason))
        self.save()
        if reason:
            raise Refused(reason)

    def bind(self):
        """Bind this batch to its registered image, clock evidence and software tree.

        Every run is checked against the clock evidence digest recorded here, so a
        driver that calls run_one directly (the endurance run) must call bind first;
        without it the digest is None and every record is rejected as
        'qualified clock evidence identity'.
        """
        self.out.mkdir(parents=True, exist_ok=True)
        board = self.reg['board']
        if sha_file(board['image']) != board['image_sha256']:
            raise Refused('the registered image file is not the declared image')
        clock = json.loads(Path(board['clock_evidence']).read_text())
        if clock.get('image_sha256') != board['image_sha256'] or clock.get('clock_hz') != board['clock_hz']:
            raise Refused('qualified clock evidence is not bound to this image and clock')
        self.state.update(clock_evidence_sha256=sha_file(board['clock_evidence']), software_tree=self.tree,
                          image_sha256=board['image_sha256'], clock_hz=board['clock_hz'])

    def execute(self):
        board = self.reg['board']
        self.bind()
        self.check_card()
        m = self.reg['plan']['measurement']['long_device_tasks']
        mode = self.reg['batch']['sampling']
        formal = {}
        if mode == 'device-rule':
            for group in self.reg['groups']:
                q = {cid: self.run_one(self.reg['configs'][cid], 'q', 0, m['qualification_samples']) for cid in group}
                if not all(r['ok'] for r in q.values()):
                    continue
                if all(r['seconds'] <= m['short_threshold_seconds'] for r in q.values()):
                    for cid in group:
                        formal[cid] = [self.run_one(self.reg['configs'][cid], f"f{m['short_timed_samples']}",
                                                    0, m['short_timed_samples'])]
                elif m['long_timed_samples'] == m['qualification_samples']:
                    formal.update({cid: [q[cid]] for cid in group})
                else:
                    for cid in group:
                        formal[cid] = [self.run_one(self.reg['configs'][cid], f"l{m['long_timed_samples']}",
                                                    0, m['long_timed_samples'])]
        else:
            for cid, tag, warmup, timed in schedule(self.reg):
                formal.setdefault(cid, []).append(self.run_one(self.reg['configs'][cid], tag, warmup, timed))
        stability = {}
        for cid, entries in formal.items():
            graph = [x for e in entries if e['ok'] for x in e.get('graph_makespan_samples') or []]
            serial = [x for e in entries if e['ok'] for x in e.get('serial_grid_samples') or []]
            stats = dict(graph_makespan_samples=graph, serial_grid_samples=serial, processes=len(entries))
            if graph:
                ordered = sorted(graph)
                stats['graph_makespan_relative_spread'] = (ordered[-1]-ordered[0])/ordered[len(ordered)//2]
            stability[cid] = stats
        failed = sorted({e['config'] for e in self.state['runs'].values() if not e['ok']})
        missing = [cid for cid in self.reg['order'] if cid not in formal or not all(r['ok'] for r in formal[cid])]
        with (self.out/'formal.tmp').open('w') as fh:
            for cid in self.reg['order']:
                for entry in formal.get(cid, []):
                    if entry['ok']:
                        fh.write(Path(entry['raw']).read_text())
        (self.out/'formal.tmp').replace(self.out/'formal.jsonl')
        summary = dict(schema='tnca-board-batch/1', batch=self.reg['batch'].get('name'), sampling=mode,
                       measurement=self.reg['plan']['measurement'], plan_sha256=sha_file(self.reg['plan_path']),
                       board=dict(image_sha256=board['image_sha256'], clock_hz=board['clock_hz'],
                                  clock_evidence_sha256=self.state['clock_evidence_sha256'],
                                  backend=board['backend'], backend_args=board['backend_args'],
                                  memory_map=board.get('memory_map')),
                       software_tree=self.tree, configurations=len(self.reg['order']),
                       formal={cid: [e['raw'] for e in formal.get(cid, [])] for cid in self.reg['order']},
                       samples=stability, failed_configurations=failed, missing_formal=missing,
                       board_wall_seconds=round(sum(e['wall_seconds'] for e in self.state['runs'].values()), 1),
                       formal_records=str(self.out/'formal.jsonl'),
                       status='complete' if not failed and not missing else 'failed-preserved',
                       performance_judgement='not-measured')
        experiment.publish(self.out/'batch.json', summary)
        self.state.update(status=summary['status'], current=None)
        self.save()
        return summary


def reference_estimates(root, sources):
    """{(point, knobs json): (process seconds, device seconds)} from records."""
    table = {}
    def add(point, knobs, process, device):
        knobs = {k: v for k, v in sorted((knobs or {}).items()) if k != CLOCK_KNOB}
        table.setdefault((point, json.dumps(knobs, sort_keys=True)), (process, device))
        table.setdefault((point, None), (process, device))
    for source in sources or []:
        if source.get('records'):
            for line in resolve(root, source['records']).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                t = (r.get('timings') or {}).get('device-cycles-serial-grid') or {}
                clock = t.get('clock_hz') or (r.get('platform') or {}).get('clock_hz') or 9e7
                add(r['point_id'], (r.get('platform') or {}).get('knobs'), (r.get('host') or {}).get('process_s'),
                    (t.get('median') or 0)/clock)
        if source.get('campaign'):
            campaign = json.loads(resolve(root, source['campaign']).read_text())
            state = json.loads(resolve(root, source['state']).read_text())
            for stem, e in state['runs'].items():
                cfg = campaign['configs'].get(stem.split('-')[0])
                if not cfg or not e.get('ok') or not stem.endswith('-q'):
                    continue
                seconds = (datetime.datetime.fromisoformat(e['ended']) -
                           datetime.datetime.fromisoformat(e['started'])).total_seconds()
                add(cfg['point']['id'], cfg['knobs'], seconds, e.get('seconds'))
    return table


def estimate(reg, root, default_seconds=20.0):
    """Expected board seconds of the registered runs from reference records."""
    table = reference_estimates(root, reg['spec'].get('estimates'))
    m = reg['plan']['measurement']['long_device_tasks']
    rows, total = [], 0.0
    def lookup(cfg):
        exact = table.get((cfg['point'], json.dumps(dict(sorted(cfg['knobs'].items())), sort_keys=True)))
        hit = exact or table.get((cfg['point'], None))
        source = 'same-configuration' if exact else 'same-point-other-configuration' if hit else 'default'
        process, device = hit if hit else (default_seconds, 0.0)
        return float(process or default_seconds), float(device or 0.0), source
    mode = reg['batch']['sampling']
    if mode == 'device-rule':
        for group in reg['groups']:
            found = {cid: lookup(reg['configs'][cid]) for cid in group}
            short = all(device <= m['short_threshold_seconds'] for _p, device, _s in found.values())
            for cid, (process, device, source) in found.items():
                seconds = process + ((process + (m['short_timed_samples']-1)*device) if short else 0.0)
                rows.append(dict(config=cid, runs=2 if short else 1, seconds=round(seconds, 1), source=source,
                                 reference_device_seconds=round(device, 4)))
                total += seconds
    else:
        for cid, tag, warmup, timed in schedule(reg):
            process, device, source = lookup(reg['configs'][cid])
            seconds = process + (warmup+timed-1)*device
            rows.append(dict(config=cid, tag=tag, seconds=round(seconds, 1), source=source))
            total += seconds
    return dict(expected_board_hours=round(total/3600, 3), runs=rows,
                method='per-process wall seconds of the nearest reference record (R40 fe1b board runs); '
                       'short configurations add one formal process; extra timed calls add their device time')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--python', type=Path, default=Path(sys.executable))
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--var', action='append', default=[], help='NAME=VALUE substituted for {NAME} in the registration')
    ap.add_argument('--dry-run', action='store_true', help='list the runs and the expected board time; run nothing')
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    variables = dict(kv.split('=', 1) for kv in args.var)
    reg = load(args.manifest, args.batch, root, out, args.python, variables)
    reg['batch'].setdefault('name', args.batch)
    if args.dry_run:
        print(json.dumps(dict(batch=args.batch, sampling=reg['batch']['sampling'],
                              configurations=len(reg['order']), first_stage=schedule(reg),
                              estimate=estimate(reg, root)), indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    tree = experiment.software_tree(root)
    if tree is None:
        raise Refused('board batches require the product software tree identity')
    batch = Batch(reg, root=root, out=out, python=args.python, retry_failed=args.retry_failed, tree=tree,
                  probe=experiment.sysfs_probe)
    summary = batch.execute()
    print(json.dumps({k: summary[k] for k in ('status', 'configurations', 'failed_configurations',
                                               'missing_formal', 'board_wall_seconds')}, indent=2))
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
