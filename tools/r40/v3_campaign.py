#!/usr/bin/env python3
"""The R40 v3 board performance campaign: plan, collect and judge.

    v3_campaign.py plan  --out DIR --candidate R40-03.wide-blocks=64 --candidate R40-03.gemm-tile=16,64,16 \
                         --compile-preflight evidence/r40/v3/compile_preflight/report.json
    v3_campaign.py run   --out DIR --xclbin X --operating-clock-evidence C
    v3_campaign.py judge --out DIR [--evidence]

Everything frozen comes from doc/r40_v3/plan.json (tools/r40/contract_v3.py
verifies it): experiments, subsets, baselines, control/candidate values,
targets and the sampling rule (doc/r40_tasks_and_acceptance_v3.md 变更五).

Device-cycle experiments: each (point, complete knob set) configuration runs
in its own ArchBench process -- the cold first call recorded apart, no extra
warm-up, one timed call (qualification). When both configurations of a point
in an experiment qualify at <= 1 s, each runs again with three timed calls and
those are the formal records; otherwise the qualification record is. One
block; a point's configurations run back to back in a seeded random order; a
configuration two experiments need is measured once. Host-latency experiments
keep v2 sampling (5 warm-up, 30 timed, three blocks) and run through
tools/r40/v3_latency_child.py, which times the protocol's request and
resident-call latencies.

The judgement is tools/r40/ablate.py's accepted comparison (every refusal
rule applies) with the v3 rule: perf-pass when the 95 % paired bootstrap lower
bound of S_geo is above 1, the v2 threshold reported as a target; R40-01.
device-time keeps its capability non-regression rule. Raw records, logs and
compiler bundles stay under DIR.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm'),
                str(ROOT/'tools/r40')]
import ablate
import contract_v3

PLAN = ROOT/'doc/r40_v3/plan.json'
# Cheap, gate-closing experiments first; the long 1-core arms of the large
# models last, so a long tail never delays the other gates.
ORDER = ('R40-02.batched', 'R40-02.resident', 'R40-03.wide-blocks', 'R40-03.gemm-tile',
         'R40-04.memory-service', 'R40-06.ports', 'R40-06.streaming-cores', 'R40-07.pipeline',
         'R40-01.device-time', 'R40-09.model', 'R40-05.three-cores', 'R40-05.two-cores',
         'R40-08.prefill')
HOST = ('request-latency', 'resident-call-latency')
# configure() echoes these compile knobs into platform.inductor, which
# ablate.config_of compares. Only the echo of the varied knob is exempted,
# and judge() checks it equals the knob on every record.
MIRRORS = {'TNCA_MAX_BLOCK': ('max_block', int), 'TNCA_GEMM_TILE': ('gemm_tile', lambda v: [int(x) for x in v.split(',')])}
# A mechanism's documented prerequisite, held constant on both sides so the
# comparison stays single-factor: python/tnca/online_attention.py refuses the
# fused SDPA template without packed-window unpack for real DMA loads.
# On the native device the fused template for the frozen D64 long-prefill
# shapes also needs window reuse to fit the 64 vector registers per role
# context (v3 compile preflight, native-stub path); the decomposed control
# then gets the same window reuse, which can only make the control faster.
PREREQUISITES = {'R40-08.prefill': dict(knobs={'TNCA_PACKED_WINDOW_UNPACK': '1', 'TNCA_WINDOW_REUSE': '1'},
                 reason='python/tnca/online_attention.py: online SDPA requires TNCA_PACKED_WINDOW_UNPACK=1 for real DMA loads; '
                        'native-path compile preflight: the fused D64 template exhausts the vector registers without TNCA_WINDOW_REUSE=1')}
SHORT_S = 1.0
DEADLINE = '200000000000'       # cycles; a watchdog, declared identically for every configuration
HOST_TIMEOUT = '28800'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan():
    contract_v3.verify()
    return json.loads(PLAN.read_text())


def config_key(point_id, knobs, klass):
    raw = json.dumps([point_id, sorted(knobs.items()), klass]).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def experiment_configs(plan, name, candidates):
    exp = plan['experiments'][name]
    knob = exp['vary'][0]
    base = dict(plan['baseline_knobs'][exp['baseline']])
    if name == 'R40-01.device-time':
        base = dict(TNCA_TRANSFER_POLICY='per-kernel', TNCA_ACTIVE_CORES='3')
    base.update(exp.get('baseline_override') or {})
    base.update(PREREQUISITES.get(name, {}).get('knobs', {}))
    candidate = exp['candidate']
    if isinstance(candidate, list):
        if candidates.get(name) not in candidate:
            raise ValueError(f'{name}: candidate must be one of the frozen ordered values {candidate}')
        candidate = candidates[name]
    sides = {}
    for side, value in (('control', exp['control']), ('candidate', candidate)):
        sides[side] = dict(base, **{knob: str(value)})
    return exp, sides


def build(plan, candidates, preflight):
    seed = plan['sample_plan']['bootstrap']['seed']
    rng = random.Random(seed)
    configs, experiments = {}, {}
    for name in ORDER:
        exp, sides = experiment_configs(plan, name, candidates)
        klass = 'host' if exp['timing'] in HOST else 'device'
        points = sorted(json.loads((ROOT/exp['subset']).read_text())['points'], key=lambda p: p['id'])
        rows = []
        for point in points:
            keys = {}
            for side, knobs in sides.items():
                key = config_key(point['id'], knobs, klass)
                keys[side] = key
                configs.setdefault(key, dict(key=key, point=dict(id=point['id'], sha256=point['sha256']),
                                             knobs=knobs, klass=klass, experiments=[]))
                configs[key]['experiments'].append(name)
            order = ['control', 'candidate']
            rng.shuffle(order)
            rows.append(dict(point=point['id'], sha256=point['sha256'], keys=keys, order=order))
        experiments[name] = dict(exp, name=name, klass=klass, points=rows, candidate_value=sides['candidate'][exp['vary'][0]])
    return dict(schema='tnca-r40-v3-campaign/1', planned=utc(), contract_sha256=contract_v3.LOCK_SHA256,
                plan_sha256=sha_file(PLAN), harness_sha256=sha_file(__file__),
                latency_child_sha256=sha_file(ROOT/'tools/r40/v3_latency_child.py'),
                candidates=candidates, compile_preflight=preflight, order=list(ORDER), prerequisites=PREREQUISITES,
                transport=dict(deadline_cycles=int(DEADLINE), host_timeout_s=float(HOST_TIMEOUT)),
                experiments=experiments, configs=configs)


# ---------------------------------------------------------------- collection

class Runner:
    def __init__(self, out, campaign, args):
        self.out, self.c, self.a = out, campaign, args
        self.state_path = out/'state.json'
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else dict(status='planned', runs={}, experiments={}, log=[]))
        self.plan = load_plan()
        d = contract_v3.delivery(self.plan)
        self.config = dict(xclbin=str(Path(args.xclbin).resolve()), image_profile='core-vrf-sync-v5',
                           core_count=3, memory_ports=3, hbm_dma=True, hbm_bytes=d['hbm_bytes'],
                           device_clock=True)

    def save(self):
        self.state['observed_utc'] = utc()
        tmp = self.out/'state.tmp'
        tmp.write_text(json.dumps(self.state, indent=1)+'\n')
        tmp.replace(self.state_path)

    def env(self, knobs, block):
        env = {k: v for k, v in os.environ.items() if not k.startswith('TNCA_')}
        env.update(CUDA_VISIBLE_DEVICES='', R40_CONTRACT='v3', PYTHONFAULTHANDLER='1',
                   TNCA_OPERATING_CLOCK_EVIDENCE=str(Path(self.a.operating_clock_evidence).resolve()),
                   TNCA_DEADLINE=DEADLINE, TNCA_HOST_TIMEOUT=HOST_TIMEOUT,
                   ARCHBENCH_BLOCK_ID=str(block))
        env['PYTHONPATH'] = os.pathsep.join(str(ROOT/d) for d in ('', 'python', 'compiler/python', 'tools/asm'))
        return env

    def command(self, cfg, tag, warmup, timed):
        stem = f"{cfg['key']}-{tag}"
        raw = self.out/'runs'/(stem+'.jsonl')
        bundle = self.out/'bundles'/stem
        if cfg['klass'] == 'host':
            cmd = [sys.executable, '-u', str(ROOT/'tools/r40/v3_latency_child.py'), '--point', cfg['point']['id'],
                   '--profile', 'core-vrf-sync-v5', '--warmup', str(warmup), '--timed', str(timed),
                   '--bundle-dir', str(bundle)]
        else:
            cmd = [sys.executable, '-u', '-m', 'archbench.run', '--platform', 'tnca', '--backend', 'fpga-u280-multicore',
                   '--backend-args', json.dumps(self.config), '--profile', 'core-vrf-sync-v5',
                   '--point', cfg['point']['id'], '--jobs', '1', '--seed', '0', '--warmup', str(warmup),
                   '--timed', str(timed), '--timeout', HOST_TIMEOUT, '--bundle-dir', str(bundle), '--out', str(raw)]
            for k, v in sorted(cfg['knobs'].items()):
                cmd += ['--set', f'{k}={v}']
        return stem, raw, cmd

    def child_env(self, cfg, block):
        env = self.env(cfg['knobs'], block)
        if cfg['klass'] == 'host':
            # What archbench.run's parent would set for its child.
            env.update(TNCA_TRITON='1', TNCA_BACKEND='fpga-u280-multicore',
                       TNCA_BACKEND_ARGS=json.dumps(self.config), **cfg['knobs'])
        return env

    def check(self, cfg, row, warmup, timed):
        plat, prov = row.get('platform', {}), row.get('provenance', {})
        d = contract_v3.delivery(self.plan)
        problems = []
        if row.get('status') != 'pass' or row.get('fallbacks'):
            problems.append(f"status {row.get('status')} {row.get('problems')}")
        if row.get('point_id') != cfg['point']['id'] or row.get('point_sha256') != cfg['point']['sha256']:
            problems.append('point identity')
        if plat.get('software_tree') != d['software_tree'] or plat.get('image_sha256') != d['image_sha256']:
            problems.append('software/image identity')
        if plat.get('backend') != 'fpga-u280-multicore' or plat.get('backend_args') != self.config:
            problems.append('backend')
        if prov.get('clock_hz') != d['clock_hz']:
            problems.append(f"clock {prov.get('clock_hz')}")
        if any(plat.get('knobs', {}).get(k) != v for k, v in cfg['knobs'].items()):
            problems.append('knobs')
        cores = int(cfg['knobs'].get('TNCA_ACTIVE_CORES', '3'))
        if prov.get('launch_map', {}).get('cores') != list(range(cores)):
            problems.append(f"launch map cores {prov.get('launch_map', {}).get('cores')}")
        timing = 'request-latency' if cfg['klass'] == 'host' else 'device-cycles-serial-grid'
        t = (row.get('timings') or {}).get(timing) or {}
        if t.get('warmup') != warmup or len(t.get('samples') or []) != timed:
            problems.append('sampling')
        return problems

    def run_one(self, cfg, tag, warmup, timed, block=0):
        stem, raw, cmd = self.command(cfg, tag, warmup, timed)
        done = self.state['runs'].get(stem)
        if done and done['ok']:
            return done
        (self.out/'runs').mkdir(exist_ok=True)
        if raw.exists():
            # A failed earlier attempt stays in its preserved copy.
            raw.rename(raw.with_suffix(f'.failed{len(list(raw.parent.glob(raw.stem+".failed*")))}'))
        log = self.out/'logs'/(stem+'.log')
        log.parent.mkdir(exist_ok=True)
        self.state.update(status='running', current=stem, current_point=cfg['point']['id']); self.save()
        started = utc()
        with log.open('w') as fh:
            proc = subprocess.run(cmd, cwd=ROOT, env=self.child_env(cfg, block), stdout=subprocess.PIPE if cfg['klass'] == 'host' else fh,
                                  stderr=fh, text=True, timeout=float(HOST_TIMEOUT)*4)
        if cfg['klass'] == 'host':
            recs = [line[len('ARCHBENCH-RECORD '):] for line in proc.stdout.splitlines()
                    if line.startswith('ARCHBENCH-RECORD ')]
            raw.write_text(''.join(r+'\n' for r in recs))
        rows = ablate.load(raw) if raw.exists() else []
        problems = [f'exit {proc.returncode}'] if proc.returncode else []
        if len(rows) != 1:
            problems.append(f'{len(rows)} records')
        else:
            problems += self.check(cfg, rows[0], warmup, timed)
        entry = dict(ok=not problems, problems=problems, raw=str(raw), log=str(log), command=cmd,
                     started=started, ended=utc(), warmup=warmup, timed=timed, block=block)
        if rows and not problems:
            row = rows[0]
            entry['raw_sha256'] = sha_file(raw)
            t = row['timings']
            if cfg['klass'] == 'device':
                entry['seconds'] = t['device-cycles-serial-grid']['median']/contract_v3.delivery(self.plan)['clock_hz']
                entry['graph_makespan_cycles'] = (t.get('device-graph-makespan') or {}).get('median')
        self.state['runs'][stem] = entry
        print(f"{utc()} {'OK  ' if entry['ok'] else 'FAIL'} {stem} {cfg['point']['id']} "
              f"{entry.get('seconds', '')} {problems}", flush=True)
        self.save()
        return entry

    def device_experiment(self, exp):
        cfgs = self.c['configs']
        for row in exp['points']:
            q = {side: self.run_one(cfgs[row['keys'][side]], 'q', 0, 1) for side in row['order']}
            if not all(r['ok'] for r in q.values()):
                raise RuntimeError(f"{exp['name']}: {row['point']} failed: "
                                   + '; '.join(f'{s}: {r["problems"]}' for s, r in q.items()))
            if all(r['seconds'] <= SHORT_S for r in q.values()):
                for side in row['order']:
                    f = self.run_one(cfgs[row['keys'][side]], 'f3', 0, 3)
                    if not f['ok']:
                        raise RuntimeError(f"{exp['name']}: {row['point']} formal {side} failed: {f['problems']}")

    def host_experiment(self, exp, sampling):
        cfgs = self.c['configs']
        rng = random.Random(self.plan['sample_plan']['bootstrap']['seed'])
        for block in range(sampling['blocks']):
            for row in exp['points']:
                order = ['control', 'candidate']
                rng.shuffle(order)
                for side in order:
                    r = self.run_one(cfgs[row['keys'][side]], f'b{block}', sampling['warmup'], sampling['timed'], block)
                    if not r['ok']:
                        raise RuntimeError(f"{exp['name']}: {row['point']} block {block} {side} failed: {r['problems']}")

    def run(self, only=None):
        from archbench.provenance import software_tree
        d = contract_v3.delivery(self.plan)
        if software_tree(refresh=True) != d['software_tree']:
            raise SystemExit('product tree is not the v3 delivery identity')
        if sha_file(self.a.xclbin) != d['image_sha256']:
            raise SystemExit('xclbin is not the v3 delivery image')
        host = self.plan['sample_plan']['board_host_latency']
        for name in self.c['order']:
            if only and name not in only:
                continue
            if self.state['experiments'].get(name, {}).get('status') == 'collected':
                continue
            exp = self.c['experiments'][name]
            self.state['experiments'][name] = dict(status='running', started=utc()); self.save()
            try:
                if exp['klass'] == 'host':
                    self.host_experiment(exp, host)
                else:
                    self.device_experiment(exp)
                self.state['experiments'][name].update(status='collected', ended=utc())
            except Exception:
                # Preserve and continue: one failed experiment must not idle the
                # board for the others. Its gate cannot pass until it is fixed.
                self.state['experiments'][name].update(status='failed-preserved', ended=utc(),
                                                       error=traceback.format_exc())
            self.save()
            if software_tree(refresh=True) != d['software_tree']:
                raise SystemExit('product tree changed during collection')
        self.state.update(status='complete', current=None); self.save()


# ------------------------------------------------------------------ judgement

def pool(rows, timing):
    """Several independent blocks of one configuration as one record."""
    import copy
    result = copy.deepcopy(rows[0])
    _, entry = ablate.timing_entry(result, timing)
    entry['samples'] = [s for r in rows for s in ablate.samples_of(r, timing)[1]]
    for key in ('median', 'p95'):
        entry.pop(key, None)
    return result


def judge(out, campaign, state, write_evidence=False):
    plan = load_plan()
    results = {}
    jdir = out/'judge'
    jdir.mkdir(exist_ok=True)
    for name in campaign['order']:
        exp = campaign['experiments'][name]
        st = state['experiments'].get(name, {})
        if st.get('status') != 'collected':
            results[name] = dict(status='not-measured' if st.get('status') != 'failed-preserved' else 'fail',
                                 why=st.get('error', 'not collected')[-2000:])
            continue
        sides = {'control': [], 'candidate': []}
        for row in exp['points']:
            for side in sides:
                key = row['keys'][side]
                if exp['klass'] == 'host':
                    stems = [f'{key}-b{b}' for b in range(plan['sample_plan']['board_host_latency']['blocks'])]
                    recs = [ablate.load(state['runs'][s]['raw'])[0] for s in stems]
                    sides[side].append(pool(recs, exp['timing']))
                else:
                    stem = f'{key}-f3' if f'{key}-f3' in state['runs'] else f'{key}-q'
                    sides[side].append(ablate.load(state['runs'][stem]['raw'])[0])
        paths = {}
        for side, rows in sides.items():
            paths[side] = jdir/f'{name}.{side}.jsonl'
            paths[side].write_text(''.join(json.dumps(r)+'\n' for r in rows))
        subset = {row['point']: row['sha256'] for row in exp['points']}
        vary = list(exp['vary'])
        mirror = MIRRORS.get(vary[0])
        if mirror:
            field, parse = mirror
            for side, rows in sides.items():
                value = parse(str(campaign['configs'][exp['points'][0]['keys'][side]]['knobs'][vary[0]]))
                bad = [r['point_id'] for r in rows
                       if (r['platform'].get('inductor') or {}).get(field) not in (value, tuple(value) if isinstance(value, list) else value)]
                if bad:
                    results[name] = dict(status='fail', why=f'platform.inductor.{field} does not echo {vary[0]} on {side}: {bad[:3]}')
                    break
            else:
                vary.append(f'platform.inductor.{field}')
            if name in results:
                continue
        try:
            res = ablate.compare(paths['control'], paths['candidate'], subset, vary,
                                 timing=exp['timing'], threshold=None, audit_bundles=True)
        except ablate.Refused as e:
            results[name] = dict(status='fail', why=f'comparison refused: {e}')
            continue
        lo = res['ci95'][0]
        if exp['rule'] == 'capability-non-regression':
            ok = lo >= exp['floor'] and all(p['speedup'] >= 1/exp['max_point_slowdown'] for p in res['points'])
            status = 'capability-pass' if ok else 'fail'
        else:
            status = 'perf-pass' if lo > 1.0 else 'no-significant-gain'
        target = exp.get('target')
        results[name] = dict(status=status, s_geo=res['s_geo'], ci95=res['ci95'],
                             s_geo_balanced_weighted=res['s_geo_balanced_weighted'],
                             target=target, target_met=(None if target is None else res['s_geo'] >= target),
                             worst=res['worst'], regress_2pct=res['regress_2pct'],
                             regress_5pct=[p['point'] for p in res['points'] if p['speedup'] < 1/1.05],
                             points=res['points'], timing=res['timing'], vary=exp['vary'], compared_vary=vary,
                             control=exp['control'], candidate=exp['candidate_value'], baseline=exp['baseline'],
                             config_a=res['config_a'], config_b=res['config_b'],
                             inputs={side: dict(path=str(p), sha256=sha_file(p)) for side, p in paths.items()})
    report = dict(schema='tnca-r40-v3-campaign-judgement/1', judged=utc(), contract_sha256=contract_v3.LOCK_SHA256,
                  harness_sha256=sha_file(__file__), identity=contract_v3.delivery(plan), experiments=results)
    (out/'judgement.json').write_text(json.dumps(report, indent=1)+'\n')
    if write_evidence:
        record_gates(plan, results, out)
    return report


def record_gates(plan, results, out):
    os.environ['R40_CONTRACT'] = 'v3'
    import evidence
    for wp in plan['work_packages']:
        for gate in wp['gates']:
            names = gate.get('experiments')
            if not names or gate.get('deferred'):
                continue
            rs = {n: results.get(n, dict(status='not-measured')) for n in names}
            statuses = {r['status'] for r in rs.values()}
            if 'not-measured' in statuses:
                continue            # never write a partial gate
            if 'fail' in statuses:
                status = 'fail'
            elif gate['kind'] == 'capability':
                status = 'capability-pass' if statuses == {'capability-pass'} else 'fail'
            else:
                status = 'perf-pass' if statuses == {'perf-pass'} else 'no-significant-gain'
            checks = []
            for n, r in rs.items():
                checks.append(dict(name=f'{n}: complete accepted comparison', ok=r['status'] != 'fail',
                                   detail=r.get('why', '')))
                if r['status'] != 'fail':
                    if gate['kind'] == 'capability':
                        checks.append(dict(name=f'{n}: non-regression floor and per-point limit',
                                           ok=r['status'] == 'capability-pass', ci95=r['ci95'], s_geo=r['s_geo']))
                    else:
                        checks.append(dict(name=f'{n}: 95% CI lower bound of S_geo > 1',
                                           ok=r['status'] == 'perf-pass', ci95=r['ci95'], s_geo=r['s_geo']))
            detail = '; '.join(f"{n}: S_geo {r['s_geo']:.4f} CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
                               + (f", v2 target {r['target']} {'met' if r['target_met'] else 'missed'}"
                                  if r.get('target') else '')
                               for n, r in rs.items() if 's_geo' in r)
            evidence.record(gate['id'], status, checks, detail=detail or 'see judgement.json',
                            identity=contract_v3.delivery(plan), experiments=rs,
                            judgement=str(out/'judgement.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate', action='append', default=[], help='EXPERIMENT=VALUE for list-valued candidates')
    p.add_argument('--compile-preflight', type=Path)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--xclbin', type=Path, required=True)
    r.add_argument('--operating-clock-evidence', type=Path, required=True)
    r.add_argument('--only', nargs='*')
    j = sub.add_parser('judge')
    j.add_argument('--out', type=Path, required=True)
    j.add_argument('--evidence', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'plan':
        plan = load_plan()
        candidates = dict(kv.split('=', 1) for kv in a.candidate)
        preflight = None
        if a.compile_preflight:
            preflight = dict(path=str(a.compile_preflight), sha256=sha_file(a.compile_preflight))
        campaign = build(plan, candidates, preflight)
        a.out.mkdir(parents=True, exist_ok=False)
        raw = (json.dumps(campaign, indent=1)+'\n').encode()
        (a.out/'campaign.json').write_bytes(raw)
        (a.out/'campaign.json').chmod(0o444)
        n = {k: sum(c['klass'] == k for c in campaign['configs'].values()) for k in ('device', 'host')}
        print(f"campaign {hashlib.sha256(raw).hexdigest()[:16]}: {len(campaign['configs'])} configurations {n}")
        return 0
    campaign = json.loads((a.out/'campaign.json').read_text())
    if campaign['contract_sha256'] != contract_v3.LOCK_SHA256:
        raise SystemExit('campaign was planned under another contract')
    if a.cmd == 'run':
        if campaign['harness_sha256'] != sha_file(__file__):
            print('note: collector changed since planning; the change is recorded in state.json', flush=True)
        runner = Runner(a.out, campaign, a)
        runner.state.setdefault('harness_sha256_by_run', []).append(dict(at=utc(), harness=sha_file(__file__),
            latency_child=sha_file(ROOT/'tools/r40/v3_latency_child.py')))
        runner.run(a.only)
        return 0
    state = json.loads((a.out/'state.json').read_text())
    report = judge(a.out, campaign, state, a.evidence)
    for name, r in report['experiments'].items():
        extra = (f" S_geo {r['s_geo']:.4f} CI [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}] target {r['target']}"
                 if 's_geo' in r else f" {r.get('why', '')[:200]}")
        print(f'{name:26s} {r["status"]:20s}{extra}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
