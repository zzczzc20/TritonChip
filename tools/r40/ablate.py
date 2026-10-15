#!/usr/bin/env python3
"""R40 A/B ablation: a comparison the protocol accepts, or the reasons it does not.

    python3 tools/r40/ablate.py --plan spec/r40/plan.json --experiment R40-02.resident \
        A.jsonl B.jsonl [--json out.json]
    python3 tools/r40/ablate.py --subset spec/r40/suites/<subset>.json --vary KNOB[,KNOB] \
        [--timing graph-makespan] A.jsonl B.jsonl

A is the configuration without the mechanism, B with it. Both are ArchBench
records (protocol 2, archbench/PROTOCOL.md; protocol-1 records are accepted
only as the frozen B39 baseline, wrapped by tools/r40/b39.py with the
provenance spec/r40/baseline.json gives them).

doc/r40_tasks_and_acceptance.md sec 3 asks that a comparison be refused,
not merely flagged, when any of these is true -- each is a way a number has
been made to look better than the machine is:

  missing       a point of the frozen subset is absent or not measured on
                either side (a slow point dropped, a failure left out)
  zero-fill     a passing record whose time is zero or empty
  mixed         one side holds more than one configuration (two images, two
                software trees, two sets of knobs)
  config        A and B differ in anything but the declared knobs: the image,
                the clock policy, a point definition (its tolerance is in the
                hash), the suite, the weights, the software beyond the knob
  provenance    a record without the chain back to its source: the software
                tree, the artifacts, the profile, the image and clock, the
                launch map
  timing        the two sides measured different things (a kernel makespan
                against a graph makespan, resident against request latency),
                or a makespan that is the sum of per-core cycles
  clock-only    the only difference is the clock: device cycles equal, so a
                ratio of seconds would be a ratio of frequencies
  warmup        a timed sample that includes compilation, autotuning or the
                warm-up of a replay

A record may carry several timing kinds (`timings`, protocol 2); the
experiment's frozen kind is the one compared, never a mix. A and B may
differ in a knob (`--vary TNCA_X`) or, declared the same way, in a platform
field (`--vary platform.image`) -- nothing else. `--audit-bundles` checks each
record's provenance bundle file by file (archbench/provenance.py), and
`--resources A.json B.json` puts the two images' resources and timing
(tools/r40/resources.py) beside the speed.

What it reports for an accepted comparison: per point A, B and A/B (the
median of each side's timed samples); S_geo (R39's definition, the geometric
mean of the per-point ratios); the balanced-weighted geometric mean, named
as such; a 95 % paired bootstrap interval of S_geo; the worst regression;
and a verdict -- perf-pass when the interval's lower bound is above 1 and the
frozen threshold is met, no-significant-gain when the lower bound is not
above 1 or the threshold is not met, fail when a point regressed past 5 %
without an explanation the plan records.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Timing kinds (archbench/PROTOCOL.md sec 3, protocol 2).
TIMINGS = ('device-cycles-serial-grid', 'kernel-makespan', 'graph-makespan',
           'resident-call-latency', 'request-latency')
MAKESPANS = ('kernel-makespan', 'graph-makespan')

# What a protocol-2 record's provenance must hold (doc sec 3.3, last item).
PROVENANCE = ('software_tree', 'artifacts', 'profile', 'capabilities', 'image',
              'clock_hz', 'launch_map')

# Platform fields that make a configuration; A and B may differ only in the
# declared knobs.
CONFIG_FIELDS = ('kind', 'profile', 'backend', 'opt_level', 'inductor', 'device_id',
                 'image', 'clock_policy', 'software_tree', 'suite', 'weights')

BOOT = 2000
ALPHA = 0.05


class Refused(Exception):
    """The comparison is not a measurement of the mechanism."""

    def __init__(self, kind, detail):
        super().__init__(f'{kind}: {detail}')
        self.kind = kind
        self.detail = detail


def load(path):
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise Refused('format', f'{path}:{n}: {e}')
    return out


def config_of(rec, vary):
    p = rec.get('platform') or {}
    fields = {v.split('.', 1)[1] for v in vary if v.startswith('platform.')}
    cfg = {k: p.get(k) for k in CONFIG_FIELDS if k in p and k not in fields}
    # A device migration may change only inductor.device, while block/tile
    # settings remain fixed. Never exempt the whole Inductor configuration.
    for field in fields:
        if '.' in field:
            parent, child = field.split('.', 1)
            if parent in cfg and isinstance(cfg[parent], dict):
                cfg[parent] = dict(cfg[parent])
                cfg[parent].pop(child, None)
    knobs = dict(p.get('knobs') or {})
    for k in vary:
        knobs.pop(k, None)
    cfg['knobs'] = knobs
    return cfg


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def timing_entry(rec, timing=None):
    """(kind, entry) of the record's timing `timing`, or its primary one.

    Protocol 2 records carry every kind they measured in `timings` and the
    primary one in `time`. A protocol-1 record from the frozen B39 baseline
    is deterministic -- one launch sequence is the measurement -- and counts
    as one sample of `device-cycles-serial-grid`."""
    t = rec.get('time') or {}
    tims = rec.get('timings') or {}
    if timing and timing in tims:
        return timing, tims[timing]
    kind = t.get('kind') or (t.get('timing') == 'serial-grid' and 'device-cycles-serial-grid')
    if timing and kind != timing:
        raise Refused('timing', f"{rec['point_id']}: no {timing} samples (it has "
                                f"{sorted(set(tims) | {kind} - {None, False})})")
    return kind, t


def samples_of(rec, timing=None):
    """The timed samples of a record, in its timing's unit, and the timing."""
    kind, t = timing_entry(rec, timing)
    if 'samples' in t:
        s = t['samples']
        if any(isinstance(x, dict) for x in s):
            bad = [x for x in s if isinstance(x, dict) and
                   (x.get('warmup') or x.get('autotune') or x.get('replay_warmup') or x.get('compile'))]
            if bad:
                raise Refused('warmup', f"{rec['point_id']}: {len(bad)} timed sample(s) include "
                                        f"warm-up, autotuning or compilation")
            vals = [x['value'] for x in s]
        else:
            vals = list(s)
    elif t.get('device') is not None:
        vals = [t['device']]
    else:
        vals = []
    for flag in ('includes_autotune', 'includes_compile', 'includes_replay_warmup'):
        if t.get(flag):
            raise Refused('warmup', f"{rec['point_id']}: the timed region {flag[9:].replace('_', ' ')}")
    return kind, vals


def check_makespan(rec):
    """A makespan must be measured end to end, never assembled from per-core
    busy time: doc sec 3.3 forbids reporting the sum of active cycles (or of
    overlapping kernels) as a completion time."""
    t = rec.get('time') or {}
    if t.get('kind') not in MAKESPANS:
        return
    per_core = t.get('per_core_active')
    if per_core and t.get('device') is not None and len(per_core) > 1:
        if abs(sum(per_core) - t['device']) <= max(1, 1e-6 * t['device']):
            raise Refused('timing', f"{rec['point_id']}: the makespan equals the sum of per-core "
                                    'active time; it was not measured end to end')
    if t.get('derived_from') in ('sum-of-cores', 'sum-of-kernels'):
        raise Refused('timing', f"{rec['point_id']}: makespan derived from {t['derived_from']}")


def side(path, subset, vary, *, legacy_ok, timing=None, audit_bundles=False):
    recs = load(path)
    by_id = {}
    for r in recs:
        by_id[r['point_id']] = r            # a rerun replaces, as archbench.report does
    cfgs = {digest(config_of(r, vary)): config_of(r, vary) for r in by_id.values()}
    if len(cfgs) != 1:
        raise Refused('mixed', f'{path}: {len(cfgs)} configurations on one side')
    cfg = next(iter(cfgs.values()))
    out = {}
    for pid, sha in subset.items():
        r = by_id.get(pid)
        if r is None:
            raise Refused('missing', f'{path}: {pid} is in the frozen subset and has no record')
        if r.get('point_sha256') != sha:
            raise Refused('config', f'{path}: {pid} was measured against another definition '
                                    f'(tolerance, shape or demand changed)')
        if r.get('status') != 'pass':
            raise Refused('missing', f"{path}: {pid} is {r.get('status')}; a frozen point that did "
                                     'not pass is not left out of a speed comparison')
        schema = r.get('schema', '')
        if schema.endswith('/1') and not legacy_ok:
            raise Refused('provenance', f'{path}: {pid} is a protocol-1 record outside the frozen '
                                        'B39 baseline')
        prov = r.get('provenance') or {}
        lack = [k for k in PROVENANCE if prov.get(k) in (None, '', [], {})]
        if lack:
            raise Refused('provenance', f"{path}: {pid} lacks {', '.join(lack)}")
        if audit_bundles and not schema.endswith('/1'):
            sys.path.insert(0, str(ROOT))
            from archbench.provenance import check_bundle
            bad = check_bundle(prov.get('bundle'), root=Path(path).parent)
            used = {l.get('artifact') for l in (prov.get('launch_map') or {}).get('launches') or []}
            files = set((prov.get('bundle') or {}).get('files') or {})
            bad += [f'artifact {a[:12]} ran but is not in the bundle' for a in used
                    if a and f'artifacts/{a}.tnca' not in files]
            if bad:
                raise Refused('provenance', f'{path}: {pid}: bundle: {bad[0]}'
                                            + (f' (+{len(bad) - 1} more)' if len(bad) > 1 else ''))
        check_makespan(r)
        kind, vals = samples_of(r, timing)
        if kind not in TIMINGS:
            raise Refused('timing', f'{path}: {pid} has no declared timing kind')
        if not vals or any(v is None or v <= 0 for v in vals):
            raise Refused('zero-fill', f'{path}: {pid} passed with an empty or zero time')
        out[pid] = dict(rec=r, kind=kind, vals=vals, family=r.get('family'),
                        clock=prov.get('clock_hz'))
    return cfg, out


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def compare(a_path, b_path, subset, vary, *, legacy_a=False, timing=None, explained=(),
            threshold=None, seed=20261010, audit_bundles=False, resources=None):
    """A verdict dict; raises Refused."""
    cfg_a, A = side(a_path, subset, vary, legacy_ok=legacy_a, timing=timing,
                    audit_bundles=audit_bundles)
    cfg_b, B = side(b_path, subset, vary, legacy_ok=False, timing=timing,
                    audit_bundles=audit_bundles)
    if digest(cfg_a) != digest(cfg_b):
        diff = sorted(k for k in set(cfg_a) | set(cfg_b) if cfg_a.get(k) != cfg_b.get(k))
        if diff == ['knobs']:
            ka, kb = cfg_a['knobs'], cfg_b['knobs']
            diff = [f'knob {k}' for k in sorted(set(ka) | set(kb)) if ka.get(k) != kb.get(k)]
        if diff == ['clock_policy'] or diff == ['image'] and _only_clock(A, B):
            raise Refused('clock-only', 'A and B differ only in the clock; a ratio of their seconds '
                                        'is a ratio of frequencies')
        raise Refused('config', f"A and B differ beyond the declared knobs: {', '.join(diff)}")
    kinds = {v['kind'] for v in A.values()} | {v['kind'] for v in B.values()}
    if len(kinds) != 1:
        raise Refused('timing', f'the sides measured different things: {sorted(kinds)}')
    kind = kinds.pop()
    if timing and kind != timing:
        raise Refused('timing', f'the experiment is frozen on {timing}; the records are {kind}')
    if _only_clock(A, B):
        raise Refused('clock-only', 'every point has equal device cycles on both sides; the '
                                    'difference is the clock')
    rows, logs, weights = [], [], {}
    fam = {}
    for pid in subset:
        fam.setdefault(A[pid]['family'], []).append(pid)
    for f, ids in fam.items():
        for pid in ids:
            weights[pid] = 1.0 / (len(fam) * len(ids))
    for pid in subset:
        ta, tb = median(A[pid]['vals']), median(B[pid]['vals'])
        sp = ta / tb
        rows.append(dict(point=pid, a=ta, b=tb, speedup=sp, n_a=len(A[pid]['vals']),
                         n_b=len(B[pid]['vals'])))
        logs.append(math.log(sp))
    sgeo = math.exp(sum(logs) / len(logs))
    wgeo = math.exp(sum(weights[r['point']] * math.log(r['speedup']) for r in rows))
    lo, hi = bootstrap(A, B, list(subset), seed)
    worst = min(rows, key=lambda r: r['speedup'])
    regress2 = [r['point'] for r in rows if r['speedup'] < 1 / 1.02]
    regress5 = [r['point'] for r in rows if r['speedup'] < 1 / 1.05 and r['point'] not in explained]
    unexplained2 = [p for p in regress2 if p not in explained]
    if regress5:
        verdict = 'fail'
    elif lo > 1.0 and (threshold is None or sgeo >= threshold) and not unexplained2:
        verdict = 'perf-pass'
    else:
        verdict = 'no-significant-gain'
    res = dict(timing=kind, points=rows, s_geo=sgeo, s_geo_balanced_weighted=wgeo,
               ci95=[lo, hi], worst=dict(point=worst['point'], speedup=worst['speedup']),
               regress_2pct=regress2, unexplained_regress_2pct=unexplained2,
               regress_5pct_unexplained=regress5, threshold=threshold, verdict=verdict,
               config_a=digest(cfg_a), config_b=digest(cfg_b), vary=list(vary))
    if resources:
        res['resources'] = resource_delta(*resources)
    return res


RESOURCE_KEYS = ('lut', 'ff', 'bram', 'uram', 'dsp')


def resource_delta(a_path, b_path):
    """The two images side by side (tools/r40/resources.py JSON): totals and
    per SLR, and setup/hold slack. A speedup is reported with what it cost."""
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    out = dict(a=a.get('image'), b=b.get('image'), total={}, slr={}, timing={})
    for k in RESOURCE_KEYS:
        va, vb = (a.get('total') or {}).get(k), (b.get('total') or {}).get(k)
        out['total'][k] = dict(a=va, b=vb, delta=(vb - va) if None not in (va, vb) else None)
    for slr in sorted(set(a.get('slr') or {}) | set(b.get('slr') or {})):
        out['slr'][slr] = {k: dict(a=(a.get('slr') or {}).get(slr, {}).get(k),
                                   b=(b.get('slr') or {}).get(slr, {}).get(k)) for k in RESOURCE_KEYS}
    for k in ('wns_ns', 'whs_ns', 'clock_mhz'):
        out['timing'][k] = dict(a=(a.get('timing') or {}).get(k), b=(b.get('timing') or {}).get(k))
    return out


def _only_clock(A, B):
    """Equal device cycles everywhere, seconds different: a clock change."""
    same = True
    for pid in A:
        ra, rb = A[pid]['rec'].get('time') or {}, B[pid]['rec'].get('time') or {}
        ca, cb = ra.get('device_cycles', ra.get('device')), rb.get('device_cycles', rb.get('device'))
        if ca is None or ca != cb:
            same = False
            break
    return same and any(A[p]['clock'] != B[p]['clock'] for p in A)


def bootstrap(A, B, ids, seed):
    """95 % paired bootstrap interval of S_geo: resample the points, and
    within each point its timed samples on both sides (a deterministic
    single-sample record contributes no within-point spread)."""
    rng = random.Random(seed)
    stats = []
    for _ in range(BOOT):
        logs = []
        for _ in ids:
            pid = ids[rng.randrange(len(ids))]
            va, vb = A[pid]['vals'], B[pid]['vals']
            sa = [va[rng.randrange(len(va))] for _ in va]
            sb = [vb[rng.randrange(len(vb))] for _ in vb]
            logs.append(math.log(median(sa) / median(sb)))
        stats.append(math.exp(sum(logs) / len(logs)))
    stats.sort()
    return stats[int(ALPHA / 2 * BOOT)], stats[int((1 - ALPHA / 2) * BOOT) - 1]


def load_subset(path):
    doc = json.loads(Path(path).read_text())
    return {e['id']: e['sha256'] for e in doc['points']}, doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--experiment', help='an experiment id of the plan')
    ap.add_argument('--subset', type=Path, help='a frozen subset file (spec/r40/suites/)')
    ap.add_argument('--vary', default='', help='the knobs A and B may differ in, comma-separated')
    ap.add_argument('--timing', choices=TIMINGS)
    ap.add_argument('--threshold', type=float)
    ap.add_argument('--legacy-a', action='store_true',
                    help='A is the frozen B39 baseline (protocol-1 records wrapped by tools/r40/b39.py)')
    ap.add_argument('--audit-bundles', action='store_true',
                    help='check every record\'s provenance bundle against its hashes')
    ap.add_argument('--resources', nargs=2, metavar=('A.json', 'B.json'),
                    help='the two images\' resource summaries (tools/r40/resources.py)')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    explained = ()
    if args.plan and args.experiment:
        plan = json.loads(args.plan.read_text())
        if plan.get('contract_version') == 2:
            from contract_v2 import verify, ROOT as CONTRACT_ROOT
            lock = verify()
            if hashlib.sha256(args.plan.read_bytes()).hexdigest() != lock['files']['spec/r40/v2/plan.json']:
                ap.error('the supplied v2 plan is not the frozen contract')
            if plan['experiments'][args.experiment].get('comparison_rule') == 'capability-non-regression':
                ap.error('use device_time_v2.py for the three-block capability comparison')
        exp = plan['experiments'][args.experiment]
        subset, _ = load_subset(ROOT / exp['subset'])
        vary = exp['vary']
        timing = exp['timing']
        threshold = exp.get('threshold')
        explained = tuple(exp.get('explained', ()))
        legacy = exp.get('legacy_a', False)
    elif args.subset:
        subset, _ = load_subset(args.subset)
        vary = [v for v in args.vary.split(',') if v]
        timing, threshold, legacy = args.timing, args.threshold, args.legacy_a
    else:
        ap.error('give --plan and --experiment, or --subset')
    try:
        res = compare(args.a, args.b, subset, vary, legacy_a=legacy, timing=timing,
                      explained=explained, threshold=threshold,
                      audit_bundles=args.audit_bundles, resources=args.resources)
    except Refused as e:
        print(f'REFUSED  {e}')
        if args.json:
            args.json.write_text(json.dumps(dict(verdict='refused', reason=e.kind,
                                                 detail=e.detail), indent=1) + '\n')
        return 2
    for r in res['points']:
        print(f"{r['point']:<56s} {r['a']:>14.6g} {r['b']:>14.6g} {r['speedup']:7.3f}")
    print(f"\nS_geo {res['s_geo']:.4f}  95% CI [{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]  "
          f"balanced-weighted {res['s_geo_balanced_weighted']:.4f}  ({res['timing']})")
    print(f"worst {res['worst']['point']} {res['worst']['speedup']:.3f}; "
          f">2% slower: {res['regress_2pct'] or 'none'}")
    if res.get('resources'):
        r = res['resources']
        print('resources (B - A): ' + ', '.join(f"{k} {v['delta']:+d}" for k, v in r['total'].items()
                                               if v['delta'] is not None)
              + f"; WNS {r['timing']['wns_ns']['a']} -> {r['timing']['wns_ns']['b']} ns")
    print(f"VERDICT {res['verdict']}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1) + '\n')
    return 0 if res['verdict'] == 'perf-pass' else 1


if __name__ == '__main__':
    sys.exit(main())
