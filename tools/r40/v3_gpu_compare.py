#!/usr/bin/env python3
"""R40-10 item 4: TNCA board records beside GPU runs of the same Torch points.

    python tools/r40/v3_gpu_compare.py \
        --tnca native/u280_tiny.jsonl native/u280_small.jsonl native/u280_medium.jsonl \
               hbm14.jsonl [large24.jsonl] \
        --gpu-archbench raw/archbench_*.jsonl.gz \
        --gpu-boundary raw/boundary_*.jsonl.gz \
        --out-dir evidence/r40/v3/gpu_comparison

doc/r40_tasks_and_acceptance.md, R40-10 acceptance 4 (kept by v3, change 7):
run the same Torch points at the same precision on an available GPU, with
the strong Inductor/CUDA-graph and the Triton-only variants, at the same
request/resident/makespan boundaries as TNCA; state the different resource
and power scope; no win/loss figures for commercial ASICs.

Inputs (plain or .gz JSONL; a path may be prefixed `SUITE=` to name its
suite, otherwise the suite is the frozen suite containing all its points):

  --tnca           ArchBench TNCA board records (archbench-torch-record/2)
  --gpu-archbench  `python -m archbench.run --platform gpu` records, the
                   primary GPU records (variants inductor-cudagraph and
                   inductor-triton as ArchBench defines them)
  --gpu-boundary   tools/r40/v3_gpu_boundaries.py records: the same points
                   timed on the GPU at TNCA's boundaries (CUPTI span, CUPTI
                   sum, host resident call latency, warm and cold)

Writes report.json and README.md. Suites without any TNCA input are
reported as `tnca-pending`; a point absent from a supplied file as
`missing`. Pairs are formed only where both sides pass with a positive
time. The report gives times on both sides and their quotient at one
boundary; it does not weight, normalise by resources or power, or
extrapolate.
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'r40-v3-gpu-comparison/1'
SUITES = ('existing', 'hbm-core', 'large')
SUITE_TITLES = {'existing': 'native77 (tiny, small, medium)', 'hbm-core': 'HBM14', 'large': 'large24'}
GPU_VARIANTS = ('inductor-cudagraph', 'inductor-triton', 'inductor-triton-bigsm')

# The boundaries: which TNCA timing is paired with which GPU timing.
BOUNDARIES = [
    dict(name='makespan',
         tnca='timings["device-graph-makespan"] median cycles / clock_hz',
         tnca_definition='first accepted launch to last device completion (sequence-checked MMIO '
                         'endpoints); host submission gaps and backpressure included; no cache flush, '
                         'after one untimed warm-up call',
         gpu='boundary timings["cupti-span-warm"] median',
         gpu_definition='first CUDA device activity start to last device activity end of one call '
                        '(CUPTI via torch.profiler); launch/scheduling gaps included; no flush, after 3 '
                        'warm-up calls',
         match='matched',
         differences=['GPU timestamps come from CUPTI with the profiler active, which can lengthen host '
                      'launch gaps of the non-graph variants; TNCA endpoints are device-clock MMIO counters',
                      'TNCA: one timed sample per point (functional inventory sampling); GPU: median of '
                      '10-25 profiled calls']),
    dict(name='serial-sum',
         tnca='timings["device-cycles-serial-grid"] median cycles / clock_hz',
         tnca_definition="sum over the call's launches of launch-accept-to-core-idle cycles; gaps excluded",
         gpu='boundary timings["cupti-sum-warm"] median',
         gpu_definition="sum of the call's device activity durations (kernels, memsets, copies); gaps "
                        'excluded; no flush',
         match='matched',
         differences=['a GPU kernel duration is its own makespan over the SMs; a TNCA launch ROI is the '
                      'launch makespan over the 3 cores']),
    dict(name='resident-call',
         tnca='timings["resident-call-latency"] median, or on the native resident path '
              'timings["request-latency"] median (0 bytes uploaded or downloaded: inputs and module '
              'resident, outputs left on the device)',
         tnca_definition='host time.monotonic() from the call until the graph is complete',
         gpu='boundary timings["resident-call-latency-warm"] median',
         gpu_definition='host time.monotonic() around fn(*inputs) and torch.cuda.synchronize(); module '
                        'and inputs resident (.cuda()), outputs left on the device; no flush',
         match='matched',
         differences=['TNCA: one sample per point; GPU: median of 20-50 calls',
                      'host CPUs differ (TNCA: acf3019 x86_64 over PCIe MMIO; GPU: GB10 Arm cores, '
                      'unified memory)']),
    dict(name='request',
         tnca='not in the native-path records (request-latency there is the resident call)',
         tnca_definition=None,
         gpu='boundary timings["request-latency"] median',
         gpu_definition='in-place host-to-device copy of every input, the call, .cpu() of every output',
         match='gpu-only',
         differences=['no TNCA counterpart in these records; GPU figure reported for completeness']),
    dict(name='archbench-gpu-primary',
         tnca='(none paired) TNCA device-cycles-serial-grid is the nearest kind',
         tnca_definition=None,
         gpu='archbench time.device (CUPTI activity sum, L2 + system cache flushed before the call), '
             'time.latency_cold / latency_warm (CUDA events around the call)',
         gpu_definition='archbench/PROTOCOL.md section 3.2',
         match='not-matched',
         differences=['cold cache on the GPU, no flush on TNCA',
                      'CUDA-event latency is device-side and includes the host time before the first '
                      'launch; TNCA has no event-based figure',
                      'reported unchanged as the primary ArchBench GPU record and as a cross-check of the '
                      'boundary harness (cupti-sum-cold should agree with it)']),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load(path):
    p = Path(path)
    raw = p.read_bytes()
    data = gzip.decompress(raw) if p.suffix == '.gz' else raw
    recs = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    return recs, dict(path=_rel(p), sha256=sha256_bytes(raw),
                      content_sha256=sha256_bytes(data) if data is not raw else None,
                      records=len(recs))


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _rel_arg(a):
    s, path = split_arg(a)
    if Path(path).exists():
        return (f'{s}=' if s else '') + _rel(path)
    return a


def load_suites(d):
    out = {}
    for s in SUITES:
        doc = json.loads((Path(d) / f'{s}.json').read_text())
        out[s] = {p['id']: p for p in doc['points']}
    return out


def split_arg(a):
    m = re.match(r'^([a-z0-9-]+)=(.+)$', a)
    if m and m.group(1) in SUITES:
        return m.group(1), m.group(2)
    return None, a


def infer_suite(ids, suites, path):
    have = [s for s in SUITES if set(ids) <= set(suites[s])]
    if len(have) == 1:
        return have[0]
    if not have:
        raise SystemExit(f'{path}: its points are not all in one frozen suite; prefix it with SUITE=')
    # the smallest suite that contains them all (hbm-core's two large SDPA points)
    return min(have, key=lambda s: len(suites[s]))


def med(t):
    return (t or {}).get('median')


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def kernel_class(name):
    n = name.lower()
    if n.startswith('triton') or 'triton_' in n:
        return 'triton'
    if 'flash' in n or 'fmha' in n:
        return 'flash-attention-library'
    if 'memset' in n or 'memcpy' in n:
        return 'memory'
    if 'cudnn' in n or 'conv' in n or 'winograd' in n:
        return 'cudnn-convolution'
    if any(k in n for k in ('gemm', 'gemv', 'cutlass', 'cublas', 'nvjet', 'xmma')):
        return 'cublas-cutlass-gemm'
    if 'at::native' in n:
        return 'aten-native'
    return 'other'


# ---- per-record extraction -------------------------------------------------

def tnca_view(r, suite_sha):
    t = r.get('timings') or {}
    plat = r.get('platform') or {}
    clock = plat.get('clock_hz')
    gm, sg = t.get('device-graph-makespan'), t.get('device-cycles-serial-grid')
    host = r.get('host') or {}
    native = (plat.get('knobs') or {}).get('TNCA_TORCH_DEVICE') == 'native'
    resident = request = None
    resident_source = None
    if 'resident-call-latency' in t:
        resident, request = t['resident-call-latency'], t.get('request-latency')
        resident_source = 'resident-call-latency'
    elif native and not host.get('upload_bytes') and not host.get('download_bytes'):
        resident = t.get('request-latency')
        resident_source = 'request-latency (native resident path, 0 bytes up/down)'
    else:
        request = t.get('request-latency')

    def sec(x):
        c = (x or {}).get('clock_hz') or clock
        return med(x) / c if med(x) and c else None
    problems = list(r.get('problems') or [])
    if r.get('point_sha256') != suite_sha:
        problems.append(f'point_sha256 {r.get("point_sha256")} differs from the frozen suite')
    status = r.get('status') if r.get('point_sha256') == suite_sha else 'definition-mismatch'
    return dict(status=status, problems=[p[:300] for p in problems],
                graph_makespan_s=sec(gm), graph_makespan_cycles=med(gm),
                serial_grid_s=sec(sg), serial_grid_cycles=med(sg),
                resident_call_s=med(resident), resident_source=resident_source,
                request_s=med(request),
                samples=len((gm or sg or {}).get('samples') or []), warmup=(gm or sg or {}).get('warmup'),
                launches=(r.get('time') or {}).get('launches'), clock_hz=clock, eta=r.get('eta'),
                max_rel_err=(r.get('correctness') or {}).get('max_rel_err'),
                gap_fraction=(1 - med(sg) / med(gm)) if med(gm) and med(sg) else None,
                timestamp=r.get('timestamp'))


def archbench_gpu_view(r, suite_sha):
    t = r.get('time') or {}
    status = r.get('status') if r.get('point_sha256') == suite_sha else 'definition-mismatch'
    return dict(status=status, problem_summary=_first_line(r.get('problems')),
                problems=[p[:1000] for p in (r.get('problems') or [])],
                device_cold_sum_s=t.get('device'), latency_cold_event_s=med(t.get('latency_cold')),
                latency_warm_event_s=med(t.get('latency_warm')),
                activities_per_call=t.get('activities_per_call'), eta=r.get('eta'),
                max_rel_err=(r.get('correctness') or {}).get('max_rel_err'),
                nvidia_smi=((r.get('platform') or {}).get('device') or {}).get('nvidia_smi'),
                timestamp=r.get('timestamp'))


def boundary_view(r, suite_sha):
    t = r.get('timings') or {}
    names = (r.get('kernels') or {}).get('names') or {}
    classes = collections.Counter()
    for n, c in names.items():
        classes[kernel_class(n)] += c
    non_triton = {k: v for k, v in classes.items() if k not in ('triton', 'memory')}
    status = r.get('status') if r.get('point_sha256') in (suite_sha, None) else 'definition-mismatch'
    if r.get('point_sha256') is None and status == 'pass':
        status = 'definition-unknown'
    sus = r.get('sustained') or {}
    span, ssum = med(t.get('cupti-span-warm')), med(t.get('cupti-sum-warm'))
    return dict(status=status, problem_summary=_first_line(r.get('problems')),
                problems=[p[:1000] for p in (r.get('problems') or [])],
                span_warm_s=span, span_cold_s=med(t.get('cupti-span-cold')),
                sum_warm_s=ssum, sum_cold_s=med(t.get('cupti-sum-cold')),
                resident_call_warm_s=med(t.get('resident-call-latency-warm')),
                resident_call_cold_s=med(t.get('resident-call-latency-cold')),
                request_s=med(t.get('request-latency')),
                activities_per_call=t.get('activities_per_call'),
                kernel_classes=dict(classes) or None,
                triton_only=(not non_triton) if names else None,
                gap_fraction=(1 - ssum / span) if span and ssum else None,
                sustained_power_w=med(sus.get('power_w')), sustained_sm_mhz=med(sus.get('sm_mhz')),
                override=r.get('override'), max_rel_err=(r.get('correctness') or {}).get('max_rel_err'),
                tnca_modules_loaded=r.get('tnca_modules_loaded'),
                triton_active_backend=r.get('triton_active_backend'))


# ---- assembly --------------------------------------------------------------

def build(args):
    suites = load_suites(args.suites_dir)
    inputs = dict(tnca=[], gpu_archbench=[], gpu_boundary=[])
    tnca = collections.defaultdict(dict)          # suite -> point -> record
    tnca_ident = collections.defaultdict(set)
    for a in args.tnca or []:
        s, path = split_arg(a)
        recs, meta = load(path)
        s = s or infer_suite([r['point_id'] for r in recs], suites, path)
        meta['suite'] = s
        inputs['tnca'].append(meta)
        for r in recs:
            if r['point_id'] in tnca[s]:
                raise SystemExit(f'{path}: {r["point_id"]} appears twice in suite {s}')
            tnca[s][r['point_id']] = r
            p = r.get('platform') or {}
            tnca_ident['software_tree'].add(p.get('software_tree'))
            tnca_ident['image_sha256'].add(p.get('image_sha256'))
            tnca_ident['clock_hz'].add(p.get('clock_hz'))
            tnca_ident['backend'].add(p.get('backend'))
            tnca_ident['profile'].add(p.get('profile'))
            tnca_ident['core_count'].add((p.get('backend_args') or {}).get('core_count'))
            tnca_ident['memory_ports'].add((p.get('backend_args') or {}).get('memory_ports'))
            tnca_ident['hbm_bytes'].add((p.get('backend_args') or {}).get('hbm_bytes'))
            tnca_ident['knobs'].add(json.dumps(p.get('knobs'), sort_keys=True))
            tnca_ident['clock_qualification'].add(p.get('clock_qualification'))
            tnca_ident['torch'].add((p.get('toolchain') or {}).get('torch'))
            tnca_ident['host'].add((r.get('env') or {}).get('host'))
    gpu_ab = collections.defaultdict(lambda: collections.defaultdict(dict))   # variant -> suite -> point
    gpu_dev = collections.defaultdict(set)
    for a in args.gpu_archbench or []:
        s, path = split_arg(a)
        recs, meta = load(path)
        s = s or infer_suite([r['point_id'] for r in recs], suites, path)
        variants = {(r.get('platform') or {}).get('variant') for r in recs}
        meta.update(suite=s, variants=sorted(variants))
        inputs['gpu_archbench'].append(meta)
        for r in recs:
            v = r['platform']['variant']
            gpu_ab[v][s][r['point_id']] = r
            d = r['platform'].get('device') or {}
            gpu_dev['name'].add(d.get('name'))
            gpu_dev['sm_count'].add(d.get('sm_count'))
            gpu_dev['l2_bytes'].add(d.get('l2_bytes'))
            gpu_dev['capability'].add(d.get('capability'))
            gpu_dev['cuda'].add(d.get('cuda'))
            if d.get('nvidia_smi'):
                gpu_dev['driver'].add(d['nvidia_smi'].split(',')[0].strip())
            env = r.get('env') or {}
            gpu_dev['torch'].add(env.get('torch'))
            gpu_dev['triton'].add(env.get('triton'))
            gpu_dev['host'].add(env.get('host'))
    gpu_bd = collections.defaultdict(dict)        # variant -> point
    for a in args.gpu_boundary or []:
        _, path = split_arg(a)
        recs, meta = load(path)
        meta['variants'] = sorted({r.get('variant') for r in recs})
        inputs['gpu_boundary'].append(meta)
        for r in recs:
            gpu_bd[r['variant']][r['point_id']] = r
            if r.get('software_tree'):
                gpu_dev['boundary_software_tree'].add(r['software_tree'])
            if r.get('harness_sha256'):
                gpu_dev['boundary_harness_sha256'].add(r['harness_sha256'])
    tnca_suites = {m['suite'] for m in inputs['tnca']}

    rows = []
    for s in SUITES:
        for pid, sp in sorted(suites[s].items()):
            row = dict(suite=s, point_id=pid, family=sp['family'], tier=sp['tier'], dtype=sp['dtype'],
                       point_sha256=sp['sha256'])
            if s not in tnca_suites:
                row['tnca'] = dict(status='tnca-pending')
            elif pid not in tnca[s]:
                row['tnca'] = dict(status='missing')
            else:
                row['tnca'] = tnca_view(tnca[s][pid], sp['sha256'])
            row['gpu'] = {}
            for v in GPU_VARIANTS:
                g = {}
                if v in gpu_ab:
                    r = gpu_ab[v][s].get(pid)
                    g['archbench'] = archbench_gpu_view(r, sp['sha256']) if r else dict(status='missing')
                if v in gpu_bd:
                    r = gpu_bd[v].get(pid)
                    g['boundary'] = boundary_view(r, sp['sha256']) if r else dict(status='missing')
                if g:
                    row['gpu'][v] = g
            row['pairs'] = pairs(row)
            rows.append(row)
    return suites, inputs, tnca_ident, gpu_dev, rows


PAIRS = (('makespan', 'graph_makespan_s', 'span_warm_s'),
         ('serial-sum', 'serial_grid_s', 'sum_warm_s'),
         ('resident-call', 'resident_call_s', 'resident_call_warm_s'))


def pairs(row):
    out = {}
    t = row['tnca']
    for v, g in row['gpu'].items():
        b = g.get('boundary') or {}
        pv = {}
        for name, tk, gk in PAIRS:
            ok = t.get('status') == 'pass' and b.get('status') == 'pass' and t.get(tk) and b.get(gk)
            pv[name] = dict(tnca_s=t.get(tk), gpu_s=b.get(gk),
                            time_ratio_tnca_over_gpu=(t[tk] / b[gk]) if ok else None,
                            paired=bool(ok))
        out[v] = pv
    return out


def aggregates(rows):
    fam = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in rows:
        for v, pv in r['pairs'].items():
            for name, p in pv.items():
                if p['paired']:
                    fam[(r['suite'], r['family'])][v][name].append((p['tnca_s'], p['gpu_s']))
                    fam[(r['suite'], '*all*')][v][name].append((p['tnca_s'], p['gpu_s']))
    out = []
    for (s, f), byv in sorted(fam.items()):
        for v, byb in sorted(byv.items()):
            for name, xs in sorted(byb.items()):
                out.append(dict(suite=s, family=f, variant=v, boundary=name, n=len(xs),
                                tnca_geomean_s=geomean([a for a, _ in xs]),
                                gpu_geomean_s=geomean([b for _, b in xs]),
                                time_ratio_geomean=geomean([a / b for a, b in xs])))
    return out


def status_counts(rows):
    out = {}
    for s in SUITES:
        rs = [r for r in rows if r['suite'] == s]
        d = dict(points=len(rs), tnca=dict(collections.Counter(r['tnca']['status'] for r in rs)))
        for v in GPU_VARIANTS:
            for k in ('archbench', 'boundary'):
                c = collections.Counter((r['gpu'].get(v, {}).get(k) or {}).get('status') for r in rs)
                if set(c) != {None}:
                    d[f'gpu {v} {k}'] = {str(kk): vv for kk, vv in c.items()}
        out[s] = d
    return out


def failures(rows):
    """Every non-passing record. A point with no record yet (`missing`) is
    counted under not_measured, not listed here."""
    out = []
    for r in rows:
        if r['tnca'].get('status') not in ('pass', 'tnca-pending', 'missing'):
            out.append(dict(suite=r['suite'], point_id=r['point_id'], side='tnca', status=r['tnca'].get('status'),
                            problem=(r['tnca'].get('problems') or [''])[0][:300]))
        for v, g in r['gpu'].items():
            for k, x in g.items():
                if x.get('status') not in ('pass', 'missing'):
                    out.append(dict(suite=r['suite'], point_id=r['point_id'], side=f'gpu {v} {k}',
                                    status=x.get('status'),
                                    problem=x.get('problem_summary') or _first_line(x.get('problems'))))
    return out


def _first_line(ps):
    if not ps:
        return ''
    p = ps[0]
    for pat in (r'NoValidChoicesError: [^\n.]*', r'OutOfMemoryError[^\n]*', r'LoweringException: [^\n]*',
                r'[A-Za-z]+(?:Error|Exception): [^\n]*'):
        m = re.search(pat, p)
        if m:
            return m.group(0)[:300]
    lines = [x for x in p.strip().splitlines() if x.strip()]
    return (lines[-1] if lines else '')[:300]


def not_measured(rows):
    c = collections.Counter()
    for r in rows:
        if r['tnca'].get('status') in ('missing', 'tnca-pending'):
            c[f'tnca {r["suite"]} ({r["tnca"]["status"]})'] += 1
        for v, g in r['gpu'].items():
            for k, x in g.items():
                if x.get('status') == 'missing':
                    c[f'gpu {v} {k} {r["suite"]}'] += 1
    return dict(sorted(c.items()))


def crosschecks(rows):
    """ArchBench's cold CUPTI sum against the boundary harness's (same
    definition, separate process), and GPU records of a point in two suites."""
    out = {}
    for v in GPU_VARIANTS:
        q = []
        for r in rows:
            g = r['gpu'].get(v) or {}
            a, b = g.get('archbench') or {}, g.get('boundary') or {}
            if a.get('status') == 'pass' and b.get('status') == 'pass' and a.get('device_cold_sum_s') \
                    and b.get('sum_cold_s'):
                q.append(b['sum_cold_s'] / a['device_cold_sum_s'])
        if q:
            q.sort()
            out[f'{v}: boundary cupti-sum-cold / archbench time.device'] = dict(
                n=len(q), geomean=geomean(q), min=q[0], median=q[len(q) // 2], max=q[-1])
    dup = collections.defaultdict(list)
    for r in rows:
        for v, g in r['gpu'].items():
            a = g.get('archbench') or {}
            if a.get('status') == 'pass':
                dup[(r['point_id'], v)].append((r['suite'], a['device_cold_sum_s']))
    out['points measured in two suites (archbench time.device)'] = [
        dict(point_id=p, variant=v, values=dict(xs)) for (p, v), xs in sorted(dup.items()) if len(xs) > 1]
    return out


def gate_summary(paths):
    """tools/r40/v3_gpu_campaign.py sidecars: every attempt and what else held the GPU."""
    out = []
    for a in paths or []:
        _, path = split_arg(a)
        recs, meta = load(path)
        att = [g for g in recs if 'attempt' in g]
        contended = [g for g in att if g.get('seen_during_run') or g.get('aborted')]
        users = sorted({v.get('user') for g in recs for d in ('seen_during_run', 'seen_while_waiting')
                        for v in (g.get(d) or {}).values()} - {None})
        meta.update(attempts=len(att), clean=sum(bool(g.get('clean')) for g in att),
                    contended_attempts=len(contended),
                    oom_retries=sum(bool(g.get('oom_retry')) for g in att),
                    retried_points=sorted({g['point_id'] for g in att if not g.get('clean')}),
                    never_clean=sorted({g['point_id'] for g in recs if g.get('outcome')}),
                    waited_s=round(sum(g.get('waited_s') or 0 for g in att), 1),
                    other_users_seen=users,
                    attempts_with_same_account_coresident=sum(bool(g.get('same_account_coresident')) for g in att),
                    same_account_coresident_pids=sorted({k for g in att
                                                         for k in (g.get('same_account_coresident') or {})}),
                    poll_s=sorted({g.get('poll_s') for g in att} - {None}),
                    quiet_s=sorted({g.get('quiet_s') for g in att} - {None}),
                    first_start_utc=min((g['start_utc'] for g in att), default=None),
                    last_end_utc=max((g['end_utc'] for g in att), default=None))
        out.append(meta)
    return out


def resources(path):
    if not path:
        return None
    p = Path(path)
    raw = p.read_bytes()
    text = (gzip.decompress(raw) if p.suffix == '.gz' else raw).decode(errors='replace')
    rows = {}
    for key in ('CLB LUTs', 'CLB Registers', 'Block RAM Tile', 'URAM', 'DSPs'):
        m = re.search(r'^\|\s*' + re.escape(key) + r'\s*\|([^\n]+)$', text, re.M)
        if m:
            cells = [c.strip() for c in m.group(1).split('|') if c.strip()]
            rows[key] = dict(used_slr0_slr1_slr2=[float(c) for c in cells[:3]],
                             percent_slr0_slr1_slr2=[float(c) for c in cells[3:6]])
    dev = re.search(r'^\| Device\s*:\s*(\S+)', text, re.M)
    state = re.search(r'^\| Design State\s*:\s*(.+?)\s*$', text, re.M)
    return dict(path=_rel(p), sha256=sha256_bytes(raw), device=dev and dev.group(1),
                design_state=state and state.group(1), utilization=rows)


def gpu_spec():
    d = json.loads((ROOT / 'archbench/specs/gpu_gb10.json').read_text())
    f = d['facts']
    return dict(source='archbench/specs/gpu_gb10.json (spec_peak facts used by the ArchBench GPU bound)',
                sm_count=f['sm_count']['value'], sm_clock_max_hz=f['sm_clock_max_hz']['value'],
                l2_bytes=f['l2_bytes']['value'], system_cache_bytes=f['system_cache_bytes']['value'],
                memory=f['memory']['value'], dram_bytes_per_s=d['rates']['dram_bytes']['value'],
                tensor_fp16_mac_per_s_upper_estimate=d['rates']['tensor_fp16_mac']['value'],
                fp32_fma_per_s=d['rates']['fp32_fma']['value'])


def _s(x):
    return sorted(x, key=lambda v: (v is None, str(v)))


def report(args):
    suites, inputs, tid, gdev, rows = build(args)
    agg = aggregates(rows)
    bp = {(r['point_id'], v): r['gpu'][v]['boundary'].get('sustained_power_w') for r in rows for v in r['gpu']
          if (r['gpu'][v].get('boundary') or {}).get('sustained_power_w')}
    b_power = list(bp.values())
    idle_power = []
    for r in rows:
        for v, g in r['gpu'].items():
            s = (g.get('archbench') or {}).get('nvidia_smi')
            if s:
                try:
                    idle_power.append(float(s.split(',')[-1].strip().rstrip('W').strip()))
                except ValueError:
                    pass
    mods = set()
    backends = set()
    for r in rows:
        for v, g in r['gpu'].items():
            b = g.get('boundary') or {}
            mods.update(b.get('tnca_modules_loaded') or [])
            if b.get('triton_active_backend'):
                backends.add(b['triton_active_backend'])
    gpu_tree = _s(gdev.get('boundary_software_tree', set()))
    tnca_tree = _s(tid.get('software_tree', set()))
    rep = dict(
        schema=SCHEMA,
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        tool=dict(path=_rel(Path(__file__)), sha256=sha256_bytes(Path(__file__).read_bytes())),
        acceptance='doc/r40_tasks_and_acceptance.md section R40-10, acceptance 4 (retained by '
                   'doc/r40_tasks_and_acceptance_v3.md change 7)',
        inputs=inputs,
        identities=dict(
            tnca={k: _s(v) for k, v in tid.items()},
            gpu={k: _s(v) for k, v in gdev.items()},
            software_tree_equal=(gpu_tree == tnca_tree) if gpu_tree and tnca_tree else None,
            gpu_cuda_path_tnca_modules=sorted(mods), gpu_triton_active_backend=sorted(backends),
            gpu_cuda_path_note=('the GPU children import no tnca.inductor module (the TNCA path\'s Inductor '
                                'configuration); Triton backend discovery imports tnca.triton_backend '
                                '{compiler, driver}, and the active Triton backend is the one listed. '
                                + (args.identity_note or ''))),
        scope=dict(
            tnca=dict(device='AMD/Xilinx Alveo U280 (xcu280), FPGA prototype image',
                      cores=_s(tid.get('core_count', set())), clock_hz=_s(tid.get('clock_hz', set())),
                      memory_ports=_s(tid.get('memory_ports', set())),
                      exposed_hbm_bytes=_s(tid.get('hbm_bytes', set())),
                      image_sha256=_s(tid.get('image_sha256', set())),
                      host=_s(tid.get('host', set())), host_link='PCIe, MMIO register protocol',
                      resources=resources(args.tnca_resources),
                      power_w=None,
                      power_note='no power measurement in the TNCA records; board power not measured'),
            gpu=dict(device=_s(gdev.get('name', set())), sm_count=_s(gdev.get('sm_count', set())),
                     l2_bytes=_s(gdev.get('l2_bytes', set())), capability=_s(gdev.get('capability', set())),
                     cuda=_s(gdev.get('cuda', set())), driver=_s(gdev.get('driver', set())),
                     torch=_s(gdev.get('torch', set())), triton=_s(gdev.get('triton', set())),
                     host=_s(gdev.get('host', set())), spec=gpu_spec(),
                     clocks='not locked (no root); the SM clock observed during sustained loops is in each '
                            'boundary record',
                     power_w=dict(
                         sustained_loop_median_of_point_medians=(sorted(b_power)[len(b_power) // 2]
                                                                 if b_power else None),
                         sustained_loop_min=min(b_power) if b_power else None,
                         sustained_loop_max=max(b_power) if b_power else None,
                         records=len(b_power),
                         archbench_record_start_snapshot_median=(sorted(idle_power)[len(idle_power) // 2]
                                                                 if idle_power else None),
                         note='nvidia-smi power.draw as the driver reports it for GB10 (not wall power, '
                              'not the whole system); sustained = back-to-back resident calls of one point '
                              'for >= 2 s, sampled every ~0.1 s (small points are host-bound and barely load '
                              'the GPU); snapshot = one reading when an ArchBench record starts, before its '
                              'warm-up, with the device between points'))),
        boundaries=BOUNDARIES,
        status=status_counts(rows),
        family_geomeans=agg,
        crosschecks=crosschecks(rows),
        failures=failures(rows),
        not_measured=not_measured(rows),
        caveats=caveats(rows, tid, gdev, args),
        notes=list(args.note or []),
        gpu_gate=gate_summary(args.gpu_gate),
        rows=rows)
    return rep


def caveats(rows, tid, gdev, args):
    c = [
        'Different resource and power scope: TNCA is a 3-core FPGA prototype at 90 MHz on a U280 card '
        '(an FPGA-fabric-limited clock); the GPU is a 48-SM GB10 at up to '
        '3 GHz sharing LPDDR5X with its CPU. Time ratios here compare these two systems as built; they '
        'are not normalised by area, clock, power or process and do not predict an ASIC.',
        'No commercial-ASIC figures are used or compared (R40-10 acceptance 4).',
        'TNCA sampling is the functional inventory plan: one untimed warm-up call and one timed call per '
        'point. Device-cycle kinds are deterministic within 0.2 % on this board (v3 change 5); the host '
        'resident-call latency is a single sample and carries host noise.',
        'Warm vs cold: TNCA timed calls follow one warm-up with no cache flush, so the paired GPU figures '
        'are the warm (no-flush) ones. ArchBench\'s primary GPU device time is cold-L2 and is reported '
        'beside them, not paired.',
        'The TNCA makespan includes host submission gaps between launches (MMIO dispatch per launch); '
        'gap_fraction = 1 - serial-sum/makespan shows how much of each makespan they are. Under '
        'inductor-cudagraph the GPU replays the whole call as one CUDA graph; under the Triton variants '
        'kernels are launched one by one from Python and their host gaps enter the GPU span the same way.',
        'Precision is the point\'s dtype on both sides (FP16 with FP32 accumulation on tensor cores / MMA; '
        'FP32 points in IEEE binary32, TF32 off) and both are checked against the same FP32 reference and '
        'tolerance.',
        'inductor-triton as ArchBench freezes it (max_autotune_gemm with backends=TRITON) has no GEMM '
        'choice on GB10: Inductor disables Triton GEMM templates below 68 SMs (is_big_gpu), so its GEMM '
        'points fail with NoValidChoicesError. These failures are kept. inductor-triton-bigsm is the same '
        'configuration with that check forced true, a supplementary variant from '
        'tools/r40/v3_gpu_boundaries.py, not an ArchBench variant.',
        '"Triton-only" covers GEMMs: in the Triton variants Inductor still lowers some operators to library '
        'kernels (e.g. SDPA to flash attention, convolution to cuDNN); kernel_classes and triton_only in '
        'each boundary row list what ran.',
        'GPU clocks are not locked and the GPU is shared; the SM clock and power during the sustained loop '
        'and the nvidia-smi state at each record start are kept in the raw records.',
    ]
    xc = [v for k, v in crosschecks(rows).items() if isinstance(v, dict)]
    if xc:
        lo, hi = min(v['min'] for v in xc), max(v['max'] for v in xc)
        c.append(f'GPU kernels of the small tiers run for about 1-10 µs; the same point measured in separate '
                 f'processes (ArchBench cold sum vs the boundary harness cold sum) differs by a factor '
                 f'{lo:.2f}-{hi:.2f}, so per-point GPU figures at that scale carry that spread.')
    if set(tid.get('software_tree', set())) != set(gdev.get('boundary_software_tree', set())):
        c.append('Software identity differs between sides: TNCA records '
                 f'{_s(tid.get("software_tree", set()))}, GPU runs {_s(gdev.get("boundary_software_tree", set()))}. '
                 + (args.identity_note or ''))
    pending = sorted({r['suite'] for r in rows if r['tnca'].get('status') == 'tnca-pending'})
    if pending:
        c.append(f'TNCA records not yet supplied for suite(s) {pending}: their rows are tnca-pending and '
                 'enter no pair or mean.')
    return c


# ---- README ----------------------------------------------------------------

def fmt(x, unit='s'):
    if x is None:
        return '—'
    if unit == 's':
        for u, f in (('s', 1), ('ms', 1e-3), ('µs', 1e-6)):
            if abs(x) >= f or u == 'µs':
                return f'{x / f:.3g} {u}'
    return f'{x:.3g}'


def j(x):
    return ', '.join(str(v) for v in x) if isinstance(x, (list, tuple, set)) else str(x)


def readme(rep, raw_files):
    L = []
    w = L.append
    w('# R40-10 item 4: GPU comparison (TNCA U280 board vs NVIDIA GB10)')
    w('')
    w(f'Generated {rep["generated_utc"]} by `{rep["tool"]["path"]}` (sha256 `{rep["tool"]["sha256"][:16]}…`). '
      'Data: `report.json` (every row, pair, aggregate and failure). Requirement: ' + rep['acceptance'] + '.')
    w('')
    ti, gi = rep['identities']['tnca'], rep['identities']['gpu']
    w('## What was run')
    w('')
    w(f'- **TNCA**: board records, image `{", ".join(x[:16] for x in ti.get("image_sha256", []) if x)}…`, '
      f'software tree {", ".join(f"`{x}`" for x in ti.get("software_tree", []))}, '
      f'{j(ti.get("core_count"))} cores, clock {j(ti.get("clock_hz"))} Hz, backend {j(ti.get("backend"))}, '
      f'profile {j(ti.get("profile"))}, knobs `{j(ti.get("knobs"))}`, host {j(ti.get("host"))}.')
    w(f'- **GPU**: {j(gi.get("name"))} ({j(gi.get("sm_count"))} SMs, L2 {j(gi.get("l2_bytes"))} B, '
      f'cc {j(gi.get("capability"))}, CUDA {j(gi.get("cuda"))}, driver {j(gi.get("driver"))}), '
      f'torch {j(gi.get("torch"))}, triton {j(gi.get("triton"))}, host {j(gi.get("host"))}; '
      f'source tree {", ".join(f"`{x}`" for x in gi.get("boundary_software_tree", []))}.')
    w('  - `python -m archbench.run --platform gpu` with `--variant inductor-cudagraph` (primary) and '
      '`--variant inductor-triton`, every point of each suite (the primary ArchBench GPU records).')
    w('  - `tools/r40/v3_gpu_boundaries.py` for the same points with variants inductor-cudagraph, '
      'inductor-triton and inductor-triton-bigsm: the timings at TNCA\'s boundaries.')
    w(f'- CUDA path and the TNCA tree: {rep["identities"]["gpu_cuda_path_note"]} '
      f'Modules seen: {rep["identities"]["gpu_cuda_path_tnca_modules"]}; active backend '
      f'{rep["identities"]["gpu_triton_active_backend"]}.')
    for n in rep.get('notes') or []:
        w(f'- {n}')
    gates = rep.get('gpu_gate') or []
    if gates:
        w('- Shared-GPU gate (`tools/r40/v3_gpu_campaign.py`): one point per process, started only when '
          'nvidia-smi listed no other user\'s compute process (a quiet window after one was seen), polled '
          'during the run; an attempt that saw one was stopped and retried. Per stage: ' + '; '.join(
              f'`{Path(g["path"]).name}`: {g["clean"]} clean of {g["attempts"]} attempts, '
              f'{g["contended_attempts"]} stopped for contention, {g["oom_retries"]} OOM retries, '
              f'waited {g["waited_s"]:.0f} s, never clean {len(g["never_clean"])}, '
              f'{g["attempts_with_same_account_coresident"]} with a same-account process resident'
              for g in gates) + '. Other users seen: ' +
          (', '.join(sorted({u for g in gates for u in g['other_users_seen']})) or 'none') +
          '. Same-account co-residents are other jobs of the measuring account that held a CUDA context '
          '(not waited for; listed in each gate line).')
    w('')
    w('## Status per suite')
    w('')
    for s, d in rep['status'].items():
        w(f'- **{SUITE_TITLES[s]}** ({d["points"]} points): TNCA {d["tnca"]}; ' +
          '; '.join(f'{k}: {v}' for k, v in d.items() if k.startswith('gpu')))
    w('')
    w('## Boundaries')
    w('')
    w('| Boundary | TNCA | GPU | Match |')
    w('|---|---|---|---|')
    for b in rep['boundaries']:
        w(f'| {b["name"]} | {b["tnca"]} | {b["gpu"]} | {b["match"]} |')
    w('')
    for b in rep['boundaries']:
        if b['differences']:
            w(f'- {b["name"]}: ' + '; '.join(b['differences']) + '.')
    w('')
    w('## Scope')
    w('')
    st, sg = rep['scope']['tnca'], rep['scope']['gpu']
    res = st.get('resources') or {}
    util = res.get('utilization') or {}
    w(f'- TNCA: {st["device"]}; {j(st["cores"])} cores at {j(st["clock_hz"])} Hz; {j(st["memory_ports"])} '
      f'memory ports; exposed HBM {j(st["exposed_hbm_bytes"])} B; host {j(st["host"])} over {st["host_link"]}. '
      f'Power: **not measured** (null).')
    if util:
        w('  - Placed utilization per SLR (SLR0/1/2, %) from `' + res['path'] + '`: ' +
          '; '.join(f'{k} {v["percent_slr0_slr1_slr2"]}' for k, v in util.items()) +
          '. The report is from the attempt03 build that produced the image; it is bound to the image by '
          'that build directory, not by a hash inside the report.')
    sp = sg['spec']
    pw = sg['power_w']
    w(f'- GPU: {j(sg["device"])}, {sp["sm_count"]} SMs, max SM clock {sp["sm_clock_max_hz"] / 1e6:.0f} MHz, '
      f'L2 {sp["l2_bytes"] >> 20} MiB + {sp["system_cache_bytes"] >> 20} MiB system cache, {sp["memory"]} '
      f'({sp["dram_bytes_per_s"] / 1e9:.1f} GB/s), shared with the CPU. Clocks {sg["clocks"]}.')
    w(f'  - Power (nvidia-smi power.draw, GPU as the driver reports it, not wall power): sustained-loop '
      f'medians across {pw["records"]} point-variant records: median {fmt(pw["sustained_loop_median_of_point_medians"], "x")} W, '
      f'range {fmt(pw["sustained_loop_min"], "x")}–{fmt(pw["sustained_loop_max"], "x")} W; '
      f'record-start snapshot (between points) median {fmt(pw["archbench_record_start_snapshot_median"], "x")} W. '
      f'No per-point energy is derived: small points barely load the GPU in the sustained loop.')
    w('')
    w('## Geometric means per family (paired points only)')
    w('')
    w('Times are geometric means over the points of a family where both sides pass; ratio = geometric mean '
      'of T_TNCA / T_GPU at that boundary (above 1: TNCA took longer). See the scope and caveats before '
      'reading a ratio.')
    w('')
    for s in SUITES:
        sel = [a for a in rep['family_geomeans'] if a['suite'] == s]
        if not sel:
            w(f'### {SUITE_TITLES[s]}: no pairs (TNCA {rep["status"][s]["tnca"]})')
            w('')
            continue
        for bname, title in (('makespan', 'TNCA device-graph-makespan vs GPU CUPTI span (warm)'),
                             ('resident-call', 'TNCA resident call latency vs GPU resident call latency (warm)')):
            w(f'### {SUITE_TITLES[s]}: {title}')
            w('')
            w('| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |')
            w('|---|---|---|---|---|---|---|---|')
            fams = sorted({a['family'] for a in sel}, key=lambda f: (f == '*all*', f))
            for f in fams:
                def get(v):
                    return next((a for a in sel if a['family'] == f and a['variant'] == v
                                 and a['boundary'] == bname), {})
                cg, tr, tb = get('inductor-cudagraph'), get('inductor-triton'), get('inductor-triton-bigsm')
                w(f'| {"**all**" if f == "*all*" else f} | {fmt(cg.get("tnca_geomean_s"))} ({cg.get("n", 0)}) | '
                  f'{fmt(cg.get("gpu_geomean_s"))} | {fmt(cg.get("time_ratio_geomean"), "x")} | '
                  f'{fmt(tr.get("gpu_geomean_s"))} ({tr.get("n", 0)}) | {fmt(tr.get("time_ratio_geomean"), "x")} | '
                  f'{fmt(tb.get("gpu_geomean_s"))} ({tb.get("n", 0)}) | {fmt(tb.get("time_ratio_geomean"), "x")} |')
            w('')
        w('TNCA column: geometric mean over the inductor-cudagraph pairs; each ratio is over its own pairs.')
        w('')
    w('All variants and the serial-sum boundary are in `report.json` (`family_geomeans`).')
    w('')
    for s in ('hbm-core', 'large'):
        rs = [r for r in rep['rows'] if r['suite'] == s]
        w(f'## Per point: {SUITE_TITLES[s]}')
        w('')
        w('Makespan: TNCA device-graph-makespan / GPU CUPTI span (warm). Resident: host resident call '
          'latency. A status replaces a time where that side did not pass.')
        w('')
        w('| Point | TNCA makespan | GPU cudagraph | GPU triton | GPU triton-bigsm | TNCA resident | '
          'GPU cudagraph resident | GPU triton-bigsm resident | TNCA gap fraction |')
        w('|---|---|---|---|---|---|---|---|---|')
        for r in rs:
            t = r['tnca']

            def tv(k):
                return fmt(t.get(k)) if t.get('status') == 'pass' else t.get('status', '—')

            def gv(v, k):
                b = (r['gpu'].get(v) or {}).get('boundary') or {}
                if not b:
                    return '—'
                return fmt(b.get(k)) if b.get('status') == 'pass' else b.get('status', '—')
            gap = t.get('gap_fraction')
            w(f'| `{r["point_id"]}` | {tv("graph_makespan_s")} | {gv("inductor-cudagraph", "span_warm_s")} | '
              f'{gv("inductor-triton", "span_warm_s")} | {gv("inductor-triton-bigsm", "span_warm_s")} | '
              f'{tv("resident_call_s")} | {gv("inductor-cudagraph", "resident_call_warm_s")} | '
              f'{gv("inductor-triton-bigsm", "resident_call_warm_s")} | '
              f'{"—" if gap is None else f"{gap:.3f}"} |')
        w('')
    w('## Failures (kept, not dropped)')
    w('')
    fl = rep['failures']
    if not fl:
        w('None.')
    else:
        by = collections.defaultdict(list)
        for f in fl:
            by[(f['side'], f['status'], f['problem'][:160])].append(f'{f["point_id"]} ({f["suite"]})')
        for (side, status, prob), pts in sorted(by.items()):
            w(f'- {side}, **{status}** ({len(pts)}): {prob}')
            w(f'  - {", ".join(pts)}')
    w('')
    nm = rep.get('not_measured') or {}
    if nm:
        w('Not measured yet (no record supplied): ' + '; '.join(f'{k}: {v}' for k, v in nm.items()) + '.')
        w('')
    w('## Cross-checks')
    w('')
    for k, v in rep['crosschecks'].items():
        if isinstance(v, dict):
            w(f'- {k}: n={v["n"]}, geomean {v["geomean"]:.3f}, median {v["median"]:.3f}, '
              f'range {v["min"]:.3f}–{v["max"]:.3f}')
        else:
            for d in v:
                w(f'- {k}: {d["point_id"]} {d["variant"]}: ' +
                  ', '.join(f'{s} {fmt(x)}' for s, x in d['values'].items()))
    w('')
    w('## Caveats')
    w('')
    for c in rep['caveats']:
        w(f'- {c}')
    w('')
    w('## Files')
    w('')
    w('| File | sha256 | Records |')
    w('|---|---|---|')
    seen = set()
    for kind in ('tnca', 'gpu_archbench', 'gpu_boundary'):
        for m in rep['inputs'][kind]:
            seen.add(m['path'])
            extra = f' (content `{m["content_sha256"]}`)' if m.get('content_sha256') else ''
            w(f'| `{m["path"]}` | `{m["sha256"]}`{extra} | {m["records"]} |')
    for p in raw_files:
        if _rel(p) not in seen:
            w(f'| `{_rel(p)}` | `{sha256_bytes(Path(p).read_bytes())}` | |')
    res = rep['scope']['tnca'].get('resources')
    if res:
        w(f'| `{res["path"]}` | `{res["sha256"]}` | |')
    w('')
    w('Gzipped inputs: the first hash is of the .gz file, the content hash of the JSONL inside.')
    w('')
    w('Regenerate (from the repository root):')
    w('')
    w('```sh')
    w(rep.get('command', ''))
    w('```')
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tnca', nargs='*', default=[])
    ap.add_argument('--gpu-archbench', nargs='*', default=[])
    ap.add_argument('--gpu-boundary', nargs='*', default=[])
    ap.add_argument('--suites-dir', default=str(ROOT / 'spec/r40/suites'))
    ap.add_argument('--tnca-resources', help='Vivado report_utilization -slr report (.rpt or .rpt.gz)')
    ap.add_argument('--extra-file', nargs='*', default=[], help='more files to list with their sha256')
    ap.add_argument('--identity-note', default='', help='a sentence on the software identities')
    ap.add_argument('--note', action='append', default=[], help='a provenance line for the README (repeatable)')
    ap.add_argument('--notes-file', help='JSON list of provenance lines for the README (kept with the raw files)')
    ap.add_argument('--gpu-gate', nargs='*', default=[],
                    help='tools/r40/v3_gpu_campaign.py gate sidecars (OUT.gate.jsonl)')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'evidence/r40/v3/gpu_comparison')
    args = ap.parse_args(argv)
    if args.notes_file:
        nf = json.loads(Path(args.notes_file).read_text())
        nf = nf if isinstance(nf, dict) else dict(notes=nf)
        args.note = list(args.note or []) + list(nf.get('notes') or [])
        if nf.get('identity_note') and not args.identity_note:
            args.identity_note = nf['identity_note']
        args.extra_file = list(args.extra_file or []) + [args.notes_file]
    rep = report(args)
    rep['command'] = ' '.join(['python3', _rel(Path(__file__))] +
                              [a if a.startswith('-') else _rel_arg(a) for a in (argv or sys.argv[1:])])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'report.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    (args.out_dir / 'README.md').write_text(readme(rep, args.extra_file))
    print(json.dumps(rep['status'], indent=1))
    print(f'{len(rep["failures"])} failure entries; -> {args.out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
