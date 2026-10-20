#!/usr/bin/env python3
"""Fit the SystemC model's free timing constants on the R40-04 calibration split.

    fit.py host --records 'GLOB' [--records ...] --root SRC --out host_fit.json
    fit.py latency --dataset D.json[.gz] --root SRC --grid L=PRED.jsonl [--grid ...] --out latency_fit.json
    fit.py report --config CFG --host host_fit.json --latency latency_fit.json --out fit_report.json

Everything structural (pipeline, LSU, transfer engine, matrix unit, NUCA
slice, NoC, AXI outlet) is read from the RTL of the board image and is not
fitted (configs/r40-board-m2.cfg; the report lists each value and where it
comes from). Two things are not in the RTL and are fitted here, from
calibration rows only (spec/r40/suites/R40-04.calibration.json); a holdout or
unlabelled row is never read by a fitting step:

  host     the host time between launches of one graph call (the board's
           device graph makespan includes it): the median inter-launch gap of
           the calibration records' measured calls, and the per-KiB staging
           rate of their host transfers (steady upload/download seconds per
           byte). host_transfer_cycles (a fixed cost per transfer) is not
           identifiable from these records and stays 0.
  latency  hbm_latency_cycles (read) = tl_hbm_write_latency_cycles, the
           AXI/HBM round trip behind an outlet that no RTL file fixes, and
           optionally a second key (tl_outlet_to_slice_cycles: memory data
           path latency outside the outlet's outstanding window). The model
           is run on the calibration configurations at a grid of values; per
           configuration the graph makespan is interpolated (linear, or
           piecewise bilinear) between grid points, and the point minimising
           the sec 3.4 ratio max(median/0.10, p95/0.20) of the calibration
           absolute errors (then the median) is chosen on a 1-cycle grid. A
           confirmation run at the chosen point is judged separately.
"""
import argparse
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def calibration_ids(root):
    return {p['id'] for p in json.loads((Path(root)/'spec/r40/suites/R40-04.calibration.json').read_text())['points']}


def holdout_ids(root):
    return {p['id'] for p in json.loads((Path(root)/'spec/r40/suites/R40-04.model-holdout.json').read_text())['points']}


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def host(a):
    cal, hold = calibration_ids(a.root), holdout_ids(a.root)
    assert not cal & hold, 'calibration and holdout overlap'
    files = sorted({f for g in a.records for f in glob.glob(g, recursive=True)})
    gaps, rates, used = [], [], []
    sources = {}
    for f in files:
        raw = Path(f).read_bytes()
        sources[f] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('point_id') not in cal:      # holdout and unlabelled rows are never read further
                continue
            if r.get('status') != 'pass' or r.get('fallbacks'):
                continue
            clock = (r.get('provenance') or {}).get('clock_hz') or a.clock_hz
            knobs = (r.get('platform') or {}).get('knobs') or {}
            t = (r.get('timings') or {}).get('device-graph-makespan') or {}
            g = [b['start'] - x['end'] for call in t.get('endpoint_records') or [] for x, b in zip(call, call[1:])]
            gaps += g
            h = r.get('host') or {}
            st = h.get('steady') or {}
            nbytes = (h.get('upload_bytes') or 0) + (h.get('download_bytes') or 0)
            secs = (st.get('upload_s') or 0) + (st.get('download_s') or 0)
            if knobs.get('TNCA_TRANSFER_POLICY') == 'per-kernel' and nbytes >= 1 << 20 and secs > 0:
                rates.append(secs / nbytes * 1024 * clock)
            used.append(dict(point=r['point_id'], knobs=knobs, gaps=len(g),
                             transfer_bytes=nbytes if nbytes >= 1 << 20 else None, source=f))
    out = dict(schema='tnca-systemc-host-fit/1', fit_rows='calibration-only', records=len(used),
               gaps=dict(n=len(gaps), median=statistics.median(gaps) if gaps else None,
                         p05=quantile(gaps, .05), p95=quantile(gaps, .95), min=min(gaps, default=None),
                         max=max(gaps, default=None)),
               staging_cycles_per_kib=dict(n=len(rates), median=statistics.median(rates) if rates else None,
                                           p05=quantile(rates, .05), p95=quantile(rates, .95)),
               parameters=dict(host_launch_cycles=round(statistics.median(gaps)) if gaps else 0,
                               host_transfer_cycles=0,
                               host_transfer_cycles_per_kib=round(statistics.median(rates)) if rates else 0),
               rows=used, sources=sources)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('records', 'gaps', 'staging_cycles_per_kib', 'parameters')}, indent=1))


def load_dataset(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def latency(a):
    cal = calibration_ids(a.root)
    ds = load_dataset(a.dataset)
    rows = [r for r in ds['rows'] if r['split'] == 'calibration' and r['point'] in cal and r.get('graph_makespan_cycles')]

    def board(point, knobs):
        c = sorted(r['graph_makespan_cycles'] for r in rows if r['point'] == point
                   and all(r['knobs'].get(k) == str(v) for k, v in knobs.items()))
        return c[len(c) // 2] if c else None
    names = a.params.split(',')
    grid = {}
    for item in a.grid:
        value, path = item.split('=', 1)
        v = tuple(int(x) for x in value.split(':'))
        assert len(v) == len(names), item
        preds = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        grid[v] = {(p['point'], json.dumps(p['knobs'], sort_keys=True)): p['graph_cycles'] for p in preds if p['point'] in cal}
    keys = sorted(set.intersection(*(set(g) for g in grid.values())))
    refs = {k: board(k[0], json.loads(k[1])) for k in keys}
    keys = [k for k in keys if refs[k]]
    xs = sorted({v[0] for v in grid})
    ys = sorted({v[1] for v in grid}) if len(names) > 1 else [None]

    def lin(x0, x1, g0, g1, x):
        return g0 if x1 == x0 else g0 + (g1 - g0) * (x - x0) / (x1 - x0)

    def bracket(vals, x):
        lo = max([v for v in vals if v <= x], default=vals[0])
        hi = min([v for v in vals if v >= x], default=vals[-1])
        if lo == hi:   # outside (or on) the grid: extrapolate along the nearest segment
            i = vals.index(lo)
            lo, hi = (vals[i - 1], vals[i]) if i > 0 else (vals[0], vals[min(1, len(vals) - 1)])
        return lo, hi

    def model_at(k, point):
        if len(names) == 1:
            x0, x1 = bracket(xs, point[0])
            return lin(x0, x1, grid[(x0,)][k], grid[(x1,)][k], point[0])
        # Piecewise bilinear on the rectangles the grid has, linear along the
        # one-parameter segments it has; None where neither defines a value.
        x, y = point
        ys_at = lambda xv: sorted(v[1] for v in grid if v[0] == xv)
        x0, x1 = bracket(xs, x)
        for (xa, xb) in ((x0, x1),):
            ya, yb = ys_at(xa), ys_at(xb)
            common = sorted(set(ya) & set(yb))
            if len(common) >= 2:
                y0, y1 = bracket(common, y)
                g0 = lin(y0, y1, grid[(xa, y0)][k], grid[(xa, y1)][k], y)
                g1 = lin(y0, y1, grid[(xb, y0)][k], grid[(xb, y1)][k], y)
                return lin(xa, xb, g0, g1, x)
            if y in common:
                return lin(xa, xb, grid[(xa, y)][k], grid[(xb, y)][k], x)
        return None

    def score(point):
        vals = [model_at(k, point) for k in keys]
        if any(v is None for v in vals):
            return None
        errs = [abs(v - refs[k]) / refs[k] for v, k in zip(vals, keys)]
        med, p95 = quantile(errs, .5), quantile(errs, .95)
        return dict(median_abs=med, p95_abs=p95, sec34=max(med / 0.10, p95 / 0.20))
    if len(names) == 1:
        candidates = [(L,) for L in range(max(1, xs[0] // 2), xs[-1] + 1)]
    else:
        ylo, yhi = min(v[1] for v in grid), max(v[1] for v in grid)
        candidates = [(L, D) for L in range(xs[0], xs[-1] + 1) for D in range(ylo, yhi + (yhi - ylo) // 2 + 1)]
    scan = {c: sc for c in candidates for sc in [score(c)] if sc}
    objective = a.objective
    best = min(scan, key=lambda c: (round(scan[c][objective], 6), scan[c]['median_abs']))
    chosen = {names[i]: best[i] for i in range(len(names))}
    if 'hbm_latency_cycles' in chosen:
        chosen['tl_hbm_write_latency_cycles'] = chosen['hbm_latency_cycles']
    out = dict(schema='tnca-systemc-latency-fit/2', fit_rows='calibration-only', parameters=names,
               grid=[list(v) for v in sorted(grid)], configurations=len(keys),
               objective=dict(sec34='max(median/0.10, p95/0.20) of calibration |graph makespan error|, then median',
                              median_abs='calibration median |graph makespan error|, then p95')[objective],
               chosen=dict(chosen, **{f'interpolated_{k}': v for k, v in scan[best].items()}),
               best_by_median=dict(zip(names, min(scan, key=lambda c: (round(scan[c]['median_abs'], 6), scan[c]['p95_abs'])))),
               grid_scores={':'.join(map(str, v)): score(v) for v in sorted(grid)},
               per_configuration=[dict(point=k[0], knobs=json.loads(k[1]), board=refs[k],
                                       model={':'.join(map(str, v)): grid[v][k] for v in sorted(grid)}) for k in keys],
               scan={':'.join(map(str, c)): sc for c, sc in scan.items()
                     if all(x % 5 == 0 for x in c) or c == best})
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('parameters', 'grid', 'configurations', 'chosen', 'best_by_median', 'grid_scores')}, indent=1))


def shift(a):
    """One slow-path calibration pass at F0 plus per-configuration slopes:
    T_i(F) = T_i(F0) + (F - F0) * S_i, with S_i = dT_i/dL taken from an
    earlier fast-path latency grid (a uniform latency shift is what F is).
    Calibration rows only, like `latency`."""
    cal = calibration_ids(a.root)
    ds = load_dataset(a.dataset)
    rows = [r for r in ds['rows'] if r['split'] == 'calibration' and r['point'] in cal and r.get('graph_makespan_cycles')]

    def board(point, knobs):
        c = sorted(r['graph_makespan_cycles'] for r in rows if r['point'] == point
                   and all(r['knobs'].get(k) == str(v) for k, v in knobs.items()))
        return c[len(c) // 2] if c else None
    f0 = int(a.base.split('=', 1)[0])
    base = {(p['point'], json.dumps(p['knobs'], sort_keys=True)): p['graph_cycles']
            for p in (json.loads(l) for l in Path(a.base.split('=', 1)[1]).read_text().splitlines() if l.strip())
            if p['point'] in cal}
    grid = json.loads(Path(a.slopes).read_text())
    assert grid['fit_rows'] == 'calibration-only'
    lo, hi = [int(x) for x in a.slope_between.split(':')]
    d = a.slope_second
    slope = {}
    for c in grid['per_configuration']:
        m = c['model']
        def at(L):
            # the earlier grid is (hbm_latency_cycles : tl_outlet_to_slice_cycles); linear in the second key
            a5, a40 = m.get(f'{L}:5'), m.get(f'{L}:40')
            if a5 is None or a40 is None:
                return None
            return a5 + (a40 - a5) * (d - 5) / 35
        t_lo, t_hi = at(lo), at(hi)
        if t_lo is not None and t_hi is not None:
            slope[(c['point'], json.dumps(c['knobs'], sort_keys=True))] = (t_hi - t_lo) / (hi - lo)
    keys = sorted(k for k in base if k in slope)
    refs = {k: board(k[0], json.loads(k[1])) for k in keys}
    keys = [k for k in keys if refs[k]]

    def score(F):
        errs = [abs(base[k] + (F - f0) * slope[k] - refs[k]) / refs[k] for k in keys]
        med, p95 = quantile(errs, .5), quantile(errs, .95)
        return dict(median_abs=med, p95_abs=p95, sec34=max(med / 0.10, p95 / 0.20))
    scan = {F: score(F) for F in range(max(0, f0 - a.span), f0 + a.span + 1)}
    best = min(scan, key=lambda F: (round(scan[F]['sec34'], 6), scan[F]['median_abs']))
    out = dict(schema='tnca-systemc-latency-fit/2', fit_rows='calibration-only', parameters=[a.param],
               method=(f'one slow-path calibration pass at {a.param}={f0}; per configuration T(F) = T({f0}) + (F - {f0}) * S, '
                       f'S = dT/d(hbm_latency_cycles) between {lo} and {hi} of the earlier fast-path grid at '
                       f'tl_outlet_to_slice_cycles={d} (calibration rows only)'),
               grid=[[f0]], configurations=len(keys),
               objective='max(median/0.10, p95/0.20) of calibration |graph makespan error|, then median',
               chosen={a.param: best, **{f'interpolated_{k}': v for k, v in scan[best].items()}},
               best_by_median={a.param: min(scan, key=lambda F: (round(scan[F]['median_abs'], 6), scan[F]['p95_abs']))},
               grid_scores={str(f0): score(f0)},
               per_configuration=[dict(point=k[0], knobs=json.loads(k[1]), board=refs[k], model={str(f0): base[k]},
                                       slope_per_cycle=slope[k]) for k in keys],
               scan={str(F): v for F, v in scan.items()})
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('configurations', 'chosen', 'best_by_median', 'grid_scores')}, indent=1))


def report(a):
    cfg = {}
    for line in Path(a.config).read_text().splitlines():
        line = line.split('#')[0].strip()
        if '=' in line:
            k, v = line.split('=', 1)
            cfg[k] = int(v)
    h = json.loads(Path(a.host).read_text())
    lat = json.loads(Path(a.latency).read_text())
    assert h['fit_rows'] == lat['fit_rows'] == 'calibration-only'
    fitted = dict(h['parameters'])
    fitted.update(lat['chosen'])
    fitted = {k: v for k, v in fitted.items() if k in cfg}
    mismatch = {k: (v, cfg[k]) for k, v in fitted.items() if cfg[k] != v}
    assert not mismatch, f'config does not carry the fitted values: {mismatch}'
    sources = json.loads(Path(a.sources).read_text()) if a.sources else {}
    params = []
    for k, v in cfg.items():
        if k in fitted:
            src = 'FIT (calibration split only): ' + ('host_fit' if k.startswith('host_') else 'latency_fit')
        else:
            src = sources.get(k, 'structure (RTL of the attempt03 image) or model control; see the M2 README')
        params.append(dict(key=k, value=v, source=src))
    out = dict(schema='tnca-systemc-fit-report/1', fit_rows='calibration-only',
               rule='parameters were fitted on rows of spec/r40/suites/R40-04.calibration.json only; '
                    'no holdout (R40-04.model-holdout.json) or unlabelled row was read by any fitting step',
               config=str(a.config), config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
               procedure=[
                   'structural constants set from the RTL of the attempt03 board image (no fitting)',
                   'matrix busy cycles set from the RTL oracle (2a+18K+6 with ABUF; m.zero 17; row ops 7)',
                   'host: median calibration inter-launch gap -> host_launch_cycles; median calibration '
                   'per-kernel staging rate -> host_transfer_cycles_per_kib',
                   (f"latency: {lat['method']}; objective: {lat['objective']}" if lat.get('method') else
                    f"latency: calibration configurations simulated at the grid {lat['grid']} of "
                    f"{', '.join(lat.get('parameters', ['hbm_latency_cycles']))}; per-configuration "
                    f"(bi)linear interpolation; objective: {lat['objective']}"),
                   *(['tl_hbm_write_latency_cycles tied to hbm_latency_cycles']
                     if 'hbm_latency_cycles' in (lat.get('parameters') or []) else []),
                   *(a.note or []),
                   'holdout evaluated once with calibrate.py at the frozen configuration'],
               fitted=fitted, host_fit=dict(gaps=h['gaps'], staging_cycles_per_kib=h['staging_cycles_per_kib'],
                                            records=h['records']),
               latency_fit=dict(parameters=lat.get('parameters'), grid=lat['grid'], configurations=lat['configurations'],
                                objective=lat['objective'], chosen=lat['chosen'], best_by_median=lat.get('best_by_median'),
                                grid_scores=lat['grid_scores']),
               excluded_from_fitting_grid=a.excluded or [],
               parameters=params)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(fitted=fitted, parameters=len(params)), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('host')
    h.add_argument('--records', action='append', required=True)
    h.add_argument('--root', required=True)
    h.add_argument('--clock-hz', dest='clock_hz', type=float, default=90e6)
    h.add_argument('--out', required=True)
    lt = sub.add_parser('latency')
    lt.add_argument('--dataset', required=True)
    lt.add_argument('--root', required=True)
    lt.add_argument('--grid', action='append', required=True, help='V1[:V2]=predictions.jsonl')
    lt.add_argument('--params', default='hbm_latency_cycles', help='the fitted keys, comma separated (one or two)')
    lt.add_argument('--objective', choices=('sec34', 'median_abs'), default='sec34')
    lt.add_argument('--out', required=True)
    sh = sub.add_parser('shift')
    sh.add_argument('--dataset', required=True)
    sh.add_argument('--root', required=True)
    sh.add_argument('--base', required=True, help='F0=predictions.jsonl (slow path, calibration)')
    sh.add_argument('--slopes', required=True, help='an earlier latency_fit.json (fast-path grid, calibration-only)')
    sh.add_argument('--slope-between', default='20:35', help='hbm_latency_cycles values the slope is taken between')
    sh.add_argument('--slope-second', type=int, default=13, help='tl_outlet_to_slice_cycles at which the slope is taken')
    sh.add_argument('--param', default='dram_fixed_cycles')
    sh.add_argument('--span', type=int, default=15)
    sh.add_argument('--out', required=True)
    r = sub.add_parser('report')
    r.add_argument('--config', required=True)
    r.add_argument('--host', required=True)
    r.add_argument('--latency', required=True)
    r.add_argument('--sources', help='JSON {key: provenance text} for the structural constants')
    r.add_argument('--excluded', action='append', help='calibration configurations not in the fitting grid (cost)')
    r.add_argument('--note', action='append', help='a procedure line to record (e.g. values kept from an earlier fit)')
    r.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(host=host, latency=latency, shift=shift, report=report)[a.cmd](a)


if __name__ == '__main__':
    main()
