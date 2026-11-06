"""Roofline figure and efficiency table from ArchBench records.

Axes: x = arithmetic intensity (FLOP/byte of required traffic, the ArchBench
`demand`), y = achieved FLOP/s = demand FLOP / device time. FLOP counts 2 per
MAC, plus fp_ops, plus sfu_ops (a transcendental counts as at least one FP
operation, ArchBench bounds rule). Bytes are required reads + writes.

Rooflines are the ArchBench spec peaks, never measurements:
  * TNCA: the strict-bound peaks of `bound.peak` (per core): the fp lanes
    (vector FPU + matrix MACs, summed) and the core's single memory port.
  * GB10: spec peaks (archbench/specs/gpu_gb10.json).
eta = bound / time is the platform-neutral efficiency (1.0 = on the bound).

    python roofline_report.py --tnca100 FORMAL.jsonl --tnca90 DIR_OR_GZ... \
        --gpu RAW.jsonl... --out OUTDIR
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median


def read(paths):
    rows = []
    for p in paths:
        p = Path(p)
        opener = gzip.open if p.suffix == '.gz' else open
        with opener(p, 'rt') as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def point(r, kind):
    d = r.get('demand') or {}
    flop = 2 * d.get('macs', 0) + d.get('fp_ops', 0) + d.get('sfu_ops', 0)
    byts = d.get('read_bytes', 0) + d.get('write_bytes', 0)
    t = r['time']
    if kind == 'tnca':
        sec = t['device'] / r['platform']['clock_hz']
    else:
        sec = t['device']
    if r.get('status') != 'pass' or not sec or not flop or not byts:
        return None
    return dict(id=r['point_id'], family=r['family'], tier=r['tier'], dtype=r['dtype'],
                flop=flop, bytes=byts, sec=sec, perf=flop / sec, ai=flop / byts,
                eta=r.get('eta'), binding=(r.get('bound') or {}).get('binding'))


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float('nan')


def summarize(name, pts, peak_flops, peak_bw):
    etas = [p['eta'] for p in pts if p['eta']]
    ridge = peak_flops / peak_bw
    mem = [p for p in pts if p['ai'] < ridge]
    cmp_ = [p for p in pts if p['ai'] >= ridge]
    # distance below the roofline: roof(ai) / perf
    for p in pts:
        roof = min(peak_flops, peak_bw * p['ai'])
        p['roof'] = roof
        p['gap'] = roof / p['perf']
    return dict(name=name, n=len(pts), ridge_flop_per_byte=ridge,
                eta_geomean=geomean(etas), eta_median=median(etas) if etas else None,
                eta_max=max(etas) if etas else None,
                gap_geomean=geomean([p['gap'] for p in pts]),
                memory_side_points=len(mem), compute_side_points=len(cmp_),
                gap_geomean_memory_side=geomean([p['gap'] for p in mem]),
                gap_geomean_compute_side=geomean([p['gap'] for p in cmp_]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tnca100', nargs='+', required=True)
    ap.add_argument('--tnca90', nargs='+', required=True)
    ap.add_argument('--gpu', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t100 = [p for p in (point(r, 'tnca') for r in read(a.tnca100)) if p]
    t90 = [p for p in (point(r, 'tnca') for r in read(a.tnca90)) if p]
    graw = read(a.gpu)
    gpts = [p for p in (point(r, 'gpu') for r in graw) if p]
    g0 = next(r for r in graw if r.get('bound'))
    gpk = g0['bound']['peak']
    r0 = next(r for r in read(a.tnca100) if r.get('bound'))
    tpk = r0['bound']['peak']

    # peaks, FLOP/s and B/s
    clk100, clk90 = 100e6, 90e6
    tnca100 = (2 * tpk['fp_lane'] * clk100, tpk['mem_bytes'] * clk100)
    tnca90 = (2 * tpk['fp_lane'] * clk90, tpk['mem_bytes'] * clk90)
    # Three-core device ceilings (board: 3 cores, 3 AXI outlets of 256 bit, 32 HBM2 pseudo-channels)
    tnca3 = (3 * tnca100[0], 3 * tnca100[1])            # 3 cores x single-core port
    axi3 = 3 * 32 * clk100                                # 3 outlets x 32 B/cycle
    hbm = 460e9                                           # U280 HBM2: 32 PC x 14.4 GB/s
    gpu_simt = (2 * gpk['fp32_fma'], gpk['dram_bytes'])
    gpu_tc = (2 * gpk['tensor_fp16_mac'], gpk['dram_bytes'])

    rep = dict(
        generated_by='tools/performance_harness/roofline_report.py',
        axes='x: FLOP/byte (required traffic); y: achieved FLOP/s = demand FLOP / device time',
        peaks=dict(tnca_100MHz=dict(flops=tnca100[0], bytes_per_s=tnca100[1]),
                   tnca_90MHz=dict(flops=tnca90[0], bytes_per_s=tnca90[1]),
                   tnca_3core_100MHz=dict(flops=tnca3[0], core_port_bytes_per_s=tnca3[1],
                                          axi_outlets_bytes_per_s=axi3, hbm_theoretical_bytes_per_s=hbm),
                   gb10_fp32_simt=dict(flops=gpu_simt[0], bytes_per_s=gpu_simt[1]),
                   gb10_fp16_tensor=dict(flops=gpu_tc[0], bytes_per_s=gpu_tc[1])),
        results=[summarize('TNCA 100 MHz (Stage 2 image, board)', t100, *tnca100),
                 summarize('TNCA 90 MHz (R40 delivery image, board)', t90, *tnca90),
                 summarize('GB10 (CUDA graph, fp32 SIMT roof)', gpts, *gpu_simt)])
    fam = {}
    for p in t100:
        fam.setdefault(p['family'], []).append(p)
    rep['tnca100_by_family'] = {k: dict(n=len(v), eta_geomean=geomean([p['eta'] for p in v if p['eta']]),
                                        gap_geomean=geomean([p['gap'] for p in v]))
                                for k, v in sorted(fam.items())}
    famg = {}
    for p in gpts:
        famg.setdefault(p['family'], []).append(p)
    rep['gb10_by_family'] = {k: dict(n=len(v), eta_geomean=geomean([p['eta'] for p in v if p['eta']]))
                             for k, v in sorted(famg.items())}
    t100d = {p['id']: p for p in t100}
    gd = {p['id']: p for p in gpts}
    both = sorted(set(t100d) & set(gd))
    rep['paired'] = dict(n=len(both),
                         tnca_eta_geomean=geomean([t100d[i]['eta'] for i in both if t100d[i]['eta']]),
                         gb10_eta_geomean=geomean([gd[i]['eta'] for i in both if gd[i]['eta']]),
                         tnca_over_gb10_eta_ratio_geomean=geomean(
                             [t100d[i]['eta'] / gd[i]['eta'] for i in both
                              if t100d[i]['eta'] and gd[i]['eta']]),
                         gb10_time_over_tnca_time_geomean=geomean(
                             [gd[i]['sec'] / t100d[i]['sec'] for i in both]))
    (out / 'roofline.json').write_text(json.dumps(rep, indent=1, default=float))
    (out / 'points.json').write_text(json.dumps(dict(tnca100=t100, tnca90=t90, gb10=gpts), indent=0, default=float))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a0 = ax[0]
    xs = np.logspace(-3, 4, 200)

    def roof(ax_, peak, color, label, ls='-'):
        ax_.plot(xs, np.minimum(peak[0], peak[1] * xs), color=color, lw=2.2, ls=ls, label=label)

    roof(a0, tnca100, '#c0392b', f'TNCA roofline @100 MHz  ({tnca100[0]/1e9:.1f} GFLOP/s, {tnca100[1]/1e9:.1f} GB/s)')
    roof(a0, tnca90, '#e67e22', f'TNCA roofline @90 MHz', ls='--')
    roof(a0, (tnca3[0], tnca3[1]), '#8e44ad', f'TNCA 3 cores @100 MHz  ({tnca3[0]/1e9:.1f} GFLOP/s, core ports {tnca3[1]/1e9:.1f} GB/s)', ls='-.')
    a0.plot(xs, np.minimum(tnca3[0], axi3 * xs), color='#16a085', lw=1.6, ls=':', label=f'TNCA 3 cores, AXI outlets ceiling ({axi3/1e9:.1f} GB/s)')
    a0.plot(xs, np.minimum(tnca3[0], hbm * xs), color='#7f8c8d', lw=1.6, ls=':', label=f'U280 HBM2 theoretical ({hbm/1e9:.0f} GB/s)')
    roof(a0, gpu_simt, '#1f4e79', f'GB10 roofline, fp32 SIMT  ({gpu_simt[0]/1e12:.1f} TFLOP/s, {gpu_simt[1]/1e9:.0f} GB/s)')
    roof(a0, gpu_tc, '#2e86c1', f'GB10 roofline, fp16 tensor  ({gpu_tc[0]/1e12:.0f} TFLOP/s)', ls='--')
    a0.scatter([p['ai'] for p in gpts], [p['perf'] for p in gpts], s=16, c='#1f4e79', alpha=.55, label=f'GB10 measured ({len(gpts)} pts)')
    a0.scatter([p['ai'] for p in t90], [p['perf'] for p in t90], s=16, c='#e67e22', alpha=.5, marker='s', label=f'TNCA 90 MHz measured ({len(t90)} pts)')
    a0.scatter([p['ai'] for p in t100], [p['perf'] for p in t100], s=20, c='#c0392b', alpha=.75, marker='o', label=f'TNCA 100 MHz measured ({len(t100)} pts)')
    a0.set_xscale('log'); a0.set_yscale('log')
    a0.set_xlim(1e-2, 1e4); a0.set_ylim(1e6, 3e14)
    a0.set_xlabel('Arithmetic intensity (FLOP per byte of required traffic)')
    a0.set_ylabel('Achieved performance (FLOP/s)')
    a0.set_title('Roofline: achieved performance of the 113-point ArchBench suite')
    a0.grid(True, which='both', alpha=.25)
    a0.legend(fontsize=8, loc='lower right')

    a1 = ax[1]
    data, labels, cols = [], [], []
    for lab, pts, c in (('TNCA\n100 MHz', t100, '#c0392b'), ('TNCA\n90 MHz', t90, '#e67e22'), ('GB10', gpts, '#1f4e79')):
        data.append([p['eta'] for p in pts if p['eta']]); labels.append(lab); cols.append(c)
    bp = a1.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True, widths=.55)
    for b, c in zip(bp['boxes'], cols):
        b.set_facecolor(c); b.set_alpha(.45)
    a1.set_yscale('log'); a1.set_ylim(5e-4, 8)
    a1.set_ylabel('eta = ArchBench bound / measured time (1.0 = on the bound)')
    a1.axhline(1.0, color='k', lw=.8, ls=':')
    for i, d in enumerate(data, 1):
        a1.text(i, max(d) * 1.25, f'geomean {geomean(d):.3f}\nn={len(d)}', ha='center', fontsize=8)
    a1.set_title('Efficiency against each platform\'s own spec bound')
    a1.grid(True, axis='y', which='both', alpha=.25)
    fig.suptitle('TNCA (3-core U280) vs NVIDIA GB10: distance from the roofline', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'roofline.png', dpi=150)
    print(json.dumps(rep['results'], indent=1, default=float))
    print(json.dumps(rep['paired'], indent=1, default=float))


if __name__ == '__main__':
    main()
