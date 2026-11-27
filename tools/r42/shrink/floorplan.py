#!/usr/bin/env python3
"""floorplan.py [TILE_SYNTH_LUT ...]  (R42-shrink, derived arithmetic, no tool run)

How many tiles fit the three SLRs of the U280 for a given tile size. Inputs are the recorded figures of the area worker's report section C
(evidence/r42/area/REPORT.md): LUT capacity at the highest CLB packing that ever routed (96.74 %: SLR0 325.9k, SLR1 361.0k, SLR2 319.1k),
non-core LUT of the run-2 link (72.3k, 85.2k, 37.6k: HBM subsystem, outlets, shell, mesh, in SLR0 mostly), routed tile = 0.983 x synthesis tile.
Each core beyond the third adds one HBM outlet in SLR0 (1.5k LUT with AXI_LINE_LUTRAM, `evidence/r42/dse/lever_costs.txt`); mesh routers beyond the
three present are not costed anywhere (stated, not assumed away): OUTLET_LUT is a parameter.
"""
import sys

CAP = [325.9e3, 361.0e3, 319.1e3]
NONCORE = [72.3e3, 85.2e3, 37.6e3]
ROUTED_PER_SYNTH = 0.983
OUTLET_LUT = 1.5e3


def fit(tile_synth, outlet=OUTLET_LUT):
    t = tile_synth * ROUTED_PER_SYNTH
    room = [c - n for c, n in zip(CAP, NONCORE)]
    # fill greedily: SLR2 and SLR1 first (more room), SLR0 last (it also pays the extra outlets)
    n2 = int(room[2] // t)
    n1 = int(room[1] // t)
    n0 = 0
    while True:
        extra = max(0, (n0 + n1 + n2 + 1) - 3)         # cores beyond the three of the image need an outlet each in SLR0
        if (n0 + 1) * t + extra * outlet <= room[0]:
            n0 += 1
        else:
            break
    return n0 + n1 + n2, (n0, n1, n2), room


def need_for(n_total, outlet=OUTLET_LUT):
    """largest tile (synthesis LUT) that allows n_total tiles, by bisection"""
    lo, hi = 50e3, 400e3
    for _ in range(60):
        mid = (lo + hi) / 2
        if fit(mid, outlet)[0] >= n_total:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == '__main__':
    sizes = [float(a) for a in sys.argv[1:]] or [253898, 248890, 240000, 230000, 220000, 200000, 180000, 160000, 140000, 127000]
    print('tile (synthesis LUT) -> routed -> cores that fit (SLR0, SLR1, SLR2)')
    for s in sizes:
        n, per, room = fit(s)
        print(f'{s/1e3:8.1f}k -> {s*ROUTED_PER_SYNTH/1e3:7.1f}k -> {n} cores {per}')
    print('largest tile (synthesis LUT) for N tiles:')
    for n in (3, 4, 5, 6, 7, 8):
        print(f'  {n} cores: {need_for(n)/1e3:7.1f}k')
