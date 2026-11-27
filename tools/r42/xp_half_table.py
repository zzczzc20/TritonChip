#!/usr/bin/env python3
"""Per-K-step cycles of one operand from run_xp_half.sh output (rtl-eng: tnca_tensor alone, fixed memory latency).
step cycles = (cycles at XN=8 - cycles at XN=4) / 4. Columns: word engine, line engine, line engine with LINE_HALF."""
import re, sys, collections
rows = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    m = re.match(r"XFER LHALF=(\d) XP=1 KS=(\d+) XB=(\d) XM=(\d+) XN=(\d+) fixlat=(\d+) word_engine_cycles=(\d+) line_engine_cycles=(\d+)", l)
    if not m: continue
    lh, ks, xb, xm, xn, lat, w, ln = map(int, m.groups())
    rows[(ks, xb, xm, lat)][(lh, xn)] = (w, ln)
name = {(2, 1): "A tile 16 x KS (plain)", (1, 0): "B tile KS x 16 (plain)", (0, 0): "W^T tile (transposed)", (0, 1): "W^T + A (one GEMM step)"}
lats = sorted({k[3] for k in rows})
print("operand                    KS   " + "   ".join(f"lat={l:<2d} word/line/half" for l in lats))
for (xb, xm) in ((2, 1), (1, 0), (0, 0), (0, 1)):
    for ks in (16, 32):
        cells = []
        for lat in lats:
            r = rows.get((ks, xb, xm, lat), {})
            try:
                w = (r[(0, 8)][0] - r[(0, 4)][0]) / 4
                l0 = (r[(0, 8)][1] - r[(0, 4)][1]) / 4
                l1 = (r[(1, 8)][1] - r[(1, 4)][1]) / 4
                cells.append(f"{w:5.0f} / {l0:4.0f} / {l1:4.0f}  ")
            except KeyError:
                cells.append("-".ljust(22))
        print(f"{name[(xb, xm)]:26s} {ks:3d}   " + "   ".join(cells))
