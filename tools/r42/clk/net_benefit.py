#!/usr/bin/env python3
"""net_benefit.py  -- derived: wall-time speed-up (S145) of a tile OOC period through the TLM clock curve, minus the cycle price of the fixes.

Inputs (recorded): TLM clock curve of R42-shrink RESULTS section 4, S145 = 2.032 / 2.448 / 2.863 at 100 / 125 / 150 MHz with the nine path fixes at +1 cycle each (-1.9 %);
today's hardware at 100 MHz = 2.072 (so the curve without the fixes' cycles is the listed values / 0.981). The image period is the OOC tile period times a derate d (1.2-1.4: the
same families are 1.4-1.6 times longer in the run-3 image; the shrunk tile packs SLR0 less); memory latency fixed in ns is inside the TLM curve.
Cycle price of this worker's bits (TLM, 77 points): PICK_REG 0.4 %, S2 split 1.55 % (stand-in tiny points: 0.32 % and 5.4 %); all other bits 0.
"""
import sys

curve = [(100.0, 2.032), (125.0, 2.448), (150.0, 2.863)]
nofix = [(f, s / 0.981) for f, s in curve]


def s145(mhz, pts):
    if mhz <= pts[0][0]:
        return pts[0][1]
    for (f0, s0), (f1, s1) in zip(pts, pts[1:]):
        if mhz <= f1:
            return s0 + (mhz - f0) / (f1 - f0) * (s1 - s0)
    f0, s0 = pts[-2]
    f1, s1 = pts[-1]
    return s1 + (mhz - f1) / (f1 - f0) * (s1 - s0)


price = {'tlm': 0.0195, 'standin': 0.0032 + 0.054}
print('| OOC tile period | derate | image period | MHz | S145 clock only | S145 minus cycle price (TLM %.2f %%) | (stand-in tiny price %.1f %%) | gain over 2.072 (TLM price) |' % (100 * price['tlm'], 100 * price['standin']))
print('|---|--:|--:|--:|--:|--:|--:|--:|')
for per in (6.67, 6.2, 5.6):
    for d in (1.2, 1.3, 1.4):
        img = per * d
        mhz = 1000.0 / img
        base = s145(mhz, nofix)
        a = base * (1 - price['tlm'])
        b = base * (1 - price['standin'])
        print('| %.2f ns | %.1f | %.2f ns | %.1f | %.3f | %.3f | %.3f | %+.1f %% |' % (per, d, img, mhz, base, a, b, 100 * (a / 2.072 - 1)))
