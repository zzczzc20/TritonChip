#!/usr/bin/env python3
"""The rounding floors behind the xl tolerances: CPU only, no device.

    xl_tolerance.py [--point ID ...] [--seeds 0 1 2] [--out FILE.json]

For each point it evaluates the binary64 reference (`Point.reference`, exact block-wise FP8 arithmetic, fp16 at stage
boundaries) and the module's CPU twin in `emu` mode (fp16 eager everywhere, binary32 block products: what a device that
rounds every stage to fp16 computes), then reports the criterion of PROTOCOL.md section 2 between them,
max|emu - ref| / max(max|ref|, 1), per seed. That number is the disagreement of two correct implementations, so a
tolerance below it would reject correct devices; the tolerance of a point is a multiple of it (PROTOCOL.md 4.2), fixed
before any device output exists. Points without a reference of their own (the fp16 ones) use the registry's rule.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--point', nargs='+')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    import torch
    from archbench import registry, xl
    from archbench.platforms.common import compare, flatten
    pts = [registry.get(i) for i in a.point] if a.point else [p for p in registry.points(tier='xl') if p.reference]
    rows = []
    for p in pts:
        for seed in a.seeds:
            t0 = time.monotonic()
            m, ins = p.make(seed)
            m.eval()
            ref = tuple(t.detach().cpu() for t in flatten(p.reference(m, ins)))
            t1 = time.monotonic()
            with xl.mode('emu'), torch.no_grad():
                got = flatten(m(*ins))
            t2 = time.monotonic()
            c = compare(got, ref, 1e9)
            per = [compare([g], [r], 1e9)['max_rel_err'] for g, r in zip(got, ref)]
            rows.append(dict(point=p.id, seed=seed, max_rel_err=c['max_rel_err'], per_output=per,
                             tolerance=p.tolerance, reference_s=round(t1 - t0, 1), emulation_s=round(t2 - t1, 1)))
            print(f'{p.id:56s} seed {seed}: emu vs ref {c["max_rel_err"]:.3e} per output {[f"{x:.2e}" for x in per]} '
                  f'(tolerance {p.tolerance}; ref {t1 - t0:.0f}s, emu {t2 - t1:.0f}s)', flush=True)
            if a.out:
                a.out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
