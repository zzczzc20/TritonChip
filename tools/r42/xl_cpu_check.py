#!/usr/bin/env python3
"""CPU numerical check of the smallest xl point of every family (no device, no board).

    xl_cpu_check.py [--out FILE.json] [--point ID ...]

An fp16 point: the module run eagerly in fp16 on the CPU against the registry reference (the same module in binary32),
the criterion of PROTOCOL.md section 2 against the point's tolerance. An FP8 point: the module's CPU twin (`emu` mode,
fp16 eager with binary32 block products) against the point's binary64 reference. (The device path itself calls the Hub
kernels, which only a Triton backend runs.) Records the peak resident memory of the process.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def smallest(points):
    by = {}
    for p in points:
        if p.family not in by or p.extra['memory']['live_bytes'] < by[p.family].extra['memory']['live_bytes']:
            by[p.family] = p
    return [by[f] for f in sorted(by)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--point', nargs='+')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--list-smallest', action='store_true', help='print the id of the smallest point of each family')
    a = ap.parse_args()
    import torch
    from archbench import registry, xl
    from archbench.platforms.common import compare, flatten, reference
    pts = [registry.get(i) for i in a.point] if a.point else smallest(registry.points(tier='xl'))
    if a.list_smallest:
        print('\n'.join(p.id for p in pts))
        return 0
    rows = []
    for p in pts:
        t0 = time.monotonic()
        ref = reference(p, 0)
        t1 = time.monotonic()
        m, ins = p.make(0)
        m.eval()
        with torch.no_grad(), (xl.mode('emu') if p.reference else xl.mode('device')):
            got = m(*ins)
        t2 = time.monotonic()
        c = compare(got, ref, p.tolerance)
        row = dict(point=p.id, family=p.family, path='emu vs binary64 reference' if p.reference else 'fp16 eager vs binary32 module',
                   ok=c['ok'], max_rel_err=c['max_rel_err'], tolerance=p.tolerance, detail=c['detail'],
                   reference_s=round(t1 - t0, 1), module_s=round(t2 - t1, 1),
                   peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
        rows.append(row)
        print(f'{"PASS" if c["ok"] else "FAIL"} {p.id:58s} err {c["max_rel_err"]:.3e} tol {p.tolerance}  '
              f'[{row["path"]}; ref {row["reference_s"]}s, module {row["module_s"]}s, rss {row["peak_rss_gib"]} GiB] {c["detail"]}',
              flush=True)
        if a.out:
            a.out.write_text(json.dumps(rows, indent=1))
    return 0 if all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
