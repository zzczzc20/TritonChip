#!/usr/bin/env python3
"""Per-SLR resource preflight of the next implementation's changed RTL (R41 S1-04 B).

    regional_preflight.py --out evidence/r41/stage1/regional_preflight.json

Inputs, all preserved evidence:
  - the A3 routed per-SLR utilization (frequency-preflight-v2 tgz),
  - paired unit OOC utilization, old A3 source vs merged M1-M4
    (evidence/r40/v3/final_image_candidate/ooc_units/units_reports.tgz),
  - the F07 (C1) paired slice OOC (evidence/r41/stage1/f07_ooc/f07-ooc-evidence.tgz).

Each changed unit's synthesized LUT/register delta is multiplied by its
instance count in each SLR of the three-core image (pblocks: one tile per
SLR; outlets and tile 0's host mux in SLR0; memory mode and MMIO with the
control block in SLR1) and added to A3's routed per-SLR counts. Unit
synthesis deltas are not placed results; CLB packing is nonlinear and is
estimated only by scaling with the LUT increase. F01-F06 and F08-F10 are
not implemented, so their cost is unknown and is listed, not estimated.
"""
import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A3 = ROOT / 'evidence/r41/stage1/kickoff/frequency-preflight-v2-evidence.tgz'
UNITS = ROOT / 'evidence/r40/v3/final_image_candidate/ooc_units/units_reports.tgz'
F07 = ROOT / 'evidence/r41/stage1/f07_ooc/f07-ooc-evidence.tgz'
SLRS = ('SLR0', 'SLR1', 'SLR2')
# unit -> instances per SLR in the B40 three-core image
PLACEMENT = {
    'tnca_vector_fpu': {'SLR0': 1, 'SLR1': 1, 'SLR2': 1},        # M4, one per core
    'tnca_line_axi': {'SLR0': 3, 'SLR1': 0, 'SLR2': 0},          # M3, three outlets
    'tnca_memory_mode': {'SLR0': 0, 'SLR1': 1, 'SLR2': 0},       # M3, control block
    'tnca_multicore_mmio': {'SLR0': 0, 'SLR1': 1, 'SLR2': 0},    # M3, control block
    'tnca_idle_host_mux': {'SLR0': 1, 'SLR1': 0, 'SLR2': 0},     # M1, tile 0
    'tnca_nuca_slice': {'SLR0': 1, 'SLR1': 1, 'SLR2': 1},        # F07 (C1), one per tile
}
# A3 routed CLB packing (frequency_preflight.md); LUT/register counts are read from the report
A3_CLB_PACKING = {'SLR0': 95.24, 'SLR1': 88.39, 'SLR2': 85.41}
NOT_IMPLEMENTED = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F08', 'F09', 'F10']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(tgz, suffix):
    with tarfile.open(tgz) as t:
        names = [m for m in t.getnames() if m.endswith(suffix)]
        if len(names) != 1:
            raise SystemExit(f'{tgz}: expected one member ending {suffix}, got {names}')
        return t.extractfile(names[0]).read().decode()


def unit_counts(text):
    out = {}
    for key, label in (('luts', r'CLB LUTs\*?'), ('registers', r'CLB Registers')):
        m = re.search(r'^\|\s*' + label + r'\s*\|\s*(\d+)', text, re.M)
        if not m:
            raise SystemExit(f'no {label} row')
        out[key] = int(m.group(1))
    return out


def slr_counts(text):
    out = {}
    for key, label in (('luts', 'CLB LUTs'), ('registers', 'CLB Registers')):
        m = re.search(r'^\|\s*' + label + r'\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', text, re.M)
        if not m:
            raise SystemExit(f'no per-SLR {label} row')
        v = [float(x) for x in m.groups()]
        out[key] = {s: dict(used=int(v[i]), percent=v[3 + i], capacity=round(v[i] / v[3 + i] * 100)) for i, s in enumerate(SLRS)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    base = slr_counts(member(A3, 'a3-slr-util-routed.rpt'))
    deltas = {}
    for unit in PLACEMENT:
        if unit == 'tnca_nuca_slice':
            old = unit_counts(member(F07, 'out/base/tnca_nuca_slice_util.rpt'))
            new = unit_counts(member(F07, 'out/c1/tnca_nuca_slice_util.rpt'))
            src = 'f07'
        else:
            old = unit_counts(member(UNITS, f'old/{unit}/{unit}_util.rpt'))
            new = unit_counts(member(UNITS, f'new/{unit}/{unit}_util.rpt'))
            src = 'm1_m4_units'
        deltas[unit] = dict(source=src, old=old, new=new,
                            delta={k: new[k] - old[k] for k in old}, instances=PLACEMENT[unit])
    slr = {}
    for s in SLRS:
        d = {k: sum(u['delta'][k] * u['instances'][s] for u in deltas.values()) for k in ('luts', 'registers')}
        lut, reg = base['luts'][s], base['registers'][s]
        slr[s] = dict(
            a3_routed=dict(luts=lut['used'], lut_percent=lut['percent'], registers=reg['used'],
                           register_percent=reg['percent'], clb_packing_percent=A3_CLB_PACKING[s]),
            delta=d,
            estimate=dict(luts=lut['used'] + d['luts'],
                          lut_percent=round((lut['used'] + d['luts']) / lut['capacity'] * 100, 3),
                          registers=reg['used'] + d['registers'],
                          register_percent=round((reg['used'] + d['registers']) / reg['capacity'] * 100, 3),
                          clb_packing_percent_scaled=round(A3_CLB_PACKING[s] * (lut['used'] + d['luts']) / lut['used'], 3)))
    doc = dict(schema='tnca-r41-regional-preflight/1',
               inputs={str(p.relative_to(ROOT)): sha(p) for p in (A3, UNITS, F07)},
               placement=PLACEMENT, units=deltas, slr=slr, not_implemented_candidates=NOT_IMPLEMENTED,
               boundary='unit synthesis deltas added to A3 routed counts; not a placed, packed or routed result; '
                        'CLB packing scaled linearly with LUTs, which understates packing nonlinearity; '
                        'F01-F06 and F08-F10 have no implementation and no estimate')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + '\n')
    for s in SLRS:
        e = slr[s]
        print(f"{s}: dLUT {e['delta']['luts']:+d} dFF {e['delta']['registers']:+d}  "
              f"LUT {e['a3_routed']['lut_percent']}% -> {e['estimate']['lut_percent']}%  "
              f"CLB packing {e['a3_routed']['clb_packing_percent']}% -> ~{e['estimate']['clb_packing_percent_scaled']}%")


if __name__ == '__main__':
    main()
