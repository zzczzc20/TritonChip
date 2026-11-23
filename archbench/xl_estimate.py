"""Device-time estimates of the xl tier: planning figures recorded in archbench/suites/xl.json, never results.

The brief (2026-11-23) asked for the device time of each point at the CURRENT baseline -- the R41 run-3 three-core
image, 100 MHz, serial-grid cycles -- estimated as the strict bound divided by an efficiency eta, so that no point
exceeds about 12 minutes and a pass about 4 hours:

    estimate = sum over the parts of a point of  bound_part / eta_part / 100 MHz

  * the bound is archbench.bounds.tnca_bound on the profile of the image's cores
    (core-vrf-sync-v5-x8-g2c4-dp), computed from the semantic demand;
  * eta of an fp16 part is that of its analogue in the 113 original points on the board: the same family's `large`
    point in evidence/r42/baseline/b41_board.json (cycles of the run-3 image) over its bound (archbench/suites/
    xl_eta.json, written by tools/r42/xl_eta.py from those files);
  * eta of an FP8 matrix part is the 0.2 of the brief: no FP8 point has been run on a board. A pessimistic column uses
    eta = bound / cycles of the SystemC/TLM model on the four real-layer shapes of evidence/r41/fp8/kernels/README.md
    (an uncalibrated model of the earlier, non-dp profile, domain `model`), which is far lower.

A composite point (an FP8 layer or decoder) is split into the parts of xl.COMPONENTS (FP8 projections, attention,
element-wise kernels, head) because their efficiencies differ by an order of magnitude; the point's own bound
(`bound_cycles`) is the strict bound of the whole demand and is what a result's eta is measured against.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from .bounds import tnca_bound

PROFILE = 'core-vrf-sync-v5-x8-g2c4-dp'
CLOCK_HZ = 100e6
ETA_FILE = Path(__file__).resolve().parent / 'suites' / 'xl_eta.json'
# an xl fp16 family -> the family of the 113 original points whose board efficiency it takes
FAMILY_ETA = {'attention_prefill': 'attention_prefill', 'attention_decode': 'attention_decode',
              'rmsnorm_residual': 'rmsnorm', 'swiglu': 'swiglu', 'rope': 'rope', 'gemv': 'gemv', 'linear': 'linear',
              'embedding': 'embedding', 'argmax': 'argmax'}


def load_eta():
    return json.loads(ETA_FILE.read_text())


def _part_eta(table, point, part):
    """(eta, pessimistic eta, basis) of one part of a point."""
    fam = table['fp16_families']
    if part == 'fp8_projections' or (part is None and point.dtype == 'fp8'):
        return (table['fp8']['primary'], table['fp8']['pessimistic']['eta'],
                'fp8 matrix: eta 0.2 (brief); pessimistic: TLM model of the Hub kernel')
    if part == 'attention':
        key = 'attention_prefill' if 'prefill' in point.family else 'attention_decode'
    elif part == 'elementwise':
        key = 'rmsnorm'
    elif part == 'head':
        key = 'gemv'
    else:
        key = FAMILY_ETA[point.family]
    return fam[key]['eta'], fam[key]['eta'], f'board eta of the {key} point of tier large (run-3 image)'


def estimate(point, table=None):
    """The planning estimate of `point`: bound, parts, seconds at 100 MHz, primary and pessimistic."""
    from .xl import COMPONENTS
    table = table or load_eta()
    L, binding, _ = tnca_bound(point, PROFILE)
    parts = []
    comps = COMPONENTS.get(point.id)
    for name, d in (comps or {None: point.demand}).items():
        shim = SimpleNamespace(id=point.id, family=point.family, kind=point.kind, demand=d,
                               dtype='fp8' if d.macs_fp8 else 'fp16')
        Lp, bp, _ = tnca_bound(shim, PROFILE)
        eta, eta_pess, basis = _part_eta(table, point, name)
        parts.append(dict(part=name or point.family, bound_cycles=Lp, binding=bp, eta=eta, eta_pessimistic=eta_pess,
                          basis=basis))
    cycles = sum(p['bound_cycles'] / p['eta'] for p in parts)
    cycles_pess = sum(p['bound_cycles'] / p['eta_pessimistic'] for p in parts)
    return dict(bound_cycles=L, binding=binding, parts=parts, cycles=cycles, seconds=cycles / CLOCK_HZ,
                cycles_pessimistic=cycles_pess, seconds_pessimistic=cycles_pess / CLOCK_HZ,
                eta_equivalent=L / cycles)
