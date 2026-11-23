"""The strict lower bound of a point on a platform: L = max over resources of
demand / capacity (roadmap sections 4.3-4.5), with the same rules on every
platform:

  * a capacity is spec_peak -- what the implementation can serve -- never a
    measurement; where a figure is uncertain the generous end is used
  * work several units can execute is divided by their SUMMED rate (a MAC
    can run on a matrix unit or on vector/SIMT lanes), so no algorithm choice
    is assumed
  * a transcendental counts as at least one FP operation (it can be built
    from FMAs), never as a slow special-function op
  * bytes: every required input read once and output written once, through
    the level data must cross on that platform under the timing protocol
    (TNCA: the core's single memory port, writes included -- it has no
    cache; GPU: DRAM, with the cache flushed before every timed call, and
    writes counted only beyond what the write-back cache can hold)

TNCA reuses tools/r38/archbench (bounds.Workload, bounds.lower_bound, peaks):
the torch-level ArchBench and the Stage 1 TileBench scoring compute a TNCA
bound with one formula. Its unit is cycles. The GPU bound is in seconds.
eta = L / T is unitless and comparable across platforms; absolute times are
compared only in seconds and only when both platforms have them.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R38 = ROOT / 'tools' / 'r38' / 'archbench'

VERSION = 'archbench-torch-bounds/2'


def _r38(name):
    """tools/r38/archbench/<name>.py, loaded by path: that package is also
    called `archbench`, and importing it by name would find this one."""
    key = f'_r38_archbench_{name}'
    spec = importlib.util.spec_from_file_location(key, R38 / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_r38_bounds = _r38('bounds')
_r38_peaks = _r38('peaks')
r38_score = _r38('score')


def tnca_peak(profile):
    if profile.endswith('/hbm'):
        from tnca.device import base_profile
        # HBM changes the backing memory, while this profile still has
        # the same single core and its shared 64-bit memory ingress.
        peak = _r38_peaks.spec_peak(base_profile(profile))
        peak['profile'] = profile
        return peak
    return _r38_peaks.spec_peak(profile)


def _fp8_matrix_rate(profile):
    """E4M3 x E4M3 MACs per cycle the matrix engine of `profile` can execute.

    The matrix engine has VLEN = 16 lanes, one output column each
    (rtl/core/tnca_matrix.v). m.mma.async advances one k per lane per cycle:
    16 MAC/cycle in every format. m.mma.dp.async (ISA sec 9C.4, capability
    MATRIX_DP, RTL parameter MAC_DP) advances `kstep` k per lane per cycle:
    4 for E4M3 x E4M3 (the DP4 mode of rtl/core/tnca_mac_lane.v, four exact
    products summed in a fixed-point cascade and rounded once into the
    binary32 accumulator), so 16 x 4 = 64 MAC/cycle. A profile without
    MATRIX_DP (or without MATRIX_FP8) keeps 16. This is the architecture's
    peak, never a measured rate; see archbench/PROTOCOL.md section 4.1.
    """
    if profile.endswith('/hbm'):
        from tnca.device import base_profile
        profile = base_profile(profile)
    caps = _r38_peaks.target_for(profile)['capabilities']
    base = _r38_peaks.spec_peak(profile)['rates']['mma_mac']
    return base * (4 if caps.get('matrix_dp') and caps.get('matrix_fp8') else 1)


def tnca_rates(point, profile):
    """The per-cycle service rates a point's bound is computed from (recorded with every result)."""
    rates = dict(tnca_peak(profile)['rates'])
    if point.demand.macs_fp8 or point.dtype == 'fp8':
        rates['mma_mac_e4m3'] = _fp8_matrix_rate(profile)
    return rates


def _tnca_bound_fp8(point, profile):
    """Strict bound of a point with E4M3 matrix work (xl tier).

    MACs run on the matrix engine at its E4M3 rate or on the vector FPU
    (VLEN FMA lanes); everything else only on the vector FPU:

        compute = max((MAC + other) / (vector + matrix_e4m3), other / vector)

    A point that also has 16-bit MACs (an attention inside an FP8 decoder
    layer) is credited the E4M3 rate for all of them: the bound can only
    get lower, never invalid. Bytes cross the core's one memory port, as for
    every TNCA point.
    """
    r = tnca_rates(point, profile)
    vector = r['fp_lane'] - r['mma_mac']
    d = point.demand
    other = d.fp_ops + d.sfu_ops
    terms = dict(fp_lane=max((d.macs + other) / (vector + r['mma_mac_e4m3']), other / vector),
                 mem_bytes=(d.read_bytes + d.write_bytes) / r['mem_bytes'])
    binding = max(terms, key=terms.get)
    return terms[binding], binding, terms


def tnca_bound(point, profile):
    """(cycles, binding resource, terms) on a TNCA profile."""
    if point.demand.macs_fp8 or point.dtype == 'fp8':
        return _tnca_bound_fp8(point, profile)
    d = point.demand
    w = _r38_bounds.Workload(point.id, point.family, point.kind,
                             fp_ops=d.macs + d.fp_ops, macs=d.macs, sfu_ops=d.sfu_ops,
                             read_bytes=d.read_bytes, write_bytes=d.write_bytes)
    return _r38_bounds.lower_bound(w, tnca_peak(profile))


def gpu_spec(name='gb10'):
    return json.loads((HERE / 'specs' / f'gpu_{name}.json').read_text())


def gpu_bound(point, name='gb10'):
    """(seconds, binding resource, terms) on a GPU spec."""
    r = {k: v['value'] for k, v in gpu_spec(name)['rates'].items()}
    d = point.demand
    if point.dtype == 'fp32':
        # IEEE binary32 only: TF32 is disabled by the protocol, so tensor
        # cores cannot serve these MACs.
        tensor, simt = 0.0, r['fp32_fma']
    elif point.dtype == 'fp8' or d.macs_fp8:
        # xl tier: FP8 tensor cores (a point with 16-bit MACs as well is credited
        # the FP8 rate for all of them, which can only lower the bound).
        tensor, simt = r['tensor_fp8_mac'], r['fp16_fma_simt']
    else:
        tensor = r['tensor_bf16_mac'] if point.dtype == 'bf16' else r['tensor_fp16_mac']
        simt = r['fp16_fma_simt']
    other = d.fp_ops + d.sfu_ops
    terms = {}
    if d.macs or other:
        # MACs may go to tensor cores or SIMT lanes; everything else only to
        # SIMT lanes. The least time any split can take:
        terms['compute'] = max((d.macs + other) / (tensor + simt), other / simt)
    if d.read_bytes or d.write_bytes:
        # Reads must come from DRAM: the cache is flushed (twice its size)
        # before every timed call. Writes need not: a write-back cache can
        # still hold them dirty when the kernel ends, so only the part that
        # cannot fit in it is a necessary DRAM transfer. Counting every write
        # let an RMSNorm whose output fits in L2 beat its "bound" (eta 1.2,
        # bound version 1).
        wb = gpu_spec(name).get('caches', {}).get('write_back_bytes', 0)
        terms['dram_bytes'] = (d.read_bytes + max(0.0, d.write_bytes - wb)) / r['dram_bytes']
    binding = max(terms, key=terms.get)
    return terms[binding], binding, terms
