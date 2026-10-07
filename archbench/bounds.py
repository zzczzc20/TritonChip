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
    return _r38_peaks.spec_peak(profile)


def tnca_bound(point, profile):
    """(cycles, binding resource, terms) on a TNCA profile."""
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
