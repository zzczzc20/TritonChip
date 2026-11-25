"""spec_peak: the most a configuration can serve per cycle, per resource.

roadmap section 4.5 keeps two sets of capacities apart. This file is the
first one only -- what the implementation and its configuration allow --
and it is never derived from a measurement. `empirical_sustained` belongs
to a microbenchmark and is not here.

Each rate carries the RTL fact it comes from. A rate that is wrong in the
generous direction makes a lower bound too small, which is safe and
merely loose; a rate wrong in the stingy direction makes the bound too
large and shows up as eta > 1, which score.py refuses to clip.

Resources:

    fp_lane    binary32 multiply/add lanes per cycle, counting EVERY unit
               that can execute one: the vector FPU (VLEN lanes, pipelined,
               one instruction per cycle) and the matrix engine (mma_mac).
               A dense MAC can run on either, so a strict bound may only
               assume their sum.
    mma_mac    matrix-engine MACs per cycle: VLEN MACs (rtl/core/tnca_matrix.v
               "16 MACs: one output column each"), times the group of
               products each forms per cycle on a target with the matrix_dp
               capability, which depends on the operand format (mma_rate,
               keyword `mma_dtype`). Enters fp_lane, and the algorithm-scoped
               bound alone ("GEMM served by the matrix unit").
    int_lane   integer/compare vector lanes (tnca_vector_alu, single cycle).
    sfu_lane   special-function lanes per cycle: one slice, one lane per
               cycle (rtl/core/tnca_sfu.v "ONE SLICE, VLEN PASSES").
               Scoped only: exp/rcp/rsqrt/log2 can also be built from FMAs,
               so the SFU rate is not a strict limit on them.
    mem_bytes  bytes per cycle through the single core memory port.
               rtl/core/tnca_mem_mux.v merges the LSU and the transfer
               engine onto ONE port of MEM_DW = 64 bits, one request per
               cycle.
    issue      instructions per cycle (target file: issue.width).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

MEM_DW_BITS = 64         # rtl/core/tnca_core.v parameter MEM_DW
SFU_SLICES = 1           # rtl/core/tnca_sfu.v: a single four-function slice

SOURCES = {
    'fp_lane': 'rtl/core/tnca_vector_fpu.v (VLEN lanes) + rtl/core/tnca_matrix.v (VLEN MACs)',
    'mma_mac': 'rtl/core/tnca_matrix.v: 16 MACs, one k per 16 cycles',
    'int_lane': 'rtl/core/tnca_vector_alu.v: VLEN lanes, single cycle',
    'sfu_lane': 'rtl/core/tnca_sfu.v: one slice, one lane per cycle',
    'mem_bytes': 'rtl/core/tnca_mem_mux.v: LSU and transfer engine share one MEM_DW port',
    'issue': 'target file issue.width',
}


def target_for(profile):
    spec = json.loads((ROOT / 'spec' / 'r1_profiles.json').read_text())['profiles'][profile]
    return json.loads((ROOT / 'compiler' / 'targets' / spec['target_file']).read_text())


# The matrix engine's MACs per cycle, per operand format, in multiples of
# VLEN (one MAC per output column). Without the matrix_dp capability a MAC
# is a tnca_fpu_multi, one product per cycle for every format: 1x. With it
# each MAC is a tnca_mac_lane executing m.mma.dp.async (ISA sec 9C.4) and
# forms a GROUP of products per cycle -- rtl/core/tnca_matrix.v, parameter
# MAC_DP: "FP32 1x, FP16/BF16/FP8 pairs 2x, E4M3 x E4M3 4x". 'e4m3' means
# both operands E4M3; a mixed FP8 pair is 2x, like e5m2.
MMA_GROUP = {'fp32': 1, 'fp16': 2, 'bf16': 2, 'fp8': 2, 'e5m2': 2, 'e4m3': 4}
MMA_DP_SOURCES = {
    'fp_lane': 'rtl/core/tnca_vector_fpu.v (VLEN lanes) + mma_mac',
    'mma_mac': 'rtl/core/tnca_matrix.v MAC_DP=1: VLEN tnca_mac_lane MACs, each a group of '
               '1 (FP32), 2 (FP16/BF16/FP8 pairs) or 4 (E4M3 x E4M3) products per cycle '
               '(m.mma.dp.async, ISA sec 9C.4)',
}


def mma_rate(target, mma_dtype=None):
    """Matrix-engine MACs per cycle on `target` for one operand format.

    Peaks were per profile only; this keyword is the smallest extension
    that lets an E4M3 point's bound use the 4x rate. `mma_dtype` None is a
    caller without a format (tools/r38/run.py's TileBench sweeps): on a
    matrix_dp target it gets the GENEROUS end, 4x, because a rate too high
    only loosens a bound while one too low shows up as eta > 1, which
    score.py refuses to clip. Without the capability every format is VLEN
    and the peak is exactly what it was.
    """
    if mma_dtype is not None and mma_dtype not in MMA_GROUP:
        raise ValueError(f'no matrix-engine rate for operand format {mma_dtype!r}')
    vlen = target['vlen']
    if not target['capabilities'].get('matrix_dp'):
        return vlen
    return vlen * (MMA_GROUP[mma_dtype] if mma_dtype else max(MMA_GROUP.values()))


def spec_peak(profile='core-sim', *, target=None, overrides=None, mma_dtype=None):
    """Per-cycle service rates for one single-core profile.

    `overrides` exists for the metric's own tests (a candidate that lowers
    its advertised peak); a real configuration's rates come from its
    target file. `mma_dtype` is the matrix operand format of the point
    being bounded ('fp32', 'fp16', 'bf16', 'e5m2', 'e4m3'): it matters only
    on a target with the matrix_dp capability (mma_rate), where the result
    also names the format (`mma_dtype`) and the lane's sources.
    """
    t = target or target_for(profile)
    if t.get('num_cores', 1) != 1:
        raise ValueError(f'{profile}: spec_peak v0 covers single-core profiles only')
    vlen = t['vlen']
    has_sfu = bool(t['capabilities'].get('sfu'))
    mma = mma_rate(t, mma_dtype)
    peak = dict(
        fp_lane=vlen + mma,
        mma_mac=mma,
        int_lane=vlen,
        sfu_lane=SFU_SLICES if has_sfu else 0,
        mem_bytes=MEM_DW_BITS // 8,
        issue=t['issue']['width'],
    )
    peak.update(overrides or {})
    out = dict(profile=profile, unit='per-cycle', rates=peak, sources=SOURCES,
               kind='spec_peak')
    if t['capabilities'].get('matrix_dp'):
        out.update(mma_dtype=mma_dtype, sources=dict(SOURCES, **MMA_DP_SOURCES))
    return out
