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
               one instruction per cycle) and the matrix engine (VLEN MACs,
               rtl/core/tnca_matrix.v "16 MACs: one output column each").
               A dense MAC can run on either, so a strict bound may only
               assume their sum.
    mma_mac    matrix-engine MACs per cycle. Used by the algorithm-scoped
               bound only ("GEMM served by the matrix unit").
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


def spec_peak(profile='core-sim', *, target=None, overrides=None):
    """Per-cycle service rates for one single-core profile.

    `overrides` exists for the metric's own tests (a candidate that lowers
    its advertised peak); a real configuration's rates come from its
    target file.
    """
    t = target or target_for(profile)
    if t.get('num_cores', 1) != 1:
        raise ValueError(f'{profile}: spec_peak v0 covers single-core profiles only')
    vlen = t['vlen']
    has_sfu = bool(t['capabilities'].get('sfu'))
    peak = dict(
        fp_lane=vlen + vlen,
        mma_mac=vlen,
        int_lane=vlen,
        sfu_lane=SFU_SLICES if has_sfu else 0,
        mem_bytes=MEM_DW_BITS // 8,
        issue=t['issue']['width'],
    )
    peak.update(overrides or {})
    return dict(profile=profile, unit='per-cycle', rates=peak, sources=SOURCES,
                kind='spec_peak')
