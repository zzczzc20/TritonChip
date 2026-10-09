"""Necessary-resource lower bounds: roadmap sections 4.3, 4.4 and 4.10.5.

A Workload is the SEMANTIC demand of one point: how many useful binary32
operations, special-function evaluations, integer/compare lanes and
required bytes it needs. It is computed from the operator's shape and
never from a compiled program, so a candidate that spills, pads or reads
twice cannot raise its own bound (appendix A test 3).

Two bounds are kept apart:

    strict   every term is a necessary condition on ANY program for this
             machine: work that several units can execute is divided by
             their summed rate, and a transcendental counts as at least one
             FP lane operation (it can be built from FMAs).
    scoped   an algorithm-scoped bound (dense MACs on the matrix unit,
             transcendentals on the SFU). Useful for diagnosis, not for
             the main efficiency table.

What is deliberately NOT a term: register-file traffic (depends on
bypass and operand holding, section 4.3), a dependency-chain latency we
cannot prove (`D` stays None), and sort/gather/attention work whose
algorithm-independent minimum v0 does not establish -- those raise
BoundUnavailable instead of inventing a number.
"""
from dataclasses import dataclass, field
import math

DTYPE_BYTES = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'e4m3': 1, 'e5m2': 1, 'fp8': 1,
               'i8': 1, 'i32': 4, 'i64': 8}


class BoundUnavailable(Exception):
    """No legal, informative lower bound exists for this workload in v0."""


@dataclass
class Workload:
    id: str
    operator: str
    category: str
    fp_ops: float = 0            # binary32 multiply/add lane operations (a MAC is ONE)
    macs: float = 0              # the subset of fp_ops that are dense MACs
    sfu_ops: float = 0           # exp/log/rcp/rsqrt evaluations
    int_ops: float = 0           # integer/compare lane operations
    read_bytes: float = 0        # each input byte the result depends on, once
    write_bytes: float = 0       # each output byte, once
    dep_cycles: float = None     # proven dependency-chain bound, or None
    bound_available: bool = True
    note: str = ''
    meta: dict = field(default_factory=dict)

    @property
    def required_bytes(self):
        return self.read_bytes + self.write_bytes


def _terms(w, rates, scoped):
    terms = {}
    if w.fp_ops or w.sfu_ops:
        # A transcendental is at least one FP lane op however it is built.
        fp = w.fp_ops + (0 if scoped else w.sfu_ops)
        if scoped and w.macs:
            # Scoped: MACs on the matrix unit, the rest on the vector FPU.
            terms['mma_mac'] = w.macs / rates['mma_mac']
            rest = w.fp_ops - w.macs
            if rest:
                terms['fp_lane'] = rest / (rates['fp_lane'] - rates['mma_mac'])
        elif fp:
            terms['fp_lane'] = fp / rates['fp_lane']
    if scoped and w.sfu_ops:
        terms['sfu_lane'] = (w.sfu_ops / rates['sfu_lane']) if rates['sfu_lane'] else math.inf
    if w.int_ops:
        terms['int_lane'] = w.int_ops / rates['int_lane']
    if w.required_bytes:
        terms['mem_bytes'] = w.required_bytes / rates['mem_bytes']
    if w.dep_cycles is not None:
        terms['dependency'] = w.dep_cycles
    return terms


def lower_bound(w, peak, *, scoped=False):
    """L = max over resources of demand / capacity, in cycles.

    Returns (cycles, binding_resource, terms). Raises BoundUnavailable when
    the workload has no v0 bound.
    """
    if not w.bound_available:
        raise BoundUnavailable(f'{w.id}: {w.note or "no v0 bound for " + w.category}')
    terms = _terms(w, peak['rates'], scoped)
    if not terms:
        raise BoundUnavailable(f'{w.id}: no demand recorded')
    binding = max(terms, key=terms.get)
    return terms[binding], binding, terms


# ---------------------------------------------------------------------------
# Semantic demand per operator class (roadmap section 4.4).
# ---------------------------------------------------------------------------

def elementwise(id, operator, n, *, inputs, outputs, fp_per_elem=0, sfu_per_elem=0,
                int_per_elem=0, dtype='fp32', out_dtype=None):
    """n elements, `inputs` arrays read and `outputs` arrays written, once each."""
    b, ob = DTYPE_BYTES[dtype], DTYPE_BYTES[out_dtype or dtype]
    return Workload(id, operator, 'elementwise', fp_ops=n * fp_per_elem,
                    sfu_ops=n * sfu_per_elem, int_ops=n * int_per_elem,
                    read_bytes=n * inputs * b, write_bytes=n * outputs * ob)


def gemm(id, operator, M, N, K, *, batch=1, in_dtype='fp16', out_dtype='fp32'):
    """Conventional dense MAC count (section 4.4); M*N*K MACs per batch."""
    ib, ob = DTYPE_BYTES[in_dtype], DTYPE_BYTES[out_dtype]
    macs = batch * M * N * K
    return Workload(id, operator, 'gemm', fp_ops=macs, macs=macs,
                    read_bytes=batch * (M * K + K * N) * ib,
                    write_bytes=batch * M * N * ob,
                    meta=dict(M=M, N=N, K=K, batch=batch))


def row_reduction(id, operator, rows, n, *, reads_per_elem=1, writes_per_row=1,
                  writes_per_elem=0, fp_per_elem=1, sfu_per_elem=0, dtype='fp32'):
    """Norms, softmax, losses: every input element read once."""
    b = DTYPE_BYTES[dtype]
    return Workload(id, operator, 'reduction', fp_ops=rows * n * fp_per_elem,
                    sfu_ops=rows * n * sfu_per_elem,
                    read_bytes=rows * n * reads_per_elem * b,
                    write_bytes=(rows * writes_per_row + rows * n * writes_per_elem) * b)


def copy(id, operator, n, *, dtype='fp32'):
    """Copy/transpose/reverse/interleave: payload bytes in and out (section 4.4)."""
    b = DTYPE_BYTES[dtype]
    return Workload(id, operator, 'copy', read_bytes=n * b, write_bytes=n * b)


def conv(id, operator, out_points, taps, in_elems, weights, *, in_dtype='fp16', out_dtype='fp16',
         in_channels=1, out_channels=1):
    """Direct convolution (declared algorithm family): one MAC per valid tap."""
    ib, ob = DTYPE_BYTES[in_dtype], DTYPE_BYTES[out_dtype]
    macs = out_points * out_channels * taps * in_channels
    return Workload(id, operator, 'conv', fp_ops=macs, macs=macs,
                    read_bytes=(in_elems * in_channels + weights) * ib,
                    write_bytes=out_points * out_channels * ob)


def unavailable(id, operator, category, why):
    return Workload(id, operator, category, bound_available=False, note=why)
