"""The roofline lower bound the evaluation scores against, for your own cases.

Copied from the evaluation's own bound code so that the L and eta you see for a
case of yours are the ones the hidden benchmark would compute for it. The
originals stay with the evaluator; a harness test keeps the copies equal.

    L = max over resources of demand / capacity        (cycles)
    eta = L / T                                         (T: measured cycles)

Demand is the SEMANTIC work of one call, independent of any program: dense
MACs by the conventional count, other floating-point operations,
transcendentals, every required input byte read once, every output byte
written once. Undercounting is safe (the bound only gets looser, eta lower);
overcounting lets a program beat its own bound, which the evaluation refuses
rather than clips.

Capacities are the WHOLE CHIP's (chip_peak), the roof the card's eta is
against, never a measurement: one core's spec_peak (the profile's target file
and RTL facts) times the device's cores, and memory through the device's
MEM_PORTS outlets of AXI_DATA_W bits. In a run the harness gives the base
machine's chip rates in TNCA_CHIP_ROOF, computed by the card's own code, and
those are used as they are; elsewhere they are formed from the device that
TNCA_BACKEND_ARGS names. Work that several units can execute is divided by
their SUMMED rate (a MAC can run on the matrix unit or on the vector lanes),
and a transcendental counts as at least one FP operation, so the strict bound
assumes no algorithm.

The roof is the base machine's in a run, whatever you change: a faster
machine shows as fewer cycles T against the same L, so eta rises.
"""
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path



def repo_root():
    """The TritonChip checkout whose spec/ and compiler/targets the bound reads.

    The skill is mounted outside the checkout, so the root is the one the
    caller works in: $TNCA_ROOT if set, else the nearest ancestor of the
    current directory with spec/r1_profiles.json (the agent works in /work),
    else the checkout this file is in (the harness's own tests).
    """
    import os
    env = os.environ.get('TNCA_ROOT')
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents, *here.parents]
    for d in candidates:
        if (d / 'spec' / 'r1_profiles.json').is_file():
            return d
    raise FileNotFoundError(
        'no TritonChip checkout found: run from inside one (the agent works in /work) '
        'or set TNCA_ROOT to it; the bound reads its spec/r1_profiles.json and compiler/targets')


DTYPE_BYTES = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'e4m3': 1, 'e5m2': 1, 'fp8': 1,
               'i8': 1, 'i32': 4, 'i64': 8}

# ---------------------------------------------------------------------------
# Demand: the five quantities the evaluation records for a point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Demand:
    """The semantic work of one call, independent of any implementation.

    macs        dense multiply-accumulates (conventional count); a causal
                attention counts only the unmasked products
    fp_ops      other binary32-class arithmetic the result needs (adds,
                multiplies, compares on values), per element; a lower
                estimate is safe, it only loosens the bound
    sfu_ops     transcendentals (exp, rsqrt, tanh, sigmoid, ...)
    read_bytes  every input byte the result depends on, once (activations,
                weights and caches)
    write_bytes every output byte, once
    """
    macs: float = 0.0
    fp_ops: float = 0.0
    sfu_ops: float = 0.0
    read_bytes: float = 0.0
    write_bytes: float = 0.0

    def __add__(self, o):
        return Demand(self.macs + o.macs, self.fp_ops + o.fp_ops, self.sfu_ops + o.sfu_ops,
                      self.read_bytes + o.read_bytes, self.write_bytes + o.write_bytes)

    def as_dict(self):
        return dict(macs=self.macs, fp_ops=self.fp_ops, sfu_ops=self.sfu_ops,
                    read_bytes=self.read_bytes, write_bytes=self.write_bytes)


# ---------------------------------------------------------------------------
# spec_peak: one core's compute rates; chip_peak: the whole chip's, memory too.
# ---------------------------------------------------------------------------

SFU_SLICES = 1           # rtl/core/tnca_sfu.v: a single four-function slice
ROOF_ENV = 'TNCA_CHIP_ROOF'   # the harness's roof for a run (its roofline.agent_roof)

SOURCES = {
    'fp_lane': 'rtl/core/tnca_vector_fpu.v (VLEN lanes) + rtl/core/tnca_matrix.v (VLEN MACs)',
    'mma_mac': 'rtl/core/tnca_matrix.v: 16 MACs, one k per 16 cycles',
    'int_lane': 'rtl/core/tnca_vector_alu.v: VLEN lanes, single cycle',
    'sfu_lane': 'rtl/core/tnca_sfu.v: one slice, one lane per cycle',
    'issue': 'target file issue.width',
}
MEM_SOURCE = ('the device description: MEM_PORTS outlets of AXI_DATA_W bits '
              '(a core\'s port is as wide on a HYBRID_LSU core, rtl/tile/tnca_tile.v CORE_DW)')


def target_for(profile):
    root = repo_root()
    spec = json.loads((root / 'spec' / 'r1_profiles.json').read_text())['profiles'][profile]
    return json.loads((root / 'compiler' / 'targets' / spec['target_file']).read_text())


def base_profile(name):
    """The architectural profile behind a memory-mode variant ('core-sim/slow',
    'x/hbm' are the same core on another memory). Copied from tnca.device."""
    return name.split('/', 1)[0]


# The matrix engine's MACs per cycle per operand format, in multiples of VLEN
# (copied from the evaluation, tools/r38/archbench/peaks.py). Without the
# matrix_dp capability every format is 1x; with it each MAC forms a group of
# products per cycle (m.mma.dp.async, ISA sec 9C.4): FP32 1x, FP16/BF16/FP8
# pairs 2x, E4M3 x E4M3 4x. 'e4m3' means both operands E4M3.
MMA_GROUP = {'fp32': 1, 'fp16': 2, 'bf16': 2, 'fp8': 2, 'e5m2': 2, 'e4m3': 4}
MMA_DP_SOURCES = {
    'fp_lane': 'rtl/core/tnca_vector_fpu.v (VLEN lanes) + mma_mac',
    'mma_mac': 'rtl/core/tnca_matrix.v MAC_DP=1: VLEN tnca_mac_lane MACs, each a group of '
               '1 (FP32), 2 (FP16/BF16/FP8 pairs) or 4 (E4M3 x E4M3) products per cycle '
               '(m.mma.dp.async, ISA sec 9C.4)',
}


def mma_rate(target, mma_dtype=None):
    """Matrix-engine MACs per cycle on `target` for one operand format; with no
    format on a matrix_dp target the generous end (4x), as the evaluation does:
    a rate too high only loosens a bound, one too low shows as eta > 1."""
    if mma_dtype is not None and mma_dtype not in MMA_GROUP:
        raise ValueError(f'no matrix-engine rate for operand format {mma_dtype!r}')
    vlen = target['vlen']
    if not target['capabilities'].get('matrix_dp'):
        return vlen
    return vlen * (MMA_GROUP[mma_dtype] if mma_dtype else max(MMA_GROUP.values()))


def spec_peak(profile='core-sim', *, target=None, overrides=None, mma_dtype=None):
    """One core's compute rates for a single-core profile, as the evaluation's
    spec_peak has them, without memory: memory is the chip's (chip_peak).
    `mma_dtype` is the matrix operand format of the case being bounded; it
    matters only on a profile with the matrix_dp capability (mma_rate)."""
    if profile.endswith('/hbm') and target is None:
        # As the evaluation has it (archbench/bounds.py tnca_peak): HBM changes the
        # backing memory, while this profile still has the same single core.
        peak = spec_peak(base_profile(profile), overrides=overrides, mma_dtype=mma_dtype)
        peak['profile'] = profile
        return peak
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
        issue=t['issue']['width'],
    )
    peak.update(overrides or {})
    out = dict(profile=profile, unit='per-cycle', rates=peak, sources=SOURCES,
               kind='spec_peak')
    if t['capabilities'].get('matrix_dp'):
        out.update(mma_dtype=mma_dtype, sources=dict(SOURCES, **MMA_DP_SOURCES))
    return out


def device_named():
    """The device the environment measures on: TNCA_BACKEND_ARGS' `device`, or None."""
    try:
        return json.loads(os.environ.get('TNCA_BACKEND_ARGS') or '{}').get('device')
    except (ValueError, AttributeError):
        return None


def chip_peak(profile='core-sim', *, mma_dtype=None, device=None):
    """The whole chip's per-cycle rates: the roof of the card's eta.

    In a run (TNCA_CHIP_ROOF set, and no `device` asked for) the base machine's,
    as the harness computed them with the card's own code: used as given, the
    profile argument aside. Otherwise one core's spec_peak times the device's
    core_count, and memory MEM_PORTS x AXI_DATA_W / 8 bytes per cycle, from
    spec/devices/<device>.json (`device`, else the one TNCA_BACKEND_ARGS names).
    """
    roof = os.environ.get(ROOF_ENV)
    if roof and device is None:
        r = json.loads(roof)
        key = mma_dtype or 'any'
        if key not in r['rates']:
            raise ValueError(f'no chip rate for operand format {mma_dtype!r}')
        return dict(profile=r.get('profile'), unit='per-cycle', rates=dict(r['rates'][key]), kind='chip',
                    cores=r.get('cores'), mma_dtype=mma_dtype,
                    source=f'{ROOF_ENV}: the base machine\'s whole chip, as the card\'s eta')
    device = device or device_named()
    if not device:
        raise BoundUnavailable('no device for the whole chip\'s roof: a run sets TNCA_CHIP_ROOF; elsewhere '
                               'TNCA_BACKEND_ARGS must name a device of spec/devices/')
    dev = json.loads((repo_root() / 'spec' / 'devices' / f'{device}.json').read_text())
    cores = int(dev['platform']['core_count'])
    core = spec_peak(profile, mma_dtype=mma_dtype)
    rates = {k: v * cores for k, v in core['rates'].items()}
    rates['mem_bytes'] = int(dev['uncore']['MEM_PORTS']) * int(dev['platform']['AXI_DATA_W']) // 8
    return dict(profile=profile, unit='per-cycle', rates=rates, kind='chip', cores=cores, mma_dtype=mma_dtype,
                source=f'spec/devices/{device}.json x the profile\'s core; memory: {MEM_SOURCE}')


# ---------------------------------------------------------------------------
# The bound.
# ---------------------------------------------------------------------------


class BoundUnavailable(Exception):
    """No legal, informative lower bound exists for this workload."""


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

    Returns (cycles, binding_resource, terms).
    """
    if not w.bound_available:
        raise BoundUnavailable(f'{w.id}: {w.note or "no v0 bound for " + w.category}')
    terms = _terms(w, peak['rates'], scoped)
    if not terms:
        raise BoundUnavailable(f'{w.id}: no demand recorded')
    binding = max(terms, key=terms.get)
    return terms[binding], binding, terms


def workload(demand, id='case', operator='case', category='case'):
    """A Workload from a Demand, exactly as the evaluation forms one for a point:
    every MAC is also one FP lane operation."""
    return Workload(id, operator, category, fp_ops=demand.macs + demand.fp_ops, macs=demand.macs,
                    sfu_ops=demand.sfu_ops, read_bytes=demand.read_bytes, write_bytes=demand.write_bytes)


def bound(demand, profile, *, scoped=False, mma_dtype=None, device=None):
    """(cycles, binding resource, terms) for a Demand against the whole chip
    (chip_peak), as the card bounds a point. `mma_dtype` is the case's matrix
    operand format (its dtype), as the evaluation passes a point's; it only
    matters on a profile with the matrix_dp capability."""
    return lower_bound(workload(demand), chip_peak(profile, mma_dtype=mma_dtype, device=device), scoped=scoped)


def achieved(demand, cycles, peak):
    """What a run of `cycles` did per cycle against each rate it is bounded by:
    {resource: {per_cycle, rate}} -- memory bytes, FP lane operations (a
    transcendental counts as one), matrix MACs."""
    w, rates = workload(demand), peak['rates']
    done = {'mem_bytes': w.required_bytes, 'fp_lane': w.fp_ops + w.sfu_ops, 'mma_mac': w.macs}
    return {k: dict(per_cycle=v / cycles, rate=rates[k])
            for k, v in done.items() if v and cycles and rates.get(k)}


def eta(L, cycles):
    """L / T. Above 1 the demand is overcounted or the count is wrong: the
    evaluation treats that as an audit failure, never as a good result."""
    return L / cycles if cycles else math.inf


# ---------------------------------------------------------------------------
# Demand by hand, per operator class.
# ---------------------------------------------------------------------------


def elementwise(id, operator, n, *, inputs, outputs, fp_per_elem=0, sfu_per_elem=0,
                int_per_elem=0, dtype='fp32', out_dtype=None):
    """n elements, `inputs` arrays read and `outputs` arrays written, once each."""
    b, ob = DTYPE_BYTES[dtype], DTYPE_BYTES[out_dtype or dtype]
    return Workload(id, operator, 'elementwise', fp_ops=n * fp_per_elem,
                    sfu_ops=n * sfu_per_elem, int_ops=n * int_per_elem,
                    read_bytes=n * inputs * b, write_bytes=n * outputs * ob)


def gemm(id, operator, M, N, K, *, batch=1, in_dtype='fp16', out_dtype='fp32'):
    """Conventional dense MAC count; M*N*K MACs per batch."""
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
    """Copy/transpose/reverse/interleave: payload bytes in and out."""
    b = DTYPE_BYTES[dtype]
    return Workload(id, operator, 'copy', read_bytes=n * b, write_bytes=n * b)


def conv(id, operator, out_points, taps, in_elems, weights, *, in_dtype='fp16', out_dtype='fp16',
         in_channels=1, out_channels=1):
    """Direct convolution: one MAC per valid tap."""
    ib, ob = DTYPE_BYTES[in_dtype], DTYPE_BYTES[out_dtype]
    macs = out_points * out_channels * taps * in_channels
    return Workload(id, operator, 'conv', fp_ops=macs, macs=macs,
                    read_bytes=(in_elems * in_channels + weights) * ib,
                    write_bytes=out_points * out_channels * ob)


# ---------------------------------------------------------------------------
# Demand from the case itself: MACs from PyTorch's FLOP counter, bytes from
# the real tensors. The evaluation checks every point's declared demand
# against exactly these two measures.
# ---------------------------------------------------------------------------


def _storage_bytes(ts):
    seen, total = set(), 0
    for t in ts:
        st = t.untyped_storage()
        if st.data_ptr() in seen:
            continue
        seen.add(st.data_ptr())
        total += t.numel() * t.element_size()
    return total


def _math_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Attention as matmul-softmax-matmul, for counting only: the FLOP counter
    has no formula for the CPU flash-attention op. This counts the DENSE
    work, masked scores included, so for a causal case it overcounts and the
    bound it gives is NOT a safe one; declare that demand by hand."""
    import torch
    s = (q @ k.transpose(-2, -1)) * (scale or q.shape[-1] ** -0.5)
    if is_causal:
        n, m = s.shape[-2], s.shape[-1]
        s = s.masked_fill(torch.ones(n, m, dtype=torch.bool).triu(m - n + 1), float('-inf'))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v


def demand_of(module, inputs, *, fp_ops=0.0, sfu_ops=0.0):
    """A Demand measured from the case: MACs = the FLOP counter's FLOPs / 2 on
    an eager run, read bytes = every input, parameter and buffer storage once,
    write bytes = every output once.

    The counter sees matrix products and convolutions, not elementwise
    arithmetic or transcendentals, so `fp_ops` and `sfu_ops` are 0 unless you
    pass them. A missing term only loosens the bound. For a causal attention
    the counter counts masked products too (see _math_sdpa): pass macs by hand
    through Demand instead.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.flop_counter import FlopCounterMode
    m = module.eval()
    real = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _math_sdpa
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            out = m(*inputs)
    finally:
        F.scaled_dot_product_attention = real
    macs = fc.get_total_flops() / 2
    outs = (out,) if isinstance(out, torch.Tensor) else tuple(t for o in out for t in ((o,) if isinstance(o, torch.Tensor) else tuple(o)))
    write = sum(t.numel() * t.element_size() for t in outs)
    read = _storage_bytes([t for t in list(inputs) if isinstance(t, torch.Tensor)]
                          + list(m.parameters()) + list(m.buffers()))
    return Demand(macs=macs, fp_ops=fp_ops, sfu_ops=sfu_ops, read_bytes=read, write_bytes=write)
