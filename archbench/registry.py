"""The registry: every ArchBench point, defined once at the PyTorch level.

A point is a torch callable, the inputs it is called with, and the semantic
demand that bounds it. Every platform -- a GPU through torch.compile, TNCA
through torch.compile -> Inductor -> Triton -> tnca, eager CPU as the
reference -- runs the SAME point object, built from the same seed, so the
values compared and the work bounded cannot differ between platforms.

Point ids are `<kind>.<family>/<tier>/<variant>`, e.g.
`op.matmul/tiny/fp16-m16k32n16`. A point's definition (its parameters, dtype,
tolerance and demand) is hashed into the frozen suite files under
archbench/suites/, so a changed definition is a different suite.

Tiers say where a point can be measured, not how important it is:

    tiny    sized for the RTL simulator (Icarus, ~500 target cycles per
            wall second): each point is at most a few hundred thousand cycles
    small   sized for the board: every kernel's arguments fit the device
            heap (384 KiB of the U50 image's 512 KiB memory)
    medium  for the board too, a moderately bigger load (R39-13): two to
            eight times small, still within the device heap per kernel
    large   GPU-realistic shapes (a ~1B-parameter decoder, 512-token prefill);
            TNCA records them as capacity-exceeded until it has the memory
    hbm     R40-00: working sets of at least 32 MiB -- four times the
            architecture's whole NUCA -- for streaming, GEMM, transpose and
            gather (archbench/hbm.py); capacity-exceeded until TNCA has HBM.
            R40's HBM suite is these and every point of the other tiers
    membound
            R41: the registered memory-bound bandwidth suite (archbench/
            membound.py, frozen as archbench/suites/r41_membound.json):
            copy, element-wise, casts, reductions and normalisations in
            fp32/fp16/bf16, small to 64 MiB-per-tensor sizes, contiguous,
            strided and broadcast layouts
    xl      R42: real-LLM-scale points beyond `large`, with FP8 block-wise
            matrix multiplication as the majority (archbench/xl.py): every
            point's live device bytes are at most 3 GiB (the board exposes
            4 GiB to a launch), declared by the point and asserted by the
            selftest
"""
from dataclasses import dataclass, field
import hashlib
import json

TIERS = ('tiny', 'small', 'medium', 'large', 'hbm', 'membound', 'xl')

# Relative tolerance on max|got - ref| / max(max|ref|, 1), by the dtype the
# point computes in. The reference is eager PyTorch in float32 on the CPU
# (fp16/bf16 values upcast exactly), so the tolerance is the low-precision
# arithmetic's, not the reference's.
TOLERANCE = {'fp32': 2e-4, 'fp16': 2e-2, 'bf16': 6e-2,
             # xl tier (archbench/xl.py): the default of a block-wise FP8 matrix multiplication with fp16
             # activations and output -- 8x the fp16 output rounding (2^-11, relative to the value); every
             # xl point states its own tolerance and how it was derived
             'fp8': 4e-3}

# 'fp8' names the matrix format of an xl point (E4M3 weights and dynamically quantised E4M3 activations,
# fp16 everywhere else); the tensors it is made of are float8_e4m3fn weights and fp16 activations.
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'fp8': 'float8_e4m3fn'}


@dataclass(frozen=True)
class Demand:
    """The semantic work of one call, independent of any implementation.

    macs        dense multiply-accumulates (conventional count, section 4.4
                of the R38 roadmap); a causal attention counts only the
                unmasked products, which is what an implementation must do
    fp_ops      other binary32-class arithmetic the result needs (adds,
                multiplies, compares on values), per element -- a lower
                estimate is safe, it only loosens the bound
    sfu_ops     transcendentals (exp, rsqrt, tanh, sigmoid, ...)
    read_bytes  every input byte the result depends on, once (activations,
                weights and caches)
    write_bytes every output byte, once
    macs_fp8    xl tier: the part of `macs` whose operands are E4M3 (a
                block-wise FP8 matrix multiplication); zero for every other
                point, and then absent from the definition
    """
    macs: float = 0.0
    fp_ops: float = 0.0
    sfu_ops: float = 0.0
    read_bytes: float = 0.0
    write_bytes: float = 0.0
    macs_fp8: float = 0.0

    def __add__(self, o):
        return Demand(self.macs + o.macs, self.fp_ops + o.fp_ops, self.sfu_ops + o.sfu_ops,
                      self.read_bytes + o.read_bytes, self.write_bytes + o.write_bytes,
                      self.macs_fp8 + o.macs_fp8)

    def as_dict(self):
        d = dict(macs=self.macs, fp_ops=self.fp_ops, sfu_ops=self.sfu_ops,
                 read_bytes=self.read_bytes, write_bytes=self.write_bytes)
        if self.macs_fp8:
            d['macs_fp8'] = self.macs_fp8
        return d


@dataclass(frozen=True)
class Point:
    id: str
    kind: str                 # 'op' | 'model'
    family: str
    tier: str
    dtype: str
    params: dict
    make: object              # make(seed) -> (callable, tuple of CPU inputs)
    demand: Demand
    track: tuple = ('llm',)   # 'llm' and/or 'general'
    tol: float = None
    note: str = ''
    # xl tier: a reference that is not "the module in binary32" (a block-wise FP8 module cannot run on the
    # CPU in the Hub kernel's Triton form): reference(module, inputs) -> tuple of CPU tensors, or None
    reference: object = None
    # xl tier: definition fields the suite hash covers beyond the above (kernel identity, memory accounting,
    # the tolerance's derivation); None for every other tier, and then absent from the definition
    extra: dict = None

    @property
    def tolerance(self):
        return self.tol if self.tol is not None else TOLERANCE[self.dtype]

    def definition(self):
        """What the suite hash covers: everything that changes the point."""
        d = dict(id=self.id, kind=self.kind, family=self.family, tier=self.tier,
                 dtype=self.dtype, params=self.params, demand=self.demand.as_dict(),
                 track=list(self.track), tolerance=self.tolerance)
        if self.extra is not None:
            d['extra'] = self.extra
        return d

    def definition_sha256(self):
        blob = json.dumps(self.definition(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


_POINTS = {}


def register(point):
    if point.tier not in TIERS:
        raise ValueError(f'{point.id}: tier {point.tier!r}')
    if point.id in _POINTS:
        raise ValueError(f'{point.id} registered twice')
    _POINTS[point.id] = point
    return point


def points(tier=None, kind=None, track=None, family=None):
    _load()
    out = []
    for p in _POINTS.values():
        if tier and p.tier not in ((tier,) if isinstance(tier, str) else tier):
            continue
        if kind and p.kind != kind:
            continue
        if track and track not in p.track:
            continue
        if family and p.family not in ((family,) if isinstance(family, str) else family):
            continue
        out.append(p)
    return sorted(out, key=lambda p: (TIERS.index(p.tier), p.kind, p.family, p.id))


def get(point_id):
    _load()
    try:
        return _POINTS[point_id]
    except KeyError:
        raise KeyError(f'no ArchBench point {point_id!r}') from None


_loaded = False


def _load():
    global _loaded
    if not _loaded:
        _loaded = True
        from . import ops, models, hbm, membound, xl    # noqa: F401  (registration side effect)
