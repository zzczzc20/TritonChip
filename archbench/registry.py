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
"""
from dataclasses import dataclass, field
import hashlib
import json

TIERS = ('tiny', 'small', 'medium', 'large', 'hbm')

# Relative tolerance on max|got - ref| / max(max|ref|, 1), by the dtype the
# point computes in. The reference is eager PyTorch in float32 on the CPU
# (fp16/bf16 values upcast exactly), so the tolerance is the low-precision
# arithmetic's, not the reference's.
TOLERANCE = {'fp32': 2e-4, 'fp16': 2e-2, 'bf16': 6e-2}

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16'}


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

    @property
    def tolerance(self):
        return self.tol if self.tol is not None else TOLERANCE[self.dtype]

    def definition(self):
        """What the suite hash covers: everything that changes the point."""
        return dict(id=self.id, kind=self.kind, family=self.family, tier=self.tier,
                    dtype=self.dtype, params=self.params, demand=self.demand.as_dict(),
                    track=list(self.track), tolerance=self.tolerance)

    def definition_sha256(self):
        blob = json.dumps(self.definition(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


_POINTS = {}
_HOLDOUT_POINTS = {}


def register(point):
    if point.tier not in TIERS:
        raise ValueError(f'{point.id}: tier {point.tier!r}')
    if point.id in _POINTS or point.id in _HOLDOUT_POINTS:
        raise ValueError(f'{point.id} registered twice')
    _POINTS[point.id] = point
    return point


def register_holdout(point):
    if point.id in _POINTS or point.id in _HOLDOUT_POINTS:
        raise ValueError(f'{point.id} registered twice')
    _HOLDOUT_POINTS[point.id] = point
    return point


def holdout_points():
    _load()
    return list(_HOLDOUT_POINTS.values())


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
    if point_id in _POINTS:
        return _POINTS[point_id]
    if point_id in _HOLDOUT_POINTS:
        return _HOLDOUT_POINTS[point_id]
    raise KeyError(f'no ArchBench point {point_id!r}') from None


_loaded = False


def _load():
    global _loaded
    if not _loaded:
        _loaded = True
        from . import ops, models, hbm, holdout    # noqa: F401  (registration side effect)
