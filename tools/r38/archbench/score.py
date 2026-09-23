"""ArchBench scores: roadmap sections 4.5-4.8 and 4.10.6.

    eta_i  = L_i / T_i                     per point, strict bound over measured time
    E      = (sum_i w_i / eta_i)^-1        weighted harmonic: distance to the ceiling
    A      = sum_i w_i eta_i               weighted arithmetic, reported beside E
    S      = sum a_i T_ref / sum a_i T     same work mix, total time ratio
    U      = sum a_i L_i / sum a_i T_i     candidate-weighted diagnostic, never a rank
    AQ     = 100 / sum_{b,e,i} p_b q_e w_i tau/L*

Rules that are code here, not advice:

  * eta > 1 beyond the stated uncertainty raises BoundViolation. Nothing is
    clipped (section 4.5).
  * A registered point that failed or was not measured makes the full-suite
    score 0 and uncertified; the supported-only value is returned beside it,
    labelled diagnostic (section 4.8).
  * Weights come from a frozen file whose hash the evaluator checks; a
    candidate supplying different weights is refused (appendix A test 9).
  * Operator mass is fixed before stages are counted: splitting an operator
    into more stages or repeating a shape does not move its weight
    (appendix A test 4).
  * Time is a quantity with a unit. Cycles and seconds convert only through
    an explicit clock (appendix A test 6).
  * Energy is measured or None. A TDP stands in for nothing (test 10).
"""
from dataclasses import dataclass, field
import hashlib
import json
import math

PASS = 'pass'
FAILURES = ('compile-failure', 'runtime-fault', 'numeric-error', 'capacity-exceeded',
            'timeout', 'not-measured', 'unsupported')


class BoundViolation(Exception):
    """A measurement faster than a strict lower bound: audit, never clip."""


class WeightMismatch(Exception):
    """The weights are not the frozen ones."""


class ProtocolMismatch(Exception):
    """A record from a different timing protocol or bound version."""


# ---------------------------------------------------------------------------
# Time with units
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Duration:
    value: float
    unit: str                    # 'cycles' | 'ns' | 's'

    def cycles(self, clock_hz=None):
        if self.unit == 'cycles':
            return self.value
        if clock_hz is None:
            raise ValueError(f'converting {self.unit} to cycles needs an explicit clock')
        return self.seconds() * clock_hz

    def seconds(self, clock_hz=None):
        if self.unit == 's':
            return self.value
        if self.unit == 'ns':
            return self.value * 1e-9
        if clock_hz is None:
            raise ValueError('converting cycles to seconds needs an explicit clock')
        return self.value / clock_hz


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """One registered point's result for one candidate."""
    id: str
    operator: str
    status: str                      # PASS or one of FAILURES
    time_cycles: float = None        # ROI makespan, launch accepted -> drained
    bound_cycles: float = None       # strict L for THIS candidate's configuration
    bound_version: str = None
    timing: str = None               # e.g. 'serial-grid'
    core_active_cycles: list = None  # per core; never summed into time
    energy_j: float = None
    energy_source: str = None        # 'measured' | None
    meta: dict = field(default_factory=dict)


def check_energy(p):
    if p.energy_j is None:
        return
    if p.energy_source != 'measured':
        raise ProtocolMismatch(f'{p.id}: energy from {p.energy_source!r} is not a measurement')


def eta(bound, time, *, uncertainty=0.0, id=''):
    if bound is None or time is None:
        return None
    if time <= 0:
        raise BoundViolation(f'{id}: non-positive time {time}')
    e = bound / time
    if e > 1 + uncertainty:
        raise BoundViolation(
            f'{id}: measured {time} cycles is below the strict bound {bound:.3f} '
            f'(eta={e:.4f}); audit the timing window, units, MAC/FLOP counting, '
            f'concurrency and initial cache state -- the record is not clipped')
    return e


# ---------------------------------------------------------------------------
# Weights (section 4.6)
# ---------------------------------------------------------------------------

def reference_weights(ref_times, calls=None):
    """w_i = a_i T_ref_i / sum_j a_j T_ref_j, from a frozen reference."""
    calls = calls or {i: 1.0 for i in ref_times}
    total = sum(calls[i] * ref_times[i] for i in ref_times)
    return {i: calls[i] * ref_times[i] / total for i in ref_times}


def balanced_weights(points_by_operator, shape_mass=None):
    """Balanced track: each operator gets 1/n_ops; inside it, the registered
    shape distribution (default uniform over SEMANTIC instances). Stages are
    not instances and never enter here."""
    ops = sorted(points_by_operator)
    w = {}
    for op in ops:
        ids = sorted(set(points_by_operator[op]))
        mass = shape_mass.get(op) if shape_mass else None
        mass = mass or {i: 1.0 for i in ids}
        tot = sum(mass[i] for i in ids)
        for i in ids:
            w[i] = mass[i] / tot / len(ops)
    return w


def weights_hash(weights):
    blob = json.dumps({k: round(v, 15) for k, v in sorted(weights.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def frozen_weights(doc):
    """doc = {'weights': {...}, 'sha256': ...}; refuses a tampered file."""
    if weights_hash(doc['weights']) != doc['sha256']:
        raise WeightMismatch('weights do not match their recorded hash')
    return doc['weights']


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

@dataclass
class Score:
    value: float
    certified: bool
    coverage_count: float
    coverage_weight: float
    supported_only: float = None     # diagnostic, never the main number
    missing: list = field(default_factory=list)
    label: str = ''


def _coverage(points, weights):
    by_id = {p.id: p for p in points}
    missing = sorted(i for i in weights if i not in by_id or by_id[i].status != PASS)
    ok = [i for i in weights if i not in missing]
    return by_id, ok, missing


def efficiency(points, weights, *, uncertainty=0.0, bound_version=None):
    """E (harmonic) and A (arithmetic) over the frozen weights."""
    by_id, ok, missing = _coverage(points, weights)
    etas = {}
    for i in ok:
        p = by_id[i]
        check_energy(p)
        if bound_version and p.bound_version != bound_version:
            raise ProtocolMismatch(f'{i}: bound version {p.bound_version} != {bound_version}')
        etas[i] = eta(p.bound_cycles, p.time_cycles, uncertainty=uncertainty, id=i)
    etas = {i: e for i, e in etas.items() if e is not None}
    unbounded = sorted(set(ok) - set(etas))
    missing_all = missing + unbounded

    def harmonic(ids):
        tw = sum(weights[i] for i in ids)
        return tw / sum(weights[i] / etas[i] for i in ids) if ids else 0.0

    def arithmetic(ids):
        tw = sum(weights[i] for i in ids)
        return sum(weights[i] * etas[i] for i in ids) / tw if ids else 0.0

    ids = sorted(etas)
    cov_w = sum(weights[i] for i in ids)
    full = not missing_all
    E = Score(harmonic(ids) if full else 0.0, full, len(ids) / len(weights), cov_w,
              supported_only=harmonic(ids), missing=missing_all, label='E')
    A = Score(arithmetic(ids) if full else 0.0, full, len(ids) / len(weights), cov_w,
              supported_only=arithmetic(ids), missing=missing_all, label='A')
    return E, A, etas


def speedup(ref_times, times, calls=None):
    """S = sum a T_ref / sum a T over the same work mix; None if incomplete."""
    calls = calls or {i: 1.0 for i in ref_times}
    if any(times.get(i) is None for i in ref_times):
        return None
    return (sum(calls[i] * ref_times[i] for i in ref_times)
            / sum(calls[i] * times[i] for i in ref_times))


def utilisation(points, calls=None):
    """U: candidate-weighted diagnostic. Not a cross-architecture rank."""
    good = [p for p in points if p.status == PASS and p.bound_cycles is not None]
    calls = calls or {p.id: 1.0 for p in good}
    return (sum(calls[p.id] * p.bound_cycles for p in good)
            / sum(calls[p.id] * p.time_cycles for p in good)) if good else None


def aq(tau, lstar, w, p_budget, q_env, *, label='AQ-RTL-canonical'):
    """AQ = 100 / sum_b p_b sum_e q_e sum_i w_i tau[b][e][i] / L*[b][e][i].

    tau[b][e][i] is None for a point not measured or failed. Any such hole
    gives AQ 0 and NOT CERTIFIED, with the supported-only value beside it.
    """
    total, covered, holes = 0.0, 0.0, []
    for b, pb in p_budget.items():
        for e, qe in q_env.items():
            for i, wi in w.items():
                t = tau.get(b, {}).get(e, {}).get(i)
                l = lstar[b][e][i]
                mass = pb * qe * wi
                if t is None:
                    holes.append((b, e, i))
                    continue
                if l / t > 1 + 1e-12:
                    raise BoundViolation(f'{b}/{e}/{i}: tau below the common bound L*')
                total += mass * t / l
                covered += mass
    supported = 100 * covered / total if total else 0.0
    certified = not holes
    return Score(100 / total if certified else 0.0, certified, None, covered,
                 supported_only=supported, missing=holes,
                 label=label + ('' if certified else ' NOT CERTIFIED'))


def roi_cycles(point):
    """The time a point is scored on: its ROI makespan. Per-core active
    cycles are evidence about the run, never a substitute for its length."""
    if point.time_cycles is None:
        return None
    if point.core_active_cycles and max(point.core_active_cycles) > point.time_cycles:
        raise ProtocolMismatch(f'{point.id}: a core was active longer than the ROI')
    return point.time_cycles


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
