"""Pre-synthesis resource bounds and counter-guided architecture screening.

Explicit capacities are optimistic physical upper bounds (not achieved BW).
Demands are lower bounds from compiled work or measured traffic. Consequently
max(demand/capacity) can reject an impossible cycle budget, but cannot certify
an improvement, placement, timing, or routing. Never scales old single-core
operator fits by core count. Unmodelled contention only makes execution worse.
"""
import math
import copy

from .proton import check_identity


def resource_screen(demand, candidate, *, identity, cycle_budget=None):
    """Demand/candidate use the same named shared or per-core resources.

    Each demand: {work: nonnegative, unit: str, source: str}.
    Each capacity: {per_cycle: positive optimistic upper bound, unit: str,
                    source: str}. Use independent keys for each actual PC,
    direction, NoC cut and core issue slot; never multiply shared bandwidth
    by cores. Dependent chain latency may be represented by a unit-capacity
    resource whose work is a proven critical-path lower bound.
    """
    check_identity(identity)
    if not demand:
        raise ValueError('empty demand cannot screen a candidate')
    terms = {}
    for name, item in demand.items():
        if name not in candidate:
            raise ValueError(f'candidate missing resource {name}; cannot silently ignore traffic')
        cap = candidate[name]
        work, rate = item['work'], cap['per_cycle']
        if (type(work) not in (int, float) or type(rate) not in (int, float)
                or not math.isfinite(work) or not math.isfinite(rate)
                or work < 0 or rate <= 0 or item['unit'] != cap['unit']
                or not item.get('source') or not cap.get('source')):
            raise ValueError(f'invalid or unsourced demand/capacity for {name}')
        terms[name] = work / rate
    if cycle_budget is not None and (not math.isfinite(cycle_budget) or cycle_budget <= 0):
        raise ValueError('cycle budget must be finite and positive')
    bound = max(terms.values())
    return dict(schema='tnca-resource-screen/1', identity=identity,
                demand=demand, candidate=candidate, lower_bound_cycles=bound,
                resource_lower_bounds=terms,
                limiting_resources=sorted(terms, key=terms.get, reverse=True),
                decision=('reject-impossible-budget' if cycle_budget is not None and bound > cycle_budget
                          else 'needs-rtl-and-calibration'), cycle_budget=cycle_budget,
                predicted_cycles=None, predicted_speedup=None,
                placement='not-measured', timing='not-measured',
                uncertainty='one-sided optimistic bound; omitted bottlenecks can only worsen it')


def counter_diagnosis(counters):
    """Exclusive cycle partition + Amdahl ceilings; overlapping flags not summed.

    Input is ONE core/program counter delta. Multi-core cumulative cycles
    must not be divided by global makespan and called utilization.
    """
    cycles, issued = counters['perf_cycles'], counters['perf_issued']
    stalls, frontend = counters['perf_stall_total'], counters['perf_x_frontend']
    exclusive = {k: v for k, v in counters.items() if k.startswith('perf_x_')
                 and k != 'perf_x_frontend'}
    values = [cycles, issued, stalls, frontend, *exclusive.values()]
    if any(type(v) is not int or v < 0 for v in values) or cycles <= 0:
        raise ValueError('counters need positive cycles and nonnegative integer deltas')
    if issued + stalls + frontend != cycles or sum(exclusive.values()) != stalls:
        raise ValueError('exclusive counter partition is inconsistent/incomplete')
    fractions = {k: v / cycles for k, v in exclusive.items()}
    fractions.update(issue=issued / cycles, frontend=frontend / cycles)
    return dict(fractions=fractions, bottlenecks=sorted(fractions, key=fractions.get, reverse=True),
        eliminate_one_component_speedup_ceiling={
            k: (cycles / (cycles - v) if v < cycles else None)
            for k, v in exclusive.items()},
        interpretation='counter attribution; overlapping mechanisms and critical paths require traces',
        performance_status='diagnostic-only')


def implementation_screen(floors, available, *, identity):
    """Check proven resource minima against physical per-region capacities.

    Keys identify actual regions/resources, e.g. SLR0/DSP or a specific SLL
    cut. Do not treat synthesis LUT counts as proven post-placement floors:
    packing/replication/optimization can change them. Wire congestion/fmax
    are unknown even when every capacity check fits.
    """
    check_identity(identity)
    if not floors:
        raise ValueError('resource minima must be supplied explicitly')
    utilization, exceeded = {}, []
    for key, item in floors.items():
        if key not in available:
            raise ValueError(f'missing physical capacity for {key}')
        cap = available[key]
        minimum, capacity = item['minimum'], cap['capacity']
        if (type(minimum) is not int or minimum < 0 or type(capacity) is not int
                or capacity <= 0 or not item.get('source') or not cap.get('source')
                or item['unit'] != cap['unit']):
            raise ValueError(f'invalid or unsourced implementation resource {key}')
        utilization[key] = minimum/capacity
        if minimum > capacity:
            exceeded.append(key)
    return dict(identity=identity, floors=floors, available=available,
                minimum_utilization=utilization, exceeded=exceeded,
                decision='reject-resource-floor' if exceeded else 'needs-placement-route',
                timing='not-measured', routability='not-measured', area_estimate=None)


def calibrated_interval(prediction, residuals):
    """90% split-conformal interval for |pred-actual|/actual residuals.

    Algebra requires pred/(1+q), pred/(1-q), not pred*(1 +/- q).
    Caller supplies a disjoint validation set from an identical target/domain;
    never consume frozen holdout records for fitting or calibration.
    """
    if (not math.isfinite(prediction) or prediction <= 0 or not residuals
            or any(not math.isfinite(r) or r < 0 for r in residuals)):
        raise ValueError('positive prediction and nonempty finite residuals required')
    rank = math.ceil((len(residuals) + 1) * .9)
    if rank > len(residuals):
        return (0.0, None)  # not enough calibration points for finite 90% coverage
    q = sorted(residuals)[rank - 1]
    return prediction / (1 + q), (prediction / (1 - q) if q < 1 else None)


class CalibratedModel:
    """Identity-bound resource model: cycles = startup + scale * bound.

    Fit and calibration records carry explicit dev/validation roles, unique
    IDs, full identity, demand and raw uninstrumented cycles. A different
    architecture is out of domain until separately calibrated; screening
    bounds remain available. Frozen holdout records are always refused.
    This is an empirical estimate, never a proof of schedulability.
    """
    def __init__(self, identity, candidate):
        check_identity(identity)
        self.identity, self.candidate = dict(identity), copy.deepcopy(candidate)
        self.groups = {}
        self.ids = set()

    def _bound(self, demand):
        result = resource_screen(demand, self.candidate, identity=self.identity)
        # Keep tied regimes distinct; do not pool memory-bound and compute-bound
        # validation residuals just because they have the same operator name.
        bounds = result['resource_lower_bounds']
        peak = result['lower_bound_cycles']
        regime = tuple(sorted(k for k, v in bounds.items() if v == peak))
        return regime, peak

    def _records(self, records, split):
        rows = []
        new_ids = set()
        for r in records:
            if (r.get('split') != split or r.get('identity') != self.identity
                    or r.get('instrumented') is not False):
                raise ValueError('fit/calibration requires matching identity, explicit split, uninstrumented data')
            rid, cycles = r['id'], r['cycles']
            if rid in self.ids or rid in new_ids:
                raise ValueError('fit/calibration IDs must be unique and disjoint')
            if type(cycles) is not int or cycles <= 0:
                raise ValueError('calibration needs raw positive device cycles')
            regime, bound = self._bound(r['demand'])
            if cycles < bound:
                raise ValueError('observed cycles violate the claimed lower bound; fix demand/capacity')
            new_ids.add(rid)
            rows.append((regime, bound, cycles, rid))
        return rows, new_ids

    def fit(self, records):
        if self.groups or self.ids:
            raise ValueError('fit once per model; create another model for another experiment')
        rows, ids = self._records(records, 'dev')
        grouped = {}
        for regime, x, y, _ in rows:
            grouped.setdefault(regime, []).append((x, y))
        groups = {}
        for regime, pairs in grouped.items():
            xs, ys = zip(*pairs)
            if len(pairs) < 3 or len(set(xs)) < 2:
                raise ValueError('each resource regime needs >=3 dev records and distinct work sizes')
            mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
            scale = max(0.0, sum((x-mx)*(y-my) for x, y in pairs) /
                        sum((x-mx)**2 for x in xs))
            startup = my - scale*mx
            if startup < 0:
                startup = 0.0
                scale = sum(x*y for x, y in pairs) / sum(x*x for x in xs)
            groups[regime] = dict(scale=scale, startup=startup, low=min(xs), high=max(xs), residuals=[])
        if not groups:
            raise ValueError('no dev records')
        self.groups, self.ids = groups, ids
        return self

    def calibrate(self, records):
        rows, ids = self._records(records, 'validation')
        updates = []
        for regime, x, y, _ in rows:
            group = self.groups.get(regime)
            if group is None or not group['low'] <= x <= group['high']:
                raise ValueError('validation outside fitted resource regime/range')
            p = max(x, group['startup'] + group['scale']*x)
            updates.append((group, abs(p-y)/y))
        for group, residual in updates:
            group['residuals'].append(residual)
        self.ids.update(ids)
        return self

    def predict(self, demand, *, identity):
        regime, x = self._bound(demand)
        group = self.groups.get(regime)
        if (identity != self.identity or group is None or not group['low'] <= x <= group['high']
                or not group['residuals']):
            return dict(status='out-of-domain', predicted_cycles=None, interval_90=None,
                        lower_bound_cycles=x, next_step='calibrate this identity/regime/range on dev + validation')
        p = max(x, group['startup'] + group['scale']*x)
        lo, hi = calibrated_interval(p, group['residuals'])
        return dict(status='calibrated-estimate', predicted_cycles=p,
                    interval_90=[max(x, lo), hi], lower_bound_cycles=x,
                    identity=self.identity, regime=list(regime),
                    performance_status='estimate; not a placement/timing or performance gate')
