#!/usr/bin/env python3
"""S1-05 acceptance: the ArchBench v0 metric checks itself.

doc/r38_stage1_tasks_and_acceptance.md S1-05. Each roadmap appendix A item
is one test, plus section 4.7's worked example, section 4.10.9 items 3 and
5, a hand-computed GEMM bound, and eta <= 1 on the measured S1-01 baseline.

Section 4.10.9 item 7 needs deliberately degraded RTL builds measured by the
S1-04 harness. Until that exists it is reported PENDING, and a PENDING test
makes this script exit non-zero: an unimplemented check is not a pass.
`--allow-pending` exists for development runs and says so in its output.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from archbench import bounds as B, canonical as K, peaks as P, score as S   # noqa: E402
from archbench import r37_workloads                                      # noqa: E402

PEAK = P.spec_peak('core-sim')
TESTS = []


class Pending(Exception):
    pass


def test(fn):
    TESTS.append(fn)
    return fn


def close(a, b, rel=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-15)


def pts(times, bounds_, status=None):
    status = status or {}
    return [S.Point(i, i.split('/')[0], status.get(i, S.PASS), times.get(i), bounds_.get(i), 'v0')
            for i in bounds_]


# --- appendix A -------------------------------------------------------------

@test
def a1_uniform_2x_faster():
    """All times halve on fixed hardware: S doubles, every eta doubles, E doubles."""
    L = {'x': 10.0, 'y': 30.0}
    T = {'x': 40.0, 'y': 90.0}
    T2 = {k: v / 2 for k, v in T.items()}
    w = {'x': .5, 'y': .5}
    E1, _, e1 = S.efficiency(pts(T, L), w)
    E2, _, e2 = S.efficiency(pts(T2, L), w)
    assert close(S.speedup(T, T2), 2.0)
    assert all(close(e2[i], 2 * e1[i]) for i in L)
    assert close(E2.value, 2 * E1.value)


@test
def a2_lower_advertised_peak_is_not_a_speedup():
    """A candidate lowering its own peak: S unchanged, E rises, AQ unchanged."""
    w_ = B.gemm('g', 'gemm', 16, 16, 16)
    low = P.spec_peak('core-sim', overrides={'fp_lane': 8, 'mem_bytes': 2})
    T = 1000.0
    L_hi, _, _ = B.lower_bound(w_, PEAK)
    L_lo, _, _ = B.lower_bound(w_, low)
    assert L_lo > L_hi                                  # own-peak efficiency looks better...
    assert S.speedup({'g': T}, {'g': T}) == 1.0         # ...the machine is not faster
    # The 1x budget is the REFERENCE configuration's ledger cost; the low-peak
    # candidate does not redefine it, so L* and hence AQ at equal time agree.
    caps = K.budget(1, K.reference_config(PEAK['rates']))
    caps_if_candidate_defined_it = K.budget(1, K.reference_config(low['rates']))
    lstar, _ = K.common_bound(w_, caps, {})
    assert lstar < K.common_bound(w_, caps_if_candidate_defined_it, {})[0]
    a = S.aq({1: {'e': {'g': T}}}, {1: {'e': {'g': lstar}}}, {'g': 1.}, {1: 1.}, {'e': 1.})
    assert close(a.value, 100 * lstar / T)


@test
def a3_redundant_traffic_does_not_raise_the_bound():
    """Semantic bytes come from the shape. Extra DRAM traffic only adds time."""
    w_ = B.copy('c', 'copy', 256)
    L, _, _ = B.lower_bound(w_, PEAK)
    T, T_redundant = 300.0, 450.0                       # the same copy reading twice
    e, er = S.eta(L, T), S.eta(L, T_redundant)
    assert er < e
    # A measured-bytes field exists on records but never feeds lower_bound.
    assert 'measured' not in B.lower_bound.__code__.co_varnames


@test
def a4_stages_and_duplicates_do_not_move_weight():
    base = S.balanced_weights({'sort': ['sort/n8'], 'add': ['add/n13']})
    staged = S.balanced_weights({'sort': ['sort/n8', 'sort/n8', 'sort/n8'], 'add': ['add/n13']})
    assert base == staged
    assert close(base['sort/n8'], .5) and close(base['add/n13'], .5)
    # A second shape of `add` splits add's mass; sort keeps its half.
    two = S.balanced_weights({'sort': ['sort/n8'], 'add': ['add/n13', 'add/n37']})
    assert close(two['sort/n8'], .5) and close(two['add/n13'] + two['add/n37'], .5)


@test
def a5_dropping_a_slow_point_fails_the_gate():
    L = {'fast': 10.0, 'slow': 10.0}
    T = {'fast': 12.0, 'slow': 100.0}
    w = {'fast': .5, 'slow': .5}
    full, _, _ = S.efficiency(pts(T, L), w)
    dropped, _, _ = S.efficiency(pts(T, L, status={'slow': 'not-measured'}), w)
    assert full.certified and not dropped.certified
    assert dropped.value == 0.0
    assert dropped.supported_only > full.value            # the tempting number...
    assert dropped.value < full.value                    # ...is not the score


@test
def a6_units_convert_consistently():
    c = S.Duration(2000, 'cycles')
    ns = S.Duration(20000, 'ns')                        # 2000 cycles at 100 MHz
    assert close(c.seconds(100e6), ns.seconds())
    assert close(ns.cycles(100e6), c.cycles())
    try:
        c.seconds()
    except ValueError:
        pass
    else:
        raise AssertionError('cycles converted to seconds without a clock')
    # MAC vs FLOP: the bound counts a MAC as one FP lane op, never two.
    g = B.gemm('g', 'gemm', 16, 16, 16)
    assert g.fp_ops == g.macs == 16 * 16 * 16


@test
def a7_faster_than_bound_is_audited_not_clipped():
    try:
        S.eta(100.0, 99.0, id='too-fast')
    except S.BoundViolation as e:
        assert 'not clipped' in str(e)
    else:
        raise AssertionError('eta > 1 was accepted')
    assert S.eta(100.0, 100.0) == 1.0


@test
def a8_active_cycles_are_not_time():
    p = S.Point('g', 'g', S.PASS, time_cycles=1000, bound_cycles=100,
                core_active_cycles=[900, 950, 800, 1000])
    assert sum(p.core_active_cycles) > p.time_cycles
    assert S.roi_cycles(p) == 1000
    bad = S.Point('g', 'g', S.PASS, time_cycles=1000, core_active_cycles=[1200])
    try:
        S.roi_cycles(bad)
    except S.ProtocolMismatch:
        pass
    else:
        raise AssertionError('a core active past the ROI was accepted')


@test
def a9_tampered_weights_are_refused():
    w = S.balanced_weights({'a': ['a/1'], 'b': ['b/1']})
    doc = {'weights': w, 'sha256': S.weights_hash(w)}
    assert S.frozen_weights(doc) == w
    forged = copy.deepcopy(doc)
    forged['weights']['a/1'] = .9
    forged['weights']['b/1'] = .1
    try:
        S.frozen_weights(forged)
    except S.WeightMismatch:
        pass
    else:
        raise AssertionError('changed weights accepted')


@test
def a10_energy_without_sensor_stays_empty():
    p = S.Point('g', 'g', S.PASS, 100, 10, 'v0')
    S.check_energy(p)                                   # None is fine
    assert p.energy_j is None
    p.energy_j, p.energy_source = 3.5, 'tdp'
    try:
        S.check_energy(p)
    except S.ProtocolMismatch:
        pass
    else:
        raise AssertionError('a TDP-derived energy was accepted')


# --- section 4.7 ------------------------------------------------------------

@test
def s47_worked_example():
    w = {'p': .5, 'q': .5}
    EA, _, ea = S.efficiency(pts({'p': 2., 'q': 2.}, {'p': 1., 'q': 1.}), w)
    EB, _, eb = S.efficiency(pts({'p': 11., 'q': 11.}, {'p': 10., 'q': 10.}), w)
    assert close(EA.value, .5) and close(EB.value, 10 / 11)
    assert close(S.speedup({'p': 11., 'q': 11.}, {'p': 2., 'q': 2.}), 5.5)
    E, A, _ = S.efficiency(pts({'p': 1 / .9, 'q': 1 / .1}, {'p': 1., 'q': 1.}), w)
    assert close(A.value, .5) and close(E.value, .18)


# --- section 4.10.9 -----------------------------------------------------------

@test
def s4109_3_bigger_design_cannot_score_in_the_smaller_budget():
    ref = K.reference_config(PEAK['rates'])
    one = K.budget(1, ref)
    assert K.fits(ref, one)
    assert not K.fits(K.scaled(ref, 2), one)
    assert K.fits(K.scaled(ref, 2), K.budget(2, ref))


@test
def s4109_5_gaming_does_not_raise_aq():
    ref = K.reference_config(PEAK['rates'])
    caps = K.budget(1, ref)
    g = B.gemm('g', 'gemm', 16, 16, 32)
    lstar, _ = K.common_bound(g, caps, {'external_bytes_per_cycle': 8})

    def AQ(t):
        return S.aq({1: {'e': {'g': t}}}, {1: {'e': {'g': lstar}}}, {'g': 1.}, {1: 1.}, {'e': 1.}).value

    base = AQ(1200.)
    assert AQ(1400.) < base                             # redundant reads cost time
    # A "fused" instruction claiming more work per issue changes nothing L* reads.
    g2 = copy.deepcopy(g)
    g2.meta['claimed_instructions'] = 1
    assert K.common_bound(g2, caps, {'external_bytes_per_cycle': 8})[0] == lstar
    # Lowering the candidate's own peak does not enter L* at all.
    assert 'peak' not in K.common_bound.__code__.co_varnames


@test
def s4109_7_degraded_configs_score_lower():
    raise Pending('needs degraded RTL builds (extra barrier, fewer RF ports) measured by S1-04')


# --- the bound itself ---------------------------------------------------------

@test
def gemm_16x16x16_fp16_hand_computed():
    # 4096 MACs over 16 (vector FPU) + 16 (matrix) lanes = 128 cycles.
    # Bytes: A and B 256 fp16 each = 1024, C 256 fp32 = 1024; 2048 / 8 = 256.
    g = B.gemm('g', 'gemm', 16, 16, 16, in_dtype='fp16', out_dtype='fp32')
    L, binding, terms = B.lower_bound(g, PEAK)
    assert terms['fp_lane'] == 128 and terms['mem_bytes'] == 256
    assert L == 256 and binding == 'mem_bytes'
    Ls, _, ts = B.lower_bound(g, PEAK, scoped=True)
    assert ts['mma_mac'] == 256 and Ls == 256


@test
def unavailable_bounds_are_refused_not_invented():
    w = r37_workloads.build()
    try:
        B.lower_bound(w['bitonic_sort'], PEAK)
    except B.BoundUnavailable:
        pass
    else:
        raise AssertionError('a sort got a bound')


@test
def aq_label_and_certification():
    w = {'a': .5, 'b': .5}
    L = {1: {'e': {'a': 10., 'b': 10.}}}
    full = S.aq({1: {'e': {'a': 20., 'b': 40.}}}, L, w, {1: 1.}, {'e': 1.})
    part = S.aq({1: {'e': {'a': 20.}}}, L, w, {1: 1.}, {'e': 1.})
    assert full.label == 'AQ-RTL-canonical' and full.certified
    assert close(full.value, 100 / (.5 * 2 + .5 * 4))
    assert part.label.endswith('NOT CERTIFIED') and part.value == 0.0
    assert 0 < full.value <= 100


@test
def measured_baseline_never_beats_its_bound():
    path = ROOT / 'sim' / 'build' / 'r38' / 'baseline_O2.json'
    if not path.exists():
        raise Pending('run `tools/r38/baseline.py --run` first (S1-01)')
    rows = {r['case']: r for r in json.loads(path.read_text())['cases']}
    work = r37_workloads.build()
    checked = 0
    for case, wl in work.items():
        if not wl.bound_available:
            continue
        row = rows.get(case)
        assert row and row['ok'], f'{case}: not in the baseline or failed'
        L, _, _ = B.lower_bound(wl, PEAK)
        S.eta(L, row['cycles'], id=case)                # raises if eta > 1
        checked += 1
    assert checked >= 30, f'only {checked} bounded cases'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-pending', action='store_true')
    args = ap.parse_args()
    failed = pending = 0
    for fn in TESTS:
        try:
            fn()
            print(f'PASS    {fn.__name__}')
        except Pending as e:
            pending += 1
            print(f'PENDING {fn.__name__}: {e}')
        except Exception as e:                           # noqa: BLE001
            failed += 1
            print(f'FAIL    {fn.__name__}: {type(e).__name__}: {e}')
    print(f'archbench: {len(TESTS) - failed - pending} pass, {failed} fail, {pending} pending')
    if pending and args.allow_pending and not failed:
        print('(development run: pending tests allowed; S1-05 is NOT accepted)')
        return 0
    return 1 if failed or pending else 0


if __name__ == '__main__':
    sys.exit(main())
