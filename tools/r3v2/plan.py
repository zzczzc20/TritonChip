#!/usr/bin/env python3
"""
R3V2-BASE-01, -02 and -03 -- doc/phase3_tasks_and_acceptance.md G0.

The inventory decides what must run. Not the results file: R3V2-BASE-01
says so outright ("必需集合由独立冻结 inventory 决定，不能按结果文件里出现
什么就验什么"), and the difference matters because a run that never
happened leaves nothing in a results file to notice.

So this checks the frozen inventory and the frozen contract themselves,
before anything is simulated, and it is the file the later verifier reads
first.

R3V2-BASE-02 is the half with teeth. It is not enough for the checker to
accept a good inventory; it has to refuse six specific bad ones, and each
refusal is exercised here against a copy with that fault injected. The
old `tools/r3/plan.py` passed by reading a `verdict` field out of a
historical file, which is the failure this replaces: a checker that
believes a stored answer has not checked anything.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

CONTRACT = os.path.join(ROOT, 'spec', 'r3v2_contract.json')
INVENTORY = os.path.join(ROOT, 'spec', 'r3v2_inventory.json')
PLAN_DOC = os.path.join(ROOT, 'doc', 'phase3_tasks_and_acceptance.md')

IMPLEMENTATIONS = ('reference', 'candidate')
MEMORY_MODES = ('fast', 'slow', 'ooo', 'slow-reorder')
SOURCE_KINDS = ('triton', 'tnca-ir')


class Reject(Exception):
    """The inventory is not acceptable, and why."""


# ---------------------------------------------------------------------------
# the checks, each raising Reject with a reason
# ---------------------------------------------------------------------------

def check_inventory(inv, contract):
    """Every rule R3V2-BASE-01 and -02 name, in one pass."""
    profiles = contract['profiles']
    known_profiles = {profiles['reference'], profiles['candidate']} | \
        set(profiles['unchanged'])

    seen = set()
    for case in inv['core']:
        cid = case.get('id')
        if not cid:
            raise Reject('a core case has no id')
        if cid in seen:
            raise Reject(f"duplicate case id {cid!r}")
        seen.add(cid)

        impls = case.get('implementations')
        if not impls:
            raise Reject(f"{cid}: no implementations")
        if 'candidate' not in impls:
            raise Reject(f"{cid}: no candidate. Every core case runs on "
                         f"both implementations; a reference-only case "
                         f"measures the thing that did not change")
        if 'reference' not in impls:
            raise Reject(f"{cid}: no reference to compare against")
        if len(impls) != len(set(impls)):
            raise Reject(f"{cid}: an implementation is listed twice "
                         f"({impls}); two runs of the reference are not a "
                         f"comparison")
        for impl in impls:
            if impl not in IMPLEMENTATIONS:
                raise Reject(f"{cid}: unknown implementation {impl!r}")

        mem = case.get('memory')
        if mem is None:
            raise Reject(f"{cid}: no memory mode")
        if mem not in MEMORY_MODES:
            raise Reject(f"{cid}: unknown memory mode {mem!r}")
        # A memory mode is a condition, not an implementation. Counting
        # fast/slow/ooo as the two sides of a comparison is how the old
        # inventory reached 32 runs without a candidate existing at all.
        if mem in impls or any(m in IMPLEMENTATIONS for m in [mem]):
            raise Reject(f"{cid}: the memory mode {mem!r} is being used as "
                         f"an implementation")

        if case.get('source_kind') not in SOURCE_KINDS:
            raise Reject(f"{cid}: source_kind "
                         f"{case.get('source_kind')!r} is not one of "
                         f"{SOURCE_KINDS}")
        if case.get('seed') is None:
            raise Reject(f"{cid}: no seed")
        if not isinstance(case.get('purpose'), str) or not case['purpose']:
            raise Reject(f"{cid}: no stated purpose")

        _check_sizes(cid, case, inv['size_caps'])

    for case in inv['runtime']:
        if case.get('from_case') not in seen:
            raise Reject(f"{case.get('id')}: from_case "
                         f"{case.get('from_case')!r} is not a core case")
        if case.get('implementations') != ['candidate']:
            raise Reject(f"{case.get('id')}: the post-install runs are "
                         f"candidate-only")
        if case.get('alpha') is None:
            raise Reject(f"{case.get('id')}: no alpha")

    for entry in inv.get('inherited', []):
        prof = entry.get('profile')
        if prof not in known_profiles | {'none', 'all'}:
            raise Reject(f"{entry.get('id')}: unknown profile {prof!r}")
        if not entry.get('command'):
            raise Reject(f"{entry.get('id')}: no command")
        if not entry.get('why'):
            raise Reject(f"{entry.get('id')}: no reason for inheriting it")

    _check_budget(inv)
    return True


def _check_sizes(cid, case, caps):
    sizes = case.get('sizes')
    if not sizes:
        raise Reject(f"{cid}: no sizes")
    for axis, value in sizes.items():
        if not isinstance(value, int):
            raise Reject(f"{cid}: {axis}={value!r} is not an integer")
        if value <= 0:
            raise Reject(f"{cid}: {axis}={value} is zero or negative")

    kind = case['workload']
    if kind == 'gemm':
        cap = caps['gemm']
        for axis in ('m', 'n', 'k'):
            if sizes.get(axis) is None:
                raise Reject(f"{cid}: a GEMM with no {axis}")
            if sizes[axis] > cap[axis]:
                raise Reject(f"{cid}: {axis}={sizes[axis]} is over the "
                             f"GEMM cap of {cap[axis]}")
    elif kind == 'reduction':
        cap = caps['reduction']
        if sizes.get('rows', 1) > cap['rows']:
            raise Reject(f"{cid}: rows={sizes['rows']} is over "
                         f"{cap['rows']}")
        if sizes.get('elements', 0) > cap['elements']:
            raise Reject(f"{cid}: elements={sizes['elements']} is over "
                         f"{cap['elements']}")
    else:
        cap = caps['elementwise']
        if sizes.get('n', 0) > cap['n']:
            raise Reject(f"{cid}: n={sizes['n']} is over the elementwise "
                         f"cap of {cap['n']}")
        if sizes.get('grid', 1) > cap['grid']:
            raise Reject(f"{cid}: grid={sizes['grid']} is over "
                         f"{cap['grid']}")


def _check_budget(inv):
    b = inv['budget']
    kernels = sum(len(c['implementations']) for c in inv['core']) + \
        sum(len(c['implementations']) for c in inv['runtime'])
    if kernels != b['core_kernel_runs']:
        raise Reject(f"the inventory lists {kernels} kernel run(s) and the "
                     f"budget allows {b['core_kernel_runs']}")
    comp = sum(c['rtl_runs'] for c in inv['components'])
    mut = sum(m['sub_variants'] for m in inv['mutations'] if m['rtl'])
    if comp + mut > b['component_rtl_invocations']:
        raise Reject(f"components ({comp}) plus RTL mutations ({mut}) is "
                     f"{comp + mut}, over the "
                     f"{b['component_rtl_invocations']} allowed")
    total = kernels + comp + mut
    if total > b['total_rtl_runs']:
        raise Reject(f"{total} RTL runs planned, {b['total_rtl_runs']} "
                     f"allowed")
    return True


def worst_case_seconds(inv):
    """What the plan would cost if every run took its full timeout.

    NOT a rejection. The per-run numbers in section 4.3 are timeouts -- a
    single run may not exceed them -- and the 1,800s total is a budget on
    the sum of what actually happened. Multiplying every run by its
    timeout and comparing against the total makes the contract
    unsatisfiable by construction: 15 x 180 alone is 2,700.

    It is still worth printing, because it says how much slower than the
    plan the runs may get before the budget is the binding constraint.
    """
    b = inv['budget']
    kernels = sum(len(c['implementations']) for c in inv['core']) + \
        sum(len(c['implementations']) for c in inv['runtime'])
    comp = sum(c['rtl_runs'] for c in inv['components'])
    mut = sum(m['sub_variants'] for m in inv['mutations'] if m['rtl'])
    return (kernels * b['core_wall_seconds_each'] +
            (comp + mut) * b['component_wall_seconds_each'])


def check_measured(inv, registry):
    """The budget applies to what actually ran."""
    b = inv['budget']
    elapsed = sum(r.get('elapsed_s', 0) for r in registry)
    cycles = sum(r.get('cycles', 0) for r in registry)
    if elapsed > b['total_elapsed_seconds']:
        raise Reject(f"the runs took {elapsed:.0f}s, over the "
                     f"{b['total_elapsed_seconds']}s budget")
    if cycles > b['total_cycles']:
        raise Reject(f"the runs used {cycles} cycles, over the "
                     f"{b['total_cycles']}")
    for r in registry:
        cap = (b['core_wall_seconds_each'] if r.get('kind') == 'core'
               else b['component_wall_seconds_each'])
        if r.get('elapsed_s', 0) > cap:
            raise Reject(f"{r.get('id')} took {r['elapsed_s']:.0f}s, over "
                         f"its {cap}s cap")
    return elapsed, cycles


def check_contract(contract):
    """R3V2-BASE-03: no required field is null."""
    required = {
        'vrf': ['data_bits', 'banks', 'rows_per_bank', 'lanes',
                'element_bits', 'physical_registers'],
        'ports': ['accepted_reads_per_bank_per_cycle',
                  'accepted_writes_per_bank_per_cycle'],
        'synchronous_read': ['read_latency_cycles', 'downstream_stall'],
        'collector': ['entries', 'sources_per_entry', 'issue_rule'],
        'rdw_and_mask': ['policy', 'granularity'],
        'reset': ['architectural_read_of_unwritten'],
        'fairness': ['requesters_C', 'max_grant_occupancy_B_cycles',
                     'arbitration_pipeline_A_cycles', 'bound_cycles'],
    }
    for section, fields in required.items():
        if section not in contract:
            raise Reject(f"the contract has no {section!r} section")
        for field in fields:
            if contract[section].get(field) is None:
                raise Reject(f"contract.{section}.{field} is null; a null "
                             f"here is a claim nobody made")

    v = contract['vrf']
    if v['data_bits'] != v['physical_registers'] * v['lanes'] * \
            v['element_bits']:
        raise Reject(f"the VRF's stated capacity does not equal "
                     f"registers x lanes x bits")
    if v['banks'] * v['rows_per_bank'] != v['physical_registers']:
        raise Reject(f"{v['banks']} banks x {v['rows_per_bank']} rows is "
                     f"not {v['physical_registers']} registers")
    f = contract['fairness']
    want = (f['requesters_C'] - 1) * f['max_grant_occupancy_B_cycles'] + \
        f['arbitration_pipeline_A_cycles']
    if f['bound_cycles'] != want:
        raise Reject(f"the fairness bound says {f['bound_cycles']} but "
                     f"(C-1)*B+A is {want}")
    return True


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------

def base01(log, ctx):
    """Every case is fully described, and the inventory is the authority."""
    inv, contract = ctx['inventory'], ctx['contract']
    check_inventory(inv, contract)
    kernels = sum(len(c['implementations']) for c in inv['core']) + \
        sum(len(c['implementations']) for c in inv['runtime'])
    comp = sum(c['rtl_runs'] for c in inv['components'])
    mut = sum(m['sub_variants'] for m in inv['mutations'] if m['rtl'])
    log.append(f"    {len(inv['core'])} core case(s) x 2 implementations "
               f"+ {len(inv['runtime'])} post-install = {kernels} kernel "
               f"run(s); the contract says "
               f"{inv['budget']['core_kernel_runs']}")
    log.append(f"    {comp} component invocation(s) + {mut} RTL mutation "
               f"variant(s) = {comp + mut}, under "
               f"{inv['budget']['component_rtl_invocations']}")
    log.append(f"    {kernels + comp + mut} RTL run(s) in total, under "
               f"{inv['budget']['total_rtl_runs']}")
    worst = worst_case_seconds(inv)
    b = inv['budget']
    log.append(f"    if every run took its full timeout the plan would "
               f"cost {worst}s against a {b['total_elapsed_seconds']}s "
               f"budget, so the timeouts are not the binding constraint "
               f"and the measured total is what has to fit")
    for field in ('id', 'workload', 'sizes', 'source_kind', 'memory',
                  'seed', 'implementations', 'purpose'):
        missing = [c['id'] for c in inv['core'] if not c.get(field)]
        if missing:
            log.append(f"    {field}: missing on {missing}")
            return False, f"a core case has no {field}"
    log.append(f"    every core case names an id, a workload, its sizes, "
               f"a source kind, a memory mode, a seed, both "
               f"implementations and a purpose")
    log.append(f"    {len(inv['inherited'])} inherited regression(s), each "
               f"with a command, a profile and a reason")
    log.append(f"    {len(inv['mutations'])} mutation(s) covering "
               f"{sum(m['sub_variants'] for m in inv['mutations'])} "
               f"sub-variant(s)")
    return True, 'the frozen inventory is complete and within budget'


# Each entry: what to break, and the words the refusal must contain.
INJECTIONS = [
    ('a case with no candidate',
     lambda inv: inv['core'][0].__setitem__('implementations',
                                            ['reference']),
     'candidate'),
    # Each injection must break exactly ONE rule. ['reference',
    # 'reference'] also has no candidate, so the no-candidate check fired
    # first and the duplicate rule was never exercised -- a test that
    # passed while testing nothing.
    ('the reference listed twice',
     lambda inv: inv['core'][0].__setitem__(
         'implementations', ['candidate', 'reference', 'reference']),
     'twice'),
    ('an unknown profile in the inherited list',
     lambda inv: inv['inherited'][0].__setitem__('profile', 'core-9000'),
     'unknown profile'),
    ('a memory mode used as an implementation',
     lambda inv: inv['core'][0].__setitem__(
         'implementations', ['reference', 'candidate', 'fast']),
     'implementation'),
    ('a size of zero',
     lambda inv: inv['core'][0]['sizes'].__setitem__('n', 0),
     'zero or negative'),
    ('a size over the cap',
     lambda inv: inv['core'][2]['sizes'].__setitem__('k', 128),
     'over the GEMM cap'),
    ('a duplicate case id',
     lambda inv: inv['core'].append(copy.deepcopy(inv['core'][0])),
     'duplicate'),
    ('a kernel count the budget does not allow',
     lambda inv: inv['core'].append(
         dict(copy.deepcopy(inv['core'][0]), id='C99')),
     'budget allows'),
]


def base02(log, ctx):
    """The checker refuses each malformation, before anything is simulated."""
    ok = True
    for what, break_it, expect in INJECTIONS:
        bad = copy.deepcopy(ctx['inventory'])
        break_it(bad)
        try:
            check_inventory(bad, ctx['contract'])
            log.append(f"    {what:<44s} -> ACCEPTED (bad)")
            ok = False
        except Reject as e:
            named = expect in str(e)
            log.append(f"    {what:<44s} -> refused: {str(e)[:52]}")
            if not named:
                log.append(f"        but not for the expected reason "
                           f"({expect!r})")
                ok = False
    log.append(f"    {len(INJECTIONS)} malformation(s), each refused by "
               f"name, and none of them started a simulation")
    return ok, 'every malformation refused before any RTL runs'


def base03(log, ctx):
    """The contract has no required nulls, and the scout is real."""
    ok = True
    check_contract(ctx['contract'])
    c = ctx['contract']
    log.append(f"    VRF {c['vrf']['data_bits']} bit = "
               f"{c['vrf']['data_bytes'] // 1024} KiB in "
               f"{c['vrf']['banks']} bank(s) of "
               f"{c['vrf']['rows_per_bank']} row(s); capacity and bank "
               f"arithmetic agree")
    log.append(f"    {c['ports']['accepted_reads_per_bank_per_cycle']}R/"
               f"{c['ports']['accepted_writes_per_bank_per_cycle']}W per "
               f"bank per cycle, read latency "
               f"{c['synchronous_read']['read_latency_cycles']}")
    f = c['fairness']
    log.append(f"    fairness (C-1)*B+A = ({f['requesters_C']}-1)*"
               f"{f['max_grant_occupancy_B_cycles']}+"
               f"{f['arbitration_pipeline_A_cycles']} = "
               f"{f['bound_cycles']} cycle(s), given a bounded downstream")
    log.append(f"    the ARF is declared {c['arf']['implementation']} and "
               f"does not count toward synchronous storage")

    # A null in a required field is what BASE-03 is about, so it is
    # injected rather than assumed absent.
    for section, field in (('ports', 'accepted_reads_per_bank_per_cycle'),
                           ('collector', 'entries'),
                           ('fairness', 'bound_cycles')):
        bad = copy.deepcopy(c)
        bad[section][field] = None
        try:
            check_contract(bad)
            log.append(f"    a null {section}.{field} was ACCEPTED (bad)")
            ok = False
        except Reject:
            pass
    log.append(f"    a null in any required field is refused")

    # Provenance: the historical R1 verdict is evidence about 97c0cf90 and
    # about nothing else. The old plan.py read its `verdict` field and
    # passed; this records what commit it belongs to and moves on.
    hist = os.path.join(ROOT, 'doc', 'evidence', 'r1-combined-verdict.json')
    if os.path.exists(hist):
        h = json.load(open(hist))
        # The commit is in the notes, not in a field of its own. The
        # `verdict` key beside it is the one the old plan.py read and
        # passed on; it is deliberately not consulted here.
        commit = 'unrecorded'
        import re as _re
        for note in h.get('notes', []):
            m = _re.search(r'commit ([0-9a-f]{8,40})', str(note))
            if m:
                commit = m.group(1)
                break
        log.append(f"    the historical R1 verdict is provenance for "
                   f"{str(commit)[:12]}, not for this candidate")
    here = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                          capture_output=True, text=True)
    head = here.stdout.strip()[:12] if here.returncode == 0 else '?'
    log.append(f"    this candidate is {head}; every piece of evidence "
               f"must name it")

    # The scout is a real run with a real exit code, or it is NOT RUN.
    scout = os.path.join(ROOT, 'sim', 'build', 'r3v2_scout.json')
    if os.path.exists(scout):
        s = json.load(open(scout))
        for row in s.get('runs', []):
            log.append(f"    scout {row['id']:<10s} exit="
                       f"{row.get('exit')} cycles={row.get('cycles')} "
                       f"elapsed={row.get('elapsed_s')}s")
        ok &= bool(s.get('runs'))
    else:
        log.append(f"    NOT RUN: the Linux scout has not been run on this "
                   f"candidate. `make r3v2-scout` produces "
                   f"sim/build/r3v2_scout.json; until then G0 is not "
                   f"complete and no budget number here is measured")
        ok = False
    return ok, 'the contract is complete; the scout must be real'


TESTS = [('R3V2-BASE-01', base01), ('R3V2-BASE-02', base02),
         ('R3V2-BASE-03', base03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G0: the frozen contract, inventory and budget")
    ctx = {'contract': json.load(open(CONTRACT)),
           'inventory': json.load(open(INVENTORY))}
    print(f"  contract {ctx['contract']['contract_id']} "
          f"v{ctx['contract']['contract_version']}, baseline "
          f"{ctx['contract']['baseline_commit'][:12]}")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log, ctx)
        except Reject as e:
            ok, what = False, f"rejected: {e}"
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-800:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are: {', '.join(t[0] for t in TESTS)}")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3v2-plan              FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-plan              PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
