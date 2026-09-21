#!/usr/bin/env python3
"""
R35-SAFE-01, SAFE-02 and MUT-01 -- phase3_5_tasks_and_acceptance.md 7.

That the inherited safety cases still hold, that each pass has a
positive case and a boundary negative one, and that every mutation the
contract lists is caught by the checker it names.

The mutation rule, from section 7: "变异用隔离副本，clean 基线必须通过;
不能用无关 build failure 算检出". So each one is injected into a copy,
the clean baseline has to pass first, and a run that merely fails to
build has not been detected by anything.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, HERE)
import r1lib as R                                          # noqa: E402
import build_tests as B                                    # noqa: E402

SPEC = json.load(open(os.path.join(ROOT, 'spec',
                                   'r35_required_cases.json')))
PROFILE = json.load(open(os.path.join(
    ROOT, 'spec', 'r35_workloads.json')))['hardware_precondition']['profile']


# ---------------------------------------------------------------------------
# SAFE-01: what R1, R2 and R3 already guarantee, at every level

INHERITED = [
    ('R1 ownership and footprint', ['make', 'r1-resource-tests']),
    ('the Triton frontend, and what it refuses', ['make', 'triton-e2e']),
    ('R2 artifact and ABI', ['make', 'r2-artifact-tests']),
    ('R2 cache identity', ['make', 'r2-cache-tests']),
    ('R2 faults', ['make', 'r2-fault-tests']),
]


def safe01(log, inherited=True):
    """The inherited cases, and the verifier running at every level.

    `inherited=False` is for the mutation harness and nothing else. N04
    breaks the ownership verifier, and re-running five inherited suites
    inside a throwaway copy tests the copy rather than the mutation: one
    of them timed out in an untouched baseline with every counter at
    zero, which is a lab that failed to build its own bench and says
    nothing about the code. REG-01 runs those suites on the real tree,
    which is where they mean something.
    """
    ok = True
    env = dict(os.environ)
    env.setdefault('TRITON_PYTHON', sys.executable)
    if not inherited:
        log.append('    the inherited suites are skipped here: this run is '
                   'the mutation harness, and REG-01 runs them on the '
                   'real tree')
    for name, argv in (INHERITED if inherited else []):
        r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=7200)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-1:] or ['']
        log.append(f"    {name:<38s} {'PASS' if r.returncode == 0 else 'FAIL'}"
                   f"   {tail[0].strip()[:46]}")
        if r.returncode != 0:
            # Keep the reason. A suite that fails and says only "FAIL"
            # sends the next reader back to re-run it by hand.
            lines = [l for l in (r.stdout + r.stderr).splitlines()
                     if l.strip()]
            for l in lines[-8:]:
                log.append(f"        {l.strip()[:96]}")
        ok &= r.returncode == 0

    # The ownership verifier runs at every optimisation level, on the
    # final physical allocation. Section 7 asks for that specifically.
    import workloads as W
    from tnca import aot
    for level in (None, 0, 1, 2):
        opts = {} if level is None else {'opt_level': level}
        art = aot.compile_aot(
            W.gemm_program(16, 16, 16,
                           a_desc={'tile': 0x2000, 'map': 0x1000},
                           b_desc={'tile': 0x3000, 'map': 0x1100},
                           out=0x4000, roles=1),
            kernel_name='g', signature={}, source_kind='tnca-ir',
            profile=PROFILE, options=opts,
            output=os.path.join(tempfile.mkdtemp(), 'g.tnca'))
        res = art.manifest['resources']
        log.append(f"    the GEMM at "
                   f"{'B0' if level is None else 'O' + str(level):<3s} "
                   f"compiled with ownership checked: "
                   f"{len(art.asm['hex'].splitlines())} word(s), "
                   f"{res.get('barriers')} barrier(s)")
    return ok, 'the inherited cases hold and ownership is checked at O0..O2'


def safe02(log):
    """A positive and a boundary negative case for each pass."""
    import workloads as W
    import r35_workloads as WW
    from tnca import aot
    from tnca.adapters import r1 as A
    from tnca_cc.optimization import (strength_reduce, fold_broadcast_compare,
                                      drop_dead_load_other,
                                      narrow_lane0_store)
    ok = True
    target, _ = A.target_for(PROFILE, A.Provenance())

    def kernel_of(src):
        k, b = A.new_kernel('probe')
        src(b, target)
        return k

    # strength_reduce: a power of two goes, a non-power of two stays.
    def mul_by(v):
        k, b = A.new_kernel('probe')
        base = b.special('arg_base')
        p = b.sload(base, 0, 'd')
        n = b.sbin('mul', p, b.const(v))
        b.store(p, b.splat(n), mask=None) if False else None
        b.exit()
        return k
    for v, expect in ((16, True), (12, False), (1, True), (3, False)):
        k = mul_by(v)
        got = strength_reduce(k)['sites']
        log.append(f"    multiply by {v:<3d} -> "
                   f"{'a shift' if got else 'left alone'}"
                   f"{'' if bool(got) == expect else '   UNEXPECTED'}")
        ok &= bool(got) == expect

    # fold_broadcast_compare: a constant folds, a dynamic value does not.
    k = kernel_of(W.tnca_reduction(15))
    n_const = fold_broadcast_compare(k)['sites']
    k2, b2 = A.new_kernel('dyn')
    base = b2.special('arg_base')
    dyn = b2.sload(base, 0, 'sw')
    idx = b2.arange()
    b2.vcmp('lt', idx, b2.splat(dyn))
    b2.exit()
    n_dyn = fold_broadcast_compare(k2)['sites']
    log.append(f"    compare against a constant: {n_const} folded; "
               f"against a loaded value: {n_dyn} folded")
    ok &= n_const > 0 and n_dyn == 0

    # drop_dead_load_other: dropped when unobservable, kept when not.
    k = kernel_of(W.tnca_reduction(15))
    n_ok = drop_dead_load_other(k)['sites']
    # A load whose value reaches a gather index is observable in every
    # lane, so `other` must stay.
    k3, b3 = A.new_kernel('obs')
    base = b3.special('arg_base')
    p = b3.sload(base, 0, 'd')
    idx = b3.arange()
    m = b3.vcmp('lt', idx, b3.splat(b3.const(15)))
    v = b3.load(p, mask=m, other=0, dtype='i32')
    b3.gather(p, v, mask=None, dtype='f32')
    b3.exit()
    r3 = drop_dead_load_other(k3)
    log.append(f"    `other` dropped where nothing observes it: {n_ok}; "
               f"where a gather reads every lane: {r3['sites']} "
               f"({r3['kept'][:1]})")
    ok &= n_ok > 0 and r3['sites'] == 0

    # narrow_lane0_store: a lane-0 broadcast narrows, a full store does not.
    k = kernel_of(W.tnca_reduction(15))
    n_lane0 = narrow_lane0_store(k)['sites']
    k4 = kernel_of(WW.tnca_view_elementwise())
    n_full = narrow_lane0_store(k4)['sites']
    log.append(f"    a lane-0 broadcast store narrows: {n_lane0}; "
               f"a full-width store does not: {n_full}")
    ok &= n_lane0 == 1 and n_full == 0

    # A dynamic argument must not be folded into a constant.
    art1 = aot.compile_aot(
        W.triton_masked_axpy(), profile=PROFILE,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, options={'opt_level': 2},
        output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))
    asm = art1.asm['asm']
    reads_alpha = 's.ld.w' in asm and 'arg_base+32' in asm
    log.append(f"    the dynamic alpha is still loaded from the argument "
               f"buffer at O2, not folded: {reads_alpha}")
    ok &= reads_alpha

    # Section 7: "重命名函数/寄存器/符号不改变优化选择或结果". The same
    # computation written with every name changed must compile to the
    # same machine code -- an optimiser that keyed on a name would be
    # making a decision it cannot justify.
    import triton
    import triton.language as tl

    @triton.jit
    def zzz_renamed(aa, bb, cc, dd, ee, QQ: tl.constexpr):
        ff = tl.program_id(0)
        gg = ff * QQ + tl.arange(0, QQ)
        hh = gg < dd
        ii = tl.load(aa + gg, mask=hh, other=0.0)
        jj = tl.load(bb + gg, mask=hh, other=0.0)
        tl.store(cc + gg, ii * ee + jj, mask=hh)

    art2 = aot.compile_aot(
        zzz_renamed, profile=PROFILE,
        signature={'aa': '*fp32', 'bb': '*fp32', 'cc': '*fp32',
                   'dd': 'i32', 'ee': 'fp32', 'QQ': 'constexpr'},
        constexprs={'QQ': 16}, options={'opt_level': 2},
        output=os.path.join(tempfile.mkdtemp(), 'b.tnca'))
    same = art1.asm['hex'] == art2.asm['hex']
    log.append(f"    the same kernel with every name changed compiles to "
               f"the same {len(art2.asm['hex'].splitlines())} word(s): "
               f"{same}")
    ok &= same
    return ok, 'each pass has a positive case and a boundary negative one'


# ---------------------------------------------------------------------------
# MUT-01

def patch(lab, rel, old, new, count=1):
    p = os.path.join(lab.dir, rel)
    text = open(p).read()
    if text.count(old) != count:
        raise AssertionError(f"{rel}: {text.count(old)} occurrence(s) of the "
                             f"mutation site, expected {count}")
    open(p, 'w').write(text.replace(old, new, count))


# Build products a lab needs but a source copy does not carry. Given per
# mutation, not globally: putting a prebuilt bench into a lab that did
# not ask for one leaves `make` with a half-populated sim/build and a
# dependency graph that no longer describes the tree, which made an
# unrelated suite fail in a clean baseline and read as "nothing can be
# concluded".
BENCH = ('sim/build/tb_core_vrf.vvp', 'sim/build/t29_vrfbank.hex')


def run_in(lab, argv, timeout=7200):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [os.path.join(lab.dir, 'python'),
         os.path.join(lab.dir, 'compiler', 'python')])
    env.setdefault('TRITON_PYTHON', sys.executable)
    return subprocess.run([sys.executable] + argv, cwd=lab.dir,
                          capture_output=True, text=True, timeout=timeout,
                          env=env)


def suite(name, case):
    return [os.path.join('tools', 'r35', name), '--only', case]


def n01(lab):
    """A wrong LinearLayout basis.

    Wrong, not malformed. The first version made every basis row map to
    `row`, which does not span the output and which upstream answers by
    ending the process -- so the checker produced no output at all and
    detected nothing. Section 7 is explicit that a build failure is not
    a detection, and a SIGABRT is less than that.

    This swaps two rows instead. The layout is still a bijection over
    sixteen positions and upstream is perfectly happy with it; it is
    just not the transpose, which is what the element-by-element
    comparison is there to notice.
    """
    patch(lab, 'tools/r35/layout_tests.py',
          "                for k in range(cbits):\n"
          "                    bases.append([0, 1 << k])\n"
          "                for k in range(rbits):\n"
          "                    bases.append([1 << k, 0])",
          "                for k in range(rbits):\n"
          "                    bases.append([1 << k, 0])\n"
          "                for k in range(cbits):\n"
          "                    bases.append([0, 1 << k])   # N01")
    return suite('layout_tests.py', 'R35-LL-01'), 'R35-LL-01'


def n02(lab):
    """Drop the tail mask: the coverage check must notice."""
    patch(lab, 'compiler/python/tnca_cc/upstream/layout.py',
          "        return {'distinct_outputs': len(hits), "
          "'expected_outputs': total,\n"
          "                'covered': len(hits) == total,",
          "        return {'distinct_outputs': len(hits), "
          "'expected_outputs': total,\n"
          "                'covered': True,   # N02: everything is covered\n"
          "                '_covered_real': len(hits) == total,")
    return suite('layout_tests.py', 'R35-LL-02'), 'R35-LL-02'


def n03(lab):
    """Mark an effectful operation Pure."""
    patch(lab, 'compiler/python/tnca_cc/optimization.py',
          "PURE_NO_EFFECT = frozenset(_is_pure.__globals__['PURE_KINDS']) | {\n"
          "    'splat', 'vmovi', 'arange', 'scmp', 'vcmp',\n"
          "}",
          "PURE_NO_EFFECT = frozenset(_is_pure.__globals__['PURE_KINDS']) | {\n"
          "    'splat', 'vmovi', 'arange', 'scmp', 'vcmp', 'store',\n"
          "}   # N03: a store is not pure")
    return suite('opt_tests.py', 'R35-OPT-03'), 'R35-OPT-03'


def n04(lab):
    """Release an async value early.

    A barrier wait is what holds an asynchronous operation's resources
    until it has finished. Calling one pure lets the dead-value sweep
    delete it, and every window and accumulator it was protecting is
    then released while the engine is still using them. The ownership
    checker runs at every optimisation level and is what must refuse
    this.
    """
    patch(lab, 'compiler/python/tnca_cc/optimization.py',
          "    'splat', 'vmovi', 'arange', 'scmp', 'vcmp',\n"
          "}",
          "    'splat', 'vmovi', 'arange', 'scmp', 'vcmp',\n"
          "    'barrier_wait',   # N04: a wait holds nothing\n"
          "}")
    return (['tools/r35/safety_tests.py', '--only', 'R35-SAFE-01',
             '--no-inherited'], 'R35-SAFE-01')


def n05(lab):
    """Ignore the bank in the cost model."""
    patch(lab, 'compiler/python/tnca_cc/bankcost.py',
          "def bank_of(phys, banks=None):\n"
          '    """Which bank a physical register lives in."""\n'
          "    if banks is None:\n"
          "        banks = hardware()['banks']\n"
          "    return int(phys) % banks",
          "def bank_of(phys, banks=None):\n"
          '    """Which bank a physical register lives in."""\n'
          "    if banks is None:\n"
          "        banks = hardware()['banks']\n"
          "    return 0   # N05: every register is in bank 0")
    return suite('alloc_tests.py', 'R35-ALLOC-01'), 'R35-ALLOC-01'


def n06(lab):
    """Leave the optimisation level out of the cache key."""
    patch(lab, 'python/tnca/aot.py',
          "DIAGNOSTIC_OPTIONS = frozenset({'debug', 'dump_dir'})",
          "DIAGNOSTIC_OPTIONS = frozenset({'debug', 'dump_dir', "
          "'opt_level', 'no_ll'})   # N06")
    return suite('package_tests.py', 'R35-PKG-02'), 'R35-PKG-02'


def n07(lab):
    """Pass a model cycle off as a measured one."""
    patch(lab, 'tools/r35/perf.py',
          "def _cycles_of(ev):\n"
          "    return getattr(ev, 'device_cycles', None)",
          "def _cycles_of(ev):\n"
          "    return 1   # N07: a number that never touched the device")
    return suite('perf.py', 'R35-PERF-01'), 'R35-PERF-01'


def plant_evidence(lab, forge=False):
    """A complete evidence tree the verifier accepts, or one it must not.

    Synthesised from the same REQUIRED map the verifier reads, so the
    clean version is genuinely acceptable and the forged one is
    genuinely a regression -- rather than two trees refused for the same
    missing-file reason, which would prove nothing about forgery.
    """
    sys.path.insert(0, os.path.join(lab.dir, 'tools', 'r35'))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        've35', os.path.join(lab.dir, 'tools', 'r35', 'verify_evidence.py'))
    ve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ve)

    ev = os.path.join(lab.dir, 'ev')
    for sub in ('runs', 'contracts', 'inherited'):
        os.makedirs(os.path.join(ev, sub), exist_ok=True)
    import hashlib as H
    frozen = {}
    for name in ('r35_workloads.json', 'r35_required_cases.json',
                 'r35_toolchain.json'):
        src = os.path.join(lab.dir, 'spec', name)
        shutil.copy2(src, os.path.join(ev, 'contracts', name))
        frozen[name] = H.sha256(open(src, 'rb').read()).hexdigest()
    shutil.copy2(os.path.join(lab.dir, 'doc', 'evidence', 'r3v2',
                              'acceptance-linux.json'),
                 os.path.join(ev, 'inherited', 'r3v2-acceptance.json'))

    inv = json.load(open(os.path.join(lab.dir, 'spec',
                                      'r35_workloads.json')))
    # A gain table that meets every one of section 6.3's conditions.
    rows = [
        {'id': 'L01', 'correct': True, 'spread': 0,
         'modes': {'B0': {'cycles': 228, 'words': 30},
                   'O2': {'cycles': 206, 'words': 24},
                   'O2-no-LL': {'cycles': 206, 'words': 24}}},
        {'id': 'L03', 'correct': True, 'spread': 0,
         'modes': {'B0': {'cycles': 215, 'words': 63},
                   'O2': {'cycles': 61, 'words': 9},
                   'O2-no-LL': {'cycles': 208, 'words': 58}}},
        {'id': 'L04', 'correct': True, 'spread': 0,
         'modes': {'B0': {'cycles': 74, 'words': 22},
                   'O2': {'cycles': 63, 'words': 9},
                   'O2-no-LL': {'cycles': 63, 'words': 9}}},
    ]
    if forge:
        # A case that is WRONG, reported as correct. Section 7 lists
        # "仅修改 summary PASS" and "漏慢 case" as things to refuse.
        rows.append({'id': 'L06', 'correct': True, 'spread': 0,
                     'modes': {'B0': {'cycles': 3241, 'words': 183},
                               'O2': {'cycles': 3900, 'words': 95},
                               'O2-no-LL': {'cycles': 3900, 'words': 95}}})
    raw, checks = {}, []
    for checker, ids in sorted(ve.REQUIRED.items()):
        cases = []
        for cid in ids:
            c = {'id': cid, 'passed': True, 'detail': 'ok'}
            if cid == 'R35-PERF-01':
                c['rows'] = rows
            cases.append(c)
        raw[checker] = cases
        checks.append({'checker': checker, 'exit': 0, 'elapsed_s': 1,
                       'cases': [dict(c) for c in cases]})
    for checker, cases in raw.items():
        json.dump(cases, open(os.path.join(ev, 'runs', f'{checker}.json'),
                              'w'), indent=2)
    json.dump({'contract_id': ve.CONTRACT_ID, 'commit': 'a' * 40,
               'clean_tree': True, 'frozen_hashes': frozen,
               'checks': checks, 'verdict': ve.SIGNOFF},
              open(os.path.join(ev, 'acceptance.json'), 'w'), indent=2)
    return inv


def n08(lab):
    """A slow case hidden behind a forged PASS.

    The evidence is planted by the setup rather than here, so the CLEAN
    baseline gets an unforged tree of the same shape. Without that, both
    runs would be refused for having no evidence at all and the forgery
    would never be the reason for anything.
    """
    def setup(lab, mutated=False):
        plant_evidence(lab, forge=mutated)
    return (['tools/r35/verify_evidence.py', 'ev', '--only', 'R35-EV-01'],
            'L06', setup, 'R35-EV-01')


# Only ALLOC-01 cross-checks the model against a running bench.
CARRY_FOR = {'N05': BENCH}

MUTATIONS = [('N01', 'a wrong LinearLayout basis', n01),
             ('N02', 'the tail-mask coverage check defeated', n02),
             ('N03', 'an effectful operation marked pure', n03),
             ('N04', 'an async value released early', n04),
             ('N05', 'the bank ignored in the cost model', n05),
             ('N06', 'the opt level left out of the cache key', n06),
             ('N07', 'a model cycle passed off as measured', n07),
             ('N08', 'a forged PASS', n08)]


def mut01(log, only=None):
    """Every mutation caught by the checker the contract names."""
    before = B.tree_hashes(ROOT)
    base = tempfile.mkdtemp(prefix='r35-mut-')
    ok = True
    detail = []
    baseline = {}
    try:
        for mid, what, fn in MUTATIONS:
            if only and mid not in only:
                continue
            carry = CARRY_FOR.get(mid, ())
            lab = B.Lab(base, carry=carry)
            caught, note = False, ''
            try:
                got = fn(lab)
                argv, marker = got[0], got[1]
                setup = got[2] if len(got) > 2 else None
                # What the evidence records as the detector. It differs
                # from the text searched for only where a checker's
                # refusal is prose rather than a case id.
                label = got[3] if len(got) > 3 else marker
                if setup:
                    setup(lab, True)
                key = (tuple(argv), tuple(carry),
                       getattr(setup, '__qualname__', ''))
                if key not in baseline:
                    clean = B.Lab(base, carry=carry)
                    try:
                        if setup:
                            setup(clean, False)
                        r = run_in(clean, argv)
                        if r.returncode != 0:
                            fh = tempfile.NamedTemporaryFile(
                                'w', suffix='.log', delete=False,
                                prefix='r35-baseline-')
                            fh.write(r.stdout + '\n--- stderr ---\n'
                                     + r.stderr)
                            fh.close()
                            baseline[key] = (r.returncode,
                                             f"full output in {fh.name}")
                        else:
                            baseline[key] = (0, '')
                    finally:
                        clean.close()
                if baseline[key][0] != 0:
                    note = (f"{marker}: the clean baseline does not pass, "
                            f"so nothing can be concluded -- "
                            f"{baseline[key][1].strip()[-150:]}")
                else:
                    r = run_in(lab, argv)
                    if r.returncode == 0:
                        note = f"{marker}: not detected"
                    else:
                        named = [l for l in r.stdout.splitlines()
                                 if marker in l and 'PASS' not in l]
                        if named:
                            caught, note = True, f"caught by {label}"
                        else:
                            note = (f"{marker}: it failed, but not in the "
                                    f"named case")
            except Exception as e:                         # noqa: BLE001
                note = f"{type(e).__name__}: {e}"
            finally:
                lab.close()
            ok &= caught
            detail.append({'id': mid, 'caught': bool(caught), 'what': what,
                           'note': note})
            log.append(f"    {mid}  {'caught' if caught else 'MISSED':<7s} "
                       f"{what}")
            log.append(f"          {note}")
    finally:
        shutil.rmtree(base, ignore_errors=True)

    after = B.tree_hashes(ROOT)
    leaked = sorted(k for k in before if before[k] != after.get(k))
    if leaked:
        log.append(f"    THE CANDIDATE TREE WAS MODIFIED: {leaked}")
        ok = False
    else:
        log.append('    the candidate source hash is unchanged by the '
                   'mutations')
    return ok, (f"{sum(d['caught'] for d in detail)}/{len(detail)} "
                f"mutation(s) caught by their named checker"), detail


TESTS = ['R35-SAFE-01', 'R35-SAFE-02', 'R35-MUT-01']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--mutations', nargs='*',
                    help='restrict MUT-01 to these mutation ids')
    ap.add_argument('--no-inherited', action='store_true',
                    help='SAFE-01 without the inherited suites. For the '
                         'mutation harness only: REG-01 runs them on the '
                         'real tree')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate B: safety, and the mutations that test the checkers")
    results = []
    for cid in TESTS:
        if args.only and cid not in args.only:
            continue
        log, extra = [], None
        try:
            if cid == 'R35-SAFE-01':
                ok, what = safe01(log, inherited=not args.no_inherited)
            elif cid == 'R35-SAFE-02':
                ok, what = safe02(log)
            else:
                ok, what, extra = mut01(log, args.mutations)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        row = dict(id=cid, passed=bool(ok), detail=what)
        if extra is not None:
            row['mutations'] = extra
        results.append(row)
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    print(f"  r35-safety-tests       {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
