#!/usr/bin/env python3
"""
REG-01 .. REG-11 -- doc/phase1_tasks_and_acceptance.md R1-08.

The regression matrix, run rather than asserted.

Two of these are inventory questions -- did the baseline's cases survive, and
did the seven that were skipped actually execute -- and are answered by
counting what the entry points report. The rest are combinations nobody had
run: thirty-six pipeline configurations, three seeds through the PRNG that
really controls the stall and reorder pattern, and the completion cases the
contract names one by one.

The seed rule is the one worth restating, because it is easy to satisfy on
paper: "必须把 seed 传到真正控制 stimulus、stall、reorder 的 PRNG，而非只记在日
志中". So every seeded case here reports the trace hash the run produced, three
seeds must produce three different ones, and the same seed must reproduce its
own. A matrix whose three seeds give three identical hashes has run one seed
three times.
"""
import argparse
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
import r1lib as R  # noqa: E402

SEEDS = (1, 7, 42)
ITERATIONS = (0, 1, 2, 3, 5, 6)
STAGES = (2, 3)
MEMORIES = ('fast', 'slow', 'ooo')

TRACE_RE = re.compile(r'memtrace=([0-9a-f]+)')


# Each `make` this suite runs is a whole entry point -- three core profiles,
# the four-tile machine, the compiler's RTL suite. On the acceptance runner
# those are tens of minutes each, and the `build` bucket's hour is a cap on
# ONE step of the gate rather than on a suite that runs several. The gate
# gives this suite its own cap (tools/r1/gate.py CASE_SUITES); this is the
# per-command one inside it.
MAKE_CAP = 7200


def make(target, *extra, timeout=None):
    return subprocess.run(['make', target, *extra], cwd=ROOT,
                          capture_output=True, text=True,
                          timeout=timeout or MAKE_CAP)


def count_lines(text, pattern):
    return len(re.findall(pattern, text, re.M))


# --------------------------------------------------------------------------
def reg01(log):
    """The baseline's semantic cases, and the seven that were skipped."""
    env = dict(os.environ, R1_STRICT='1')
    r = subprocess.run([sys.executable, 'compiler/test/run_tests.py'],
                       cwd=ROOT, capture_output=True, text=True, env=env,
                       timeout=R.CAPS['build'])
    unit_ok = r.returncode == 0 and 'skipped' not in r.stdout
    log.append(f"    compiler-unit under R1_STRICT=1: "
               f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'no output'}")

    # The seven that used to be skipped, at the entry point that can run them.
    core = make('triton-e2e')
    system = make('e2e')
    n_core = count_lines(core.stdout, r'^  BASE-TRITON-CORE-\S+\s+PASS')
    n_sys = count_lines(system.stdout, r'^  BASE-TRITON-SYSTEM-\S+\s+PASS')
    log.append(f"    the seven skips, at their real entry point: "
               f"{n_core} single-core + {n_sys} whole-machine")
    if core.returncode or system.returncode:
        log.append('      ' + (core.stdout + system.stdout).strip()[-400:])
    return (unit_ok and n_core == 5 and n_sys == 2), \
        'no skips in compiler-unit; the seven run and are named'


def reg02(log):
    """The original compiler RTL, IR and Triton suites, all of them."""
    rtl = make('compiler-rtl')
    m = re.search(r'compiler-rtl\s+PASS\s+(\d+) tests', rtl.stdout)
    n_rtl = int(m.group(1)) if m else 0
    ir = subprocess.run([sys.executable, 'compiler/test/IR/run_ir_tests.py'],
                        cwd=ROOT, capture_output=True, text=True,
                        timeout=R.CAPS['build'])
    mi = re.search(r'(\d+) file\(s\), (\d+) RUN case\(s\)', ir.stdout)
    n_files, n_runs = (int(mi.group(1)), int(mi.group(2))) if mi else (0, 0)
    log.append(f"    compiler-rtl: {n_rtl} tests (baseline 67)")
    log.append(f"    IR: {n_files} file(s), {n_runs} RUN case(s) "
               f"(baseline 2 files)")
    if rtl.returncode or ir.returncode:
        log.append('      ' + (rtl.stdout + ir.stdout).strip()[-400:])
    return (rtl.returncode == 0 and ir.returncode == 0
            and n_rtl >= 67 and n_files >= 2), \
        'the baseline suites are intact and larger, never smaller'


def reg03(log):
    """Every core program under all three memories, faults exact."""
    ok = True
    for target, tag in (('core', 'fast'), ('core-slowmem', 'slow'),
                        ('core-ooo', 'ooo')):
        r = make(target)
        n_pass = count_lines(r.stdout, r'^  t\d+\S*.*\bPASS\b')
        n_fail = count_lines(r.stdout, r'\bFAIL\b')
        log.append(f"    {target}: {n_pass} program(s) PASS, {n_fail} FAIL")
        if r.returncode:
            log.append('      ' + r.stdout.strip()[-500:])
        ok &= (r.returncode == 0 and n_fail == 0 and n_pass >= 22)

    # The expected-fault programs are the interesting half: a clean run is
    # the failure, and the trap code has to be the exact one.
    # Program -> (trap code, the plusargs the Makefile runs it with). The
    # args matter: t05 is a per-role fault and needs every role launched,
    # while the other three are single-role programs, and launching a second
    # context into the same code makes both roles arm the same barrier, so
    # one of them races past its wait and reaches the `trap #99` that is
    # supposed to be unreachable. That reports TRAP_USER and hides the engine
    # fault the case is about.
    faults = {'t05_fault': ('01', ['+allroles']),
              't17_descfault': ('04', []),
              't23_amofault': ('03', []),
              't35_tdesc_fpad2': ('04', [])}
    for prog, (code, args) in faults.items():
        hexf = os.path.join(ROOT, 'sim', 'build', f'{prog}.hex')
        bench = os.path.join(ROOT, 'sim', 'build', 'tb_core.vvp')
        if not (os.path.exists(hexf) and os.path.exists(bench)):
            log.append(f"    {prog}: image or bench missing")
            ok = False
            continue
        # Run WITHOUT +expectfault: the raw verdict must be the trap itself,
        # so the harness is not the thing deciding what happened.
        r = subprocess.run(['vvp', bench, f'+hex={hexf}', '+maxcycles=200000',
                            *args],
                           capture_output=True, text=True,
                           timeout=R.CAPS['rtl'])
        m = re.search(r'FAIL: trap code=0x([0-9a-f]{2})', r.stdout)
        got = m.group(1) if m else None
        log.append(f"    {prog}: raises trap 0x{got} (contract 0x{code})")
        ok &= (got == code)
    return ok, 'all core programs on fast/slow/ooo; exact traps'


def reg04(log):
    """iterations x stages x memory, and the seeds on the random models."""
    import test_pipeline as T
    case = T.TestPipelineOnRTL('test_six_iterations')
    case.setUp()

    ok = True
    ran = 0
    for mem in MEMORIES:
        for stages in STAGES:
            for it in ITERATIONS:
                try:
                    case._run(it, stages=stages, bench=mem)
                    ran += 1
                except AssertionError as e:
                    ok = False
                    log.append(f"    FAIL it={it} stages={stages} mem={mem}: "
                               f"{str(e).splitlines()[0][:110]}")
                except unittest.SkipTest as e:
                    ok = False
                    log.append(f"    NOT RUN it={it} stages={stages} "
                               f"mem={mem}: {e}")
    log.append(f"    {ran} of {len(MEMORIES)*len(STAGES)*len(ITERATIONS)} "
               f"combinations ran and passed")

    # The seeds, on the two models where a seed changes anything. Same
    # configuration, three seeds, three different schedules -- then seed 1
    # again, which must reproduce its own.
    for mem in ('slow', 'ooo'):
        hashes = {}
        for sd in SEEDS:
            out = case._run(6, stages=2, bench=mem, seed=sd)
            m = TRACE_RE.search(out)
            hashes[sd] = m.group(1) if m else None
        replay = TRACE_RE.search(case._run(6, stages=2, bench=mem, seed=SEEDS[0]))
        distinct = len(set(hashes.values())) == len(SEEDS)
        reproduced = replay and replay.group(1) == hashes[SEEDS[0]]
        log.append(f"    {mem}: " + ', '.join(f"seed {k} -> {v[:12]}"
                                              for k, v in hashes.items()))
        log.append(f"    {mem}: three distinct schedules = {distinct}, "
                   f"seed {SEEDS[0]} replays = {bool(reproduced)}")
        ok &= distinct and bool(reproduced)
    return ok, '36 combinations, plus three seeds that really differ'


def reg05(log):
    """GEMM: the baseline cases, and the new footprint tests."""
    import test_gemm  # noqa: F401
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('test_gemm')
    with open(os.devnull, 'w') as devnull:
        res = unittest.TextTestRunner(verbosity=0, stream=devnull).run(suite)
    log.append(f"    test_gemm: {res.testsRun} test(s), "
               f"{len(res.failures)+len(res.errors)} failed, "
               f"{len(res.skipped)} skipped")
    for c, tb in res.failures + res.errors:
        log.append(f"      {c}: {tb.strip().splitlines()[-1][:110]}")

    # The new half: the K16/32/64 footprint computation, from the instruction.
    r = subprocess.run([sys.executable, 'tools/r1/resource_tests.py',
                        '--only', 'FP-01', 'FP-02', 'FP-03', 'FP-04',
                        'FP-05', 'FP-06'],
                       cwd=ROOT, capture_output=True, text=True,
                       timeout=R.CAPS['build'])
    log.append(f"    footprints: "
               f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
    return (res.wasSuccessful() and not res.skipped and r.returncode == 0), \
        'GEMM cases kept, with the footprint tests added'


def _seeded_entry(log, target, id_prefix):
    """One of the seeded RTL entries, with its per-seed trace hashes."""
    r = make(target)
    lines = [l for l in r.stdout.splitlines() if id_prefix in l]
    for l in lines:
        log.append('    ' + l.strip()[:120])
    return r.returncode == 0 and len(lines) >= len(SEEDS), r


def reg06(log):
    """NUCA under three memories and three seeds, coverage asserted."""
    ok, r = _seeded_entry(log, 'nuca', 'BASE-NUCA-')
    # The bench prints its own coverage; a run whose counters stayed at zero
    # fails inside the bench rather than here. What this adds is the seed
    # evidence: three different traffic traces per configuration.
    hashes = re.findall(r'seed=(\d+) trace_hash=([0-9a-f]+)', r.stdout)
    by_seed = {}
    for sd, h in hashes:
        by_seed.setdefault(sd, set()).add(h)
    distinct = len({h for _s, h in hashes}) >= len(SEEDS)
    log.append(f"    {len(hashes)} seeded run(s), "
               f"{len({h for _s, h in hashes})} distinct traffic trace(s)")
    if not ok:
        log.append('      ' + r.stdout.strip()[-400:])
    return ok and distinct, 'three memories x three seeds, counters triggered'


def reg07(log):
    """Mesh: 20,000 packets per seed, every pair, real back-pressure."""
    ok, r = _seeded_entry(log, 'mesh', 'BASE-MESH-')
    pairs = re.findall(r'pairs_covered=(\d+)/(\d+) backpressure_cycles=(\d+)',
                       r.stdout)
    hashes = re.findall(r'seed=(\d+) trace_hash=([0-9a-f]+)', r.stdout)
    full = all(a == b and int(c) > 0 for a, b, c in pairs) and pairs
    log.append(f"    coverage per seed: {pairs}")
    distinct = len({h for _s, h in hashes}) == len(SEEDS)
    if not ok:
        log.append('      ' + r.stdout.strip()[-400:])
    return bool(ok and full and distinct), \
        'all 32 src/dst/VN pairs, back-pressure applied, seeds differ'


def reg08(log):
    """The whole machine: two programs, single-role and all-roles."""
    r = make('top')
    lines = [l for l in r.stdout.splitlines()
             if re.search(r'\b(PASS|FAIL)\b', l)]
    for l in lines:
        log.append('    ' + l.strip()[:120])
    n = len([l for l in lines if 'PASS' in l])
    return (r.returncode == 0 and n >= 4), \
        't10_mesh and t24_amo_mesh, single-role and all-roles'


def reg09(log):
    """Two consumers on core-4r: the LAST release frees the window."""
    ok = True
    # The compiler half, which is OWN-07: refused after one of two releases.
    r = subprocess.run([sys.executable, 'tools/r1/resource_tests.py',
                        '--only', 'OWN-07'],
                       cwd=ROOT, capture_output=True, text=True,
                       timeout=R.CAPS['build'])
    log.append(f"    compiler: {r.stdout.strip().splitlines()[-1][:110]}")
    ok &= r.returncode == 0

    # The RTL half, on the four-context profile, under both random models
    # and all three seeds. The contract asks for slow and ooo specifically,
    # and the reason is the point of the case: on a fast in-order memory the
    # two consumers finish so close together that reclaiming on the FIRST
    # release and reclaiming on the LAST produce the same answer.
    # Build the benches this case needs, the way reg03 runs `make core`.
    # They are build products; a suite that assumes somebody built them by
    # hand passes on a developer's machine and reports "not built" on a clean
    # acceptance environment, which is where it matters.
    b = make('core4')
    if b.returncode != 0:
        log.append('    make core4 failed: ' + b.stdout.strip()[-300:])
        return False, 'the four-context benches could not be built'

    import test_pipeline as T
    case = T.TestPipelineOnRTL('test_two_consumers_on_four_contexts')
    case.setUp()
    for bench in ('rctx4', 'rctx4-slow', 'rctx4-ooo'):
        seeds = (None,) if bench == 'rctx4' else SEEDS
        traces = {}
        for sd in seeds:
            try:
                out = case._two_consumers(bench=bench, seed=sd)
                m = TRACE_RE.search(out)
                traces[sd] = m.group(1) if m else '-'
            except AssertionError as e:
                ok = False
                traces[sd] = 'FAIL'
                log.append(f"      {bench} seed {sd}: "
                           f"{str(e).splitlines()[0][:110]}")
            except unittest.SkipTest as e:
                ok = False
                traces[sd] = 'NOT RUN'
                log.append(f"      {bench}: {e}")
        log.append(f"    {bench}: " + ', '.join(
            f"seed {k} -> {v[:12]}" for k, v in traces.items()))
        if bench != 'rctx4':
            distinct = len(set(traces.values())) == len(SEEDS)
            log.append(f"    {bench}: three distinct schedules = {distinct}")
            ok &= distinct
    return ok, 'the last release frees the window, in the compiler and on RTL'


COLLIDE_TEMPLATE = os.path.join(ROOT, 'tb', 'core', 'progs',
                                't27_collide.s.in')
DONE_RE = re.compile(r'\[done\] cyc=(\d+) (txf|mtx)')
COLLIDE_RE = re.compile(r'barcollide=(\d+)')


def _run_collide(delay, tmp, *, trace=False):
    """Assemble and run the collision program with `delay` nops."""
    import tempfile
    text = open(COLLIDE_TEMPLATE).read()
    if text.count('@DELAY@') != 1:
        # It was two, once: the placeholder also appeared in the comment above
        # the code, so the nops were inserted into the prose and the assembler
        # found a second `.org 0`. A template with two holes is a template
        # with one bug.
        raise AssertionError(
            f"{COLLIDE_TEMPLATE} has {text.count('@DELAY@')} @DELAY@ "
            "placeholders; it must have exactly one")
    text = text.replace(
        '@DELAY@', ''.join('    nop\n' for _ in range(delay)).rstrip('\n'))
    src = os.path.join(tmp, f'collide{delay}.s')
    hexf = os.path.join(tmp, f'collide{delay}.hex')
    with open(src, 'w') as fh:
        fh.write(text)
    a = subprocess.run([sys.executable, 'tools/asm/tnca_asm.py', src,
                        '-o', hexf, '--size', '16384'],
                       cwd=ROOT, capture_output=True, text=True)
    if a.returncode != 0:
        return None
    argv = ['vvp', os.path.join(ROOT, 'sim', 'build', 'tb_core.vvp'),
            f'+hex={hexf}', '+maxcycles=200000']
    if trace:
        argv.append('+donetrace')
    r = subprocess.run(argv, capture_output=True, text=True,
                       timeout=R.CAPS['rtl'])
    last = {}
    for cyc, who in DONE_RE.findall(r.stdout):
        last[who] = int(cyc)
    m = COLLIDE_RE.search(r.stdout)
    return dict(passed='PASS' in r.stdout and 'FAIL' not in r.stdout,
                collisions=int(m.group(1)) if m else -1,
                done=last)


def _completion_collision(log):
    """Solve for the delay that makes both engines finish together."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix='r1-collide-')
    a = _run_collide(0, tmp, trace=True)
    b = _run_collide(200, tmp, trace=True)
    if not (a and b and 'txf' in a['done'] and 'mtx' in a['done']):
        log.append("    could not measure the two completion latencies")
        return False, 'unmeasured'
    # The transfer is issued `delay` cycles later and the matrix command is
    # not, so the gap closes one cycle per nop (to within issue effects).
    slope = ((b['done']['txf'] - a['done']['txf']) / 200.0) or 1.0
    guess = int(round((a['done']['mtx'] - a['done']['txf']) / slope))
    log.append(f"    matrix completes at cycle {a['done']['mtx']}, transfer "
               f"at {a['done']['txf']} + {slope:.2f}/nop -> try {guess}")
    found = []
    for d in range(max(0, guess - 8), guess + 9):
        r = _run_collide(d, tmp)
        if r is None:
            continue
        if not r['passed']:
            log.append(f"    delay {d}: the program FAILED, which is what a "
                       f"dropped completion looks like")
            return False, 'dropped'
        if r['collisions'] > 0:
            found.append((d, r['collisions']))
    if found:
        log.append(f"    same-cycle completions at delay(s) "
                   f"{[d for d, _c in found]}, counted "
                   f"{[c for _d, c in found]} time(s), barrier still released")
        return True, 'collided'
    log.append(f"    no same-cycle completion in delays "
               f"{max(0, guess-8)}..{guess+8}: the three-port sum in "
               f"rtl/core/tnca_barrier.v is still unexercised")
    return False, 'never collided'


def reg10(log):
    """Error completion and same-cycle completion, both counted."""
    import test_pipeline as T
    ok = True
    for name in ('test_error_propagation', 'test_predicated_off_async_still_completes',
                 'test_delayed_completion', 'test_out_of_order_returns'):
        case = T.TestPipelineOnRTL(name)
        with open(os.devnull, 'w') as devnull:
            res = unittest.TextTestRunner(verbosity=0, stream=devnull).run(
                unittest.TestSuite([case]))
        state = ('PASS' if res.wasSuccessful() and not res.skipped
                 else ('SKIPPED' if res.skipped else 'FAIL'))
        log.append(f"    {name}: {state}")
        for _c, tb in res.failures + res.errors:
            log.append(f"      {tb.strip().splitlines()[-1][:110]}")
        for _c, why in res.skipped:
            log.append(f"      {why}")
        ok &= (state == 'PASS')

    # Two completions on one barrier in one cycle, arranged rather than hoped
    # for. See tb/core/progs/t27_collide.s.in: the delay is solved for from a
    # measured run, because a constant would stop colliding the first time
    # either engine's timing changed -- and then this case would pass while
    # testing nothing, which is the failure mode the whole package is about.
    hit, detail = _completion_collision(log)
    ok &= hit

    # The engine's error completion, on real hardware: t35_tdesc_fpad2 issues
    # a transfer the engine refuses (padding mode 2 with an element out of
    # bounds), and the waiter must be released with the barrier's error bit
    # rather than left hanging. (t26_stfault did this with t.st.async until
    # the engine learned to perform stores.)
    hexf = os.path.join(ROOT, 'sim', 'build', 't35_tdesc_fpad2.hex')
    bench = os.path.join(ROOT, 'sim', 'build', 'tb_core.vvp')
    if os.path.exists(hexf) and os.path.exists(bench):
        r = subprocess.run(['vvp', bench, f'+hex={hexf}',
                            '+maxcycles=200000', '+expectfault=04'],
                           capture_output=True, text=True,
                           timeout=R.CAPS['rtl'])
        woke = 'PASS' in r.stdout and 'timeout' not in r.stdout
        log.append(f"    an error completion releases the waiter: {woke}")
        ok &= woke
    else:
        log.append("    t35_tdesc_fpad2 image not built")
        ok = False
    return ok, 'error and collision completions counted exactly once'


def reg11(log):
    """The real Triton closure, single core and whole machine."""
    core = make('triton-e2e')
    system = make('e2e')
    for r in (core, system):
        for l in r.stdout.splitlines():
            if re.search(r'\b(PASS|FAIL|SKIP)\b', l):
                log.append('    ' + l.strip()[:120])
    return (core.returncode == 0 and system.returncode == 0
            and 'SKIP' not in core.stdout and 'SKIP' not in system.stdout), \
        'Triton source through the compiler onto the RTL, checked in Python'


TESTS = [('REG-01', reg01), ('REG-02', reg02), ('REG-03', reg03),
         ('REG-04', reg04), ('REG-05', reg05), ('REG-06', reg06),
         ('REG-07', reg07), ('REG-08', reg08), ('REG-09', reg09),
         ('REG-10', reg10), ('REG-11', reg11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R1-08 regression matrix")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-regression          FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-regression          PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
