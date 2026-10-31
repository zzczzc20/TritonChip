#!/usr/bin/env python3
"""
Run the compiler test suites and report in the regression's format.

Split into two groups because they need different things. The unit group runs
with nothing installed; the RTL group needs sim/build/tb_core.vvp, so it runs
after `make core`.

  run_tests.py --no-rtl   unit tests only
  run_tests.py --rtl      the suites that run compiled output on the RTL
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

# Suites that assemble and run their output on the Icarus model of the core.
RTL_SUITES = ['test_rtl_closure', 'test_gemm', 'test_pipeline',
              'test_partition']

# Suites that need the real Triton frontend, which lives in its own
# virtualenv. They are NOT skipped here -- they are reported by
# `make triton-e2e` and `make e2e` through run_suite.py, one line per test.
#
# This is the audit's F09 at its smallest: "76 tests, 7 skipped" was 69 tests,
# and the seven were these, running elsewhere under an entry point that
# printed a count instead of names. R1-08 REG-01 allows the deduplication and
# forbids the disappearance, so they leave this discovery and gain IDs of
# their own rather than being skipped under an interpreter that was never
# going to have Triton.
TRITON_SUITES = ['test_triton_e2e', 'test_e2e_system', 'test_pointwise_tiles']


def main():
    rtl = '--rtl' in sys.argv
    label = 'compiler-rtl' if rtl else 'compiler-unit'

    loader = unittest.TestLoader()
    if rtl:
        sys.path.insert(0, HERE)
        suite = unittest.TestSuite(
            loader.loadTestsFromName(name) for name in RTL_SUITES)
    else:
        rtl_files = {f"{n}.py" for n in RTL_SUITES + TRITON_SUITES}
        suite = loader.discover(
            HERE, pattern='test_*.py',
            top_level_dir=HERE)
        # discover() has no exclusion, so drop the RTL suites by module name.
        def keep(test):
            mod = type(test).__module__
            return f"{mod}.py" not in rtl_files
        flat = []

        def walk(s):
            for t in s:
                if isinstance(t, unittest.TestSuite):
                    walk(t)
                elif keep(t):
                    flat.append(t)
        walk(suite)
        suite = unittest.TestSuite(flat)

    with open(os.devnull, 'w') as devnull:
        result = unittest.TextTestRunner(verbosity=0, stream=devnull).run(suite)

    n = result.testsRun
    bad = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    # unittest reports an expected failure and an unexpected pass as success.
    # doc/phase1_tasks_and_acceptance.md forbids both from producing an R1
    # PASS, so they are counted here rather than left to the framework.
    xfail = len(getattr(result, 'expectedFailures', []))
    xpass = len(getattr(result, 'unexpectedSuccesses', []))
    strict = os.environ.get('R1_STRICT') == '1'

    if bad:
        for case, tb in result.failures + result.errors:
            print(f"  FAIL {case}")
            print('    ' + tb.strip().replace('\n', '\n    ')[:3000])
        print(f"  {label:<22s} FAIL   {bad}/{n} failed")
        return 1

    if strict and (skipped or xfail or xpass):
        # "76 tests, 7 skipped" is 69 tests. Say so, and name them: a skip
        # that nobody can see is a requirement nobody is tracking.
        for case, why in result.skipped:
            print(f"  SKIP {case}: {why}")
        print(f"  {label:<22s} FAIL   {skipped} skipped, {xfail} xfail, "
              f"{xpass} xpass (R1_STRICT=1: none of these is a pass)")
        return 1

    extra = f", {skipped} skipped" if skipped else ""
    if xfail or xpass:
        extra += f", {xfail} xfail, {xpass} xpass"
    print(f"  {label:<22s} PASS   {n} tests{extra}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
