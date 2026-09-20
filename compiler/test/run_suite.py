#!/usr/bin/env python3
"""
Run one test module and report ONE LINE PER TEST.

  run_suite.py <module> <label> <id-prefix>

The audit's F09 in its smallest form: `compiler-unit` reported "76 tests, 7
skipped" for years. The seven were the Triton suites, which do run -- under
`make triton-e2e` and `make e2e`, in the virtualenv that has Triton -- but
that entry point printed `Ran 5 tests ... OK` and the inventory recorded one
ID for five tests. Two numbers, neither of which said which cases executed.

So the suites are removed from `compiler-unit` (deduplication, which R1-08
REG-01 allows) and reported here individually (disappearance, which it does
not). `make compiler` now says 69 tests and 0 skipped, and every one of the
seven has an ID of its own.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    module, label, prefix = sys.argv[1:4]

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module)

    flat = []

    def walk(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            else:
                flat.append(t)
    walk(suite)

    strict = os.environ.get('R1_STRICT') == '1'
    bad = 0
    skipped = 0
    with open(os.devnull, 'w') as devnull:
        runner = unittest.TextTestRunner(verbosity=0, stream=devnull)
        for test in flat:
            name = test.id().split('.')[-1]
            result = runner.run(unittest.TestSuite([test]))
            if result.failures or result.errors:
                bad += 1
                status = 'FAIL'
            elif result.skipped:
                skipped += 1
                status = 'SKIP'
            elif getattr(result, 'unexpectedSuccesses', None):
                bad += 1
                status = 'FAIL'
            elif getattr(result, 'expectedFailures', None):
                bad += 1
                status = 'FAIL'
            else:
                status = 'PASS'
            print(f"  {prefix}-{name:<28s} {status}")
            for _case, tb in result.failures + result.errors:
                print('    ' + tb.strip().replace('\n', '\n    ')[:2000])
            for _case, why in result.skipped:
                print(f"    skipped: {why}")

    n = len(flat)
    if bad or (strict and skipped):
        print(f"  {label:<22s} FAIL   {bad} failed, {skipped} skipped "
              f"of {n}")
        return 1
    print(f"  {label:<22s} PASS   {n} test(s)"
          + (f", {skipped} skipped" if skipped else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
