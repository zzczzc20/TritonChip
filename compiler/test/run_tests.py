#!/usr/bin/env python3
"""Run every compiler unit test and report in the regression's format."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pattern = 'test_*.py'
    if '--rtl' in sys.argv:
        # The closure test needs sim/build/tb_core.vvp, which is built by
        # `make core`. Running it separately keeps the fast checks early.
        pattern = 'test_rtl_closure.py'
    elif '--no-rtl' in sys.argv:
        pattern = 'test_[!r]*.py'
    suite = unittest.TestLoader().discover(HERE, pattern=pattern)
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)
    label = 'compiler-rtl' if '--rtl' in sys.argv else 'compiler-unit'
    n = result.testsRun
    bad = len(result.failures) + len(result.errors)
    if bad:
        for case, tb in result.failures + result.errors:
            print(f"  FAIL {case}")
            print('    ' + tb.strip().replace('\n', '\n    '))
        print(f"  {label:<22s} FAIL   {bad}/{n} failed")
        return 1
    print(f"  {label:<22s} PASS   {n} tests")
    return 0


if __name__ == '__main__':
    sys.exit(main())
