#!/usr/bin/env python3
"""Run every compiler unit test and report in the regression's format."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    suite = unittest.TestLoader().discover(HERE, pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)
    n = result.testsRun
    bad = len(result.failures) + len(result.errors)
    if bad:
        for case, tb in result.failures + result.errors:
            print(f"  FAIL {case}")
            print('    ' + tb.strip().replace('\n', '\n    '))
        print(f"  compiler-unit          FAIL   {bad}/{n} failed")
        return 1
    print(f"  compiler-unit          PASS   {n} tests")
    return 0


if __name__ == '__main__':
    sys.exit(main())
