#!/usr/bin/env python3
"""Compile the directed fragment fixtures and report static K-loop traffic.

Requires the real Triton frontend. This does not execute kernels or predict
cycle speedup; use the standard Torch/RTL and board campaigns separately.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--tests-root', type=Path, required=True,
                        help='root containing compiler/test/test_fragment_residency.py')
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path[:0] = [str(root / 'compiler/python'), str(root),
                   str(root / 'compiler/test'), str(args.tests_root / 'compiler/test')]
    # Load this source tree's compiler before the fixture adjusts sys.path.
    import tnca_cc.frontend  # noqa: F401
    from archbench.provenance import software_tree
    from test_fragment_residency import FragmentResidencyTests
    case = FragmentResidencyTests()
    records = []
    for columns, old_use in ((16, False), (32, False), (32, True)):
        ops = case.compile(columns, old_use=old_use)
        body = case.loop_ops(ops)
        records.append(dict(
            columns=columns, old_use=old_use,
            body_ops=dict(sorted(Counter(op.kind for op in body).items())),
            whole_ir_sha256=hashlib.sha256('\n'.join(map(repr, ops)).encode()).hexdigest()))
    print(json.dumps(dict(schema='r40-followup-fragment-static/1',
                          software_tree=software_tree(root, refresh=True),
                          records=records,
                          not_measured=['numerics', 'latency', 'board performance']), indent=2))


if __name__ == '__main__':
    main()
