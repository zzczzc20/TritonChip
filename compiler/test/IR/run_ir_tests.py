#!/usr/bin/env python3
"""
Run the dialect IR tests (compiler_overview.md sec 16.2, "IR 单元测试").

The test files are written in ordinary lit format, so they will run unchanged
under llvm-lit once a build has it. The prebuilt LLVM package Triton publishes
ships FileCheck but not lit, so this interprets the RUN: lines directly rather
than adding a dependency the toolchain pin does not cover.

Substitutions: %s (the test file), tnca-opt and FileCheck (absolute paths).

THREE THINGS THIS HAS TO GET RIGHT (doc/phase1_tasks_and_acceptance.md R1-02)

  - A RUN line is `tnca-opt ... | FileCheck ...`. Without `pipefail` the
    pipeline reports FileCheck's status, so a crashing tnca-opt whose empty
    output happens to satisfy a CHECK-NOT passes. RUN-09 in the gate self-test
    fails if that guard is removed.
  - A missing tool is a SKIP while developing and a FAILURE when the run is
    meant to be evidence. `R1_STRICT=1` makes it an error; the contract
    forbids a skip from producing an R1 PASS.
  - The unit of account is a RUN line, not a file. "2 files passed" says
    nothing about how many checks ran, and the contract requires every case in
    both files to execute.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

OPT = os.path.join(ROOT, 'compiler', 'build', 'tools', 'tnca-opt', 'tnca-opt')
LLVM = os.path.expanduser(
    os.environ.get('TNCA_LLVM', '~/.triton/llvm/llvm-macos-arm64'))
FILECHECK = os.path.join(LLVM, 'bin', 'FileCheck')

RUN_RE = re.compile(r'^\s*//\s*RUN:\s*(.*)$')


def run_lines(path):
    out = []
    with open(path) as fh:
        for line in fh:
            m = RUN_RE.match(line)
            if m:
                out.append(m.group(1).strip())
    return out


def run_file(path):
    lines = run_lines(path)
    if not lines:
        return True, 'no RUN line'

    for cmd in lines:
        cmd = (cmd.replace('%s', path)
                  .replace('tnca-opt', OPT)
                  .replace('FileCheck', FILECHECK))
        r = subprocess.run(['/bin/bash', '-o', 'pipefail', '-c', cmd],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip()
    return True, f"{len(lines)} RUN line(s)"


def main():
    strict = os.environ.get('R1_STRICT') == '1'
    for what, path in (('tnca-opt', OPT), ('FileCheck', FILECHECK)):
        if not os.path.exists(path):
            if strict:
                print(f"  compiler-ir            FAIL   {what} not found at "
                      f"{path} (R1_STRICT=1: a missing tool is not a skip)")
                return 1
            print(f"  compiler-ir            SKIP   {what} not found at {path}")
            return 0

    tests = sorted(f for f in os.listdir(HERE) if f.endswith('.mlir'))
    failed = 0
    ncases = 0
    for name in tests:
        ncases += len(run_lines(os.path.join(HERE, name)))
        ok, detail = run_file(os.path.join(HERE, name))
        if not ok:
            failed += 1
            print(f"  FAIL {name}")
            print('    ' + detail.replace('\n', '\n    ')[:2000])
    if not tests or not ncases:
        print("  compiler-ir            FAIL   no IR cases discovered")
        return 1
    if failed:
        print(f"  compiler-ir            FAIL   {failed}/{len(tests)} files")
        return 1
    print(f"  compiler-ir            PASS   {len(tests)} file(s), "
          f"{ncases} RUN case(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
