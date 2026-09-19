#!/usr/bin/env python3
"""
Run the dialect IR tests (compiler_overview.md sec 16.2, "IR 单元测试").

The test files are written in ordinary lit format, so they will run unchanged
under llvm-lit once a build has it. The prebuilt LLVM package Triton publishes
ships FileCheck but not lit, so this interprets the RUN: lines directly rather
than adding a dependency the toolchain pin does not cover.

Substitutions: %s (the test file), tnca-opt and FileCheck (absolute paths).
Skips with a clear message if tnca-opt has not been built.
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


def run_file(path):
    lines = []
    with open(path) as fh:
        for line in fh:
            m = RUN_RE.match(line)
            if m:
                lines.append(m.group(1).strip())
    if not lines:
        return True, 'no RUN line'

    for cmd in lines:
        cmd = (cmd.replace('%s', path)
                  .replace('tnca-opt', OPT)
                  .replace('FileCheck', FILECHECK))
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip()
    return True, f"{len(lines)} RUN line(s)"


def main():
    if not os.path.exists(OPT):
        print("  compiler-ir            SKIP   tnca-opt not built "
              "(see compiler/README.md)")
        return 0
    if not os.path.exists(FILECHECK):
        print(f"  compiler-ir            SKIP   FileCheck not found at {FILECHECK}")
        return 0

    tests = sorted(f for f in os.listdir(HERE) if f.endswith('.mlir'))
    failed = 0
    for name in tests:
        ok, detail = run_file(os.path.join(HERE, name))
        if not ok:
            failed += 1
            print(f"  FAIL {name}")
            print('    ' + detail.replace('\n', '\n    ')[:2000])
    if failed:
        print(f"  compiler-ir            FAIL   {failed}/{len(tests)} files")
        return 1
    print(f"  compiler-ir            PASS   {len(tests)} IR test file(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
