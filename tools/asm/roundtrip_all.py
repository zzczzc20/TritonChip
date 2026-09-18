#!/usr/bin/env python3
"""
Assemble -> disassemble -> re-assemble every given source and require the
original encoding back. This is the check that keeps three things honest at
once: the encoding tables, the assembler's field placement and the
disassembler's reading of them. A field that two of the three agree on
incorrectly still shows up, because the RTL decoder is tested separately
against the same tables.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIS = os.path.join(HERE, 'tnca_dis.py')

def main(paths):
    total = bad = 0
    for p in paths:
        out = subprocess.run([sys.executable, DIS, '--roundtrip', p],
                             capture_output=True, text=True)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
        n = int(line.split()[1]) if line.startswith('round-trip:') else 0
        m = int(line.split()[3]) if line.startswith('round-trip:') else 1
        total += n
        bad += m
        if m or out.returncode:
            print(f"  asm round-trip        FAIL   {os.path.basename(p)}")
            print(out.stdout.rstrip())
            print(out.stderr.rstrip())
    if bad:
        return 1
    print(f"  asm round-trip        PASS   {total} instructions, "
          f"{len(paths)} programs, 0 mismatches")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
