#!/usr/bin/env python3
"""Accumulators and barriers named by compiled TNCA programs (R41 G2).

  asm_resource_scan.py DIR [DIR ...]

Scans every *.s under the given directories (kernel dumps kept as evidence)
and counts the distinct accumulator registers (aN) and barrier registers (bN)
each program names. Prints the distribution and the maximum, with the file.
A program's names are per context; the hardware index is the name plus the
context's slice offset (RSLICE), so a context needs a slice at least this
large.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ACC = re.compile(r'\ba(\d+)\b')
BAR = re.compile(r'\bb(\d+)\b')


def scan(path):
    acc, bar = set(), set()
    for line in path.read_text(errors='replace').splitlines():
        code = line.split('//', 1)[0]
        if not code.strip() or code.lstrip().startswith(('.', '#')) or code.rstrip().endswith(':'):
            continue
        acc |= {int(m) for m in ACC.findall(code)}
        bar |= {int(m) for m in BAR.findall(code)}
    return acc, bar


def main(dirs):
    files = sorted({p.resolve() for d in dirs for p in Path(d).rglob('*.s')})
    da, db = Counter(), Counter()
    worst = {'acc': (0, None), 'bar': (0, None)}
    for f in files:
        acc, bar = scan(f)
        da[len(acc)] += 1
        db[len(bar)] += 1
        if len(acc) > worst['acc'][0]:
            worst['acc'] = (len(acc), f)
        if len(bar) > worst['bar'][0]:
            worst['bar'] = (len(bar), f)
    print(f'{len(files)} programs')
    print('accumulators named per program:', dict(sorted(da.items())),
          'max', worst['acc'][0], worst['acc'][1])
    print('barriers named per program:', dict(sorted(db.items())),
          'max', worst['bar'][0], worst['bar'][1])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
