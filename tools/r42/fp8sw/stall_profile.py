#!/usr/bin/env python3
"""Per-instruction waiting time of a kernel from the issue probe (domain: single-core RTL simulation).

    python3 stall_profile.py KERNEL.s PROBE.iprobe [--loop LABEL] [--chunk N]

PROBE.iprobe is written by tools/r42/fp8sw/issue_probe.v ("I <cycle> r<rctx> <pc hex>" per issued instruction). The cycles between
two consecutive issues of one Role Context are attributed to the later instruction (it waited in R for them). Prints, for the
instructions of the loop LABEL (a K loop, all iterations), the wait cycles per mnemonic and per chunk of N instructions in
program order. A `.li` is four words. Only meaningful when one context runs (one program), where the gaps are that context's.
"""
import re
import sys
from collections import Counter, defaultdict


def parse_asm(path):
    insts, labels = [], {}          # insts: (mnemonic, line text), labels: name -> instruction index
    words = []                      # word index -> instruction index
    for raw in open(path):
        s = raw.split('//')[0].strip()
        if not s:
            continue
        m = re.match(r'^([A-Za-z_][\w.]*):(.*)$', s)
        if m:
            labels[m.group(1)] = len(insts)
            s = m.group(2).strip()
            if not s:
                continue
        mn = s.split()[0]
        if mn.startswith('@'):
            mn = s.split()[1]
        words.extend([len(insts)] * (4 if mn == '.li' else 1))
        insts.append((mn, s))
    return insts, labels, words


def main():
    args = sys.argv[1:]
    loop = 'for_26'
    chunk = 400
    if '--loop' in args:
        i = args.index('--loop'); loop = args[i + 1]; del args[i:i + 2]
    if '--chunk' in args:
        i = args.index('--chunk'); chunk = int(args[i + 1]); del args[i:i + 2]
    asm, probe = args
    insts, labels, words = parse_asm(asm)
    ev = []
    for line in open(probe):
        if line.startswith('I '):
            _, cyc, ctx, pc = line.split()
            ev.append((int(cyc), ctx, int(pc, 16) // 8))
    lo = labels[loop]
    # the loop ends at the first backward branch to it
    hi = next(i for i, (mn, s) in enumerate(insts) if i > lo and mn.startswith('br') and s.replace(',', ' ').split()[-1] == loop)
    total = Counter(); wait = Counter(); n_by = Counter()
    chunk_wait = defaultdict(int); chunk_n = defaultdict(int)
    last = {}
    per_class = defaultdict(int)
    for cyc, ctx, w in ev:
        k = words[w] if w < len(words) else None
        prev = last.get(ctx)
        last[ctx] = cyc
        if k is None or prev is None:
            continue
        gap = cyc - prev - 1
        if lo <= k <= hi:
            mn = insts[k][0]
            wait[mn] += gap; n_by[mn] += 1
            chunk_wait[(k - lo) // chunk] += gap; chunk_n[(k - lo) // chunk] += 1
    trips = max(1, sum(1 for c, x, w in ev if words[w] == lo) // 1)
    tot_w = sum(wait.values()); tot_n = sum(n_by.values())
    print(f'loop {loop}: static instructions {hi - lo + 1}, trips {trips}, dynamic {tot_n}, wait cycles {tot_w} '
          f'({tot_w / max(1, tot_n):.2f} per instruction), cycles {tot_n + tot_w}')
    print('by mnemonic (count, wait, wait/inst):')
    for mn, w in wait.most_common(18):
        print(f'  {mn:14s} {n_by[mn]:7d} {w:8d} {w / n_by[mn]:6.2f}')
    print(f'by chunk of {chunk} static instructions (static count, dynamic count, wait, cycles per trip):')
    for c in sorted(chunk_n):
        print(f'  {c * chunk:6d}-{(c + 1) * chunk - 1:6d} {chunk_n[c]:7d} {chunk_wait[c]:8d} {(chunk_n[c] + chunk_wait[c]) / trips:9.1f}')


if __name__ == '__main__':
    main()
