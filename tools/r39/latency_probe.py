#!/usr/bin/env python3
"""R39-01: what a dependence costs on the core, measured on the RTL.

The scheduler needs one number per producer: how many cycles after it issues
an instruction that reads its result can issue. The target files carry two
latencies (FP and MUL) written for the flip-flop register file; on the banked
file a result also has to be written into its bank before the scoreboard lets
a reader fetch it, so those numbers are not what the machine does. This
measures it instead.

Each probe is a chain: N links, each reading what the previous one wrote, so
the core can do nothing but wait. Run at two lengths, the difference divided
by the difference in links is the cycles per link with the prologue and the
exit cancelled out -- the issue-to-dependent-issue distance. The same
instructions with no dependence between them give the issue rate, which is
the floor a schedule can reach.

    python3 tools/r39/latency_probe.py [--bench sim/build/tb_core_vrf.vvp] [--json out.json]

Memory latency is the bench's model (1 cycle), not the board's; load-to-use
on the board is measured through kernels, not here.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASM = os.path.join(ROOT, 'tools', 'asm', 'tnca_asm.py')

PROLOGUE = """\
.org 0
start:
    .li       s2, #1
    .li       s3, #3
    .li       s5, #5
    .li       s6, #6
    .li       s7, #7
    .li       s8, #8
    v.iota    v1
    v.movi    v2, #2
    v.movi    v3, #3
    v.movi    v5, #5
    v.movi    v6, #6
    v.movi    v7, #7
    v.movi    v9, #9
    v.movi    v10, #10
    v.movi    v11, #11
    v.movi    v13, #13
    v.movi    v14, #14
    v.movi    v15, #15
    v.cmp.lt  p2, v1, v2
"""

# name -> (one link, as lines; what it measures). Registers: v1/v5/v9/v13 are
# bank 1, v2/v6/v10/v14 bank 2, v3/v7/v11/v15 bank 3 (physical register
# number mod 4, role context 0).
DEP = {
    's.add':      (['s.add     s2, s2, s3'], 'scalar ALU -> scalar ALU'),
    's.mul':      (['s.mul     s2, s2, s3'], 'scalar MUL -> scalar ALU'),
    'v.add':      (['v.add     v1, v1, v2'], 'vector int ALU -> vector'),
    'v.mul':      (['v.mul     v1, v1, v2'], 'vector int MUL -> vector'),
    'vf.add':     (['vf.add    v1, v1, v2'], 'FP add -> vector'),
    'vf.mul':     (['vf.mul    v1, v1, v2'], 'FP mul -> vector'),
    'vf.fma':     (['vf.fma    v1, v1, v2, v3'], 'FP fma -> vector'),
    'sfu':        (['v.rsqrt.approx v1, v1'], 'SFU -> vector'),
    'v.cmp+sel':  (['v.cmp.lt  p1, v1, v2', '@p1 v.sel     v1, v3, v1'],
                   'compare -> predicate reader, then select -> compare (a pair)'),
    'p.and':      (['p.and     p1, p1, p2'], 'predicate logic -> predicate'),
    'splat+red':  (['v.splat   v1, s2', 'v.reduce.add s2, v1'],
                   'scalar -> splat, then reduction -> scalar (a pair)'),
    'red.b':      (['v.reduce.add.b v1, v1'], 'broadcast reduction -> vector'),
    'extract+splat': (['v.extract s2, v1, #3', 'v.splat   v1, s2'],
                      'lane extract -> scalar, then splat -> vector (a pair)'),
    's->v':       (['s.addi    s2, s2, #1', 'v.splat   v1, s2'],
                   'scalar ALU -> splat (independent splats: the pair minus 1)'),
}

# The same instructions without a dependence: the issue rate.
INDEP = {
    's.add':  ['s.add     s5, s2, s3', 's.add     s6, s2, s3', 's.add     s7, s2, s3',
               's.add     s8, s2, s3'],
    'v.add':  ['v.add     v5, v2, v3', 'v.add     v6, v3, v1', 'v.add     v7, v1, v2',
               'v.add     v9, v2, v3', 'v.add     v10, v3, v1', 'v.add     v11, v1, v2'],
    'v.add same bank': ['v.add     v5, v9, v13', 'v.add     v6, v10, v14',
                        'v.add     v7, v11, v15'],
    'vf.add': ['vf.add    v5, v2, v3', 'vf.add    v6, v3, v1', 'vf.add    v7, v1, v2',
               'vf.add    v9, v2, v3', 'vf.add    v10, v3, v1', 'vf.add    v11, v1, v2'],
    'vf.fma': ['vf.fma    v5, v2, v3, v1', 'vf.fma    v6, v3, v1, v2',
               'vf.fma    v7, v1, v2, v3'],
    'v+s alternating': ['v.add     v5, v2, v3', 's.add     s5, s2, s3',
                        'v.add     v6, v3, v1', 's.add     s6, s2, s3'],
}


def build(body, n, tmp, name):
    src = PROLOGUE + ''.join('    ' + body[i % len(body)] + '\n' for i in range(n)) + '    exit\n'
    s = os.path.join(tmp, f'{name}_{n}.s')
    h = os.path.join(tmp, f'{name}_{n}.hex')
    open(s, 'w').write(src)
    r = subprocess.run([sys.executable, ASM, s, '-o', h], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'{name}: assembly failed: {r.stderr[-400:]}')
    return h


def run(bench, hexfile):
    r = subprocess.run(['vvp', bench, f'+hex={hexfile}', '+maxcycles=200000'],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    out = r.stdout
    if 'PASS' not in out:
        raise SystemExit(f'{hexfile}: did not pass:\n{out[-600:]}')
    m = re.search(r'cycles=(\d+) issued=(\d+) hazard=(\d+)', out)
    x = {k: int(v) for k, v in re.findall(r'\bx_(\w+)=(\d+)', out) if k != 'sum'}
    fe = re.search(r'frontend=(\d+)', out)
    if fe:
        x['frontend'] = int(fe.group(1))
    return int(m.group(1)), int(m.group(2)), x


def per_link(bench, body, tmp, name, lo, hi):
    """Cycles per instruction of `body` in steady state, and the hazard
    cycles per instruction, from two lengths."""
    c1, i1, x1 = run(bench, build(body, lo, tmp, name))
    c2, i2, x2 = run(bench, build(body, hi, tmp, name))
    stalls = {k: (x2[k] - x1.get(k, 0)) / (hi - lo) for k in x2}
    return (c2 - c1) / (hi - lo), {k: round(v, 2) for k, v in stalls.items() if v}


def fmt(stalls):
    return ' '.join(f'{k}={v:g}' for k, v in sorted(stalls.items(), key=lambda kv: -kv[1])) or '-'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', default=os.path.join(ROOT, 'sim', 'build', 'tb_core_vrf.vvp'))
    ap.add_argument('--json')
    args = ap.parse_args()
    res = {'bench': os.path.relpath(args.bench, ROOT), 'dependent': {}, 'independent': {}}
    with tempfile.TemporaryDirectory(prefix='r39-lat-') as tmp:
        print(f"{'dependent chain':<16s} {'cyc/insn':>9s}  stall cycles per insn, by cause; what")
        for name, (body, what) in DEP.items():
            n = len(body)
            cpi, hpi = per_link(args.bench, body, tmp, re.sub(r'\W', '_', name), 16 * n, 80 * n)
            res['dependent'][name] = dict(cycles_per_insn=cpi, stalls_per_insn=hpi,
                                          insns_per_link=n, what=what)
            print(f'{name:<16s} {cpi:>9.2f}  {fmt(hpi)}; {what}')
        print(f"\n{'independent':<16s} {'cyc/insn':>9s}  stall cycles per insn, by cause")
        for name, body in INDEP.items():
            n = len(body)
            cpi, hpi = per_link(args.bench, body, tmp, 'i_' + re.sub(r'\W', '_', name), 8 * n, 40 * n)
            res['independent'][name] = dict(cycles_per_insn=cpi, stalls_per_insn=hpi)
            print(f'{name:<16s} {cpi:>9.2f}  {fmt(hpi)}')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=1)
            f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
