#!/usr/bin/env python3
"""Directed end-to-end checks of m.mma.dp.async (ISA sec 9C.4) in the native executor.

Each program is assembled by the product assembler and executed by tnca-chip
with the profile key `mtx_dp=1`; the 16x16 accumulator is read back with
a.vread and compared byte for byte with an independent exact oracle
(tools/model/fdp.py: Fractions, groups of 4 for E4M3 x E4M3 and 2 for every
other pair, RO_48). Covered: all 16 (A format, B format) pairs at K16/K32/K64,
accumulating onto a nonzero C; the same kernels with m.mma.async for the
equal-format pairs (cycle comparison, model domain, uncalibrated); the
capability being off (`mtx_dp=0`): the opcode faults ILLEGAL_INSTRUCTION; and
the reserved modifier bits.

    validate_matrix_dp_chip.py --binary BUILD/tnca-chip --out DIR [--extra-latency N]
"""
import argparse
import json
from pathlib import Path
import random
import struct
import sys

from run import ROOT, run, sha
from validate import config
from validate_chip import assemble
from validate_matrix import DT, SUFFIX, element, f32, flat

sys.path.insert(0, str(ROOT/'tools/model'))
import fdp  # noqa: E402

NAME = {0: 'f16', 1: 'bf16', 2: 'e5m2', 3: 'e4m3'}
WIDTH = {0: 16, 1: 16, 2: 8, 3: 8}


def oracle(fa, fb, A, B, C, K):
    group = 4 if (NAME[fa], NAME[fb]) == ('e4m3', 'e4m3') else 2
    out = []
    for m in range(16):
        for n in range(16):
            acc = C[m * 16 + n]
            for k0 in range(0, K, group):
                g = fdp.group_sum(NAME[fa], NAME[fb], [A[m * K + k] for k in range(k0, k0 + group)],
                                  [B[k * 16 + n] for k in range(k0, k0 + group)])
                if g[0] == 'fin':
                    g = ('fin', fdp.ro(g[1]), g[2])
                acc = fdp.add_round(g, acc)
            out.append(acc)
    return out


def program(mnemonic, K, fa, fb, a_regs, b_regs):
    sfx = f'{SUFFIX[K]}' + ('' if (fa, fb) == (0, 0) else f'.{NAME[fa]}' + ('' if fa == fb else f'.{NAME[fb]}'))
    if mnemonic == 'm.mma.async':
        assert fa == fb
        sfx = f'{SUFFIX[K]}' + ('' if fa == 0 else f'.{NAME[fa]}')
    loads = '\n'.join(f'    v.ld.w v{i}, [s20, #{64*i}]' for i in range(a_regs))
    loads += '\n' + '\n'.join(f'    v.ld.w v{a_regs+i}, [s21, #{64*i}]' for i in range(b_regs))
    return f'''
    .li s20, #0x40000
    .li s21, #0x48000
    .li s22, #0x50000
    .li s23, #0x58000
    .li s1, #0
cload:
    v.ld.w v0, [s22, #0]
    a.vwrite a3, s1, v0
    s.addi s22, s22, #64
    s.addi s1, s1, #1
    s.cmpi.lt s2, s1, #16
    br.nz s2, cload
{loads}
    mb.arm s6, b2, #0, #1
    mb.arm s7, b3, #0, #1
    {mnemonic}.{sfx}.sf a3, v0, v{a_regs}, s6
    mb.wait s7
    mb.wait s6
    .li s1, #0
dump:
    a.vread v2, a3, s1
    v.st.w [s23, #0], v2
    s.addi s23, s23, #64
    s.addi s1, s1, #1
    s.cmpi.lt s2, s1, #16
    br.nz s2, dump
    exit
'''


def one(out, binary, cfg, name, mnemonic, fa, fb, K, rng, expect=True):
    ba, bb = WIDTH[fa] // 8, WIDTH[fb] // 8
    A = [element(rng, fa) for _ in range(16 * K)]
    B = [element(rng, fb) for _ in range(K * 16)]
    C = [f32(rng) for _ in range(256)]
    a_regs, b_regs = 16 * K * ba // 64, K * 16 * bb // 64
    text = program(mnemonic, K, fa, fb, a_regs, b_regs)
    code = out/f'{name}.hex'
    (out/f'{name}.s').write_text(text)
    assemble(text, code)
    files = {}
    payload = (('a', flat(A, ba)), ('b', flat(B, bb)), ('c', struct.pack('<256I', *C)))
    for key, data in payload:
        files[key] = out/f'{name}-{key}.bin'
        files[key].write_bytes(data)
    lines = [f"code {code}", f"data 0x40000 {files['a']}", f"data 0x48000 {files['b']}", f"data 0x50000 {files['c']}",
             'launch 1 0 0 1 1 1', 'fence']
    if expect:
        want = oracle(fa, fb, A, B, C, K)
        files['expected'] = out/f'{name}-expected.bin'
        files['expected'].write_bytes(struct.pack('<256I', *want))
        lines.append(f"expect 0x58000 {files['expected']}")
    script = out/f'{name}.commands'
    script.write_text('\n'.join(lines) + '\n')
    return run(binary, cfg, script, out/name, mode='chip')


def validate(binary, out, extra_latency):
    out.mkdir(parents=True, exist_ok=False)
    cfg = config(out/'chip.cfg', trace_events=10000, mtx_dp=1, mtx_dp_extra_latency=extra_latency)
    cfg_off = config(out/'chip-off.cfg', trace_events=10000)
    rng = random.Random(6013)
    results = dict(pairs={}, fma_compare={})
    for fa in range(4):
        for fb in range(4):
            for K in (16, 32, 64):
                a_regs, b_regs = 16 * K * (WIDTH[fa] // 8) // 64, K * 16 * (WIDTH[fb] // 8) // 64
                if a_regs + b_regs > 62:
                    continue          # does not fit the 64 logical vregs; not a model limit
                name = f'dp-{NAME[fa]}-{NAME[fb]}-k{K}'
                r = one(out, binary, cfg, name, 'm.mma.dp.async', fa, fb, K, rng)
                assert r['status'] == 'completed-prediction', (name, (out/name/'stderr.log').read_text()[-400:])
                c = r['summary']['counters']
                assert c['core_0_matrix_dp_commands'] == 1 and c['core_0_mma_commands'] == 1, c
                results['pairs'][name] = dict(cycles=r['summary']['visible_cycles'],
                                              matrix_engine_busy_cycles=c['core_0_matrix_engine_busy_cycles'])
                if fa == fb:
                    name2 = f'fma-{NAME[fa]}-k{K}'
                    r2 = one(out, binary, cfg_off, name2, 'm.mma.async', fa, fb, K, rng, expect=False)
                    assert r2['status'] == 'completed-prediction', name2
                    c2 = r2['summary']['counters']
                    results['fma_compare'][f'{NAME[fa]}-k{K}'] = dict(
                        fma_cycles=r2['summary']['visible_cycles'],
                        fma_matrix_engine_busy_cycles=c2['core_0_matrix_engine_busy_cycles'],
                        dp_cycles=results['pairs'][name]['cycles'],
                        dp_matrix_engine_busy_cycles=results['pairs'][name]['matrix_engine_busy_cycles'],
                        busy_ratio=round(c2['core_0_matrix_engine_busy_cycles'] /
                                         results['pairs'][name]['matrix_engine_busy_cycles'], 3))
    # negative control: one flipped bit in an expected word must fail the run
    name = 'dp-e4m3-e4m3-k16'
    exp = out/f'{name}-expected.bin'
    bad = bytearray(exp.read_bytes())
    bad[100] ^= 1
    (out/'neg-expected.bin').write_bytes(bytes(bad))
    (out/'neg.commands').write_text((out/f'{name}.commands').read_text().replace(str(exp), str(out/'neg-expected.bin')))
    r = run(binary, cfg, out/'neg.commands', out/'neg', mode='chip')
    assert r['status'] == 'failed', 'a corrupted expectation was accepted'
    results['negative_control'] = 'corrupted expected word rejected'
    # capability off: the opcode is ILLEGAL_INSTRUCTION
    r = one(out, binary, cfg_off, 'dp-off', 'm.mma.dp.async', 3, 3, 16, rng, expect=False)
    log = (out/'dp-off'/'stderr.log').read_text()
    assert r['status'] == 'failed' and 'ILLEGAL_INSTRUCTION' in log and 'MATRIX_DP' in log, log[-500:]
    results['capability_off'] = 'ILLEGAL_INSTRUCTION: m.mma.dp.async without CAPABILITY0.MATRIX_DP'
    # reserved mod[9:7] bit set: illegal even with the capability
    word = (0x40 << 56) | (0x02 << 48) | (3 << 40) | (0 << 32) | (20 << 24) | (6 << 16) | (1 << 7)
    code = out/'dp-reserved.hex'
    code.write_text(f'{word:016x}\n')
    script = out/'dp-reserved.commands'
    script.write_text(f'code {code}\nlaunch 1 0 0 1 1 1\n')
    r = run(binary, cfg, script, out/'dp-reserved', mode='chip')
    log = (out/'dp-reserved'/'stderr.log').read_text()
    assert r['status'] == 'failed' and 'ILLEGAL_INSTRUCTION' in log, log[-500:]
    results['reserved_modifier'] = 'ILLEGAL_INSTRUCTION'
    report = dict(schema='tnca-matrix-dp-chip-validation/1', domain='SystemC/TLM model', status='passed',
                  mtx_dp_extra_latency=extra_latency, results=results,
                  oracle='tools/model/fdp.py (exact Fractions, RO_48)', oracle_sha256=sha(ROOT/'tools/model/fdp.py'),
                  binary_sha256=sha(binary), performance_gate='not-measured', timing='uncalibrated model')
    (out/'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--extra-latency', type=int, default=1)
    a = p.parse_args()
    print(validate(a.binary, a.out, a.extra_latency)['status'])
