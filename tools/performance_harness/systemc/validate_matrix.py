#!/usr/bin/env python3
"""Directed end-to-end checks of the M1 instructions in the native executor.

Each program is assembled by the product assembler and executed by
tnca-chip; results are compared byte-for-byte with independent Python
references (tools/model/fp32.py mma_ref/fma_ref with floatfmt widening):

  * m.mma.async for FP16/BF16/E5M2/E4M3 at K16/K32/K64, accumulating onto a
    C written with a.vwrite, then a.mul.row and a.fma.row (masked lanes),
    read back with a.vread; result barrier and source-free barrier.
  * RSLICE: two contexts sliced in halves name a0/b0 and get disjoint
    physical accumulators and barriers; RPAIR aliasing is covered by
    validate_roles.py.
  * fence ordering and the architectural trap path (TRAP_USER is a launch
    FAULT, not a model refusal or a silent no-op).
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

sys.path.insert(0, str(ROOT/'tools/model'))
import floatfmt  # noqa: E402
from fp32 import fma_ref, mma_ref  # noqa: E402

DT = {0: 'f16', 1: 'bf16', 2: 'e5m2', 3: 'e4m3'}
SUFFIX = {16: 'k16', 32: 'k32', 64: 'k64'}


def widen(dtype):
    name = DT[dtype]
    return lambda bits: floatfmt.convert(name, 'f32', bits)


def element(rng, dtype):
    if dtype == 0:
        return (rng.getrandbits(1) << 15) | (rng.randint(9, 19) << 10) | rng.getrandbits(10)
    if dtype == 1:
        return (rng.getrandbits(1) << 15) | (rng.randint(115, 135) << 7) | rng.getrandbits(7)
    if dtype == 2:
        return (rng.getrandbits(1) << 7) | (rng.randint(9, 21) << 2) | rng.getrandbits(2)
    return (rng.getrandbits(1) << 7) | (rng.randint(3, 12) << 3) | rng.getrandbits(3)


def f32(rng):
    return (rng.getrandbits(1) << 31) | (rng.randint(118, 134) << 23) | rng.getrandbits(23)


def flat(elements, nbytes):
    return b''.join(e.to_bytes(nbytes, 'little') for e in elements)


def mma_case(out, binary, cfg, name, dtype, K, rng):
    nbytes = 1 if dtype >= 2 else 2
    A = [element(rng, dtype) for _ in range(16 * K)]
    B = [element(rng, dtype) for _ in range(K * 16)]
    C = [f32(rng) for _ in range(256)]
    X = [f32(rng) for _ in range(16)]
    s_mul, s_fma = f32(rng), f32(rng)
    mask_mul, mask_fma = 0b1011011100101101, 0xffff
    a_regs, b_regs = 16 * K * nbytes // 64, K * 16 * nbytes // 64
    dt_suffix = '' if dtype == 0 else '.' + DT[dtype]
    loads = '\n'.join(f'    v.ld.w v{i}, [s20, #{64*i}]' for i in range(a_regs))
    loads += '\n' + '\n'.join(f'    v.ld.w v{a_regs+i}, [s21, #{64*i}]' for i in range(b_regs))
    program = f'''
    .li s20, #0x40000
    .li s21, #0x48000
    .li s22, #0x50000
    .li s23, #0x58000
    .li s24, #0x5c000
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
    m.mma.async.{SUFFIX[K]}{dt_suffix}.sf a3, v0, v{a_regs}, s6
    mb.wait s7
    mb.wait s6
    .li s8, #3
    .li s9, #{s_mul}
    .li s10, #{mask_mul}
    p.frombits p2, s10
    @p2 a.mul.row a3, s8, s9
    v.ld.w v1, [s24, #0]
    .li s8, #5
    .li s9, #{s_fma}
    a.fma.row a3, s8, v1, s9
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
    code = out/f'{name}.hex'
    (out/f'{name}.s').write_text(program)
    assemble(program, code)
    want = mma_ref(A, B, C, 16, 16, K, widen=widen(dtype))
    want[3*16:4*16] = [fma_ref(want[48+n], s_mul, 0, mul_only=True) if (mask_mul >> n) & 1 else want[48+n] for n in range(16)]
    want[5*16:6*16] = [fma_ref(want[80+n], s_fma, X[n]) for n in range(16)]
    files = {}
    for key, payload in (('a', flat(A, nbytes)), ('b', flat(B, nbytes)), ('c', struct.pack('<256I', *C)),
                         ('x', struct.pack('<16I', *X)), ('expected', struct.pack('<256I', *want))):
        files[key] = out/f'{name}-{key}.bin'
        files[key].write_bytes(payload)
    script = out/f'{name}.commands'
    script.write_text(f"code {code}\ndata 0x40000 {files['a']}\ndata 0x48000 {files['b']}\n"
                      f"data 0x50000 {files['c']}\ndata 0x5c000 {files['x']}\nlaunch 1 0 0 1 1 1\nfence\n"
                      f"expect 0x58000 {files['expected']}\n")
    r = run(binary, cfg, script, out/name, mode='chip')
    assert r['status'] == 'completed-prediction', out/name/'stderr.log'
    counters = r['summary']['counters']
    assert counters['core_0_mma_commands'] == 1 and counters['core_0_class_a_vread'] == 16
    return dict(dtype=DT[dtype], K=K, cycles=r['summary']['visible_cycles'],
                matrix_engine_busy_cycles=counters['core_0_matrix_engine_busy_cycles'])


def slice_case(out, binary, cfg, rng):
    # Two contexts, two slices (mask bit 31, log2 = 1): each names a0 and b0.
    program = '''
    s.rdspecial s1, role_id
    s.addi s2, s1, #1
    s.shli s3, s2, #20            // per-role scale: (role + 1) << 20 as f32 bits
    s.addi s3, s3, #0x3f80
    s.shli s3, s3, #16
    s.lshri s3, s3, #16
    .li s4, #0x3f800000
    s.shli s5, s1, #23
    s.add s4, s4, s5              // 1.0 for role 0, 2.0 for role 1
    v.splat v0, s4
    m.zero a0
    .li s6, #0
init:
    a.vwrite a0, s6, v0
    s.addi s6, s6, #1
    s.cmpi.lt s7, s6, #16
    br.nz s7, init
    mb.arm s8, b0, #1, #0
    mb.arrive b0
    mb.wait s8
    .li s9, #0x40000000           // 2.0
    .li s6, #7
    a.mul.row a0, s6, s9
    s.shli s10, s1, #10
    .li s11, #0x60000
    s.add s11, s11, s10
    .li s6, #0
dump:
    a.vread v1, a0, s6
    v.st.w [s11, #0], v1
    s.addi s11, s11, #64
    s.addi s6, s6, #1
    s.cmpi.lt s7, s6, #16
    br.nz s7, dump
    exit
'''
    code = out/'slice.hex'
    (out/'slice.s').write_text(program)
    assemble(program, code)
    want = b''
    for role in range(2):
        one = 0x3f800000 + (role << 23)
        rows = [[one] * 16 for _ in range(16)]
        rows[7] = [fma_ref(one, 0x40000000, 0, mul_only=True)] * 16
        want += b''.join(struct.pack('<16I', *r) for r in rows)
    expected = out/'slice-expected.bin'
    expected.write_bytes(want)
    script = out/'slice.commands'
    mask = 0x3 | (1 << 31) | (1 << 29)
    script.write_text(f'code {code}\nlaunch_mask 1 0 0 1 1 1 {mask:#x}\nfence\nexpect 0x60000 {expected}\n')
    r = run(binary, cfg, script, out/'slice', mode='chip')
    assert r['status'] == 'completed-prediction', out/'slice/stderr.log'
    # Without slicing both contexts would name the same accumulator: a
    # negative control must observe the clobber.
    script = out/'slice-unsliced.commands'
    script.write_text(f'code {code}\nlaunch_mask 1 0 0 1 1 1 0x3\nfence\nexpect 0x60000 {expected}\n')
    r2 = run(binary, cfg, script, out/'slice-unsliced', mode='chip')
    assert r2['status'] == 'failed', 'unsliced contexts unexpectedly kept private accumulators'
    return dict(sliced=r['summary']['visible_cycles'], unsliced_failed=True)


def fault_cases(out, binary, cfg):
    results = {}
    for name, program, text in [
            ('trap', '.li s1, #0x40000\nfence.release.device\ntrap #7\nexit\n', 'MODEL_FAULT status=1 code=64'),
            ('illegal', None, 'ILLEGAL_INSTRUCTION')]:
        code = out/f'{name}.hex'
        if program:
            assemble(program, code)
        else:
            code.write_text(f'{(0x01 << 56) | (0x00 << 48) | (70 << 40):016x}\n')   # s.mov s70: scalar index out of range
        script = out/f'{name}.commands'
        script.write_text(f'code {code}\nlaunch 1 0 0 1 1 1\n')
        r = run(binary, cfg, script, out/name, mode='chip')
        log = (out/name/'stderr.log').read_text()
        assert r['status'] == 'failed' and text in log, (name, log[-500:])
        results[name] = text
    return results


def validate(binary, out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = config(out/'chip.cfg', trace_events=10000)
    rng = random.Random(9071)
    results = dict(mma={})
    for dtype in range(4):
        for K in (16, 32, 64):
            name = f'mma-{DT[dtype]}-k{K}'
            results['mma'][name] = mma_case(out, binary, cfg, name, dtype, K, rng)
    results['rslice'] = slice_case(out, binary, cfg, rng)
    results['faults'] = fault_cases(out, binary, cfg)
    report = dict(schema='tnca-native-matrix-validation/1', status='passed', results=results,
                  oracle='tools/model/fp32.py mma_ref/fma_ref with tools/model/floatfmt.py widening (exact Fractions)',
                  oracle_sha256={p: sha(ROOT/'tools/model'/p) for p in ('fp32.py', 'floatfmt.py')},
                  performance_gate='not-measured', timing='uncalibrated')
    (out/'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(validate(a.binary, a.out)['status'])
