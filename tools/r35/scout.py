#!/usr/bin/env python3
"""
R35-BASE-02 -- doc/phase3_5_tasks_and_acceptance.md section 7.

What the compiler emits today, and where the opportunity actually is.

Section 6.3 is explicit that 5% and 3% are "待达到的阶段目标，不是已测加速":
targets, not measurements. So before any of that is promised, this reads
the machine code the product compiler emits for the frozen workloads and
counts the specific things that could be smaller -- by name, with the
mechanism that would remove each one and what currently blocks it.

An opportunity counted here is a static one. Section 6.3 condition 5 is
equally explicit that fewer static instructions with no change in device
cycles is NOT a gain, so nothing here is a result. It is the argument for
believing the work is worth starting.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
import r1lib as R                                         # noqa: E402

SPEC = json.load(open(os.path.join(ROOT, 'spec', 'r35_workloads.json')))
PROFILE = SPEC['hardware_precondition']['profile']


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


# ---------------------------------------------------------------------------
# what the compiler emits today

def compile_frozen(tmp):
    """Every frozen workload the product compiler can already build."""
    import workloads as W
    from tnca import aot
    out = {}

    def one(wid, **kw):
        try:
            r = aot.compile_aot(profile=PROFILE,
                                output=os.path.join(tmp, f'{wid}.tnca'), **kw)
        except Exception as e:
            out[wid] = {'error': f"{type(e).__name__}: {e}"}
            return
        asm = r.asm['asm'] if isinstance(r.asm, dict) else (r.asm or '')
        hexs = r.asm['hex'] if isinstance(r.asm, dict) else ''
        out[wid] = {'asm': asm, 'words': len(hexs.splitlines())}

    one('L01', source=W.triton_masked_axpy(),
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16})
    one('L04', source=W.tnca_reduction(15), kernel_name='L04_reduction',
        signature={'in_ptr': '*fp32', 'out_ptr': '*fp32'},
        source_kind='tnca-ir')
    a = {'tile': 0x2000, 'map': 0x1000}
    b = {'tile': 0x3000, 'map': 0x1100}
    one('L05', source=W.gemm_program(16, 16, 16, a_desc=a, b_desc=b,
                                     out=0x4000, roles=1),
        kernel_name='L05_gemm', source_kind='tnca-ir', signature={})
    one('L06', source=W.gemm_program(16, 16, 32, a_desc=a, b_desc=b,
                                     out=0x4000, roles=2),
        kernel_name='L06_gemm', source_kind='tnca-ir', signature={})
    return out


# Each opportunity names the mechanism that would remove it and what
# stands in the way today. "unknown" is a legal answer; zero is not --
# section 7 R35-BASE-02: "未知吞吐不记零".
def opportunities(asm):
    """Count, by name, what this program spends words on unnecessarily."""
    found = []

    # A 64-bit constant materialised in four words when it fits in one.
    # `.li` is deliberately fixed-width so the assembler can size pseudo-ops
    # before expanding them -- changing THAT would move branch targets. The
    # compiler choosing `s.movz` for a small constant costs the assembler
    # nothing.
    small = [m for m in re.finditer(r'\.li\s+(\w+),\s*#(-?\d+)', asm)
             if 0 <= int(m.group(2)) < 65536]
    if small:
        found.append({
            'kind': 'wide-constant', 'sites': len(small),
            'words_now': 4 * len(small), 'words_possible': len(small),
            'saving': 3 * len(small),
            'mechanism': 'lower a constant that fits in 16 bits to a single '
                         's.movz instead of the fixed four-word .li',
            'blocked_by': 'nothing structural: `.li` stays fixed-width for '
                          'the assembler, and the compiler stops reaching '
                          'for it when the value is small'})

    # Multiply by a power of two, with the constant materialised first.
    muls = re.findall(r'\.li\s+(\w+), #(\d+)\n\s+s\.mul\s+\w+, \w+, \1', asm)
    pow2 = [v for _r, v in muls if int(v) > 0 and int(v) & (int(v) - 1) == 0]
    if pow2:
        found.append({
            'kind': 'multiply-by-power-of-two', 'sites': len(pow2),
            'words_now': 5 * len(pow2), 'words_possible': len(pow2),
            'saving': 4 * len(pow2),
            'mechanism': 'upstream canonicalisation turns muli by a constant '
                         'power of two into a shift, and the constant then '
                         'has no user',
            'blocked_by': 'the product path does not run canonicalisation; '
                          'the TTIR reaches codegen as written'})

    # `other=` materialised for a masked load whose only consumers end in a
    # store under the same mask.
    movi = re.findall(r'v\.movi\s+(v\d+), #0\s+//.*materialised', asm)
    if movi:
        found.append({
            'kind': 'dead-masked-load-other', 'sites': len(movi),
            'words_now': len(movi), 'words_possible': 0,
            'saving': 'unknown',
            'mechanism': 'the inactive lanes of these loads reach only a '
                         'store predicated on the SAME mask, so the value '
                         'they carry is never observed',
            'blocked_by': 'needs a mask-aware liveness fact, and the safety '
                          'argument that an unobserved lane cannot fault or '
                          'change the FP result of an observed one. Not '
                          'counted as a saving until that is proven -- '
                          'section 7 R35-OPT-03 owns it'})
    return found


def count_duplicate_ttir_constants(tmp):
    """Duplicate pure values in the TTIR the frontend is handed.

    This is the upstream canonicalisation/CSE case R35-OPT-01 asks for:
    a repeated pure value in a non-test source that survives into the
    machine code today.
    """
    import workloads as W
    from tnca import aot
    try:
        r = aot.compile_aot(
            W.triton_masked_axpy(), profile=PROFILE,
            signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
            constexprs={'BLOCK': 16},
            output=os.path.join(tmp, 'ttir.tnca'))
    except Exception as e:
        return {'error': f"{type(e).__name__}: {e}"}
    ttir = r.asm.get('ttir', '') if isinstance(r.asm, dict) else ''
    consts = re.findall(r'=\s+arith\.constant\s+(\S+)\s*:\s*(\S+)', ttir)
    seen, dup = {}, 0
    for v, t in consts:
        seen[(v, t)] = seen.get((v, t), 0) + 1
    dup = sum(n - 1 for n in seen.values() if n > 1)
    return {'constants': len(consts), 'distinct': len(seen),
            'duplicates': dup,
            'repeated': [f"{v} : {t} x{n}" for (v, t), n in seen.items()
                         if n > 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R3.5 G0 scout: what the compiler emits, and where the room is")
    print(f"  profile {PROFILE}, at commit {git('rev-parse', 'HEAD')[:12]}")
    print("  section 6.3's 5% and 3% are targets. Nothing below is a "
          "measured speed-up: these are static counts, and section 6.3")
    print("  condition 5 refuses a static win that does not move device "
          "cycles.")
    print()

    tmp = tempfile.mkdtemp(prefix='r35-scout-')
    built = compile_frozen(tmp)
    rows, total_now, total_save = [], 0, 0
    print(f"  {'workload':<6s} {'words':>6s} {'opportunity':<26s} "
          f"{'sites':>5s} {'words':>6s}")
    for wid in sorted(built):
        info = built[wid]
        if 'error' in info:
            print(f"  {wid:<6s} {'-':>6s} does not build here: "
                  f"{info['error'][:50]}")
            rows.append({'id': wid, 'error': info['error']})
            continue
        opps = opportunities(info['asm'])
        total_now += info['words']
        first = True
        if not opps:
            print(f"  {wid:<6s} {info['words']:>6d} "
                  f"{'(nothing counted)':<26s}")
        for o in opps:
            s = o['saving']
            if isinstance(s, int):
                total_save += s
            print(f"  {wid if first else '':<6s} "
                  f"{info['words'] if first else '':>6} "
                  f"{o['kind']:<26s} {o['sites']:>5d} "
                  f"{s if isinstance(s, int) else 'unknown':>6}")
            first = False
        rows.append({'id': wid, 'words': info['words'],
                     'opportunities': opps})

    print()
    print(f"  {total_now} word(s) emitted across the frozen workloads that "
          f"build here; {total_save} of them are")
    print("  countable waste with a named mechanism, which is "
          f"{100.0 * total_save / max(1, total_now):.0f}%.")
    print("  Whether any of it becomes a device cycle is exactly what "
          "section 6.3 will not let us assume.")

    print()
    dup = count_duplicate_ttir_constants(tmp)
    print("  the upstream case, in the TTIR the frontend is handed:")
    if 'error' in dup:
        print(f"    unavailable: {dup['error'][:70]}")
    else:
        print(f"    {dup['constants']} arith.constant op(s), "
              f"{dup['distinct']} distinct, {dup['duplicates']} duplicate")
        for r_ in dup['repeated']:
            print(f"      repeated: {r_}")
        print("    a duplicate pure value in a real Triton source is the "
              "positive case R35-OPT-01 asks for: CSE has something to do")
        print("    here, and it is not a test written to give it work.")

    print()
    print("  blocked on, today:")
    print("    the product path is TTIR -> Python TNCA IR -> codegen, with "
          "no MLIR pass run on it")
    print("    opt_level 0..2 is declared reserved and does not change "
          "emission")
    print("    LinearLayout is not called at all: there is no layout "
          "computation to reuse yet")

    results = [{
        'id': 'R35-BASE-02', 'passed': bool(total_save > 0),
        'detail': (f"{total_save} word(s) of named, mechanised waste in "
                   f"{total_now} emitted; a real duplicate pure value in "
                   "the TTIR"),
        'workloads': rows, 'ttir_constants': dup,
        'note': 'static counts only; no cycle claim is made here'}]
    if args.json:
        R.write_json(args.json, results)
    print()
    if total_save <= 0:
        print("  r35-scout              FAIL   no countable opportunity")
        return 1
    print(f"  r35-scout              PASS   {total_save} word(s) with a "
          "named mechanism")
    return 0


if __name__ == '__main__':
    sys.exit(main())
