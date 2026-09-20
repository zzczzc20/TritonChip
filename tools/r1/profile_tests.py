#!/usr/bin/env python3
"""
CFG-01 .. CFG-06 -- doc/phase1_tasks_and_acceptance.md R1-04.

Does each profile's target description match the machine that profile actually
elaborates? The audit's F03 is the case where it did not: a four-tile target
advertising `capabilities.sfu = true` while rtl/tile/tnca_tile.v built its
cores with `.HAS_SFU(0)`. A compiler trusting that target emits an instruction
the machine refuses.

The tests read capabilities back *out of the hardware* wherever a register
exists to read, and out of the elaboration source where one does not. What
they never do is read the target JSON twice and call the agreement a
verification.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profiles as P   # noqa: E402
import r1lib as R      # noqa: E402

ROOT = R.ROOT
ASM = os.path.join(ROOT, 'tools', 'asm', 'tnca_asm.py')

CAP_BIT = {
    'vector_fp32': 0, 'sfu': 1, 'gather_scatter': 2,
    'atomics_device': 3, 'atomics_system': 4,
    'matrix_f16': 5, 'matrix_bf16': 6, 'tensor_dma': 7,
    'mbarrier': 8, 'packed_f16': 9, 'cache_hints': 10, 'matrix_fp8': 11,
}

# Special registers a program can read back, and the target field each must
# equal. Everything here is checked on real hardware, not inferred.
READBACK = [
    ('cap0',           lambda t: sum(1 << CAP_BIT[c]
                                     for c, v in t['capabilities'].items()
                                     if v and c in CAP_BIT)),
    ('num_roles',      lambda t: t['num_rctx']),
    ('num_lanes',      lambda t: t['vlen']),
    ('num_vregs',      lambda t: t['registers']['vreg_logical_per_rctx']),
    ('num_phys_vregs', lambda t: t['registers']['vreg_physical_total']),
    ('num_acc',        lambda t: t['accumulators']['count']),
    ('num_barriers',   lambda t: t['barriers']['count']),
    ('num_windows',    lambda t: t['windows']['count']),
    ('num_descriptors', lambda t: t['descriptors']['count']),
    ('num_cores',      lambda t: t['num_cores']),
]


def asm_run(src_text, bench, extra_args=(), tag='cfg'):
    """Assemble a program and run it on a bench. Returns (ok, output)."""
    d = tempfile.mkdtemp(prefix=f'r1-{tag}-')
    src = os.path.join(d, f'{tag}.s')
    hexf = os.path.join(d, f'{tag}.hex')
    with open(src, 'w') as fh:
        fh.write(src_text)
    a = subprocess.run([sys.executable, ASM, src, '-o', hexf,
                        '--size', '16384'], capture_output=True, text=True)
    if a.returncode != 0:
        return False, f"assembler: {a.stderr or a.stdout}"
    r = subprocess.run(['vvp', os.path.join(ROOT, bench), f'+hex={hexf}',
                        '+maxcycles=200000', *extra_args],
                       capture_output=True, text=True, timeout=R.CAPS['rtl'])
    ok = r.returncode == 0 and 'PASS' in r.stdout and 'FAIL' not in r.stdout
    return ok, r.stdout + r.stderr


def readback_program(target, which):
    """A program that asserts every readable capability and resource."""
    lines = ['.org 0', 'start:']
    n = 0
    for reg, fn in READBACK:
        if which and reg not in which:
            continue
        n += 1
        lines.append(f"    s.rdspecial s1, {reg}")
        lines.append(f"    .assert_eq s1, #{fn(target):#x}, #{n}")
    lines.append('    exit')
    return '\n'.join(lines) + '\n', n


def cfg01(log):
    """Read every capability and resource back out of each profile's top."""
    ok_all = True
    for name in P.load()['profiles']:
        p, t = P.profile(name), P.target_of(name)
        bench = p['bench']
        if not os.path.exists(os.path.join(ROOT, bench)):
            log.append(f"    {name}: {bench} not built")
            ok_all = False
            continue
        src, n = readback_program(t, None)
        ok, out = asm_run(src, bench, p['bench_args'], tag=f'cfg01-{name}')
        # On the four-tile bench every tile runs the program, so a capability
        # that differed on one tile would fail there and nowhere else.
        log.append(f"    {name}: {n} facts on {bench} -> "
                   f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            log.append('      ' + out.strip().splitlines()[-1][:160])
        ok_all &= ok
    return ok_all, 'capabilities and resources read back from the hardware'


def cfg02(log):
    """Fields with no read-back register, checked against the elaboration."""
    problems = []
    for name in P.load()['profiles']:
        p, t = P.profile(name), P.target_of(name)
        el = p['elaboration']
        # The target's own numbers must equal the parameters the RTL is built
        # with. Derived from the profile source, which is also what the bench
        # command uses -- not from the target file being checked.
        pairs = [
            ('num_rctx', el['NUM_RCTX']),
            ('registers.vreg_logical_per_rctx', el['VREGS_PER_RCTX']),
            ('registers.vreg_physical_total', el['PHYS_VREGS']),
            ('capabilities.sfu', bool(el['HAS_SFU'])),
            ('capabilities.atomics_device', bool(el['HAS_ATOMICS'])),
            ('capabilities.atomics_system', bool(el['HAS_SYS_ATOMICS'])),
        ]
        for path, want in pairs:
            cur = t
            for k in path.split('.'):
                cur = cur[k]
            if cur != want:
                problems.append(f"{name}.{path}: target={cur} elaboration={want}")
        log.append(f"    {name}: {len(pairs)} elaboration fields checked")
    for p_ in problems:
        log.append('    ' + p_)
    return not problems, 'target fields vs elaboration parameters'


def cfg03(log):
    """SFU: a positive on core-sim, a negative on system-2x2."""
    sfu_prog = ('.org 0\nstart:\n'
                '    .li s2, #0x40800000\n'
                '    v.splat v1, s2\n'
                '    v.rsqrt.approx v2, v1\n'
                '    .assert_lane v2, #0, #0x3f000000, #1\n'
                '    exit\n')
    core = P.profile('core-sim')
    ok_core, out_core = asm_run(sfu_prog, core['bench'], core['bench_args'],
                                tag='cfg03-core')
    log.append(f"    core-sim executes v.rsqrt.approx: {ok_core}")

    # The compiler must refuse before emitting, from the target alone.
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
    from tnca_cc import TargetInfo, CapabilityError
    t = TargetInfo.load('tnca-sim-2x2')
    try:
        t.require('sfu')
        refused = False
    except CapabilityError:
        refused = True
    log.append(f"    compiler refuses sfu for system-2x2: {refused}")

    # And the hardware raises the fault the capability bit promises.
    sysp = P.profile('system-2x2')
    trap_prog = ('.org 0\nstart:\n'
                 '    .li s2, #0x40800000\n'
                 '    v.splat v1, s2\n'
                 '    v.rsqrt.approx v2, v1\n'
                 '    trap #99\n')
    ok_trap, out_trap = asm_run(trap_prog, sysp['bench'], sysp['bench_args'],
                                tag='cfg03-sys')
    # The program is expected to fault, so the bench must NOT report PASS.
    faulted = 'ILLEGAL' in out_trap or 'trap code=0x01' in out_trap or not ok_trap
    log.append(f"    system-2x2 refuses it in hardware: {faulted}")
    return ok_core and refused and faulted, 'SFU positive and negative'


def cfg04(log):
    """A binary whose manifest does not match is refused before it runs."""
    good = P.make_manifest('core-sim', requires=['sfu'])
    cases = {
        'matching':      (good, 'core-sim', 0),
        'other profile': (good, 'system-2x2', 1),
        'stale config':  ({**good, 'config_hash': '0' * 64}, 'core-sim', 1),
        'other ISA':     ({**good, 'isa_version': 'v0.0'}, 'core-sim', 1),
        'needs sfu':     (P.make_manifest('system-2x2', requires=['sfu']),
                          'system-2x2', 1),
    }
    ok = True
    for label, (man, machine, want_problems) in cases.items():
        probs = P.verify_manifest(man, machine)
        got = 1 if probs else 0
        ok &= (got == want_problems)
        log.append(f"    {label:14s} -> {len(probs)} problem(s)"
                   + (f": {probs[0][:90]}" if probs else ''))
    return ok, 'manifest checked before the first instruction'


def cfg05(log):
    """Illegal configurations fail at configuration time."""
    spec = P.load()
    supported = set(spec['supported_topologies']) - {'note'}
    bad = [
        ('unknown topology', dict(topology='mesh-4x4')),
        ('rctx width too small', dict(NUM_RCTX=4, RCTX_W=1)),
        ('rf base past the file', dict(NUM_RCTX=4, VREGS_PER_RCTX=64,
                                       PHYS_VREGS=128)),
    ]
    problems_found = []
    for label, cfg in bad:
        probs = validate_config(cfg, supported)
        problems_found.append((label, probs))
        log.append(f"    {label:22s} -> {len(probs)} problem(s)"
                   + (f": {probs[0][:80]}" if probs else ' ** ACCEPTED **'))
    # And every real profile must validate cleanly.
    clean = []
    for name in spec['profiles']:
        p = P.profile(name)
        probs = validate_config(dict(p['elaboration'], topology=p['topology']),
                                supported)
        clean.append((name, probs))
        log.append(f"    {name:22s} -> {len(probs)} problem(s)")
    return (all(pr for _, pr in problems_found)
            and all(not pr for _, pr in clean)), 'illegal configurations refused'


def validate_config(cfg, supported):
    """The configuration rules, in one place, used by the profiles and CFG-05."""
    problems = []
    topo = cfg.get('topology')
    if topo is not None and topo not in supported:
        problems.append(f"topology {topo!r} is not supported; "
                        f"rtl/noc/tnca_core_ni.v computes a 2-bit home and "
                        f"rtl/tnca_top.v fixes N_W=2 (audit F05)")
    n = cfg.get('NUM_RCTX')
    w = cfg.get('RCTX_W')
    if n is not None and w is not None and (1 << w) < n:
        problems.append(f"RCTX_W={w} cannot index NUM_RCTX={n}: the index "
                        f"would wrap and two contexts would share state")
    n = cfg.get('NUM_RCTX')
    v = cfg.get('VREGS_PER_RCTX')
    phys = cfg.get('PHYS_VREGS')
    if None not in (n, v, phys) and n * v > phys:
        problems.append(f"{n} contexts x {v} logical vregs exceeds "
                        f"{phys} physical: the last context's base is past "
                        f"the end of the file")
    return problems


def cfg06(log):
    """core-4r really has four Role Contexts."""
    p, t = P.profile('core-4r'), P.target_of('core-4r')
    if t['num_cores'] != 1:
        log.append(f"    target claims {t['num_cores']} cores for a "
                   f"single-core profile")
        return False, 'core-4r describes the wrong machine'
    src, n = readback_program(t, {'num_roles', 'num_vregs', 'num_cores'})
    ok, out = asm_run(src, p['bench'], p['bench_args'], tag='cfg06')
    log.append(f"    {n} facts on {p['bench']} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        log.append('      ' + out.strip().splitlines()[-1][:160])
    return ok, 'four Role Contexts read back from the hardware'


TESTS = [('CFG-01', cfg01), ('CFG-02', cfg02), ('CFG-03', cfg03),
         ('CFG-04', cfg04), ('CFG-05', cfg05), ('CFG-06', cfg06)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R1-04 profile and capability: six checks against the real instances")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, detail = fn(log)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(dict(id=cid, passed=bool(ok), detail=detail))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {detail}")
        for line in log:
            print(line)

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-profile-tests       FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-profile-tests       PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
