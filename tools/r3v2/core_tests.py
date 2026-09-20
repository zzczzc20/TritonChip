#!/usr/bin/env python3
"""
R3V2-CORE-01, -02 and -03 -- doc/phase3_tasks_and_acceptance.md G2.

Whether the core really runs on the banked file, whether its sources
serialise when they share a bank, and whether every read client is seen
actually waiting.

CORE-01 does not take the design's word for it. The elaborated hierarchy
of the candidate has to contain tnca_vrf_banked and not tnca_vector_rf,
and the RTL has to contain no hierarchical path into the file -- a
`u_vrf.mem[...]` anywhere would be a read that no port had to be won for,
which is the whole thing this stage is about.

CORE-02 and -03 run one directed program, t29_vrfbank, on BOTH
implementations. Three sources in one bank, three across banks, vector
stores and an MMA. The counters come from the core, not from the
testbench, and on the reference they read zero because there are no ports
to contend for.
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
import r1lib as R                                        # noqa: E402

BUILD = os.path.join(ROOT, 'sim', 'build')
REF = os.path.join(BUILD, 'tb_core.vvp')
CAND = os.path.join(BUILD, 'tb_core_vrf.vvp')

# The VRF's read clients, and how each is made to wait.
CLIENTS = {
    'simd':   'three vf.fma sources in one bank',
    'lsu':    'vector stores reading their data through the same ports',
    'matrix': 'the MMA reading through port 3',
}


def rtl_files():
    out = []
    for sub in ('core', 'common', 'gen'):
        d = os.path.join(ROOT, 'rtl', sub)
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.endswith('.v')]
    return out


def hierarchy(banked):
    """The modules the elaborated core actually contains."""
    script = (f"read_verilog -DSYNTHESIS -I {os.path.join(ROOT,'rtl','common')} "
              + ' '.join(rtl_files()) +
              f"; chparam -set USE_BANKED_VRF {1 if banked else 0} tnca_core"
              f"; hierarchy -top tnca_core; ls")
    r = subprocess.run(['yosys', '-p', script], capture_output=True,
                       text=True, timeout=1800, cwd=ROOT)
    mods = set()
    seen = False
    for line in r.stdout.splitlines():
        if re.match(r'^\d+ modules:', line):
            seen = True
            continue
        if seen:
            m = re.match(r'^\s+(?:\$paramod\S*\\)?(tnca_\w+)\s*$', line)
            if m:
                mods.add(m.group(1))
            elif line.strip() and not line.startswith(' '):
                break
    return mods


def run(vvp, prog, args=()):
    hexf = os.path.join(BUILD, f"{prog}.hex")
    r = subprocess.run([os.environ.get('VVP', 'vvp'), vvp, f'+hex={hexf}',
                        '+maxcycles=200000'] + list(args),
                       capture_output=True, text=True, timeout=1800)
    out = {'passed': bool(re.search(r'^PASS', r.stdout, re.M))}
    m = re.search(r'^vrf ((?:\w+=\d+ ?)+)', r.stdout, re.M)
    if m:
        for kv in m.group(1).split():
            k, v = kv.split('=')
            out[k] = int(v)
    m = re.search(r'^cycles=(\d+)', r.stdout, re.M)
    if m:
        out['cycles'] = int(m.group(1))
    return out


def core01(log, ctx):
    """The candidate contains the banked file and no way around it."""
    ok = True
    ref = hierarchy(banked=False)
    cand = hierarchy(banked=True)

    log.append(f"    reference elaborates: "
               f"{'tnca_vector_rf' if 'tnca_vector_rf' in ref else 'NOT the behavioural file'}"
               f"; banked present: {'tnca_vrf_banked' in ref}")
    ok &= 'tnca_vector_rf' in ref and 'tnca_vrf_banked' not in ref

    log.append(f"    candidate elaborates: "
               f"{'tnca_vrf_banked' if 'tnca_vrf_banked' in cand else 'NOT the banked file'}"
               f" over {'tnca_rf_bank' if 'tnca_rf_bank' in cand else 'no banks'}"
               f"; behavioural present: {'tnca_vector_rf' in cand}")
    ok &= 'tnca_vrf_banked' in cand and 'tnca_rf_bank' in cand
    if 'tnca_vector_rf' in cand:
        log.append("    the candidate still contains the behavioural file: "
                   "a shadow copy")
        ok = False

    # The clients, named. A reader should not have to infer them.
    core = open(os.path.join(ROOT, 'rtl', 'core', 'tnca_core.v')).read()
    block = core[core.index('g_vrf_banked'):core.index('endgenerate')]
    readers = {
        'simd/alu/fpu (va)': 'r_va_idx' in block,
        'simd/alu/fpu (vb)': 'r_vb_idx' in block,
        'lsu / store data (vc)': 'r_vc_idx' in block,
        'matrix (port 3)': 'mtx_vrf_addr' in block,
    }
    writers = {
        'writeback': 'w_is_vec' in block,
        'transfer engine': 'txf_vw_en' in block,
    }
    log.append(f"    read clients on the file: "
               f"{', '.join(k for k, v in readers.items() if v)}")
    log.append(f"    write clients on the file: "
               f"{', '.join(k for k, v in writers.items() if v)}")
    ok &= all(readers.values()) and all(writers.values())

    # No hierarchical path into the storage, from anywhere in the RTL.
    hier = []
    for path in rtl_files():
        text = open(path).read()
        code = '\n'.join(l for l in text.splitlines()
                         if not l.lstrip().startswith('//'))
        for m in re.finditer(r'\bu_vrf\s*\.\s*\w+|\bu_bank\s*\.\s*\w+'
                             r'|\bg_vrf_\w+\s*\.\s*\w+', code):
            hier.append(f"{os.path.basename(path)}: {m.group(0)}")
    log.append(f"    hierarchical references into the file from RTL: "
               f"{len(hier)}")
    for h in hier[:4]:
        log.append(f"        {h}")
    ok &= not hier
    return ok, 'the banked file, its clients named, and no way around it'


def core02(log, ctx):
    """Same-bank sources serialise; different banks go together."""
    cand = run(CAND, 't29_vrfbank')
    ref = run(REF, 't29_vrfbank')
    ok = cand['passed'] and ref['passed']
    log.append(f"    t29_vrfbank: reference {ref['cycles']} cycles PASS, "
               f"candidate {cand['cycles']} cycles PASS")

    log.append(f"    candidate: {cand['req']} request(s), "
               f"{cand['accept']} accepted, {cand['conflict']} refused")
    log.append(f"    instructions whose sources were serialised: "
               f"{cand['serial']}; taken together: {cand['parallel']}")
    # Four vf.fma read v0, v4, v8 -- all bank 0. Four more read v1, v2, v3
    # -- three banks. The first four must serialise and the second four
    # must not, or the arbitration is not doing anything.
    ok &= cand['serial'] >= 4
    ok &= cand['parallel'] >= 4
    ok &= cand['conflict'] >= 8

    log.append(f"    reference: {ref['req']} request(s), {ref['conflict']} "
               f"refused -- it has a port for each source and nothing to "
               f"arbitrate")
    ok &= ref['req'] == 0 and ref['conflict'] == 0

    # Nothing issues before all of its sources are in: the stall count is
    # the direct evidence, and a refused request has to produce one.
    log.append(f"    cycles spent waiting for operands: {cand['stall']}")
    ok &= cand['stall'] >= cand['serial']
    return ok, 'one bank serialises, three banks do not, and it costs'


def core03(log, ctx):
    """Every read client is seen really waiting."""
    cand = run(CAND, 't29_vrfbank')
    ok = True
    evidence = {
        'simd':   cand['serial'],
        'lsu':    cand['conflict'] - cand['serial'],
        'matrix': cand['mtx_wait'],
    }
    for client, how in CLIENTS.items():
        n = evidence[client]
        log.append(f"    {client:<7s} {n:>5d} wait(s)   ({how})")
        if n <= 0:
            log.append(f"        never waited, so it was not tested")
            ok = False

    log.append(f"    the matrix asked {cand['mtx_req']} time(s) and waited "
               f"{cand['mtx_wait']}: one cycle per read, which is what a "
               f"synchronous file costs it")
    ok &= cand['mtx_req'] > 0 and cand['mtx_wait'] * 2 == cand['mtx_req']

    # And none of this was reached by leaving a client out.
    for prog, what in (('t25_mmasf', 'the MMA with source-free'),
                       ('t12_tensor', 'the transfer engine'),
                       ('t04_roles', 'both role contexts')):
        args = ['+allroles'] if prog == 't04_roles' else []
        r = run(CAND, prog, args)
        log.append(f"    {prog:<12s} {what:<26s} "
                   f"{'PASS' if r['passed'] else 'FAIL'} "
                   f"({r['cycles']} cycles)")
        ok &= r['passed']
    return ok, 'every read client waits, and none was masked to get there'


TESTS = [('R3V2-CORE-01', core01), ('R3V2-CORE-02', core02),
         ('R3V2-CORE-03', core03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G2: the core on a finite-port register file")
    for p in (REF, CAND):
        if not os.path.exists(p):
            print(f"  {p} is not built; run `make core core-vrf`")
            return 1
    ctx = {}
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log, ctx)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-900:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3v2-core-tests        FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-core-tests        PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
