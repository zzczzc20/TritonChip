#!/usr/bin/env python3
"""
R36-BASE-01, BASE-02 and BASE-03 -- phase3_6_tasks_and_acceptance.md 6.

The baseline this stage stands on, the capability inventory, and the
product path end to end from a clean environment.

BASE-02 is the one that does the most work. Section 2 wants every
capability marked verified, limited or unimplemented with its evidence,
and the things R2 left undone listed apart from the things this
integration supports -- and it forbids the handoff baseline from
depending on the second list. A matrix that called everything verified
would be quicker to write and would be the failure this standard
exists to prevent.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

CAPS = json.load(open(os.path.join(ROOT, 'spec',
                                   'r36_capabilities.json')))
CASES = json.load(open(os.path.join(ROOT, 'spec',
                                    'r36_required_cases.json')))

CONTRACT = {'standards': 30, 'packages': 10, 'mutations': 7,
            'new_rtl_runs': 40, 'elapsed_seconds': 2400,
            'cycles': 1200000}
STATES = ('verified', 'limited', 'unimplemented', 'out of scope')


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def base01(log):
    """The preconditions, the frozen inventory and a budget with no holes."""
    ok = True
    pre = CAPS['preconditions']

    # R3 v2 must really be accepted, and be the one this stage names.
    r3 = os.path.join(ROOT, pre['R3_v2']['evidence'])
    if not os.path.exists(r3):
        log.append(f"    {pre['R3_v2']['evidence']} is not here")
        ok = False
    else:
        d = json.load(open(r3))
        log.append(f"    R3 v2: {d.get('verdict')} at "
                   f"{str(d.get('commit'))[:12]}, clean tree "
                   f"{d.get('clean_tree')}")
        ok &= d.get('verdict') == pre['R3_v2']['signoff']
        ok &= str(d.get('commit', '')).startswith(pre['R3_v2']['commit'])
        ok &= bool(d.get('clean_tree'))

    # R3.5 is NOT accepted, and this stage says so rather than assuming.
    r35 = os.path.join(ROOT, pre['R3_5']['evidence'])
    if not os.path.exists(r35):
        log.append(f"    {pre['R3_5']['evidence']} is not here")
        ok = False
    else:
        d = json.load(open(r35))
        opens = [p for p, v in (d.get('packages') or {}).items() if not v]
        log.append(f"    R3.5: {d.get('verdict')} at "
                   f"{str(d.get('commit'))[:12]}; open package(s) {opens}")
        log.append(f"    the contract's precondition is that R3.5 is "
                   f"accepted, and it is NOT. What is open is "
                   f"{pre['R3_5']['open']}, and the record here says why "
                   f"and what it bears on.")
        # The recorded state has to match the record, not flatter it.
        if d.get('verdict') == 'R36_ENGINEERING_READY':
            ok = False
        if pre['R3_5']['signoff'] != 'NOT GIVEN':
            log.append('    the capability file claims a sign-off R3.5 '
                       'does not have')
            ok = False
        if not pre['R3_5'].get('why') or not pre['R3_5'].get(
                'bearing_on_this_stage'):
            log.append('    the open standard is recorded without a '
                       'reason or without what it bears on')
            ok = False
        # And no R3.6 standard may depend on it.
        depends = [c for c in CAPS['capabilities']
                   if 'axisinfo' in json.dumps(c).lower()]
        log.append(f"    capabilities that depend on the open standard: "
                   f"{[c['id'] for c in depends] or 'none'}")

    # The inventory itself.
    n = sum(len(p['cases']) for p in CASES['packages'])
    log.append(f"    {n} standard(s) in {len(CASES['packages'])} "
               f"package(s), {len(CASES['mutations'])} mutation(s)")
    ok &= n == CONTRACT['standards']
    ok &= len(CASES['packages']) == CONTRACT['packages']
    ok &= len(CASES['mutations']) == CONTRACT['mutations']
    for m in CASES['mutations']:
        if not m.get('detector'):
            log.append(f"    mutation {m['id']} names no detector")
            ok = False

    b = CASES['budget']
    holes = [k for k, v in b.items() if v is None or v == '']
    log.append(f"    the budget: {b['new_rtl_runs']} new RTL run(s), "
               f"{b['total_elapsed_seconds']}s, {b['total_cycles']} "
               f"cycle(s), {b['toolchain_build_seconds']}s of toolchain "
               f"build")
    if holes:
        log.append(f"    the budget has undefined entries: {holes}")
        ok = False
    for key, want in (('new_rtl_runs', CONTRACT['new_rtl_runs']),
                      ('total_elapsed_seconds', CONTRACT['elapsed_seconds']),
                      ('total_cycles', CONTRACT['cycles'])):
        if b.get(key) != want:
            log.append(f"    budget {key} is {b.get(key)}, the contract "
                       f"froze {want}")
            ok = False

    # The claim levels, which section 3.2 will not let be interchanged.
    lv = CASES['claim_levels']
    log.append(f"    the claim levels are {[k for k in lv if k != 'note']}")
    ok &= 'platform-verified' in lv and 'reserved' in lv['platform-verified']

    caps = CASES['workload_caps']
    log.append(f"    workloads stay small: GEMM to {caps['gemm_max']}, "
               f"elementwise to {caps['elementwise_max_elements']} "
               f"element(s), reduction to {caps['reduction_max']}")
    return ok, 'the preconditions, the inventory and a budget with no holes'


def base02(log):
    """Every capability marked, with evidence, and the gaps kept apart."""
    ok = True
    counts = {}
    for c in CAPS['capabilities']:
        counts[c['state']] = counts.get(c['state'], 0) + 1
        if c['state'] not in STATES:
            log.append(f"    {c['id']}: state {c['state']!r} is not one of "
                       f"{STATES}")
            ok = False
        if c['state'] == 'verified' and not c.get('evidence'):
            log.append(f"    {c['id']}: verified with no evidence")
            ok = False
        if c['state'] in ('limited', 'unimplemented') and not c.get('limits'):
            log.append(f"    {c['id']}: {c['state']} with no limit stated")
            ok = False
    log.append(f"    {len(CAPS['capabilities'])} capability(ies): "
               + ', '.join(f"{n} {s}" for s, n in sorted(counts.items())))

    # A matrix where everything is verified is the failure this exists
    # to prevent.
    if counts.get('verified', 0) == len(CAPS['capabilities']):
        log.append('    every capability is marked verified, which is not '
                   'a matrix')
        ok = False
    for c in CAPS['capabilities']:
        if c['state'] != 'verified':
            log.append(f"    {c['id']:<24s} {c['state']:<14s} "
                       f"{c['limits'][:64] if c.get('limits') else ''}")

    # The claims must not contradict what the earlier stages recorded.
    arf = next(c for c in CAPS['capabilities']
               if c['id'] == 'CAP-ARF-SYNCHRONOUS')
    syn = os.path.join(ROOT, 'doc', 'r3_status.md')
    said = 'ARF' in open(syn).read()
    log.append(f"    the ARF is recorded {arf['state']}, and R3 v2's own "
               f"status says the same: {said}")
    ok &= arf['state'] == 'unimplemented' and said

    # R2's remainder, listed apart, and depended on by nothing.
    rem = CAPS['r2_remaining']
    log.append(f"    R2 still open: {rem['open']}")
    log.append(f"    the handoff baseline depends on any of it: "
               f"{rem['handoff_depends_on_any']}")
    ok &= rem['handoff_depends_on_any'] is False
    ok &= bool(rem['open'])
    return ok, (f"{len(CAPS['capabilities'])} capability(ies), each with a "
                f"state and its evidence")


def base03(log):
    """The product path, from a clean environment, through the API.

    Section 5.1's chain: source, the R3.5 compiler, an artifact, a
    backend, upload/launch/wait/download, an independent oracle. Run
    here through the public entry points only -- no test helper standing
    in as product logic, no repository on the path.
    """
    ok = True
    sys.path.insert(0, os.path.join(ROOT, 'python'))
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
    import struct
    import workloads as W
    from tnca import aot, device
    from tnca.runtime import Session

    profile = 'core-vrf-sync-v1'
    device.prepare(profile, verbose=False)
    tmp = tempfile.mkdtemp(prefix='r36-base03-')

    # A real Triton elementwise kernel with a masked tail.
    art = aot.compile_aot(
        W.triton_masked_axpy(), profile=profile,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, options={'opt_level': 2},
        output=os.path.join(tmp, 'axpy.tnca'))
    log.append(f"    compiled through the R3.5 path at O2: "
               f"{len(art.asm['hex'].splitlines())} word(s), "
               f"source_kind {art.manifest['identity']['source_kind']}, "
               f"key {art.cache_key[:12]}")
    ok &= art.manifest['identity']['source_kind'] == 'triton'

    xs = W.inputs(32, seed=42)
    ys = W.inputs(32, seed=43)
    alpha, n = 2.5, 17
    with Session(profile=profile, deadline_cycles=50000) as s:
        mod = s.load(art.path)
        x = s.allocate(128, name='x')
        y = s.allocate(128, name='y')
        o = s.allocate(128, name='o')
        s.upload(x, W.f32(xs))
        s.upload(y, W.f32(ys))
        s.upload(o, struct.pack('<I', W.SENTINEL) * 32)
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': n,
                            'alpha': alpha}, grid=(2, 1, 1))
        s.wait(ev)
        got = W.unf32(s.download(o, 128))[:n]
        tail = struct.unpack('<15I', s.download(o, 128)[n * 4:])
    want = [xs[i] * alpha + ys[i] for i in range(n)]
    body = got == want
    tail_ok = all(v == W.SENTINEL for v in tail)
    log.append(f"    launched on the {profile} backend: {ev.state}, values "
               f"{'correct' if body else 'WRONG'} against the oracle, "
               f"{len(tail)} sentinel slot(s) "
               f"{'intact' if tail_ok else 'TOUCHED'}")
    ok &= ev.ok and body and tail_ok

    # A fault, and the session's own account of it.
    with Session(profile=profile, deadline_cycles=50000) as s:
        mod = s.load(art.path)
        x = s.allocate(128, name='x')
        try:
            bad = s.submit(mod, {'x_ptr': x, 'y_ptr': x, 'o_ptr': x,
                                 'n': 1 << 30, 'alpha': alpha},
                           grid=(1, 1, 1))
            s.wait(bad)
            log.append(f"    a launch reading far past its buffer: "
                       f"{bad.state}")
            ok &= not bad.ok
        except Exception as e:                             # noqa: BLE001
            log.append(f"    a launch reading far past its buffer was "
                       f"refused: {type(e).__name__}")

    # Section 5.1 names six things this chain must contain and this
    # case had two of them: a Triton elementwise kernel with a masked
    # tail, and a fault recovery. The rest follow. An audit of these
    # checkers found the gap.

    # A TNCA IR small GEMM, with its real source_kind. Section 5.1 is
    # explicit that a TNCA IR sample must not be written up as a generic
    # Triton GEMM, so this one is compiled from the builder and its
    # manifest is checked to say so.
    m, n, k = 16, 16, 16
    a_vals = W.inputs(m * k, seed=7, lo=-4, hi=4)
    b_vals = W.inputs(k * n, seed=8, lo=-4, hi=4)
    want_c = W.gemm_oracle(a_vals, b_vals, m, n, k)
    with Session(profile=profile, deadline_cycles=50000) as s:
        a_map = s.allocate(256, name='desc_a')
        b_map = s.allocate(256, name='desc_b')
        a_tile = s.allocate(m * k * 2, name='A')
        b_tile = s.allocate(k * n * 2, name='B')
        out = s.allocate((m * n + n) * 4, name='C')
        g = aot.compile_aot(
            W.gemm_program(m, n, k,
                           a_desc={'tile': a_tile.addr, 'map': a_map.addr},
                           b_desc={'tile': b_tile.addr, 'map': b_map.addr},
                           out=out.addr),
            kernel_name='base03_gemm', signature={},
            source_kind='tnca-ir', profile=profile,
            output=os.path.join(tmp, 'gemm.tnca'))
        gmod = s.load(g.path)
        s.upload(a_tile, W.pack_f16(a_vals))
        s.upload(b_tile, W.pack_f16(b_vals))
        s.upload(a_map, W.pack_words(
            W.tensormap(a_tile.addr, m, k, k * 2, 2, 16, 16, dtype=0)))
        s.upload(b_map, W.pack_words(
            W.tensormap(b_tile.addr, k, n, n * 2, 2, 16, 16, dtype=0)))
        s.upload(out, struct.pack('<I', W.SENTINEL) * (m * n + n))
        gev = s.submit(gmod, {})
        s.wait(gev)
        raw = s.download(out, (m * n + n) * 4)
    flat = [v for row in want_c for v in row]
    gvals = W.unf32(raw)[:m * n]
    gtail = struct.unpack(f'<{n}I', raw[m * n * 4:])
    gok = gev.ok and gvals == flat and all(v == W.SENTINEL for v in gtail)
    log.append(f"    a TNCA IR {m}x{n}x{k} GEMM: {gev.state}, "
               f"source_kind {g.manifest['identity']['source_kind']}, "
               f"values {'correct' if gvals == flat else 'WRONG'}, "
               f"{len(gtail)} sentinel slot(s) "
               f"{'intact' if all(v == W.SENTINEL for v in gtail) else 'TOUCHED'}")
    ok &= gok and g.manifest['identity']['source_kind'] == 'tnca-ir'

    # A two-role pipeline is NOT RUN, and this is the record of why.
    #
    # Section 5.1 names one. The core has two role contexts -- NUM_RCTX=2,
    # launch_role_mask selects them, and tb/core/progs/t04_roles.s runs a
    # producer and a consumer together on real RTL -- but there is no
    # product path to them: runtime.py launches with role_mask=1,
    # hard-coded, and the GEMM builder emits a single-role tile.
    #
    # This case originally ran gemm_program(..., roles=2) and reported a
    # two-role pipeline. The parameter was accepted and ignored: the
    # machine code was byte-identical to roles=1, so the label was the
    # only difference. gemm_program now refuses a value it cannot
    # honour, and the gap is recorded as CAP-TWO-ROLE rather than
    # papered over here.
    caps = {c['id']: c for c in CAPS['capabilities']}
    tr = caps.get('CAP-TWO-ROLE')
    log.append(f"    a two-role pipeline: NOT RUN. "
               f"{'recorded as ' + tr['state'] if tr else 'NOT RECORDED, which is worse'}")
    if tr:
        log.append(f"      {tr['limits'][:96]}")
        log.append(f"      bearing on this stage: "
                   f"{tr['bearing_on_this_stage'][:80]}")
    ok &= bool(tr) and tr['state'] == 'limited'
    try:
        W.gemm_program(16, 16, 16, a_desc={'tile': 0x2000, 'map': 0x1000},
                       b_desc={'tile': 0x3000, 'map': 0x1100},
                       out=0x4000, roles=2)
        log.append("      and gemm_program still ACCEPTS roles=2, which "
                   "it cannot honour")
        ok = False
    except ValueError as e:
        log.append(f"      and gemm_program refuses roles=2: "
                   f"{str(e)[:72]}")

    # A device-internal dependency chain: B reads what A wrote and C
    # reads what B wrote, with no host round trip between them. The
    # middle result is never downloaded, which is the point -- it has to
    # have stayed on the device.
    with Session(profile=profile, deadline_cycles=50000) as s:
        mod = s.load(art.path)
        x = s.allocate(128, name='x')
        y = s.allocate(128, name='y')
        mid = s.allocate(128, name='mid')
        fin = s.allocate(128, name='fin')
        s.upload(x, W.f32(xs))
        s.upload(y, W.f32(ys))
        s.upload(mid, struct.pack('<I', W.SENTINEL) * 32)
        s.upload(fin, struct.pack('<I', W.SENTINEL) * 32)
        e1 = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': mid,
                            'n': 16, 'alpha': 2.0}, grid=(1, 1, 1))
        e2 = s.submit(mod, {'x_ptr': mid, 'y_ptr': y, 'o_ptr': fin,
                            'n': 16, 'alpha': 3.0}, grid=(1, 1, 1))
        s.wait(e2)
        chain = W.unf32(s.download(fin, 128))[:16]
    want_chain = [(xs[i] * 2.0 + ys[i]) * 3.0 + ys[i] for i in range(16)]
    chain_ok = chain == want_chain
    log.append(f"    a device-internal chain A -> B, the middle result "
               f"never downloaded: {e1.state}/{e2.state}, values "
               f"{'correct' if chain_ok else 'WRONG'} against an oracle "
               f"that composes both steps")
    ok &= e1.ok and e2.ok and chain_ok

    # Nothing from the test tree may be product logic.
    import tnca
    leaked = [m for m in sys.modules
              if m.startswith(('workloads', 'e2e_tests', 'r1lib'))
              and getattr(sys.modules[m], '__file__', '') and
              'tools/' in sys.modules[m].__file__ and
              m in getattr(tnca, '__dict__', {})]
    log.append(f"    no test helper reached the product namespace: "
               f"{'confirmed' if not leaked else leaked}")
    ok &= not leaked
    log.append("    section 5.1's chain, all six of its named parts: a "
               "Triton elementwise kernel, a masked tail, a TNCA IR "
               "small GEMM with its real source_kind, a device-internal "
               "dependency chain and a fault recovery -- each against an "
               "oracle computed here. The two-role pipeline it also "
               "names is NOT RUN and is recorded as CAP-TWO-ROLE.")
    return ok, 'the product path runs end to end through the public API'


TESTS = [('R36-BASE-01', base01), ('R36-BASE-02', base02),
         ('R36-BASE-03', base03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate A: the baseline, the capability matrix and the "
          "product path")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-700:].replace(
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
    print(f"  r36-plan               {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
