#!/usr/bin/env python3
"""
R3V2-E2E-01, -02 and -03 -- doc/phase3_tasks_and_acceptance.md G4.

The six frozen workloads on both implementations, the post-install runs on
the candidate, and the refusals that must happen before anything launches.

Every expected value comes from tools/r3v2/workloads.py, computed in
Python from the inputs. Section 4.1 says outright that reference and
candidate agreeing is not enough: two runs of the same wrong thing agree
perfectly. Each output is filled with a NaN sentinel first and flanked by
canaries, so "the masked tail was not written" and "the masked tail holds
the right value" stay different observations.
"""
import argparse
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402
import workloads as W                                    # noqa: E402

from tnca import aot, device                             # noqa: E402
from tnca.errors import TncaError                        # noqa: E402
from tnca.runtime import Session                         # noqa: E402

INVENTORY = json.load(open(os.path.join(ROOT, 'spec',
                                        'r3v2_inventory.json')))
REF = 'core-sim'
CAND = 'core-vrf-sync-v1'


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def profile_for(case, impl):
    base = REF if impl == 'reference' else CAND
    return base + ('/slow' if case['memory'] == 'slow-reorder' else '')


def compile_case(case, tmp, profile=REF):
    """One artifact per case per implementation.

    Not one shared artifact. R3V2-E2E-03 requires a reference artifact and
    a candidate device to be refused in BOTH directions before anything
    launches, so the two are deliberately not portable: the profile is
    part of the artifact's identity and the implementations are different
    profiles, whatever they share architecturally.
    """
    cid = case['id']
    out = os.path.join(tmp, f"{cid}-{profile.replace('/', '_')}.tnca")
    if case['workload'] == 'vecadd':
        return aot.compile_aot(
            W.triton_vecadd(),
            signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                       'n': 'i32', 'BLOCK': 'constexpr'},
            constexprs={'BLOCK': 16}, profile=profile, output=out)
    if case['workload'] == 'masked_axpy':
        return aot.compile_aot(
            W.triton_masked_axpy(),
            signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
            constexprs={'BLOCK': case['sizes']['block']},
            profile=profile, output=out)
    if case['workload'] == 'reduction':
        return aot.compile_aot(
            W.tnca_reduction(case['sizes']['valid']),
            kernel_name=f"{cid}_reduction",
            signature={'in_ptr': '*fp32', 'out_ptr': '*fp32'},
            source_kind='tnca-ir', profile=profile, output=out)
    raise NotImplementedError(case['workload'])


def run_gemm(case, impl, log, tmp):
    """C03-C05: allocate first, compile against those addresses, run.

    The GEMM bakes its tile and descriptor addresses in, so the buffers
    have to exist before the code does. That is the opposite order from
    the elementwise cases and is why they do not share a path.
    """
    cid, sizes = case['id'], case['sizes']
    m, n, k = sizes['m'], sizes['n'], sizes['k']
    prof = profile_for(case, impl)
    device.prepare(prof, verbose=False)

    a_vals = W.inputs(m * k, seed=case['seed'], lo=-4, hi=4)
    b_vals = W.inputs(k * n, seed=case['seed'] + 1, lo=-4, hi=4)
    want = W.gemm_oracle(a_vals, b_vals, m, n, k)

    with Session(profile=prof, deadline_cycles=50000) as s:
        s.seeds = {'values': case['seed'], 'memory': case['seed']}
        a_map = s.allocate(256, name='desc_a')
        b_map = s.allocate(256, name='desc_b')
        a_tile = s.allocate(m * k * 2, name='A')
        b_tile = s.allocate(k * n * 2, name='B')
        out = s.allocate((m * n + n) * 4, name='C')
        hi = s.allocate(64, name='canary-hi')

        prog = W.gemm_program(m, n, k,
                              a_desc={'tile': a_tile.addr,
                                      'map': a_map.addr},
                              b_desc={'tile': b_tile.addr,
                                      'map': b_map.addr},
                              out=out.addr)
        art = aot.compile_aot(prog, kernel_name=f"{cid}_gemm",
                              signature={}, source_kind='tnca-ir',
                              profile=device.base_profile(prof),
                              output=os.path.join(
                                  tmp, f"{cid}-{impl}.tnca"))
        mod = s.load(art.path)

        s.upload(a_tile, W.pack_f16(a_vals))
        s.upload(b_tile, W.pack_f16(b_vals))
        s.upload(a_map, W.pack_words(
            W.tensormap(a_tile.addr, m, k, k * 2, 2, 16, 16, dtype=0)))
        s.upload(b_map, W.pack_words(
            W.tensormap(b_tile.addr, k, n, n * 2, 2, 16, 16, dtype=0)))
        # The tile, then one row past it, all sentinel: a masked store
        # that leaked would still satisfy every value check.
        s.upload(out, struct.pack('<I', W.SENTINEL) * (m * n + n))
        s.upload(hi, W.CANARY)

        ev = s.submit(mod, {})
        s.wait(ev)
        raw = s.download(out, (m * n + n) * 4)
        canary_ok = s.download(hi, 64) == W.CANARY

    got = W.unf32(raw)
    flat = [v for row in want for v in row]
    body_ok = got[:m * n] == flat
    tail = struct.unpack(f'<{n}I', raw[m * n * 4:])
    tail_ok = all(v == W.SENTINEL for v in tail)
    ok = ev.ok and body_ok and tail_ok and canary_ok
    log.append(f"      {impl:<10s} {prof:<22s} {ev.state:<10s} "
               f"{m}x{n}x{k} values {'ok' if body_ok else 'WRONG'}, "
               f"{len(tail)} sentinel slot(s) "
               f"{'intact' if tail_ok else 'TOUCHED'}, "
               f"canary {'intact' if canary_ok else 'CLOBBERED'}, "
               f"source_kind={mod.manifest['identity']['source_kind']}")
    if not body_ok:
        bad = next((i for i in range(m * n) if got[i] != flat[i]), 0)
        log.append(f"        element {bad}: {got[bad]} != {flat[bad]}")
    ok &= mod.manifest['identity']['source_kind'] == 'tnca-ir'
    return ok, got[:m * n]


def run_case(case, impl, art, log):
    """Compile once, run on one implementation, check against the oracle."""
    cid = case['id']
    sizes = case['sizes']
    prof = profile_for(case, impl)
    device.prepare(prof, verbose=False)

    # For the reduction `n` is how many lanes are VALID, not how many the
    # buffer holds. The oracle summed all sixteen while the program summed
    # fifteen, and both implementations produced the same wrong-looking
    # answer -- which is precisely why section 4.1 refuses to accept the
    # two agreeing with each other as evidence.
    n = sizes.get('valid') or sizes.get('n') or sizes.get('elements')
    slots = 16 * max(1, sizes.get('grid', 1))
    xs = W.inputs(max(slots, n), seed=case['seed'])
    ys = W.inputs(max(slots, n), seed=case['seed'] + 1)
    alpha = 2.5

    with Session(profile=prof, deadline_cycles=50000) as s:
        s.seeds = {'values': case['seed'], 'memory': case['seed']}
        mod = s.load(art.path)
        lo = s.allocate(64, name='canary-lo')
        x = s.allocate(slots * 4, name='x')
        y = s.allocate(slots * 4, name='y')
        o = s.allocate(slots * 4, name='o')
        hi = s.allocate(64, name='canary-hi')
        s.upload(lo, W.CANARY)
        s.upload(hi, W.CANARY)
        s.upload(x, W.f32(xs[:slots]))
        s.upload(y, W.f32(ys[:slots]))
        s.upload(o, struct.pack('<I', W.SENTINEL) * slots)

        if case['workload'] == 'reduction':
            args = {'in_ptr': x, 'out_ptr': o}
        elif case['workload'] == 'vecadd':
            args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': n}
        else:
            args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': n,
                    'alpha': alpha}
        grid = (sizes.get('grid', 1), 1, 1)
        ev = s.submit(mod, args, grid=grid)
        s.wait(ev)
        raw = s.download(o, slots * 4)
        canaries = (s.download(lo, 64), s.download(hi, 64))
        cycles = getattr(ev, 'cycles', None)

    got = W.unf32(raw)
    want = W.ORACLES[cid](xs, ys, n, alpha)
    touched = len(want)
    body_ok = got[:touched] == want
    tail = struct.unpack(f'<{slots - touched}I', raw[touched * 4:]) \
        if slots > touched else ()
    tail_ok = all(v == W.SENTINEL for v in tail)
    canary_ok = canaries == (W.CANARY, W.CANARY)
    ok = ev.ok and body_ok and tail_ok and canary_ok

    log.append(f"      {impl:<10s} {prof:<22s} {ev.state:<10s} "
               f"values {'ok' if body_ok else 'WRONG'}, "
               f"{len(tail)} sentinel slot(s) "
               f"{'intact' if tail_ok else 'TOUCHED'}, "
               f"canaries {'intact' if canary_ok else 'CLOBBERED'}")
    if not body_ok:
        bad = next((i for i in range(touched) if got[i] != want[i]), 0)
        log.append(f"        element {bad}: {got[bad]} != {want[bad]}")
    return ok, got[:touched]


def e2e01(log, ctx):
    """The frozen cases, both implementations, against an oracle."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    ran = 0
    done = []
    for case in INVENTORY['core']:
        log.append(f"    {case['id']}: {case['workload']} "
                   f"{case['sizes']}  memory={case['memory']} "
                   f"seed={case['seed']}")
        results = {}
        for impl in case['implementations']:
            if case['workload'] == 'gemm':
                good, vals = run_gemm(case, impl, log, ctx['tmp'])
            else:
                base = REF if impl == 'reference' else CAND
                art = compile_case(case, ctx['tmp'], base)
                good, vals = run_case(case, impl, art, log)
            ok &= good
            ran += 1
            results[impl] = vals
        same = results.get('reference') == results.get('candidate')
        log.append(f"      the two implementations agree with each other: "
                   f"{same} (and both with the oracle above, which is the "
                   f"claim that matters)")
        ok &= same
        done.append(case['id'])
    log.append(f"    {ran} run(s) over {len(done)} case(s): "
               f"{', '.join(done)}")
    ctx['ran'] = ran
    return ok, f'{ran} runs, values, mask, sentinel and canaries'


def e2e02(log, ctx):
    """One artifact, three scalars, one session, one device process."""
    if not have_triton():
        return False, 'needs the locked Triton'
    from tnca.cache import Cache
    ok = True
    tmp = ctx['tmp']
    cache = Cache(os.path.join(tmp, 'cache'))
    case = next(c for c in INVENTORY['core'] if c['id'] == 'C02')

    compiles = {'n': 0}
    real = aot.compile_aot

    def counted(*a, **kw):
        compiles['n'] += 1
        return real(*a, **kw)

    aot.compile_aot = counted
    try:
        art = compile_case(case, tmp, CAND)
        first = compiles['n']
        device.prepare(CAND, verbose=False)
        n = case['sizes']['n']
        slots = 16 * case['sizes']['grid']
        xs = W.inputs(slots, seed=case['seed'])
        ys = W.inputs(slots, seed=case['seed'] + 1)
        code_hashes = set()
        with Session(profile=CAND, deadline_cycles=50000) as s:
            mod = s.load(art.path)
            code_hashes.add(mod.manifest['code']['segments'][0]['sha256'])
            x = s.allocate(slots * 4, name='x')
            y = s.allocate(slots * 4, name='y')
            s.upload(x, W.f32(xs))
            s.upload(y, W.f32(ys))
            for alpha in [c['alpha'] for c in INVENTORY['runtime']]:
                o = s.allocate(slots * 4, name=f'o{alpha}')
                s.upload(o, struct.pack('<I', W.SENTINEL) * slots)
                ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                    'n': n, 'alpha': alpha},
                              grid=(case['sizes']['grid'], 1, 1))
                s.wait(ev)
                got = W.unf32(s.download(o, slots * 4))[:n]
                want = [xs[i] * alpha + ys[i] for i in range(n)]
                log.append(f"    alpha={alpha:<5} {ev.state:<10s} "
                           f"correct={got == want}")
                ok &= ev.ok and got == want
                code_hashes.add(
                    mod.manifest['code']['segments'][0]['sha256'])
            log.append(f"    one session, one device process "
                       f"({s.dev.seq} transport request(s) in it)")
        extra = compiles['n'] - first
        log.append(f"    compiler invocations after the first compile: "
                   f"{extra}")
        log.append(f"    distinct code hashes across the three launches: "
                   f"{len(code_hashes)}")
        ok &= extra == 0 and len(code_hashes) == 1
    finally:
        aot.compile_aot = real
    return ok, 'three scalars, one artifact, one device, no recompile'


def e2e03(log, ctx):
    """A mismatched artifact is refused before anything launches."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    tmp = ctx['tmp']
    case = next(c for c in INVENTORY['core'] if c['id'] == 'C01')
    ref_art = compile_case(case, tmp, REF)
    cand_art = compile_case(case, tmp, CAND)

    # Both directions, which is what the contract asks for: a reference
    # artifact on a candidate device AND a candidate artifact on a
    # reference device, each refused before a single launch.
    for sess_prof, art, what in (
            (CAND, ref_art, 'a reference artifact on the candidate'),
            (REF, cand_art, 'a candidate artifact on the reference')):
        device.prepare(sess_prof, verbose=False)
        with Session(profile=sess_prof, deadline_cycles=50000) as s:
            before = s.dev.seq
            try:
                s.load(art.path)
                log.append(f"    {what:<42s} ACCEPTED (bad)")
                ok = False
            except TncaError as e:
                log.append(f"    {what:<42s} [{e.code}] after "
                           f"{s.dev.seq - before} device request(s)")
                ok &= s.dev.seq == before
            # and the matching one still loads on the same session
            right = cand_art if sess_prof == CAND else ref_art
            s.load(right.path)
            log.append(f"      and the matching artifact loads on that "
                       f"same session")

    # A stale cache entry must not be taken for the other profile either.
    log.append(f"    the profile is part of the cache key, so a "
               f"{REF} entry is never served for a {CAND} compile "
               f"(CACHE-01 covers the key's dimensions)")

    # A runtime-only environment can load an artifact with no checkout.
    log.append(f"    the artifact carries its own manifest and code; "
               f"loading needs no source tree (PKG-03 covers the install)")
    return ok, 'a wrong profile refused before any launch'


TESTS = [('R3V2-E2E-01', e2e01), ('R3V2-E2E-02', e2e02),
         ('R3V2-E2E-03', e2e03)]

NOT_YET = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G4: the frozen workloads, both implementations")
    ctx = {'tmp': tempfile.mkdtemp(prefix='r3v2-e2e-')}
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
            log.append('    ' + traceback.format_exc()[-1200:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    for cid, why in sorted(NOT_YET.items()):
        if args.only and cid not in args.only:
            continue
        results.append(dict(id=cid, passed=False, detail=f"NOT RUN: {why}"))
        print(f"  {cid}  NOT RUN  {why}", flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3v2-e2e-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-e2e-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
