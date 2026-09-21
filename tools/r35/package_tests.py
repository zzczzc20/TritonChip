#!/usr/bin/env python3
"""
R35-PKG-01, PKG-02 and OBS-01 -- phase3_5_tasks_and_acceptance.md 7.

The public API at each level, the cache key, and what the compiler will
tell you about what it did.

PKG-02's first clause is the one that matters most and is easiest to get
wrong: the optimisation level has to be part of the cache key. If it is
not, a kernel compiled at O0 is served for an O2 request, and the
measurement that follows is of the wrong artifact.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, HERE)
import r1lib as R                                          # noqa: E402

SPEC = json.load(open(os.path.join(ROOT, 'spec', 'r35_workloads.json')))
PROFILE = SPEC['hardware_precondition']['profile']
SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32',
       'alpha': 'fp32', 'BLOCK': 'constexpr'}


def axpy(**opts):
    import workloads as W
    from tnca import aot
    return aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                           signature=SIG, constexprs={'BLOCK': 16},
                           options=opts,
                           output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))


def pkg01(log):
    """The public API at each level, and three launches from one artifact."""
    import workloads as W
    from tnca import device
    from tnca.runtime import Session
    ok = True

    for level in (None, 0, 1, 2):
        opts = {} if level is None else {'opt_level': level}
        art = axpy(**opts)
        log.append(f"    {'B0' if level is None else 'O' + str(level):<3s} "
                   f"compiles to {len(art.asm['hex'].splitlines())} word(s), "
                   f"key {art.cache_key[:12]}")

    # L01 at O2, three dynamic launches, one artifact, one session, one
    # device process, and the compiler not invoked again.
    from tnca import aot
    real = aot.compile_aot
    calls = {'n': 0}

    def counted(*a, **kw):
        calls['n'] += 1
        return real(*a, **kw)
    aot.compile_aot = counted
    try:
        art = counted(W.triton_masked_axpy(), profile=PROFILE,
                      signature=SIG, constexprs={'BLOCK': 16},
                      options={'opt_level': 2},
                      output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))
        first = calls['n']
        device.prepare(PROFILE, verbose=False)
        xs = W.inputs(32, seed=42)
        ys = W.inputs(32, seed=43)
        hashes = set()
        with Session(profile=PROFILE, deadline_cycles=50000) as s:
            mod = s.load(art.path)
            hashes.add(mod.manifest['code']['segments'][0]['sha256'])
            x = s.allocate(128, name='x')
            y = s.allocate(128, name='y')
            s.upload(x, W.f32(xs))
            s.upload(y, W.f32(ys))
            for alpha in (2.5, -1.0, 0.25):
                o = s.allocate(128, name=f'o{alpha}')
                s.upload(o, struct.pack('<I', W.SENTINEL) * 32)
                ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                    'n': 17, 'alpha': alpha},
                              grid=(2, 1, 1))
                s.wait(ev)
                got = W.unf32(s.download(o, 128))[:17]
                want = [xs[i] * alpha + ys[i] for i in range(17)]
                log.append(f"    alpha={alpha:<5} {ev.state:<10s} "
                           f"correct={got == want}")
                ok &= ev.ok and got == want
                hashes.add(mod.manifest['code']['segments'][0]['sha256'])
            log.append(f"    one session, one device process "
                       f"({s.dev.seq} transport request(s))")
        extra = calls['n'] - first
        log.append(f"    compiler invocations after the first: {extra}; "
                   f"distinct code hashes across the three: {len(hashes)}")
        ok &= extra == 0 and len(hashes) == 1
    finally:
        aot.compile_aot = real
    return ok, 'O0..O2 compile, and three launches share one artifact'


def pkg02(log):
    """The cache key, and what must and must not change it."""
    from tnca.cache import Cache
    ok = True

    keys = {}
    for tag, opts in (('B0', {}), ('O0', {'opt_level': 0}),
                      ('O1', {'opt_level': 1}), ('O2', {'opt_level': 2}),
                      ('O2-no-LL', {'opt_level': 2, 'no_ll': True})):
        keys[tag] = axpy(**opts).cache_key
    for tag, k in keys.items():
        log.append(f"    {tag:<9s} {k[:16]}")
    distinct = len(set(keys.values()))
    # B0 and O0 emit the same code and are the same request, so they may
    # share a key. Every other pair must differ: a key that did not
    # separate O1 from O2 would serve one for the other.
    log.append(f"    {distinct} distinct key(s) across 5 mode(s)")
    ok &= keys['O1'] != keys['O2']
    ok &= keys['O2'] != keys['O2-no-LL']
    ok &= keys['B0'] != keys['O2']
    log.append(f"    O1 != O2: {keys['O1'] != keys['O2']}; "
               f"O2 != O2-no-LL: {keys['O2'] != keys['O2-no-LL']}; "
               f"B0 != O2: {keys['B0'] != keys['O2']}")

    # The same request twice is the same key.
    again = axpy(opt_level=2).cache_key
    log.append(f"    recompiling the same request is deterministic: "
               f"{again == keys['O2']}")
    ok &= again == keys['O2']

    # A diagnostic option must not change it.
    diag = axpy(opt_level=2, debug=True).cache_key
    dumped = axpy(opt_level=2,
                  dump_dir=tempfile.mkdtemp()).cache_key
    log.append(f"    debug and dump_dir leave it alone: "
               f"{diag == keys['O2'] and dumped == keys['O2']}")
    ok &= diag == keys['O2'] and dumped == keys['O2']

    # A corrupted entry is refused rather than served.
    d = tempfile.mkdtemp()
    cache = Cache(os.path.join(d, 'c'))
    import workloads as W
    from tnca import aot
    a = aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                        signature=SIG, constexprs={'BLOCK': 16},
                        options={'opt_level': 2}, cache=cache,
                        output=os.path.join(d, 'a.tnca'))
    entries = [os.path.join(dp, f) for dp, _dn, fn in os.walk(
        os.path.join(d, 'c')) for f in fn if f.endswith('.tnca')]
    if entries:
        with open(entries[0], 'r+b') as fh:
            fh.seek(64)
            fh.write(b'\xde\xad\xbe\xef')
        try:
            aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                            signature=SIG, constexprs={'BLOCK': 16},
                            options={'opt_level': 2}, cache=cache,
                            output=os.path.join(d, 'b.tnca'))
            log.append('    a corrupted cache entry was served')
            served = True
        except Exception as e:                             # noqa: BLE001
            log.append(f"    a corrupted cache entry is refused: "
                       f"{type(e).__name__}")
            served = False
        ok &= not served
    else:
        log.append('    the cache wrote no entry to corrupt')
        ok = False

    # Runtime-only: loading an artifact must not need the compiler.
    art = a.path
    # Section 4.2: loading an artifact must not import Triton, LLVM or
    # LinearLayout, and must not need them installed. The compiler may
    # depend on upstream; the runtime may not.
    probe = (
        "import sys;"
        f"sys.path.insert(0, {os.path.join(ROOT, 'python')!r});"
        "import tnca.artifact as A;"
        f"m = A.load({art!r});"
        "man = m[0] if isinstance(m, tuple) else m;"
        "print('loaded', (man.get('identity') or {}).get('kernel_name'));"
        "print('triton imported:', 'triton' in sys.modules);"
        "print('tnca_cc imported:', any(k.startswith('tnca_cc') "
        "for k in sys.modules))")
    r = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                       text=True, timeout=300,
                       env={**os.environ, 'PYTHONPATH': ''})
    for line in (r.stdout or r.stderr).strip().splitlines()[-3:]:
        log.append(f"    {line}")
    ok &= 'triton imported: False' in r.stdout
    ok &= 'tnca_cc imported: False' in r.stdout
    return ok, 'the level is in the key, and a runtime-only load is clean'


def obs01(log):
    """What the compiler reports about what it did."""
    ok = True
    d = tempfile.mkdtemp()
    art = axpy(opt_level=2, dump_dir=d, debug=True)
    stages = sorted(art.asm)
    log.append(f"    the stages dumped are {stages}")
    for want in ('ttir', 'ttir_opt', 'tnir', 'asm', 'hex'):
        if want not in stages:
            log.append(f"    {want} is missing")
            ok = False

    # The dump must not change the code.
    plain = axpy(opt_level=2)
    log.append(f"    dumping and debug leave the code alone: "
               f"{plain.asm['hex'] == art.asm['hex']}")
    ok &= plain.asm['hex'] == art.asm['hex']

    # The report says which optimisations ran, and a not-applicable one
    # says why rather than going missing.
    # The provenance is under optional_metadata: it describes the
    # compile rather than the interface, so it sits outside the part of
    # the manifest a loader has to understand.
    prov = (art.manifest.get('optional_metadata') or {}).get(
        'provenance') or {}
    interesting = {k: v for k, v in prov.items() if k.startswith('opt_')}
    log.append(f"    the manifest records what ran: {interesting}")
    ok &= 'opt_pipeline' in interesting or 'opt_strength_reduced' in \
        interesting

    # The measured resources are real, not the profile's limits.
    res = art.manifest['resources']
    log.append(f"    resources measured from the emitted code: "
               f"{ {k: v for k, v in sorted(res.items()) if isinstance(v, int)} }")

    # A GEMM from the TNCA IR path must not be labelled Triton.
    import workloads as W
    from tnca import aot
    g = aot.compile_aot(
        W.gemm_program(16, 16, 16, a_desc={'tile': 0x2000, 'map': 0x1000},
                       b_desc={'tile': 0x3000, 'map': 0x1100},
                       out=0x4000, roles=1),
        kernel_name='g', signature={}, source_kind='tnca-ir',
        profile=PROFILE, options={'opt_level': 2},
        output=os.path.join(tempfile.mkdtemp(), 'g.tnca'))
    kind = g.manifest['identity']['source_kind']
    log.append(f"    the GEMM's source_kind is {kind!r}, not mislabelled "
               f"as Triton: {kind == 'tnca-ir'}")
    ok &= kind == 'tnca-ir'
    log.append(f"    and it carries no .ttir, because it never had one: "
               f"{'ttir' not in g.asm}")
    ok &= 'ttir' not in g.asm
    return ok, 'the stages, the report, and nothing mislabelled'


TESTS = [('R35-PKG-01', pkg01), ('R35-PKG-02', pkg02), ('R35-OBS-01', obs01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate D: the product API, the cache and the report")
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
    print(f"  r35-package-tests      {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
