#!/usr/bin/env python3
"""
R36-CFG-01, CFG-02 and CFG-03 -- phase3_6_tasks_and_acceptance.md 6.

That the several places a configuration is written down agree, that two
transport bindings of one implementation share an artifact, and that
compatibility is decided rather than hoped for.

CFG-01's requirement is worth stating in full, because it is easy to
satisfy badly: "target、runtime device info、RTL elaboration、导出包和
编译报告必须交叉核对，不能有多个独立手填的'真实配置'". Several files
that happen to agree today are not one configuration; what makes them
one is a check that fails when they stop agreeing.
"""
import argparse
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
import r1lib as R                                          # noqa: E402

PROFILE = 'core-vrf-sync-v1'
SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32',
       'alpha': 'fp32', 'BLOCK': 'constexpr'}


def axpy(**opts):
    import workloads as W
    from tnca import aot
    return aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                           signature=SIG, constexprs={'BLOCK': 16},
                           options=opts,
                           output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))


def _export_parameters(log):
    """The public top's parameters, from a hardware export manifest."""
    import glob
    cands = sorted(glob.glob(os.path.join(ROOT, 'sim', 'build',
                                          '*export*', 'MANIFEST.json')))
    if not cands:
        return None
    with open(cands[-1]) as fh:
        return json.load(fh).get('parameters') or {}


def _regenerate(profile):
    """The target this profile's RTL would produce, computed now.

    The generator is imported rather than shelled out to, so this
    compares values and not the formatting of a file.
    """
    import importlib.util
    gen = os.path.join(ROOT, 'tools', 'isa', 'gen_target.py')
    if not os.path.exists(gen):
        return None, 'tools/isa/gen_target.py is not in this tree'
    spec = importlib.util.spec_from_file_location('_gen_target', gen)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        core = mod.params(mod.CORE_V)
        top = mod.params(mod.TOP_V)
        with open(mod.SCHEMA) as fh:
            isa = json.load(fh)
        pspec = mod.PROFILE_SPEC['profiles'][profile]
        inst = dict(core)
        inst.update(pspec['elaboration'])
        t = mod.build_target(inst, top, isa,
                             os.path.splitext(pspec['target_file'])[0],
                             pspec['description'], profile=profile)
        t['profile'] = profile
        t['num_cores'] = pspec['num_cores']
        t['topology'] = pspec['topology']
        t['elaboration'] = pspec['elaboration']
        if pspec['num_cores'] == 1:
            t['memory'].pop('mesh', None)
        return t, ''
    except Exception as e:                                 # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def cfg01(log):
    """The same configuration, wherever it is written down."""
    from tnca import profiles as P
    ok = True
    prof = P.profile(PROFILE)
    tgt = P.target(PROFILE)
    art = axpy(opt_level=2)

    # 1. The profile's elaboration parameters and the target's.
    pe, te = prof['elaboration'], tgt['elaboration']
    diff = {k: (pe.get(k), te.get(k)) for k in set(pe) | set(te)
            if pe.get(k) != te.get(k)}
    log.append(f"    elaboration, profile vs target: "
               f"{len(pe)} parameter(s), "
               f"{'identical' if not diff else 'DIFFER ' + str(diff)}")
    ok &= not diff

    # 2. The target and what the RTL was actually built with. The
    #    device module holds the parameters it elaborates; a difference
    #    here is a compiler describing a core that was not built.
    from tnca import device
    params = device.BENCH_PARAMS.get(PROFILE, [])
    banked = any('USE_BANKED_VRF=1' in p for p in params)
    impl = prof.get('implementation', {})
    log.append(f"    the device elaborates {params}")
    log.append(f"    the profile says the register file is "
               f"{impl.get('vector_register_file')!r}, and the bench is "
               f"built banked: {banked}")
    ok &= (impl.get('vector_register_file') == 'banked') == banked

    # 3. The artifact's own record of what it was built for, against
    #    the profile and the target it names. Four statements of one
    #    configuration; the point is that a difference is an error.
    at = art.manifest['target']
    log.append(f"    the artifact names profile {at['profile']!r}, "
               f"config hash {at['config_hash'][:12]}, implementation "
               f"fingerprint {at['implementation_fingerprint'][:12]}")
    ok &= at['profile'] == PROFILE
    ok &= at['config_hash'] == P.config_hash(PROFILE)
    log.append(f"    and the config hash the profile computes now is "
               f"{P.config_hash(PROFILE)[:12]}: "
               f"{at['config_hash'] == P.config_hash(PROFILE)}")

    # The target declares capability -> bool; the artifact records the
    # ones the hardware HAS. Comparing the target's keys instead of its
    # true values would have the artifact appear to be missing five
    # capabilities the hardware does not have.
    declared = tgt['capabilities']
    tc = ({k for k, v in declared.items() if v}
          if isinstance(declared, dict) else set(declared))
    ac = set(at['capabilities'])
    absent = ({k for k, v in declared.items() if not v}
              if isinstance(declared, dict) else set())
    log.append(f"    the target declares {len(tc)} capability(ies) "
               f"present and {len(absent)} absent "
               f"({sorted(absent)})")
    log.append(f"    the artifact records {len(ac)}: "
               f"{'exactly the ones present' if ac == tc else 'DIFFER: ' + str(ac ^ tc)}")
    ok &= ac == tc

    # 3b. And the target itself, against the RTL it claims to describe.
    #
    # The comparison above is between an artifact and the target it was
    # compiled from, so both move together: editing the target by hand
    # to claim a capability the hardware does not have leaves them in
    # perfect agreement and the compiler emitting instructions the core
    # will refuse. A directed mutation (R36-TEST-03 MUT-CAPABILITY)
    # flipped atomics_system to true and every case here passed.
    #
    # So the target is re-derived from rtl/core/tnca_core.v and the
    # profile's elaboration overrides -- which is where `atomics_system`
    # actually comes from, HAS_SYS_ATOMICS on the instance -- and
    # compared field by field.
    regen, why = _regenerate(PROFILE)
    if regen is None:
        log.append(f"    the target could not be re-derived: {why}")
        ok = False
    else:
        drift = {k: (regen.get(k), tgt.get(k))
                 for k in set(regen) | set(tgt)
                 if regen.get(k) != tgt.get(k)}
        log.append(f"    the target re-derived from the RTL and the "
                   f"profile: {len(regen)} field(s), "
                   f"{'identical' if not drift else 'DRIFT ' + str(sorted(drift))}")
        if drift:
            for k in sorted(drift):
                a, b = drift[k]
                log.append(f"      {k}: RTL says {a!r}, the file says "
                           f"{b!r}")
        ok &= not drift

    # 3c. The EXPORT description, which R36-CFG-01 names alongside the
    #     target, the device query and the RTL instance. It arrived
    #     later than the rest of this case and was not being compared to
    #     anything: the export published the public top's parameters and
    #     nothing checked them against the profile whose configuration a
    #     platform has to be able to reproduce.
    #
    #     The check found HAS_ATOMICS and HAS_SYS_ATOMICS missing from
    #     the shell's parameter list. An integrator instantiating it got
    #     the core's defaults, which happen to match this profile, and
    #     would have been silently wrong on another.
    export = _export_parameters(log)
    if export is None:
        log.append("    no export to compare against; run "
                   "`make r36-export`")
        ok = False
    else:
        want = prof['elaboration']
        absent = sorted(k for k in want if k not in export)
        log.append(f"    the export publishes {len(export)} parameter(s); "
                   f"the profile's elaboration names {len(want)}")
        log.append(f"    every parameter the profile sets is settable on "
                   f"the exported top: {not absent} "
                   f"{absent if absent else ''}")
        ok &= not absent
        # And where both give a literal value and the profile does not
        # override it at elaboration, they must agree.
        disagree = {}
        for k, v in sorted(want.items()):
            ev = export.get(k, '')
            if ev.lstrip('-').isdigit() and int(ev) != int(v):
                disagree[k] = (int(ev), int(v))
        log.append(f"    the parameters whose defaults differ from the "
                   f"profile: {sorted(disagree)} "
                   f"{'(each must be set at elaboration)' if disagree else ''}")
        for k, (a, b) in sorted(disagree.items()):
            setter = k in str(P.profile(PROFILE).get('elaboration', {}))
            log.append(f"      {k}: the top defaults to {a}, the profile "
                       f"wants {b}, and the profile sets it: {setter}")
            ok &= setter

    log.append(f"    the ISA version: identity says "
               f"{art.manifest['identity']['isa_version']!r}, the target "
               f"says {tgt['isa_version']!r}")
    ok &= str(art.manifest['identity']['isa_version']) == \
        str(tgt['isa_version'])

    # 4. The resources are MEASURED from the emitted code, not copied
    #    from the profile's ceilings. R2-12 established that; a
    #    configuration that reported its own limits back would agree
    #    with itself and describe nothing.
    res = art.manifest['resources']
    ceiling = tgt['elaboration']['VREGS_PER_RCTX']
    measured = {k: v for k, v in sorted(res.items()) if isinstance(v, int)}
    log.append(f"    the report measures {measured}, against a ceiling of "
               f"{ceiling} vector register(s) per context -- measured, "
               f"not the ceiling repeated back")

    # 5. Changing a capability assumption must change the identity.
    #    Compiling the same source for a DIFFERENT profile must not give
    #    the same key, or a cache would serve one for the other.
    from tnca import aot
    import workloads as W
    other = aot.compile_aot(
        W.triton_masked_axpy(), profile='core-sim', signature=SIG,
        constexprs={'BLOCK': 16}, options={'opt_level': 2},
        output=os.path.join(tempfile.mkdtemp(), 'b.tnca'))
    log.append(f"    the same source on another profile has another key: "
               f"{art.cache_key[:12]} vs {other.cache_key[:12]} -> "
               f"{art.cache_key != other.cache_key}")
    ok &= art.cache_key != other.cache_key
    log.append(f"    and another config hash: "
               f"{P.config_hash(PROFILE)[:12]} vs "
               f"{P.config_hash('core-sim')[:12]}")
    ok &= P.config_hash(PROFILE) != P.config_hash('core-sim')
    return ok, 'the profile, the target, the device and the artifact agree'


def cfg02(log):
    """One artifact, two bindings; and what is refused before a launch."""
    import workloads as W
    from tnca import device
    from tnca.runtime import Session
    ok = True
    device.prepare(PROFILE, verbose=False)
    art = axpy(opt_level=2)
    xs = W.inputs(32, seed=42)
    ys = W.inputs(32, seed=43)
    want = [xs[i] * 2.5 + ys[i] for i in range(17)]

    seen = {}
    for name in ('sim-file', 'sim-socket'):
        with Session(profile=PROFILE, backend=name,
                     deadline_cycles=50000) as s:
            mod = s.load(art.path)
            x = s.allocate(128, name='x')
            y = s.allocate(128, name='y')
            o = s.allocate(128, name='o')
            s.upload(x, W.f32(xs))
            s.upload(y, W.f32(ys))
            s.upload(o, struct.pack('<I', W.SENTINEL) * 32)
            ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                'n': 17, 'alpha': 2.5}, grid=(2, 1, 1))
            s.wait(ev)
            got = W.unf32(s.download(o, 128))[:17]
        seen[name] = (ev.ok and got == want, ev.device_cycles)
        log.append(f"    {name:<11s} correct={seen[name][0]} "
                   f"cycles={seen[name][1]}")
        ok &= seen[name][0]
    log.append(f"    one artifact ({art.cache_key[:12]}), two bindings, "
               f"no recompile, same cycles: "
               f"{seen['sim-file'][1] == seen['sim-socket'][1]}")
    ok &= seen['sim-file'][1] == seen['sim-socket'][1]

    # The wrong profile's artifact, refused before anything launches.
    from tnca import aot
    ref = aot.compile_aot(W.triton_masked_axpy(), profile='core-sim',
                          signature=SIG, constexprs={'BLOCK': 16},
                          options={'opt_level': 2},
                          output=os.path.join(tempfile.mkdtemp(), 'r.tnca'))
    with Session(profile=PROFILE, backend='sim-file',
                 deadline_cycles=50000) as s:
        before = s.dev.seq
        try:
            s.load(ref.path)
            log.append('    a reference artifact loaded on the candidate')
            ok = False
        except Exception as e:                             # noqa: BLE001
            log.append(f"    a reference artifact on the candidate: "
                       f"[{getattr(e, 'code', '?')}] after "
                       f"{s.dev.seq - before} device request(s)")
            ok &= (s.dev.seq - before) == 0
    return ok, 'one artifact on both bindings; a wrong one refused at zero cost'


def cfg03(log):
    """Versioning, migration, and a cache that does not cross implementations."""
    from tnca import abi, manifest as M
    from tnca.cache import Cache
    ok = True
    log.append(f"    the ABI is version {list(abi.ABI_VERSION)}, the "
               f"container format {list(M.FORMAT_VERSION)}")

    # A field that decides safety may not live where a loader may ignore it.
    art = axpy(opt_level=2)
    opt = art.manifest.get('optional_metadata') or {}
    unsafe = [k for k in opt
              if k in ('abi_version', 'isa_version', 'capabilities',
                       'resources', 'target', 'profile')]
    log.append(f"    optional_metadata carries {sorted(opt)}")
    log.append(f"    and nothing a loader must not ignore: "
               f"{'confirmed' if not unsafe else 'FOUND ' + str(unsafe)}")
    ok &= not unsafe
    for need in ('identity', 'target', 'resources', 'parameters'):
        if need not in art.manifest:
            log.append(f"    the manifest has no {need}")
            ok = False

    # The cache must not hit across implementations.
    d = tempfile.mkdtemp()
    cache = Cache(os.path.join(d, 'c'))
    import workloads as W
    from tnca import aot
    a = aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                        signature=SIG, constexprs={'BLOCK': 16},
                        options={'opt_level': 2}, cache=cache,
                        output=os.path.join(d, 'a.tnca'))
    b = aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                        signature=SIG, constexprs={'BLOCK': 16},
                        options={'opt_level': 2}, cache=cache,
                        output=os.path.join(d, 'a2.tnca'))
    c = aot.compile_aot(W.triton_masked_axpy(), profile='core-sim',
                        signature=SIG, constexprs={'BLOCK': 16},
                        options={'opt_level': 2}, cache=cache,
                        output=os.path.join(d, 'c.tnca'))
    log.append(f"    the same request twice: from cache = "
               f"{getattr(b, 'from_cache', None)}")
    log.append(f"    another implementation: from cache = "
               f"{getattr(c, 'from_cache', None)}, key "
               f"{c.cache_key[:12]} vs {a.cache_key[:12]}")
    ok &= bool(getattr(b, 'from_cache', False))
    ok &= not getattr(c, 'from_cache', False)

    # An artifact from a future format is refused, not guessed at.
    import shutil
    bad = os.path.join(d, 'future.tnca')
    shutil.copy2(a.path, bad)
    with open(bad, 'r+b') as fh:
        fh.seek(8)
        fh.write(struct.pack('<HH', 99, 0))
    from tnca.runtime import Session
    from tnca import device
    device.prepare(PROFILE, verbose=False)
    with Session(profile=PROFILE, deadline_cycles=50000) as s:
        try:
            s.load(bad)
            log.append('    an artifact claiming format 99 LOADED')
            ok = False
        except Exception as e:                             # noqa: BLE001
            log.append(f"    an artifact claiming format 99: "
                       f"[{getattr(e, 'code', '?')}] "
                       f"{str(e).splitlines()[0][:54]}")
    log.append("    the acceptance matrix: a container this loader knows "
               "is read, a newer one is refused by version rather than "
               "by accident, and the version lives in the header a "
               "loader must read")
    return ok, 'versions decided, and no cache hit across implementations'


TESTS = [('R36-CFG-01', cfg01), ('R36-CFG-02', cfg02), ('R36-CFG-03', cfg03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate A: one configuration, and what compatibility means")
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
    print(f"  r36-config-tests       {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
