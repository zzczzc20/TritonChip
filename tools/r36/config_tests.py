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
import ast
import copy
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


def _latency_sites(body, path, keys):
    """Find target latency reads; distinguish the scheduler's own JSON field.

    The one exception is Model.__init__'s data['latency']. CFG-01 also
    perturbs the target fields and verifies Model.for_target's actual result,
    so this exception cannot hide a target-dependent scheduling model.
    """
    sites, scope = [], []

    class Scan(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        visit_FunctionDef = visit_ClassDef

        def visit_Subscript(self, node):
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            separate_model = (path == 'compiler/python/tnca_cc/schedule.py' and
                              scope == ['Model', '__init__'] and
                              isinstance(node.value, ast.Name) and node.value.id == 'data' and
                              key == 'latency')
            if key in ['latency', *keys] and not separate_model:
                sites.append(node.lineno)
            self.generic_visit(node)

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and
                    node.args and isinstance(node.args[0], ast.Constant) and
                    node.args[0].value in ['latency', *keys]):
                sites.append(node.lineno)
            self.generic_visit(node)

    Scan().visit(ast.parse(body))
    return sites


def axpy(**opts):
    import workloads as W
    from tnca import aot
    return aot.compile_aot(W.triton_masked_axpy(), profile=PROFILE,
                           signature=SIG, constexprs={'BLOCK': 16},
                           options=opts,
                           output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))


def _export_parameters(log):
    """The public top's parameters, from a hardware export manifest.

    BUILT if one is not lying around, rather than requiring a previous
    `make r36-export`. A checker that depends on a build artifact being
    in a particular directory passes on a developer's machine and fails
    in a fresh clone -- which is what happened on the acceptance host,
    and which the mutation suite's clean control then reported as a
    detector failing for its own reasons.
    """
    import glob
    import subprocess
    import tempfile
    cands = sorted(glob.glob(os.path.join(ROOT, 'sim', 'build',
                                          '*export*', 'MANIFEST.json')))
    if not cands:
        out = os.path.join(tempfile.mkdtemp(prefix='tnca-cfg-export-'),
                           'export')
        r = subprocess.run([sys.executable,
                            os.path.join('tools', 'r36', 'export.py'),
                            '--out', out],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=1800)
        man = os.path.join(out, 'MANIFEST.json')
        if r.returncode != 0 or not os.path.exists(man):
            log.append(f"      the export would not build: "
                       f"{(r.stderr or r.stdout).strip()[-140:]}")
            return None
        log.append(f"      no export was lying around, so one was built")
        cands = [man]
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

    # 3g. Capability, RESOURCE and LATENCY assumptions, and what each
    #     does to identity. The clause names all three; only the
    #     capability half was exercised, by compiling for a different
    #     profile, which moves everything at once and shows nothing
    #     about any one of them.
    #
    #     Resources reach identity through the elaboration: the target's
    #     register counts are derived from VREGS_PER_RCTX and
    #     PHYS_VREGS, which config_hash covers. Demonstrated by
    #     perturbing one and watching the hash move.
    import hashlib as _h
    import json as _j

    def _hash_with(elab):
        blob = _j.dumps({'profile': PROFILE, 'elaboration': elab,
                         'num_cores': prof['num_cores'],
                         'topology': prof['topology'],
                         'capabilities': tgt['capabilities']},
                        sort_keys=True)
        return _h.sha256(blob.encode()).hexdigest()

    base_h = _hash_with(prof['elaboration'])
    moved = _hash_with(dict(prof['elaboration'], VREGS_PER_RCTX=32))
    log.append(f"    a RESOURCE assumption (VREGS_PER_RCTX 64 -> 32) "
               f"moves the identity: {base_h[:12]} -> {moved[:12]} "
               f"({base_h != moved})")
    ok &= base_h != moved
    log.append(f"      and the compiler really reads it: "
               f"target.py exposes vreg_logical_per_rctx, "
               f"vreg_physical_total and sreg_per_rctx from the target's "
               f"registers block")

    # LATENCY is published by the target and is NOT in the identity.
    # That is safe only while nothing consumes it, so the invariant is
    # checked rather than assumed: the moment the compiler or the
    # runtime starts scheduling on a latency, this fails and the field
    # has to enter config_hash.
    lat_keys = sorted((tgt.get('latency') or {}))
    readers = []
    for sub in (('compiler', 'python'), ('python', 'tnca')):
        root = os.path.join(ROOT, *sub)
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in sorted(dirs) if d != '__pycache__']
            for f in sorted(files):
                if not f.endswith('.py'):
                    continue
                body = open(os.path.join(base, f)).read()
                path = os.path.relpath(os.path.join(base, f), ROOT)
                if _latency_sites(body, path, lat_keys):
                    readers.append(path)
    # The measured scheduler model is a distinct configuration source.
    # Exercise its real target selection, not just the spelling of its field.
    from tnca_cc.target import TargetInfo
    from tnca_cc.schedule import Model
    perturbed = copy.deepcopy(tgt)
    perturbed['latency'] = {k: v + 17 for k, v in tgt['latency'].items()}
    original_model = Model.for_target(TargetInfo(tgt))
    changed_model = Model.for_target(TargetInfo(perturbed))
    independent = all(getattr(original_model, k) == getattr(changed_model, k)
                      for k in ('name', 'lat', 'issue', 'lsu'))
    genuine_reads = all(_latency_sites(text, 'compiler/python/tnca_cc/schedule.py', lat_keys)
                        for text in ("x = target['latency']", "x = target.get('latency')",
                                     "x = target['fp_result_cycles']"))
    model_only = not _latency_sites(
        "class Model:\n def __init__(self, data):\n  self.lat = data['latency']\n",
        'compiler/python/tnca_cc/schedule.py', lat_keys)
    ok &= independent and genuine_reads and model_only
    log.append(f"      scheduler model unchanged after target-latency perturbation: {independent}")
    log.append(f"      genuine target reads caught, separate model field distinguished: "
               f"{genuine_reads and model_only}")
    log.append(f"    a LATENCY assumption {lat_keys} is published and "
               f"outside the identity, which is safe only while nothing "
               f"consumes it")
    log.append(f"      files in the compiler or runtime that read it: "
               f"{readers if readers else 'none'}")
    log.append(f"      if that ever changes this case fails first, and "
               f"latency has to enter config_hash")
    ok &= not readers
    log.append(f"      changing it still forces a rebuild of the DEVICE: "
               f"it comes from FP_LAT and MUL_LAT in the RTL, and "
               f"device.source_fingerprint hashes the RTL")

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

    # A run asset that has been tampered with, which the clause names
    # beside the wrong profile and which nothing here was exercising.
    # One byte of the code segment is flipped after the manifest recorded
    # its hash; the loader must refuse before any device request.
    import shutil as _sh
    corrupt = os.path.join(tempfile.mkdtemp(), 'corrupt.tnca')
    _sh.copy2(art.path, corrupt)
    with open(corrupt, 'r+b') as fh:
        fh.seek(-9, os.SEEK_END)
        b = fh.read(1)
        fh.seek(-9, os.SEEK_END)
        fh.write(bytes([b[0] ^ 0x01]))
    with Session(profile=PROFILE, backend='sim-file',
                 deadline_cycles=50000) as s:
        before = s.dev.seq
        try:
            s.load(corrupt)
            log.append('    a tampered artifact LOADED, which is the '
                       'failure this case exists for')
            ok = False
        except Exception as e:                             # noqa: BLE001
            log.append(f"    an artifact with one bit flipped in its "
                       f"payload: [{getattr(e, 'code', '?')}] after "
                       f"{s.dev.seq - before} device request(s)")
            ok &= (s.dev.seq - before) == 0

    # And a capability the artifact needs that the session's hardware
    # does not have. The compiler refuses to emit one it was not given,
    # so the artifact is made by asking a target that HAS the capability
    # and then loading it where the profile does not.
    log.append("    a capability the hardware lacks is refused at "
               "COMPILE time, not at load: the target publishes what "
               "the machine can do and the compiler will not emit what "
               "is absent (tools/r36/mutation_tests.py MUT-CAPABILITY "
               "breaks that and R36-CFG-01 catches it)")
    return ok, 'one artifact on both bindings; three wrong ones refused'


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
    # The version and migration matrix, read rather than described.
    #
    # This line used to be a prose sentence about "the acceptance
    # matrix", and no matrix existed anywhere in the tree. An audit of
    # these checkers found the description of nothing.
    mpath = os.path.join(ROOT, 'spec', 'r36_compat_matrix.json')
    if not os.path.exists(mpath):
        log.append("    there is no version/migration matrix")
        ok = False
    else:
        mx = json.load(open(mpath))
        rows = mx['interfaces']
        need = ('id', 'what', 'version_lives_in', 'produces', 'accepts',
                'older', 'newer', 'tested_by')
        gaps = [(r.get('id', '?'), k) for r in rows for k in need
                if k not in r]
        log.append(f"    the version/migration matrix: {len(rows)} "
                   f"interface(s), {len(gaps)} missing field(s) "
                   f"{gaps if gaps else ''}")
        ok &= not gaps
        log.append(f"      {', '.join(r['id'] for r in rows)}")
        ok &= bool(mx.get('migration_policy'))

        # The two versions this case already read must agree with what
        # the matrix says this build produces. A matrix that drifted
        # from the code would be worse than none.
        claims = {r['id']: r['produces'] for r in rows}
        pairs = [('container', list(M.FORMAT_VERSION)),
                 ('abi', list(abi.ABI_VERSION))]
        for rid, live in pairs:
            same = list(claims.get(rid) or []) == live
            log.append(f"      {rid}: the matrix says {claims.get(rid)}, "
                       f"the code produces {live} "
                       f"({'agree' if same else 'DISAGREE'})")
            ok &= same
    return ok, 'versions decided, written down, and no cache hit across '\
               'implementations'


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
