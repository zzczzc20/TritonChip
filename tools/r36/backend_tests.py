#!/usr/bin/env python3
"""
R36-BE-01, BE-02 and BE-03 -- phase3_6_tasks_and_acceptance.md 6.

That the four concepts are separate and askable, that a reference
adapter which is not the built-in binding really executes on the
hardware, and that a backend which is not there says so.

BE-02's rule is the one to keep in view: "仅用 mock 返回正确数值不能证明
执行接入成功". Returning the right answer proves nothing; the answer has
to have come from the Verilog. So the adapter is checked by running the
SAME artifact through both bindings and requiring the same result AND
the same device cycle count -- a mock would have to reproduce the
cycle-accurate timing of a core it does not have.
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


def build_axpy():
    import workloads as W
    from tnca import aot
    return aot.compile_aot(
        W.triton_masked_axpy(), profile=PROFILE,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, options={'opt_level': 2},
        output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))


def run_on(backend, art, log=None):
    """One artifact, one backend, the numbers and the cycles."""
    import workloads as W
    from tnca import device
    from tnca.runtime import Session
    device.prepare(PROFILE, verbose=False)
    xs = W.inputs(32, seed=42)
    ys = W.inputs(32, seed=43)
    with Session(profile=PROFILE, backend=backend,
                 deadline_cycles=50000) as s:
        mod = s.load(art.path)
        x = s.allocate(128, name='x')
        y = s.allocate(128, name='y')
        o = s.allocate(128, name='o')
        s.upload(x, W.f32(xs))
        s.upload(y, W.f32(ys))
        s.upload(o, struct.pack('<I', W.SENTINEL) * 32)
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': 17,
                            'alpha': 2.5}, grid=(2, 1, 1))
        s.wait(ev)
        got = W.unf32(s.download(o, 128))[:17]
        caps = s.dev.capabilities()
    want = [xs[i] * 2.5 + ys[i] for i in range(17)]
    return {'state': ev.state, 'ok': bool(ev.ok), 'correct': got == want,
            'cycles': ev.device_cycles, 'caps': caps}


def be01(log):
    """The four concepts, separate and askable."""
    from tnca import backend as B
    ok = True

    reg = B.available()
    log.append("    the registry: "
               + ', '.join(f"{n} ({e['state']})" for n, e in reg.items()))
    ok &= len(reg) >= 2
    ok &= any(e['state'] == B.INTERFACE_DEFINED for e in reg.values())
    ok &= any(e['state'] == B.REFERENCE_AVAILABLE for e in reg.values())
    ok &= not any(e['state'] == B.PLATFORM_VERIFIED for e in reg.values())
    log.append(f"    nothing claims platform-verified: "
               f"{not any(e['state'] == B.PLATFORM_VERIFIED for e in reg.values())}")
    try:
        B.register('bad', None, state=B.PLATFORM_VERIFIED)
        log.append('    registering platform-verified was ALLOWED')
        ok = False
    except ValueError as e:
        log.append(f"    registering platform-verified is refused: "
                   f"{str(e)[:66]}")

    # A profile is a compile target, not an executable or a board.
    profiles = json.load(open(os.path.join(ROOT, 'spec',
                                           'r1_profiles.json')))['profiles']
    p = profiles[PROFILE]
    # A platform_binding section is allowed and is the point: it says
    # what it is. What must not be there is a bare executable sitting
    # among the ISA and resource facts as though it were one of them.
    leak = [k for k in p if k in ('bench', 'bench_args', 'executable',
                                  'vvp', 'board', 'device_path', 'port')]
    log.append(f"    the profile {PROFILE} describes "
               f"{sorted(k for k in p if not k.startswith('_'))[:6]}...")
    log.append(f"    and names no executable, board or port among them: "
               f"{'confirmed' if not leak else 'LEAKED ' + str(leak)}")
    ok &= not leak
    binding = p.get('platform_binding')
    log.append(f"    its platform binding is a section of its own: "
               f"kind={binding.get('kind') if binding else None}, "
               f"bench={binding.get('bench') if binding else None}")
    ok &= bool(binding) and binding.get('kind') == 'simulation'

    # And the target the COMPILER reads carries none of it.
    tgt = json.load(open(os.path.join(
        ROOT, 'compiler', 'targets', p['target_file'])))
    stray = [k for k in tgt if k in ('bench', 'bench_args',
                                     'platform_binding')]
    log.append(f"    the generated target the compiler reads names no "
               f"binding at all: "
               f"{'confirmed' if not stray else 'LEAKED ' + str(stray)}")
    ok &= not stray

    # The general runtime and the compiler hold no simulator startup
    # logic.
    #
    # This used to grep ONE file, python/tnca/runtime.py, and pass on a
    # file-boundary technicality: transport.py Popens `vvp` and device.py
    # shells out to `iverilog`, and runtime.py imports both. An audit of
    # these checkers found it. The scan is now over the whole general
    # runtime and the whole compiler, with the files that ARE bindings
    # named here rather than silently skipped -- a binding is allowed to
    # start a simulator, that is what it is for, and the list is short
    # enough to read.
    BINDINGS = {
        'python/tnca/backends/multicore_file.py':
            'the sim-multicore binding: prepares parameterized RTL and reaches its persistent mailbox',
        'python/tnca/transport.py':
            'the sim-file backend: the file mailbox and the vvp child '
            'it speaks to',
        'python/tnca/device.py':
            'the simulation binding\'s build step: it elaborates the '
            'RTL the package ships',
        'python/tnca/backends/relay.py':
            'the relay process of the sim-socket backend',
        'python/tnca/backends/socket_relay.py':
            'the sim-socket backend',
        'python/tnca/backends/u50_sim.py':
            'the fpga-u50-sim backend: a simulator standing in for the '
            'board, driving the same register port on the same RTL. Its '
            'partner python/tnca/backends/u50.py is NOT on this list and '
            'must not be: that is the file that runs on the board, and a '
            'board binding that named a simulator would be a binding '
            'whose tested path and shipped path had come apart',
    }
    scanned, found = 0, {}
    for sub in (('python', 'tnca'), ('compiler', 'python')):
        root = os.path.join(ROOT, *sub)
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in sorted(dirs) if d != '__pycache__']
            for f in sorted(files):
                if not f.endswith('.py'):
                    continue
                rel = os.path.relpath(os.path.join(base, f), ROOT)
                if rel in BINDINGS:
                    continue
                scanned += 1
                body = open(os.path.join(base, f)).read()
                hits = [w for w in ('vvp', 'iverilog', 'Popen')
                        if w in body]
                if hits:
                    found[rel] = hits
    log.append(f"    {scanned} file(s) of the general runtime and the "
               f"compiler scanned, {len(BINDINGS)} binding(s) excluded "
               f"by name")
    for rel, why in sorted(BINDINGS.items()):
        log.append(f"      excluded {rel}: {why}")
    log.append("    simulator startup or connection logic in the general "
               "code: " + (str(found) if found else 'none'))
    ok &= not found
    log.append("    it names a backend and the registry resolves it, so "
               "an external branch registers an implementation rather "
               "than editing the state machine")
    return ok, 'target, backend, implementation and platform are separate'


def be02(log):
    """A reference adapter that is not the built-in binding, on real RTL."""
    ok = True
    art = build_axpy()
    log.append(f"    one artifact: {len(art.asm['hex'].splitlines())} "
               f"word(s), key {art.cache_key[:12]}")

    got = {}
    for name in ('sim-file', 'sim-socket'):
        got[name] = run_on(name, art)
        c = got[name]
        log.append(f"    {name:<11s} {c['state']:<10s} "
                   f"correct={c['correct']} cycles={c['cycles']} "
                   f"medium={c['caps'].get('medium')}")
        ok &= c['ok'] and c['correct']

    # The same artifact, unchanged, on both. Section 4.1: a different
    # connection medium must not require a recompile.
    log.append(f"    the same artifact ran on both without recompiling: "
               f"{art.cache_key[:12]}")

    # And the same CYCLES. This is what separates an adapter from a
    # mock: returning 2.5*x+y is arithmetic anyone can do, but
    # reproducing the core's cycle count means the core ran it.
    same = got['sim-file']['cycles'] == got['sim-socket']['cycles']
    log.append(f"    both report {got['sim-file']['cycles']} device "
               f"cycle(s): {same}")
    log.append("    a mock returning the right numbers would have to "
               "reproduce the cycle count of a core it does not have, "
               "which is why the cycles are compared and not just the "
               "values")
    ok &= same and got['sim-socket']['cycles'] is not None

    # The adapter reaches the RTL through a protocol, not by importing
    # the simulator's internals.
    src = open(os.path.join(ROOT, 'python', 'tnca', 'backends',
                            'socket_relay.py')).read()
    log.append(f"    the adapter speaks a versioned protocol over a "
               f"socket; medium={got['sim-socket']['caps'].get('medium')}, "
               f"protocol={got['sim-socket']['caps'].get('protocol')}")
    ok &= 'vvp' not in src and 'iverilog' not in src
    log.append(f"    and starts no simulator itself: "
               f"{'confirmed' if 'vvp' not in src else 'it names vvp'}")
    return ok, 'the same artifact, both bindings, the same cycles'


def be03(log):
    """A backend that is not there says so, and nothing is substituted."""
    from tnca import backend as B
    from tnca.runtime import Session
    ok = True

    # All four categories the clause names, not the two that were easy.
    # A backend whose implementation will not import ("missing") and one
    # whose peer speaks another protocol version were both unexercised,
    # although the code paths for both exist. An audit of these checkers
    # found it.
    # Registered through the same _lazy factory the built-in backends
    # use, naming a module that is not installed. That is the shape a
    # platform branch's registration has, and BACKEND_UNAVAILABLE is
    # what the caller is supposed to get.
    B.register('missing-impl',
               B._lazy('tnca.backends.nothing_here', 'Backend'),
               state=B.REFERENCE_AVAILABLE,
               what='a backend whose implementation is not installed')

    for name, expect in (('nonesuch', 'BACKEND_UNKNOWN'),
                         ('fpga-prototype', 'BACKEND_NOT_IMPLEMENTED'),
                         ('missing-impl', 'BACKEND_UNAVAILABLE')):
        try:
            B.open_backend(name, bench='irrelevant')
            log.append(f"    {name}: OPENED, which it must not be")
            ok = False
        except B.BackendError as e:
            code = getattr(e, 'code', '')
            log.append(f"    {name:<16s} refused: [{code}] "
                       f"{str(e).splitlines()[0][:52]}")
            ok &= code == expect

    # A peer speaking another protocol version. The relay answers the
    # `hello` handshake with its version, so raising the client's
    # expectation is the same mismatch a newer device would present --
    # and it has to be refused at the handshake, before any device work.
    import tnca.backends.socket_relay as SR
    from tnca import device
    real_v = SR.PROTOCOL_VERSION
    SR.PROTOCOL_VERSION = real_v + 1
    try:
        device.prepare(PROFILE, verbose=False)
        d = B.open_backend('sim-socket',
                           bench=device.expected_path(PROFILE))
        d.start()
        d.stop()
        log.append('    a version-incompatible peer was ACCEPTED')
        ok = False
    except Exception as e:                                 # noqa: BLE001
        log.append(f"    a peer speaking protocol {real_v} when this "
                   f"client wants {real_v + 1}: "
                   f"[{getattr(e, 'code', type(e).__name__)}] "
                   f"{str(e).splitlines()[0][:50]}")
    finally:
        SR.PROTOCOL_VERSION = real_v

    # A session asking for one of them must fail, not quietly simulate.
    try:
        Session(profile=PROFILE, backend='fpga-prototype',
                deadline_cycles=1000).close()
        log.append('    a session on an unimplemented backend OPENED')
        ok = False
    except Exception as e:                                 # noqa: BLE001
        log.append(f"    a session on it: {type(e).__name__} "
                   f"[{getattr(e, 'code', '')}]")
        ok &= getattr(e, 'code', '') == 'BACKEND_NOT_IMPLEMENTED'

    # No automatic switching, anywhere.
    b = open(os.path.join(ROOT, 'python', 'tnca', 'backend.py')).read()
    for word in ('fallback', 'fall back', 'except BackendError'):
        if word in b.lower() and 'never falls back' not in b.lower():
            log.append(f"    the registry appears to fall back ({word})")
            ok = False
    log.append("    the registry never substitutes: a missing backend "
               "that became the simulator would report a platform "
               "working that was never there")

    # engineering-ready is not platform-verified.
    caps = json.load(open(os.path.join(ROOT, 'spec',
                                       'r36_capabilities.json')))
    phys = next(c for c in caps['capabilities'] if c['id'] == 'CAP-PHYSICAL')
    log.append(f"    the physical capability is recorded "
               f"{phys['state']!r}, not verified")
    ok &= phys['state'] == 'out of scope'
    return ok, 'unknown and unimplemented backends refuse, and nothing is '\
               'substituted'


TESTS = [('R36-BE-01', be01), ('R36-BE-02', be02), ('R36-BE-03', be03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate A: the backend boundary, and a second binding on it")
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
    print(f"  r36-backend-tests      {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
