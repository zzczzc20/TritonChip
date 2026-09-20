#!/usr/bin/env python3
"""
OBS-01, OBS-02 and OBS-04 -- doc/phase2_tasks_and_acceptance.md R2-12.

OBS-01 is the one with teeth: "没有测量/模型支持的成本标为 unknown，不能填0
冒充零开销." A report that says a kernel uses zero scratch when nobody
measured scratch is not a conservative report, it is a wrong one, and it
reads exactly like a measured zero. So each number is cross-checked against
something computed a different way -- the manifest against the report, and
the instruction count against the size of the code segment, which the
compiler did not derive from it.

OBS-03 is NOT RUN and says why.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                       # noqa: E402

from tnca import aot, artifact, manifest as M           # noqa: E402
from tnca.errors import TncaError                       # noqa: E402
from tnca.runtime import Session                        # noqa: E402

BLOCK = 16
SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'}
INSTR_BYTES = 8            # ISA sec 3: every instruction is 64 bits


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def build(d, name='k.tnca', profile='core-sim'):
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    import kernels_triton as K
    return aot.compile_aot(K.axpy_kernel, signature=SIG,
                           constexprs={'BLOCK': BLOCK}, profile=profile,
                           output=os.path.join(d, name))


def obs01(log):
    """The report, the manifest and the emitted code agree; nothing fakes 0."""
    if not have_triton():
        return False, 'needs the locked Triton'
    d = tempfile.mkdtemp(prefix='r2-obs1-')
    res = build(d)
    man, segments = artifact.load(res.path)
    rep = res.report['resources']
    mres = man['resources']
    ok = True

    # 1. the manifest carries what the report measured
    pairs = [('rf_private_per_role.role0', mres['rf_private_per_role']['role0'],
              rep['vector_regs_used']),
             ('accumulators', mres['accumulators'], rep['accumulators_used']),
             ('barriers', mres['barriers'], rep['barriers_used']),
             ('descriptors', mres['descriptors'], rep['descriptors_used'])]
    for name, in_man, in_rep in pairs:
        if in_man != in_rep:
            log.append(f"    {name}: manifest {in_man} != report {in_rep}")
            ok = False
    log.append(f"    manifest and report agree on {len(pairs)} resource(s)")

    # 2. the instruction count against the code segment, which was measured
    #    from the assembly and the machine code independently
    code = segments['text']
    from_code = len(code) // INSTR_BYTES
    log.append(f"    {rep['assembly_instructions']} assembly instruction(s), "
               f"{rep['machine_instructions']} machine instruction(s), "
               f"{len(code)} byte(s) of code / {INSTR_BYTES} = {from_code}")
    if rep['machine_instructions'] != from_code:
        log.append("    the report disagrees with the code segment (bad)")
        ok = False
    # Expansion never shrinks: `.li` becomes up to four words, so the
    # machine count is at least the assembly count. If it were below, one
    # of the two is counting something it should not.
    if rep['machine_instructions'] < rep['assembly_instructions']:
        log.append("    fewer machine instructions than assembly lines (bad)")
        ok = False
    else:
        log.append(f"    expansion accounts for "
                   f"{rep['machine_instructions'] - rep['assembly_instructions']}"
                   f" extra word(s), and never fewer")

    # 3. every count is within the profile's limit, and is not the limit
    lim = rep['limits']
    if rep['vector_regs_used'] > lim['vector_regs']:
        log.append("    more vector registers used than the profile has")
        ok = False
    if rep['vector_regs_used'] == lim['vector_regs']:
        log.append("    the count equals the profile limit -- this is the "
                   "old bug, the limit reported as the usage")
        ok = False
    log.append(f"    {rep['vector_regs_used']} of {lim['vector_regs']} "
               f"vector registers: a usage, not the limit")

    # 4. an unmeasured cost is 'unknown', never 0
    unmeasured = ['commands_outstanding', 'scratch_bytes']
    for field in unmeasured:
        v = mres[field]
        if v == 0 or v == '0':
            log.append(f"    {field} reported as 0 with nothing measuring "
                       f"it (bad)")
            ok = False
        else:
            log.append(f"    {field:<22s} -> {v!r}")

    # 5. and a measurable one IS a number, so 'unknown' is not a blanket
    if not isinstance(rep['async_commands_issued'], int):
        log.append("    async_commands_issued is not a number either -- "
                   "'unknown' everywhere is not a report")
        ok = False
    else:
        log.append(f"    async_commands_issued  -> "
                   f"{rep['async_commands_issued']} (counted, so 'unknown' "
                   f"is not a blanket)")

    # 6. the counter can be non-zero: a measurement that cannot move is not
    #    a measurement. Checked against assembly that does use transfers.
    probe = os.path.join(ROOT, 'tb', 'core', 'progs', 't26_stfault.s')
    if os.path.exists(probe):
        body = aot._strip_asm(open(probe).read())
        n = sum(1 for line in body if aot._ASYNC.match(line))
        log.append(f"    the same counter on t26_stfault.s -> {n} async "
                   f"command(s)")
        if n == 0:
            log.append("    it never fires, so 0 above means nothing (bad)")
            ok = False
    return ok, 'measured, cross-checked, and unknown where unmeasured'


def obs02(log):
    """Four forced errors: a code, a place, and no bare traceback."""
    if not have_triton():
        return False, 'needs the locked Triton'
    d = tempfile.mkdtemp(prefix='r2-obs2-')
    res = build(d)
    ok = True
    cases = []

    # (a) a bad IR program -- refused at compile time, naming the kernel
    def broken(b, target):
        raise ValueError('this program does not build')

    cases.append(('bad IR', lambda: aot.compile_aot(
        broken, signature={'p': '*fp32'}, source_kind='tnca-ir',
        profile='core-sim', output=os.path.join(d, 'bad.tnca'))))

    # (b) an unknown compile option -- a misspelling, not a new feature
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    import kernels_triton as K
    cases.append(('unknown option', lambda: aot.compile_aot(
        K.axpy_kernel, signature=SIG, constexprs={'BLOCK': BLOCK},
        profile='core-sim', options={'opt_levle': 2},
        output=os.path.join(d, 'o.tnca'))))

    # (c) the wrong ABI major -- an artifact from a future toolchain
    forged = os.path.join(d, 'future.tnca')
    man, segs = artifact.load(res.path)
    man = dict(man)
    man['identity'] = dict(man['identity'])
    man['identity']['abi_version'] = [99, 0]
    artifact.write(forged, M.seal({k: v for k, v in man.items()
                                   if k != 'manifest_hash'}), segs)

    # (d) the wrong profile
    wrong = os.path.join(d, 'wrong.tnca')
    man2 = dict(man)
    man2['identity'] = dict(man2['identity'])
    man2['identity']['abi_version'] = [1, 0]
    man2['target'] = dict(man2['target'])
    man2['target']['profile'] = 'system-2x2'
    artifact.write(wrong, M.seal({k: v for k, v in man2.items()
                                  if k != 'manifest_hash'}), segs)

    s = Session(profile='core-sim', deadline_cycles=200000, host_timeout=180)
    try:
        cases.append(('wrong ABI major', lambda: s.load(forged)))
        cases.append(('wrong profile', lambda: s.load(wrong)))
        # (e) a resource the profile does not have
        cases.append(('a grid past the cap',
                      lambda: s.submit(s.load(res.path),
                                       {'x_ptr': s.allocate(64, name='a'),
                                        'y_ptr': s.allocate(64, name='b'),
                                        'o_ptr': s.allocate(64, name='c'),
                                        'n': 16, 'alpha': 1.0},
                                       grid=(100000, 1, 1))))
        for what, call in cases:
            try:
                call()
                log.append(f"    {what:<22s} -> ACCEPTED (bad)")
                ok = False
            except TncaError as e:
                # A stable code, a place, and a remedy: the three things a
                # user needs and a traceback does not give.
                has = bool(e.code) and bool(getattr(e, 'remedy', None) or
                                            getattr(e, 'where', None))
                log.append(f"    {what:<22s} -> [{e.code}] "
                           f"{e.message[:44]}")
                if not has:
                    log.append(f"        no remedy and no location")
                    ok = False
            except Exception as e:
                log.append(f"    {what:<22s} -> BARE {type(e).__name__}: "
                           f"{str(e)[:40]} (bad)")
                ok = False
    finally:
        s.poisoned = True
        s.close()
    return ok, 'five user errors, five codes, no bare traceback'


def obs03(log):
    """A failure is saved, replayed, and fails the same way."""
    if not have_triton():
        return False, 'needs the locked Triton'
    from tnca import failure

    d = tempfile.mkdtemp(prefix='r2-obs3-')
    ok = True

    # A kernel that traps, through the product compiler.
    def program(b, target):
        v = b.const(1)
        b.assert_eq(v, 0xDEAD, 7)      # 1 != 0xDEAD
        b.exit()

    art = aot.compile_aot(program, kernel_name='trapper', signature={},
                          profile='core-sim', source_kind='tnca-ir',
                          output=os.path.join(d, 'trap.tnca')).path

    bundle = os.path.join(d, 'bundle')
    s = Session(profile='core-sim', deadline_cycles=200000, host_timeout=180)
    try:
        s.seeds = {'values': 42, 'memory': 7}
        mod = s.load(art)
        ev = s.submit(mod, {})
        s.wait(ev)
        log.append(f"    the original run: {ev.state} "
                   f"[{getattr(ev.error, 'code', None)}]")
        ok &= ev.state == 'FAILED'
        s.capture_failure(ev, bundle, note='OBS-03')
    finally:
        s.poisoned = True
        s.close()

    index = failure.read(bundle)
    present = sorted(os.listdir(bundle))
    log.append(f"    the bundle holds: {', '.join(present)}")
    for field in ('artifact', 'kernel', 'profile', 'grid', 'parameters',
                  'seeds', 'trace', 'outcome', 'replay_command'):
        if field not in index:
            log.append(f"    the bundle has no {field!r} (bad)")
            ok = False
    log.append(f"    seeds recorded: {index['seeds']}")
    log.append(f"    {len(index['trace'])} transport request(s) traced")
    log.append(f"    replay command: {index['replay_command']}")

    # Nothing that cannot reproduce is presented as though it should.
    text = json.dumps(index)
    for volatile in ('host_pid', 'timestamp', 'started_at', 'wall_clock'):
        if volatile in text:
            log.append(f"    the bundle records {volatile}, which the "
                       f"contract exempts from replay (bad)")
            ok = False

    # The command the bundle advertises must exist. `tnca-sim` lost its
    # `prepare` and `doctor` subcommands to a bad edit and nothing noticed,
    # because nothing ran them: a CLI is only as tested as its entry points.
    # The entry points are read from pyproject rather than listed here, so
    # this tracks what the package declares instead of what I remembered.
    import contextlib
    import io

    from tnca import cli

    scripts = {}
    text = open(os.path.join(ROOT, 'pyproject.toml')).read()
    block = text.split('[project.scripts]', 1)[1].split('[', 1)[0]
    for line in block.splitlines():
        if '=' in line and ':' in line:
            name, target = line.split('=', 1)
            scripts[name.strip()] = target.strip().strip('"').split(':')[1]
    log.append(f"    declared entry points: {', '.join(sorted(scripts))}")

    for prog, func_name in sorted(scripts.items()):
        main = getattr(cli, func_name, None)
        if main is None:
            log.append(f"    {prog}: pyproject names {func_name}, which "
                       f"tnca.cli does not define (bad)")
            ok = False
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(buf):
                main(['--help'])
        except SystemExit:
            pass
        # argparse prints the subcommands in braces on the usage line.
        usage = buf.getvalue()
        names = []
        if '{' in usage:
            names = usage.split('{', 1)[1].split('}', 1)[0].split(',')
        reachable = []
        for want in [n.strip() for n in names if n.strip()]:
            buf2 = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf2), \
                        contextlib.redirect_stderr(buf2):
                    main([want, '--help'])
            except SystemExit as e:
                if e.code in (0, None):
                    reachable.append(want)
            except Exception:
                pass
        log.append(f"    {prog:<9s} {len(reachable)}/{len(names)} "
                   f"subcommand(s) reachable: {', '.join(reachable)}")
        if len(reachable) != len(names) or not names:
            ok = False

    advertised = index['replay_command'].split()
    if advertised[0] not in scripts:
        log.append(f"    the bundle advertises {advertised[0]!r}, which is "
                   f"not a command this package installs (bad)")
        ok = False

    # The replay: a new session, from the bundle alone.
    saved, got = index['outcome'], None
    _idx, got = failure.replay(bundle,
                               workdir=os.path.join(d, 'replay-dev'))
    same, why = failure.same_failure(saved, got)
    log.append(f"    replayed:         {got['state']} [{got['code']}]")
    log.append(f"    {'same failure' if same else 'DIFFERENT'}: {why}")
    ok &= same

    # And the replay must be a real run, not a recording played back: a
    # bundle whose artifact is replaced with a working kernel must NOT
    # report the saved failure.
    control = os.path.join(d, 'control')
    shutil.copytree(bundle, control)
    good = build(d, 'good.tnca')
    shutil.copy2(good.path, os.path.join(control, index['artifact']))
    with open(os.path.join(control, 'bundle.json')) as fh:
        ci = json.load(fh)
    ci['parameters'] = [
        {'name': 'x_ptr', 'type': 'device_ptr64', 'size': 64},
        {'name': 'y_ptr', 'type': 'device_ptr64', 'size': 64},
        {'name': 'o_ptr', 'type': 'device_ptr64', 'size': 64},
        {'name': 'n', 'type': 'i32', 'value': 16},
        {'name': 'alpha', 'type': 'f32', 'value': 1.0}]
    with open(os.path.join(control, 'bundle.json'), 'w') as fh:
        json.dump(ci, fh)
    _ci, control_out = failure.replay(control,
                                      workdir=os.path.join(d, 'control-dev'))
    same_c, _ = failure.same_failure(saved, control_out)
    log.append(f"    the same bundle with a working kernel: "
               f"{control_out['state']} -- "
               f"{'still reported the saved failure (bad)' if same_c else 'a real run, not a recording'}")
    ok &= not same_c
    return ok, 'saved, replayed, and the same failure both times'


def obs04(log):
    """The shipped examples run, against the package, with no test imports."""
    ex = os.path.join(ROOT, 'examples')
    scripts = sorted(f for f in os.listdir(ex) if f.endswith('.py'))
    log.append(f"    {len(scripts)} example(s): {', '.join(scripts)}")
    ok = len(scripts) >= 3
    if not ok:
        log.append("    the contract asks for at least three")

    # None of them may reach into the test tree: an example that imports a
    # test module is not an example of using the package.
    for f in scripts:
        text = open(os.path.join(ex, f)).read()
        for bad in ('kernels_triton', 'devsim', 'sys.path.insert',
                    'tools.r2', 'import baseline'):
            if bad in text:
                log.append(f"    {f} reaches into the test tree ({bad})")
                ok = False
    log.append("    none imports a test module or edits sys.path")

    if not have_triton():
        log.append("    (running them needs the locked Triton)")
        return False, 'needs the locked Triton'

    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(ROOT, 'python')
    for f in scripts:
        r = subprocess.run([sys.executable, os.path.join(ex, f)],
                           capture_output=True, text=True, timeout=3600,
                           cwd=tempfile.mkdtemp(prefix='r2-obs4-'), env=env)
        last = [l for l in r.stdout.splitlines() if l.strip()]
        log.append(f"    {f:<24s} exit={r.returncode}  "
                   f"{last[-1][:44] if last else r.stderr[-44:]}")
        if r.returncode != 0:
            ok = False
    return ok, 'three examples, run through the package'


TESTS = [('OBS-01', obs01), ('OBS-02', obs02), ('OBS-03', obs03),
         ('OBS-04', obs04)]

NOT_YET = {}


def _case_ids():
    ids = set()
    for name, value in list(globals().items()):
        if name.startswith('_') or not name.isupper():
            continue
        if isinstance(value, dict):
            ids |= {k for k in value if isinstance(k, str) and '-' in k}
        elif isinstance(value, (list, tuple)):
            for item in value:
                if (isinstance(item, (list, tuple)) and item
                        and isinstance(item[0], str) and '-' in item[0]):
                    ids.add(item[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-12 resource reporting, diagnostics and examples")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-800:].replace(
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
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are:")
        for cid in sorted(_case_ids()):
            print(f"      {cid}")
        return 2

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-obs-tests           FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-obs-tests           PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
