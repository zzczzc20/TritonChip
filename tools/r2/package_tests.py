#!/usr/bin/env python3
"""
PKG-01, PKG-02, PKG-04 and PKG-05 -- doc/phase2_tasks_and_acceptance.md R2-02.

Everything here runs against a wheel installed into a FRESH virtual
environment, with the checkout removed from `sys.path`, from `PYTHONPATH`
and from the working directory. That combination is the whole point: a
package that works only because the repository happens to be importable is
not a package, and every one of these cases has a way of accidentally
passing if the repo leaks in.

So the subprocesses below are given an environment built from scratch, they
run from a temporary directory (one of them with a space in its name, which
PKG-02 asks for), and the first thing each does is assert that no module it
imports came from the checkout.

PKG-03 -- running with no Triton and no compiler -- already passes and
lives in the same wheel build; it is re-checked here as the twin, because
the other four mean nothing if the basic case regressed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                       # noqa: E402

WHEEL_DIR = os.path.join(ROOT, 'sim', 'build', 'dist')

# From the lock rather than written here: a second copy of a version number
# is a second thing to forget to update.
with open(os.path.join(ROOT, 'spec', 'r1_toolchain.lock.json')) as _fh:
    LOCKED_TRITON = json.load(_fh)['triton']['version']


def _run(argv, **kw):
    kw.setdefault('capture_output', True)
    kw.setdefault('text', True)
    kw.setdefault('timeout', 3600)
    return subprocess.run(argv, **kw)


def clean_env(venv, extra=None):
    """An environment with no path back to the checkout."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'TNCA_DEVICE', 'TNCA_MANIFEST_SCHEMA',
                        'PYTHONHOME', 'TRITON_PYTHON')}
    env['PATH'] = os.path.join(venv, 'bin') + os.pathsep + env.get('PATH', '')
    env['VIRTUAL_ENV'] = venv
    env.update(extra or {})
    return env


def build_wheel(log):
    """One wheel, built the way the release does."""
    # The wheel is built by the interpreter running this suite: the
    # Makefile's default PYTHON may be one without pip, and a wheel
    # that was not built is not a wheel that passed.
    r = _run(['make', 'r2-package', f'PYTHON={sys.executable}'],
             cwd=ROOT)
    wheels = sorted(f for f in os.listdir(WHEEL_DIR)
                    if f.endswith('.whl')) if os.path.isdir(WHEEL_DIR) else []
    if not wheels:
        log.append(f"    no wheel built: {r.stdout[-300:]}{r.stderr[-300:]}")
        return None
    path = os.path.join(WHEEL_DIR, wheels[-1])
    log.append(f"    wheel: {wheels[-1]} ({os.path.getsize(path)} bytes)")
    return path


def make_venv(base, wheel, log, name='env'):
    venv = os.path.join(base, name)
    r = _run([sys.executable, '-m', 'venv', venv])
    if r.returncode != 0:
        log.append(f"    venv failed: {r.stderr[-200:]}")
        return None
    pip = os.path.join(venv, 'bin', 'pip')
    # --no-index: nothing may come from the network, and nothing may come
    # from a developer's cache either (PKG-05).
    r = _run([pip, 'install', '-q', '--no-cache-dir', wheel])
    if r.returncode != 0:
        log.append(f"    install failed: {r.stderr[-400:]}")
        return None
    return venv


# The probe each case runs INSIDE the fresh environment. It asserts its own
# isolation before doing anything else, so a pass cannot come from the
# checkout being importable.
PROBE_HEAD = '''
import json, os, sys
bad = [p for p in sys.path if %(root)r in os.path.abspath(p or '.')]
if bad:
    print(json.dumps({'error': 'the checkout is on sys.path', 'paths': bad}))
    raise SystemExit(3)
import tnca
if %(root)r in os.path.abspath(tnca.__file__):
    print(json.dumps({'error': 'tnca imported from the checkout',
                      'file': tnca.__file__}))
    raise SystemExit(3)
'''


def probe(venv, body, cwd, log, env_extra=None, expect=0):
    src = (PROBE_HEAD % {'root': ROOT}) + body
    path = os.path.join(cwd, 'probe.py')
    with open(path, 'w') as fh:
        fh.write(src)
    r = _run([os.path.join(venv, 'bin', 'python'), path], cwd=cwd,
             env=clean_env(venv, env_extra))
    out = r.stdout.strip().splitlines()
    data = {}
    for line in reversed(out):
        try:
            data = json.loads(line)
            break
        except ValueError:
            continue
    if r.returncode != expect:
        log.append(f"        exit={r.returncode} (wanted {expect}) "
                   f"{data or (r.stderr or r.stdout)[-200:]}")
    return r, data


def pkg01(log, ctx):
    """Two fresh environments, compile -> run -> download, repo removed."""
    outcomes = []
    for name in ('envA', 'envB'):
        venv = make_venv(ctx['base'], ctx['wheel'], log, name=name)
        if venv is None:
            return False, 'the wheel would not install'
        work = tempfile.mkdtemp(prefix=f'pkg01-{name}-')
        # compile needs Triton; install it into this environment only.
        pip = os.path.join(venv, 'bin', 'pip')
        t = _run([pip, 'install', '-q', '--no-cache-dir',
                  f"triton=={LOCKED_TRITON}"])
        if t.returncode != 0:
            # Not swallowed: a compile case that cannot compile has not
            # passed, and saying why beats an import error three lines on.
            log.append(f"    {name}: the locked Triton would not install "
                       f"here: {(t.stderr or t.stdout).strip()[-200:]}")
            return False, 'the locked Triton is not installable here'
        body = '''
import struct, tempfile
import triton, triton.language as tl
from tnca import aot, device
from tnca.runtime import Session

@triton.jit
def k(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = tl.load(y_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, x * alpha + y, mask=m)

device.prepare('core-sim', verbose=False)
art = aot.compile_aot(k, signature={'x_ptr':'*fp32','y_ptr':'*fp32',
    'o_ptr':'*fp32','n':'i32','alpha':'fp32','BLOCK':'constexpr'},
    constexprs={'BLOCK':16}, profile='core-sim',
    output=tempfile.mkdtemp() + '/k.tnca')
with Session(profile='core-sim') as s:
    mod = s.load(art.path)
    x, y, o = (s.allocate(64, name=c) for c in 'xyo')
    s.upload(x, b''.join(struct.pack('<f', 2.0) for _ in range(16)))
    s.upload(y, b''.join(struct.pack('<f', 1.0) for _ in range(16)))
    ev = s.submit(mod, {'x_ptr':x,'y_ptr':y,'o_ptr':o,'n':16,'alpha':3.0})
    s.wait(ev)
    got = list(struct.unpack('<16f', s.download(o, 64)))
print(json.dumps({'state': ev.state, 'first': got[0],
                  'all': got == [7.0]*16, 'tnca': tnca.__file__}))
'''
        r, data = probe(venv, body, work, log)
        good = r.returncode == 0 and data.get('all') is True
        log.append(f"    {name}: compile -> run -> download  "
                   f"{data.get('state')} first={data.get('first')} "
                   f"correct={data.get('all')}")
        outcomes.append(good)
        ctx.setdefault('venvs', {})[name] = venv
    return all(outcomes), 'two fresh non-editable installs, end to end'


def _need_venv(log, ctx, name):
    """The environment an earlier case was supposed to leave behind.

    Raising KeyError here made three cases report `KeyError: 'venvs'`
    when the real reason was one line above: the locked Triton has no
    wheel for this platform, so PKG-01 never built an environment. A
    cascade that hides its own cause is worse than a single failure,
    because the reader debugs the wrong thing.
    """
    venv = (ctx.get('venvs') or {}).get(name)
    if venv is None:
        log.append(f"    no {name}: an earlier case did not build it, and "
                   f"its reason is the one to read")
    return venv


def pkg02(log, ctx):
    """No test imports, no checkout sim/build, any cwd, spaces in the path."""
    venv = _need_venv(log, ctx, 'envA')
    if venv is None:
        return False, 'the environment this case needs was never built'
    ok = True
    # A directory with a space, which is the one PKG-02 names.
    spaced = os.path.join(ctx['base'], 'a path with spaces')
    os.makedirs(spaced, exist_ok=True)
    body = '''
import struct, tempfile
from tnca import device
from tnca.runtime import Session
from tnca import aot
# Every module the package loaded, and where from.
import tnca.runtime, tnca.transport, tnca.artifact, tnca.manifest
mods = {n: m.__file__ for n, m in sys.modules.items()
        if n.startswith('tnca')}
leaked = {n: f for n, f in mods.items() if %(root)r in os.path.abspath(f)}
# The device must come from the install, not from the checkout's sim/build.
bench = device.locate('core-sim') or device.prepare('core-sim', verbose=False)
print(json.dumps({'modules': len(mods), 'leaked': leaked,
                  'bench': bench,
                  'bench_in_checkout': %(root)r in os.path.abspath(bench),
                  'test_modules': [n for n in sys.modules
                                   if n in ('kernels_triton', 'devsim',
                                            'baseline', 'r1lib')],
                  'cwd': os.getcwd()}))
''' % {'root': ROOT}
    r, data = probe(venv, body, spaced, log)
    log.append(f"    cwd with a space: {os.path.basename(data.get('cwd',''))!r}")
    log.append(f"    {data.get('modules')} tnca module(s) loaded, "
               f"{len(data.get('leaked') or {})} from the checkout")
    log.append(f"    device bench: {data.get('bench')}")
    log.append(f"    test modules imported: {data.get('test_modules')}")
    ok &= r.returncode == 0
    ok &= not data.get('leaked')
    ok &= data.get('bench_in_checkout') is False
    ok &= data.get('test_modules') == []
    return ok, 'no checkout module, no checkout bench, a cwd with a space'


def pkg04(log, ctx):
    """Missing assets, a read-only output, and CLI == API."""
    venv = _need_venv(log, ctx, 'envA')
    if venv is None:
        return False, 'the environment this case needs was never built'
    work = tempfile.mkdtemp(prefix='pkg04-')
    ok = True

    body = '''
import stat, struct, tempfile
from tnca import device
from tnca.errors import TncaError
out = {}

# (a) a device that is not there
os.environ['TNCA_DEVICE'] = '/nonexistent/none.vvp'
try:
    from tnca.runtime import Session
    Session(profile='core-sim', bench='/nonexistent/none.vvp')
    out['missing_device'] = 'ACCEPTED'
except TncaError as e:
    out['missing_device'] = e.code
except Exception as e:
    out['missing_device'] = 'BARE ' + type(e).__name__
os.environ.pop('TNCA_DEVICE', None)

# (b) an output path that cannot be written
ro = tempfile.mkdtemp()
os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
try:
    from tnca import artifact
    artifact.write(os.path.join(ro, 'x.tnca'), {'a': 1}, {'text': b'x'})
    out['readonly_output'] = 'ACCEPTED'
except TncaError as e:
    out['readonly_output'] = e.code
except Exception as e:
    out['readonly_output'] = 'BARE ' + type(e).__name__
os.chmod(ro, stat.S_IRWXU)

# (c) an artifact that is not one
bad = os.path.join(tempfile.mkdtemp(), 'bad.tnca')
open(bad, 'wb').write(b'not a tnca artifact at all')
try:
    from tnca import artifact as A
    A.load(bad)
    out['bad_artifact'] = 'ACCEPTED'
except TncaError as e:
    out['bad_artifact'] = e.code
except Exception as e:
    out['bad_artifact'] = 'BARE ' + type(e).__name__

# (d) the CLI and the API must agree about the same bad artifact
from tnca import cli
import contextlib, io
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = cli.aot_main(['inspect', bad, '--json'])
except SystemExit as e:
    rc = e.code
out['cli_rc'] = rc
out['cli_said'] = buf.getvalue()[-200:]
print(json.dumps(out))
'''
    r, data = probe(venv, body, work, log)
    for field, label in (('missing_device', 'a device that is not there'),
                         ('readonly_output', 'an unwritable output path'),
                         ('bad_artifact', 'a file that is not an artifact')):
        code = data.get(field)
        log.append(f"    {label:<32s} -> {code}")
        if not code or str(code).startswith('BARE') or code == 'ACCEPTED':
            ok = False
    cli_rc = data.get('cli_rc')
    log.append(f"    the CLI on the same bad artifact -> exit {cli_rc}, "
               f"and the API raised {data.get('bad_artifact')}")
    if cli_rc in (0, None):
        log.append("    the CLI succeeded where the API refused (bad)")
        ok = False
    if data.get('bad_artifact') and str(data['bad_artifact']) not in \
            str(data.get('cli_said', '')):
        log.append(f"    the CLI did not report the same code")
        ok = False
    return ok, 'four failures, stable codes, and the CLI agrees with the API'


def pkg05(log, ctx):
    """Licence, version, schemas, assets; reinstall without a cache."""
    venv = _need_venv(log, ctx, 'envB')
    if venv is None:
        return False, 'the environment this case needs was never built'
    work = tempfile.mkdtemp(prefix='pkg05-')
    ok = True
    body = '''
import tnca, glob
from importlib import metadata
from tnca import _data
d = metadata.distribution('tnca')
files = [str(f) for f in (d.files or [])]
root = os.path.dirname(tnca.__file__)
present = {}
for want in (('spec', 'r2_manifest_v1.schema.json'),
             ('spec', 'r2_abi_v1.json'),
             ('spec', 'r2_container_v1.json'),
             ('spec', 'r2_transport_v1.json'),
             ('isa', 'tnca-isa-v0.1.json'),
             ('tb', 'device', 'tnca_device.v'),
             ('rtl', 'core', 'tnca_core.v')):
    p = _data.path(*want)
    present['/'.join(want)] = os.path.exists(p)
meta = d.metadata
print(json.dumps({
    'version': d.version,
    'license': (meta.get('License-Expression') or meta.get('License') or
                ('LICENSE' if any('LICENSE' in f for f in files) else None)),
    'license_files': sorted(f for f in files if 'LICENSE' in f.upper()),
    'assets': present,
    'profiles': sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(
                           os.path.dirname(_data.path('spec', 'x')),
                           '*profile*'))),
}))
'''
    r, data = probe(venv, body, work, log)
    log.append(f"    version {data.get('version')}, licence "
               f"{data.get('license')}, files "
               f"{data.get('license_files')}")
    ok &= bool(data.get('version'))
    ok &= bool(data.get('license')) or bool(data.get('license_files'))
    missing = [k for k, v in (data.get('assets') or {}).items() if not v]
    log.append(f"    runtime assets present: "
               f"{len(data.get('assets') or {}) - len(missing)}"
               f"/{len(data.get('assets') or {})}"
               + (f"  MISSING {missing}" if missing else ''))
    ok &= not missing

    # Uninstall and reinstall with no cache at all: nothing may be coming
    # from a developer's ~/.cache.
    pip = os.path.join(venv, 'bin', 'pip')
    u = _run([pip, 'uninstall', '-y', '-q', 'tnca'])
    i = _run([pip, 'install', '-q', '--no-cache-dir', '--no-index',
              ctx['wheel']])
    log.append(f"    uninstall exit={u.returncode}, reinstall with "
               f"--no-index --no-cache-dir exit={i.returncode}")
    ok &= u.returncode == 0 and i.returncode == 0
    r2, data2 = probe(venv, 'print(json.dumps({"ok": True}))', work, log)
    log.append(f"    the reinstalled package imports: "
               f"{data2.get('ok') is True}")
    ok &= data2.get('ok') is True
    return ok, 'licence, version, assets, and a reinstall from nothing'


def pkg06(log, ctx):
    """The wheel ships the sources in the tree, byte for byte.

    This case exists because the alternative was found the hard way. A
    setuptools `build/` tree had been committed to the repository, and
    build_py only recopies a source that is NEWER than its copy, so the
    wheel was assembled from a months-old snapshot: the shipped transport
    wrote six mailbox fields while the shipped device read eight, and the
    device sat there for the full ten-minute host timeout with an empty
    log. Every packaging case before this one passed, because none of them
    compared what was in the wheel against what was in the tree.
    """
    import zipfile
    ok = True
    checked = stale = 0
    problems = []
    with zipfile.ZipFile(ctx['wheel']) as z:
        names = z.namelist()
        for name in names:
            # tnca/ and tnca_cc/ come from the tree; data/ is staged from
            # rtl, tb, spec and isa, and is compared too.
            if name.startswith('tnca/data/'):
                src = os.path.join(ROOT, 'python', 'tnca', 'data',
                                   name[len('tnca/data/'):])
            elif name.startswith('tnca/'):
                src = os.path.join(ROOT, 'python', name)
            elif name.startswith('tnca_cc/'):
                src = os.path.join(ROOT, 'compiler', 'python', name)
            elif name in ('tnca_asm.py', 'tnca_isa.py'):
                src = os.path.join(ROOT, 'python', '_staged', name)
            else:
                continue          # dist-info, metadata
            if not os.path.exists(src):
                problems.append(f"{name}: shipped, but not in the tree")
                continue
            checked += 1
            with open(src, 'rb') as fh:
                on_disk = fh.read()
            if z.read(name) != on_disk:
                stale += 1
                problems.append(f"{name}: differs from {os.path.relpath(src, ROOT)}")
    log.append(f"    {checked} shipped file(s) compared against the tree")
    if problems:
        for pr in problems[:10]:
            log.append(f"        {pr}")
        log.append(f"    {len(problems)} file(s) do not match")
        ok = False
    else:
        log.append("    every one identical: the wheel is this tree, not an "
                   "older one")

    # And the two ends of the mailbox protocol must agree about how many
    # fields it has -- the specific mismatch that made this case necessary.
    import re as _re
    with zipfile.ZipFile(ctx['wheel']) as z:
        tr = z.read('tnca/transport.py').decode()
        dv = z.read('tnca/data/tb/device/tnca_device.v').decode()
    m = _re.search(r'fh\.write\(f"([^"]*)\\n"\)', tr)
    wrote = len(m.group(1).split()) if m else 0
    m2 = _re.search(r'\$fscanf\(fh,\s*"([^"]*)"', dv)
    reads = len(m2.group(1).split()) if m2 else 0
    log.append(f"    the shipped host writes {wrote} mailbox field(s); the "
               f"shipped device reads {reads}")
    if wrote != reads or wrote == 0:
        log.append("    the two ends of the protocol disagree (bad)")
        ok = False

    # A gate nobody has seen fail is a gate nobody has tested -- which is
    # how the stale wheel survived every other packaging case. A copy of
    # the wheel gets one shipped file changed the way the stale one was
    # changed (the mailbox line shortened from eight fields to six), and
    # the comparison must notice. Synthesised rather than fetched from
    # git, so it also runs from an exported tree with no .git.
    control = os.path.join(ctx['base'], 'control.whl')
    victim = 'tnca/transport.py'
    with zipfile.ZipFile(ctx['wheel']) as src, \
            zipfile.ZipFile(control, 'w') as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == victim:
                data = data.replace(
                    b'{a0} {a1} {a2} {a3} {a4} {a5}',
                    b'{a0} {a1} {a2} {a3}')
            dst.writestr(item, data)
    with zipfile.ZipFile(control) as z:
        with open(os.path.join(ROOT, 'python', victim), 'rb') as fh:
            caught = z.read(victim) != fh.read()
        ctl = z.read(victim).decode()
        m3 = _re.search(r'fh\.write\(f"([^"]*)\\n"\)', ctl)
        ctl_fields = len(m3.group(1).split()) if m3 else 0
    log.append(f"    a control wheel whose host writes {ctl_fields} field(s) "
               f"instead of {wrote}: "
               f"{'caught' if caught else 'NOT CAUGHT (bad)'}")
    ok &= caught and ctl_fields != reads
    return ok, 'the wheel is built from this tree, and its two ends agree'


TESTS = [('PKG-06', pkg06), ('PKG-01', pkg01), ('PKG-02', pkg02),
         ('PKG-04', pkg04), ('PKG-05', pkg05)]

NOT_YET = {}


def _case_ids():
    return {t[0] for t in TESTS} | set(NOT_YET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-02 installation and packaging")
    log0 = []
    wheel = build_wheel(log0)
    for line in log0:
        print(line, flush=True)
    if wheel is None:
        print("  r2-package-tests       FAIL   no wheel")
        return 1

    base = tempfile.mkdtemp(prefix='r2-pkg-')
    ctx = {'wheel': wheel, 'base': base}
    results = []
    try:
        for cid, fn in TESTS:
            if args.only and cid not in args.only:
                continue
            log = []
            try:
                ok, what = fn(log, ctx)
            except Exception as e:
                import traceback
                ok, what = False, f"{type(e).__name__}: {e}"
                log.append('    ' + traceback.format_exc()[-800:].replace(
                    '\n', '\n    '))
            results.append(dict(id=cid, passed=bool(ok), detail=what))
            print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}",
                  flush=True)
            for line in log:
                print(line, flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are: {', '.join(sorted(_case_ids()))}")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-package-tests       FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-package-tests       PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
