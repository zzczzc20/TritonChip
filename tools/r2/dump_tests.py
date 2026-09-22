#!/usr/bin/env python3
"""
DUMP-01 .. DUMP-06 -- per-stage IR and assembly dumps.

These are NOT contract cases. doc/phase2_tasks_and_acceptance.md does not
ask for dumping; this is a feature with its own tests, and they are
labelled so nobody counts them toward R2's 84.

What they check is mostly that the dump is a view and not an influence.
The easy way to build this is for `--dump-dir` to flow into the cache key
and the manifest, and then turning a dump on hands you a different
artifact than the one you were trying to look at.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

from tnca import aot, dump                                # noqa: E402
from tnca.errors import TncaError                         # noqa: E402

SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'}


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def build(d, name='k.tnca', **kw):
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    import kernels_triton as K
    return aot.compile_aot(K.axpy_kernel, signature=SIG,
                           constexprs={'BLOCK': 16}, profile='core-sim',
                           output=os.path.join(d, name), **kw)


def dump01(log):
    """Every stage is produced, named the way a compiler names it."""
    if not have_triton():
        return False, 'needs the locked Triton'
    d = tempfile.mkdtemp(prefix='r2-dump1-')
    res = build(d, options={'dump_dir': os.path.join(d, 'dd')})
    ok = True
    log.append(f"    stages in memory: {', '.join(sorted(res.asm))}")
    for stage in dump.ALWAYS:
        if stage not in res.asm:
            log.append(f"    {stage}: missing from a Triton compile")
            ok = False
    # A Triton source always has a .ttir. It has no optimised one at the
    # default level, because nothing ran -- and a file written anyway
    # would report an optimisation that did not happen.
    if 'ttir' not in res.asm:
        log.append('    ttir: missing from a Triton compile')
        ok = False
    if 'ttir_opt' in res.asm:
        log.append('    ttir_opt: written at the default level, where no '
                   'pass ran')
        ok = False
    else:
        log.append('    ttir_opt: absent at the default level, which is '
                   'where no pass ran')
    d1 = tempfile.mkdtemp(prefix='r2-dump1o-')
    at_o1 = build(d1, options={'dump_dir': os.path.join(d1, 'dd'),
                               'opt_level': 1})
    if 'ttir_opt' not in at_o1.asm:
        log.append('    ttir_opt: missing at opt_level 1, where the '
                   'upstream passes did run')
        ok = False
    else:
        o1_files = sorted(os.listdir(os.path.join(d1, 'dd', 'axpy_kernel')))
        log.append(f"    at opt_level 1: {', '.join(o1_files)}")
        if 'axpy_kernel.opt.ttir' not in o1_files:
            log.append('    the optimised IR was not written to its own '
                       'file')
            ok = False
    where = os.path.join(d, 'dd', 'axpy_kernel')
    on_disk = sorted(os.listdir(where))
    log.append(f"    files written:    {', '.join(on_disk)}")
    for want in ('axpy_kernel.ttir', 'axpy_kernel.tnir', 'axpy_kernel.s',
                 'axpy_kernel.hex', 'axpy_kernel.json'):
        if want not in on_disk:
            log.append(f"    {want} was not written")
            ok = False

    # Each file has to hold its own stage and not another's. A dump where
    # every file is the assembly would pass a "five files exist" check.
    # The whole file, not the first 200 characters: a .ttir opens with a
    # block of #loc declarations and does not reach `tt.func` until line
    # eight, so a small window reported the stage as wrong when it was
    # the look that was too short.
    heads = {}
    for stage, path in res.dumped.items():
        if stage == 'json':
            continue
        heads[stage] = open(path).read()
    checks = [('ttir', 'tt.func'), ('tnir', 'tnca.kernel'),
              ('asm', 's.rdspecial'), ('hex', '0')]
    for stage, marker in checks:
        present = marker in heads.get(stage, '')
        log.append(f"    {stage + ':':<6s} contains {marker!r}: {present}")
        ok &= present

    meta = json.load(open(res.dumped['json']))
    log.append(f"    the json names the kernel, the profile and the key: "
               f"{meta.get('kernel_name')}, {meta.get('profile')}, "
               f"{str(meta.get('cache_key'))[:12]}")
    ok &= meta.get('kernel_name') == 'axpy_kernel'
    ok &= set(meta.get('stages', {})) == set(res.asm)
    return ok, 'four stages, each in its own file, plus the metadata'


def dump02(log):
    """A dump changes neither the cache key nor the artifact."""
    if not have_triton():
        return False, 'needs the locked Triton'
    d = tempfile.mkdtemp(prefix='r2-dump2-')
    plain = build(d, 'plain.tnca')
    dumped = build(d, 'dumped.tnca',
                   options={'dump_dir': os.path.join(d, 'dd')})
    debug = build(d, 'debug.tnca', options={'debug': True})
    semantic = build(d, 'opt.tnca', options={'opt_level': 2})
    ok = True

    log.append(f"    no options   {plain.cache_key[:16]}")
    log.append(f"    --dump-dir   {dumped.cache_key[:16]}")
    log.append(f"    debug=True   {debug.cache_key[:16]}")
    log.append(f"    opt_level=2  {semantic.cache_key[:16]}")
    ok &= plain.cache_key == dumped.cache_key == debug.cache_key
    ok &= plain.cache_key != semantic.cache_key
    if plain.cache_key != dumped.cache_key:
        log.append("    a dump changed the key, so it changed the artifact")

    # And the path must not be IN the artifact. It is a fact about one
    # debugging session, and a distributable file should not carry
    # somebody's home directory.
    man = json.dumps(dumped.manifest, sort_keys=True, default=str)
    leaked = d in man
    log.append(f"    the dump path appears in the manifest: {leaked}")
    ok &= not leaked
    log.append(f"    identity.options records "
               f"{dumped.manifest['identity']['options']} for the dumped "
               f"build and "
               f"{semantic.manifest['identity']['options']} for the "
               f"opt_level one")
    ok &= dumped.manifest['identity']['options'] == {}
    ok &= semantic.manifest['identity']['options'] == {'opt_level': 2}
    return ok, 'the same key, the same artifact, no path inside it'


def dump03(log):
    """Asking for a dump runs the compiler even when the cache would hit."""
    if not have_triton():
        return False, 'needs the locked Triton'
    import cache_tests
    d = tempfile.mkdtemp(prefix='r2-dump3-')
    from tnca.cache import Cache
    counter = cache_tests.CompilerCounter() \
        if hasattr(cache_tests, 'CompilerCounter') else None
    c = Cache(os.path.join(d, 'cache'))

    first = build(d, 'one.tnca', cache=c)
    second = build(d, 'two.tnca', cache=c)
    log.append(f"    a second identical compile: "
               f"{'served from the cache' if second.from_cache else 'compiled again'}")
    ok = second.from_cache

    third = build(d, 'three.tnca', cache=c,
                  options={'dump_dir': os.path.join(d, 'dd')})
    log.append(f"    the same compile asking for a dump: "
               f"{'served from the cache' if third.from_cache else 'compiled again'}")
    ok &= not third.from_cache
    ok &= bool(third.dumped)
    log.append(f"    and it wrote {len(third.dumped)} file(s); a cache hit "
               f"would have written none")

    # The same, through the environment rather than the option.
    os.environ['TNCA_ALWAYS_COMPILE'] = '1'
    try:
        fourth = build(d, 'four.tnca', cache=c)
        log.append(f"    TNCA_ALWAYS_COMPILE=1: "
                   f"{'served from the cache' if fourth.from_cache else 'compiled again'}")
        ok &= not fourth.from_cache
    finally:
        os.environ.pop('TNCA_ALWAYS_COMPILE', None)
    return ok, 'a dump implies a compile; the cache still works without one'


def dump04(log):
    """The environment turns it on, and an explicit directory wins."""
    if not have_triton():
        return False, 'needs the locked Triton'
    d = tempfile.mkdtemp(prefix='r2-dump4-')
    ok = True
    log.append(f"    nothing set        -> {dump.requested({})}")
    ok &= dump.requested({}) is None

    env_dir = os.path.join(d, 'from-env')
    os.environ['TNCA_KERNEL_DUMP'] = '1'
    os.environ['TNCA_DUMP_DIR'] = env_dir
    try:
        log.append(f"    TNCA_KERNEL_DUMP=1 -> {dump.requested({})}")
        ok &= dump.requested({}) == env_dir
        opt_dir = os.path.join(d, 'from-option')
        log.append(f"    and an explicit dump_dir wins -> "
                   f"{dump.requested({'dump_dir': opt_dir})}")
        ok &= dump.requested({'dump_dir': opt_dir}) == opt_dir

        res = build(d, 'env.tnca')
        wrote = bool(res.dumped) and env_dir in next(iter(res.dumped.values()))
        log.append(f"    a compile with only the environment set wrote "
                   f"{len(res.dumped)} file(s) under it: {wrote}")
        ok &= wrote
    finally:
        os.environ.pop('TNCA_KERNEL_DUMP', None)
        os.environ.pop('TNCA_DUMP_DIR', None)

    for off in ('0', 'false', 'no', ''):
        os.environ['TNCA_KERNEL_DUMP'] = off
        if dump.requested({}) is not None:
            log.append(f"    TNCA_KERNEL_DUMP={off!r} turned it ON")
            ok = False
    os.environ.pop('TNCA_KERNEL_DUMP', None)
    log.append("    '0', 'false', 'no' and '' all leave it off")
    return ok, 'the environment, and an option that overrides it'


def dump05(log):
    """A tnca-ir program has no TTIR, and the tool says so."""
    d = tempfile.mkdtemp(prefix='r2-dump5-')

    def program(b, target):
        v = b.const(1)
        b.exit()

    res = aot.compile_aot(program, kernel_name='irdemo', signature={},
                          source_kind='tnca-ir', profile='core-sim',
                          output=os.path.join(d, 'ir.tnca'),
                          options={'dump_dir': os.path.join(d, 'dd')})
    ok = True
    log.append(f"    stages from a tnca-ir source: {sorted(res.asm)}")
    ok &= 'ttir' not in res.asm and 'tnir' in res.asm
    files = sorted(os.listdir(os.path.join(d, 'dd', 'irdemo')))
    log.append(f"    files written: {files}")
    ok &= not any(f.endswith('.ttir') for f in files)
    log.append("    no empty .ttir, which would suggest it had one")

    # And the CLI refuses to print a stage that does not exist, by name.
    from tnca import cli
    import contextlib
    import io
    buf = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.aot_main(['compile', f"{__file__}:_unused", '--emit', 'ttir'])
    except TncaError as e:
        code = e.code
    except SystemExit:
        code = 'SystemExit'
    except Exception as e:
        code = type(e).__name__
    log.append(f"    asking the CLI for a stage it cannot make -> {code}")
    ok &= code is not None
    return ok, 'no TTIR where there was none, and a named refusal'


def dump06(log):
    """The command line: -S, --emit and --dump-dir."""
    if not have_triton():
        return False, 'needs the locked Triton'
    d = tempfile.mkdtemp(prefix='r2-dump6-')
    src = os.path.join(d, 'kern.py')
    with open(src, 'w') as fh:
        fh.write('''import triton
import triton.language as tl


@triton.jit
def axpy(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = tl.load(y_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, x * alpha + y, mask=m)
''')
    base = [sys.executable, '-c',
            'from tnca.cli import aot_main; import sys; '
            'sys.exit(aot_main(sys.argv[1:]))', 'compile', f"{src}:axpy",
            '--signature', 'x_ptr=*fp32', '--signature', 'y_ptr=*fp32',
            '--signature', 'o_ptr=*fp32', '--signature', 'n=i32',
            '--signature', 'alpha=fp32', '--signature', 'BLOCK=constexpr',
            '--constexpr', 'BLOCK=16']
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(ROOT, 'python')
    env.pop('TNCA_KERNEL_DUMP', None)
    ok = True

    for flags, marker, what in (
            (['-S'], 's.rdspecial', '-S prints assembly'),
            (['--emit', 'tnir'], 'tnca.kernel', '--emit tnir prints the IR'),
            (['--emit', 'ttir'], 'tt.func', '--emit ttir prints Triton IR')):
        r = subprocess.run(base + flags + ['-o', os.path.join(d, 'x.tnca')],
                           capture_output=True, text=True, timeout=1800,
                           env=env)
        good = r.returncode == 0 and marker in r.stdout
        log.append(f"    {what:<32s} -> exit {r.returncode}, "
                   f"{len(r.stdout.splitlines())} line(s), "
                   f"{marker!r} present: {marker in r.stdout}")
        ok &= good

    # --emit must not leave files behind: the caller asked to see, not store
    stray = [f for f in os.listdir(d) if f.endswith(('.s', '.tnir', '.ttir'))]
    log.append(f"    --emit wrote no stray stage files: {not stray}")
    ok &= not stray

    dd = os.path.join(d, 'dd')
    r = subprocess.run(base + ['--dump-dir', dd,
                               '-o', os.path.join(d, 'y.tnca')],
                       capture_output=True, text=True, timeout=1800, env=env)
    files = sorted(os.listdir(os.path.join(dd, 'axpy'))) \
        if os.path.isdir(os.path.join(dd, 'axpy')) else []
    log.append(f"    --dump-dir wrote {files}")
    log.append(f"    and said where: "
               f"{'dump' in r.stdout}")
    # The set, not the count. A count passes just as happily when a
    # stage is renamed as when it is written, and says nothing about
    # which one went missing.
    want = {'axpy.ttir', 'axpy.inlined.ttir', 'axpy.tnir', 'axpy.s',
            'axpy.hex', 'axpy.json'}
    missing, extra = want - set(files), set(files) - want
    if missing or extra:
        log.append(f"    missing {sorted(missing)}, unexpected "
                   f"{sorted(extra)}")
    ok &= not missing and not extra and r.returncode == 0 \
        and 'dump' in r.stdout
    return ok, '-S, --emit and --dump-dir all behave'


TESTS = [('DUMP-01', dump01), ('DUMP-02', dump02), ('DUMP-03', dump03),
         ('DUMP-04', dump04), ('DUMP-05', dump05), ('DUMP-06', dump06)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("IR and assembly dumps (a feature's own tests, not contract cases)")
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
            log.append('    ' + traceback.format_exc()[-900:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are: {', '.join(t[0] for t in TESTS)}")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-dump-tests          FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-dump-tests          PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
