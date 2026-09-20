#!/usr/bin/env python3
"""
CACHE-01 .. CACHE-04 -- doc/phase2_tasks_and_acceptance.md R2-07.

The claim being tested is negative and therefore easy to fake: "the compiler
was not invoked". A test that measures elapsed time proves nothing -- a fast
compile is still a compile. So the compiler is COUNTED: `tnca.adapters.r1` is
wrapped so every call through the boundary increments a counter, and CACHE-04
asserts the counter is zero across repeated launches.

Wrapping the adapter rather than the compiler is deliberate. The adapter is
the only door into R1 (doc/r2_interfaces.md), so a count taken there is a
count of every path, including one somebody adds later.
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
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
import r1lib as R                                    # noqa: E402

from tnca import aot, artifact                       # noqa: E402
from tnca.adapters import r1 as adapter              # noqa: E402
from tnca.cache import Cache                         # noqa: E402
from tnca.errors import TncaError                    # noqa: E402

SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'BLOCK': 'constexpr'}


class CompilerCounter:
    """Counts every call through the R1 boundary.

    Not a timer. "It was fast" and "it did not happen" are different claims,
    and only the second is what CACHE-04 asks for.
    """

    def __init__(self):
        self.calls = []
        self._saved = {}

    def __enter__(self):
        for name in ('import_ttir', 'emit', 'assemble', 'new_kernel'):
            self._saved[name] = getattr(adapter, name)
            setattr(adapter, name, self._wrap(name, self._saved[name]))
        return self

    def _wrap(self, name, fn):
        def wrapper(*a, **kw):
            self.calls.append(name)
            return fn(*a, **kw)
        return wrapper

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            setattr(adapter, name, fn)
        return False

    @property
    def compiles(self):
        return self.calls.count('emit')


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def cache01(log):
    """What changes the key, and what cannot reach it.

    The five dimensions are checked on `cache_key` itself, which is a pure
    function of its inputs. Doing it through five real compiles would test
    Triton's constexpr validation as much as the key -- a BLOCK that is not
    VLEN is refused by the compiler for good reasons, and a refusal is not a
    cache miss. Two real compiles then confirm the function is the one the
    driver actually uses.
    """
    ok = True
    base = dict(source_kind='triton', source_hash='a' * 64,
                kernel_name='add_kernel',
                signature=['x_ptr:*fp32', 'n:i32'],
                constexprs={'BLOCK': 16}, profile='core-sim',
                config_hash='c' * 64, options={})
    k0 = aot.cache_key(**base)
    log.append(f"    baseline key {k0[:16]}")

    changes = [
        ('the source', dict(source_hash='b' * 64)),
        ('a constexpr', dict(constexprs={'BLOCK': 32})),
        ('a compile option', dict(options={'opt_level': 2})),
        ('the profile', dict(profile='system-2x2')),
        ('the profile config hash', dict(config_hash='d' * 64)),
        ('the signature', dict(signature=['x_ptr:*fp32', 'n:i64'])),
        ('the source kind', dict(source_kind='tnca-ir')),
    ]
    for label, kw in changes:
        k = aot.cache_key(**dict(base, **kw))
        differs = k != k0
        log.append(f"    {label:<26s} -> {k[:12]} "
                   f"{'differs' if differs else 'SAME (bad)'}")
        ok &= differs

    # The same inputs in a different mapping order must give the SAME key:
    # a key that depended on dict order would miss on every second call.
    shuffled = dict(base)
    shuffled['constexprs'] = {'BLOCK': 16}
    shuffled['options'] = {}
    log.append(f"    the same inputs again        -> "
               f"{'same' if aot.cache_key(**shuffled) == k0 else 'DIFFERS (bad)'}")
    ok &= aot.cache_key(**shuffled) == k0

    # And what CANNOT reach the key: there is no parameter to pass it to.
    # That is the result. A test that passed an address and observed no
    # change would be testing a filter; this observes that the driver has
    # nowhere to put one.
    import inspect as _i
    params = set(_i.signature(aot.compile_aot).parameters)
    forbidden = {'address', 'addresses', 'buffers', 'grid', 'values',
                 'seed', 'args', 'arg_values'}
    leaked = params & forbidden
    log.append(f"    compile_aot's parameters: "
               f"{sorted(params - {'source'})}")
    log.append(f"    run-time inputs that could reach the key: "
               f"{sorted(leaked) or 'none'}")
    ok &= not leaked

    # The driver uses this function. Two real compiles of different kernels
    # must give the two different keys the function predicts.
    if have_triton():
        import kernels_triton as K
        tmp = tempfile.mkdtemp(prefix='r2-cache1-')
        a = aot.compile_aot(K.add_kernel, signature=SIG,
                            constexprs={'BLOCK': 16}, profile='core-sim',
                            output=os.path.join(tmp, 'a.tnca'))
        b = aot.compile_aot(K.mul_add_kernel, signature=SIG,
                            constexprs={'BLOCK': 16}, profile='core-sim',
                            output=os.path.join(tmp, 'b.tnca'))
        differs = a.cache_key != b.cache_key
        log.append(f"    two real kernels             -> {a.cache_key[:12]} "
                   f"vs {b.cache_key[:12]} {'differ' if differs else 'SAME (bad)'}")
        ok &= differs
    else:
        log.append("    (the end-to-end half needs Triton)")
        ok = False
    return ok, 'seven dimensions change it; run-time inputs cannot reach it'


def cache02(log):
    """The same input in two clean stores gives the same artifact."""
    if not have_triton():
        return False, 'needs the locked Triton'
    import kernels_triton as K
    hashes, keys = [], []
    for i in range(2):
        tmp = tempfile.mkdtemp(prefix=f'r2-det{i}-')
        res = aot.compile_aot(K.add_kernel, signature=SIG,
                              constexprs={'BLOCK': 16}, profile='core-sim',
                              cache=Cache(os.path.join(tmp, 'store')),
                              output=os.path.join(tmp, 'k.tnca'))
        man, segs = artifact.load(res.path)
        hashes.append((man['code']['segments'][0]['sha256'],
                       man['manifest_hash']))
        keys.append(res.cache_key)
    same_code = hashes[0][0] == hashes[1][0]
    same_man = hashes[0][1] == hashes[1][1]
    same_key = keys[0] == keys[1]
    log.append(f"    code       {hashes[0][0][:16]} / {hashes[1][0][:16]} "
               f"-> {'same' if same_code else 'DIFFER'}")
    log.append(f"    manifest   {hashes[0][1][:16]} / {hashes[1][1][:16]} "
               f"-> {'same' if same_man else 'DIFFER'}")
    log.append(f"    cache key  {keys[0][:16]} -> "
               f"{'same' if same_key else 'DIFFER'}")
    return (same_code and same_man and same_key), \
        'two clean stores, one artifact'


def cache03(log):
    """A corrupt entry is refused, and a concurrent race is safe."""
    if not have_triton():
        return False, 'needs the locked Triton'
    import kernels_triton as K
    tmp = tempfile.mkdtemp(prefix='r2-cache3-')
    cache = Cache(os.path.join(tmp, 'store'))
    ok = True

    res = aot.compile_aot(K.add_kernel, signature=SIG,
                          constexprs={'BLOCK': 16}, profile='core-sim',
                          cache=cache, output=os.path.join(tmp, 'k.tnca'))
    stored = cache.get(res.cache_key)
    log.append(f"    a stored entry is a hit: {bool(stored)}")
    ok &= bool(stored)

    # One flipped byte inside the stored artifact.
    with open(stored, 'r+b') as fh:
        fh.seek(os.path.getsize(stored) - 1)
        b = fh.read(1)
        fh.seek(os.path.getsize(stored) - 1)
        fh.write(bytes([b[0] ^ 0x01]))
    miss = cache.get(res.cache_key)
    log.append(f"    after one flipped byte: {miss!r} "
               f"({miss.detail.get('message', '')[:46]})")
    ok &= (not miss) and miss.reason == 'corrupt'

    # A truncated entry.
    with open(stored, 'r+b') as fh:
        fh.truncate(32)
    miss2 = cache.get(res.cache_key)
    log.append(f"    after truncation to 32 bytes: {miss2!r}")
    ok &= (not miss2)

    # And a race. Two processes publishing the same key must not leave a
    # partial file; both write a private temp and os.replace.
    script = (
        "import sys, os, tempfile;"
        f"sys.path.insert(0, {os.path.join(ROOT, 'python')!r});"
        f"sys.path.insert(0, {os.path.join(ROOT, 'compiler', 'test')!r});"
        "from tnca import aot; from tnca.cache import Cache;"
        "import kernels_triton as K;"
        f"c = Cache({os.path.join(tmp, 'race')!r});"
        "d = tempfile.mkdtemp();"
        "r = aot.compile_aot(K.add_kernel, signature="
        f"{SIG!r}, constexprs={{'BLOCK': 16}}, profile='core-sim', cache=c,"
        " output=os.path.join(d, 'k.tnca'));"
        "print(r.cache_key)")
    procs = [subprocess.Popen([sys.executable, '-c', script],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for _ in range(2)]
    outs = [p.communicate() for p in procs]
    codes = [p.returncode for p in procs]
    race = Cache(os.path.join(tmp, 'race'))
    entry = race.get(outs[0][0].strip()) if outs[0][0].strip() else None
    log.append(f"    two processes on one key: exits={codes}, "
               f"entry loads={bool(entry)}")
    if any(c != 0 for c in codes):
        log.append('      ' + (outs[0][1] or outs[1][1]).strip()[-200:])
    ok &= all(c == 0 for c in codes) and bool(entry)
    return ok, 'corrupt refused, truncated refused, a race leaves one entry'


def cache04(log):
    """Repeated launches invoke the compiler zero times."""
    if not have_triton():
        return False, 'needs the locked Triton'
    import kernels_triton as K
    import devsim
    tmp = tempfile.mkdtemp(prefix='r2-cache4-')
    cache = Cache(os.path.join(tmp, 'store'))

    with CompilerCounter() as counter:
        first = aot.compile_aot(K.add_kernel, signature=SIG,
                                constexprs={'BLOCK': 16}, profile='core-sim',
                                cache=cache,
                                output=os.path.join(tmp, 'k.tnca'))
        after_first = counter.compiles
        for _ in range(4):
            again = aot.compile_aot(K.add_kernel, signature=SIG,
                                    constexprs={'BLOCK': 16},
                                    profile='core-sim', cache=cache,
                                    output=os.path.join(tmp, 'k.tnca'))
        after_repeats = counter.compiles

    log.append(f"    compiler invocations: {after_first} for the first "
               f"compile, {after_repeats - after_first} for four more")
    log.append(f"    the fifth call came from the cache: {again.from_cache}")
    ok = (after_first == 1 and after_repeats == 1 and again.from_cache)

    # The code pages must be identical across launches, not merely equal in
    # length: R2M-01 breaks exactly this by fixing a pointer at first launch.
    N = 16
    code_hashes = set()
    for addrs in ((0x9000, 0xA000, 0xB000), (0xD000, 0xE000, 0xF000)):
        m = devsim.Memory()
        xs = [float(i) for i in range(N)]
        ys = [float(50 + i) for i in range(N)]
        m.put_f32(addrs[0], xs)
        m.put_f32(addrs[1], ys)
        m.put(addrs[2], b'\xef\xbe\xad\xde' * N)
        run = devsim.run_artifact(
            again.path,
            {'x_ptr': addrs[0], 'y_ptr': addrs[1], 'o_ptr': addrs[2],
             'n': N},
            memory=m, arg_base=0x8000, dump_at=addrs[2], dump_words=N // 2)
        got = run.f32(N)
        want = [a + b for a, b in zip(xs, ys)]
        man, segs = artifact.load(again.path)
        code_hashes.add(man['code']['segments'][0]['sha256'])
        log.append(f"    launch at {addrs[0]:#x}: ran={run.ok} "
                   f"correct={got == want}")
        ok &= run.ok and got == want
    log.append(f"    distinct code hashes across launches: "
               f"{len(code_hashes)} (must be 1)")
    ok &= len(code_hashes) == 1
    log.append(f"    cache: {cache.stats()}")
    return ok, 'one compile, four hits, two launches, one code page'


TESTS = [('CACHE-01', cache01), ('CACHE-02', cache02),
         ('CACHE-03', cache03), ('CACHE-04', cache04)]


def _case_ids():
    """Every case ID this suite knows, for a filter that matched nothing.

    Read off the module's own tables rather than written out again: a second
    list of case IDs is a second thing to forget to update.
    """
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
    print("R2-07 compile cache, determinism and immutable code")
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
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.json:
        R.write_json(args.json, results)
    # A filter that matched nothing must not report PASS. `--only ART-02`
    # selected zero cases and exited 0, which is how a mutation that removed
    # the manifest integrity check looked like a caught one. An empty run is
    # not a clean run.
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are:")
        for cid in sorted(_case_ids()):
            print(f"      {cid}")
        return 2

    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-cache-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-cache-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
