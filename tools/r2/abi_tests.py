#!/usr/bin/env python3
"""
ABI-01 -- doc/phase2_tasks_and_acceptance.md R2-06.

    "六种 ABI 类型按独立 byte-level oracle 校验 offsets/大小/端序；整数边界、
     f32 ±0/有限值/允许的 NaN/Inf 均保持合同"

The oracle below builds the bytes by hand -- shifting and masking an integer,
one byte at a time -- and never calls `tnca.abi`. That is the whole point: an
encoder checked against itself proves only that it is deterministic. Where the
two disagree, the failure says which byte and what each produced.

The f32 cases are the ones worth being careful about. A decimal round trip
loses -0.0 (it comes back as +0.0) and flattens a NaN payload, so the oracle
takes the bit pattern from `struct` and the comparison is on bits, never on
value: `float('nan') != float('nan')` would make a NaN test pass for the
wrong reason, and `-0.0 == 0.0` would make the sign test pass for no reason
at all.
"""
import argparse
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                   # noqa: E402

from tnca import abi                                # noqa: E402
from tnca.errors import ArgumentError, SignatureError  # noqa: E402


# --------------------------------------------------------------------------
# The oracle. No import from tnca.abi, by design.
# --------------------------------------------------------------------------
SLOT = 8


def oracle_le(value, nbytes):
    """`value` as `nbytes` little-endian bytes, built one byte at a time."""
    out = bytearray()
    v = value & ((1 << (8 * nbytes)) - 1)
    for i in range(nbytes):
        out.append((v >> (8 * i)) & 0xFF)
    return bytes(out)


def oracle_slot(type_name, value):
    """One slot, from the written contract rather than from the encoder."""
    if type_name in ('i32', 'u32'):
        return oracle_le(value, 4) + b'\x00' * 4
    if type_name in ('i64', 'u64', 'device_ptr64'):
        return oracle_le(value, 8)
    if type_name == 'f32':
        # The bit pattern, from struct, which is not the module under test.
        bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        return oracle_le(bits, 4) + b'\x00' * 4
    raise AssertionError(f"the oracle has no case for {type_name!r}")


def oracle_block(params, values):
    out = bytearray()
    for name, type_name in params:
        out += oracle_slot(type_name, values[name])
    return bytes(out)


# --------------------------------------------------------------------------
CASES = [
    # (type, value, why this value)
    ('i32', 0, 'zero'),
    ('i32', 1, 'one'),
    ('i32', -1, 'all ones in the low half, and the high half still zero'),
    ('i32', 2147483647, 'the largest i32'),
    ('i32', -2147483648, 'the smallest i32'),
    ('u32', 0, 'zero'),
    ('u32', 4294967295, 'the largest u32'),
    ('i64', -9223372036854775808, 'the smallest i64'),
    ('i64', 9223372036854775807, 'the largest i64'),
    ('u64', 18446744073709551615, 'the largest u64'),
    ('device_ptr64', 0x1000, 'an ordinary address'),
    ('device_ptr64', 0xFFFFFFFFFFFFFFF8, 'the top of the address space'),
    ('f32', 0.0, '+0'),
    ('f32', -0.0, '-0 -- the case a decimal round trip loses'),
    ('f32', 1.0, 'one'),
    ('f32', -2.5, 'an ordinary negative'),
    ('f32', 1.1754943508222875e-38, 'the smallest normal'),
    ('f32', 1.401298464324817e-45, 'the smallest subnormal'),
    ('f32', 3.4028234663852886e+38, 'the largest finite'),
    ('f32', float('inf'), 'positive infinity'),
    ('f32', float('-inf'), 'negative infinity'),
    ('f32', float('nan'), 'a quiet NaN'),
]


def abi01(log):
    """Every type, against the oracle, byte by byte."""
    ok = True

    # Offsets and sizes first: they decide where everything else lands.
    for n in (0, 1, 2, 7):
        want = n * SLOT
        got = abi.block_size(n)
        if got != want:
            log.append(f"    block_size({n}) = {got}, oracle {want}")
            ok = False
    for i in (0, 1, 5, 31):
        if abi.offset_of(i) != i * SLOT:
            log.append(f"    offset_of({i}) = {abi.offset_of(i)}, "
                       f"oracle {i * SLOT}")
            ok = False
    log.append(f"    offsets and sizes: slot={SLOT}, offset=index*{SLOT}, "
               f"empty block = 0 bytes")

    for type_name, value, why in CASES:
        want = oracle_slot(type_name, value)
        got = abi.encode_slot(type_name, value)
        if got != want:
            log.append(f"    {type_name} {value!r}: encoder {got.hex()} "
                       f"oracle {want.hex()}  ({why})")
            ok = False
            continue
        # And back. For a float, compare BITS: nan != nan and -0.0 == 0.0
        # would both pass for the wrong reason.
        back = abi.decode_slot(type_name, got)
        if type_name == 'f32':
            same = (struct.pack('<f', back) == struct.pack('<f', float(value)))
        else:
            same = (back == value)
        if not same:
            log.append(f"    {type_name} {value!r} decoded as {back!r} ({why})")
            ok = False

    log.append(f"    {len(CASES)} value(s) across "
               f"{len(set(t for t, _v, _w in CASES))} type(s) match the oracle "
               f"byte for byte, and survive a round trip")

    # The whole block, in signature order.
    params = [('p', 'device_ptr64'), ('n', 'i32'), ('alpha', 'f32'),
              ('big', 'u64')]
    values = {'p': 0x2000, 'n': -17, 'alpha': -0.0, 'big': 2 ** 63}
    want = oracle_block(params, values)
    got = abi.encode_block(params, values)
    if got != want:
        log.append(f"    block: {got.hex()} vs oracle {want.hex()}")
        ok = False
    else:
        log.append(f"    a 4-parameter block is {len(got)} bytes and matches "
                   f"the oracle: {got.hex()}")

    # The high half is checked on the way IN too, or a producer that left
    # residue there would never be caught.
    dirty = abi.encode_slot('i32', 5)[:4] + b'\xde\xad\xbe\xef'
    try:
        abi.decode_slot('i32', dirty)
        log.append("    a non-zero high half decoded without complaint (bad)")
        ok = False
    except ArgumentError as e:
        log.append(f"    a non-zero high half -> {e.message}")

    # Out of range is an error. This is the case where a truncation would be
    # invisible: 2**31 as i32 wraps to the smallest negative number.
    for type_name, value in (('i32', 2 ** 31), ('i32', -2 ** 31 - 1),
                             ('u32', -1), ('u32', 2 ** 32),
                             ('i64', 2 ** 63), ('u64', 2 ** 64)):
        try:
            abi.check_value('v', type_name, value)
            log.append(f"    {type_name} accepted {value} (bad: it would wrap "
                       f"to {oracle_le(value, 4 if '32' in type_name else 8)!r})")
            ok = False
        except ArgumentError:
            pass
    log.append("    every out-of-range integer refused rather than truncated")

    # A float for an integer parameter, and a bool for anything.
    for type_name, value in (('i32', 1.5), ('i32', True), ('u64', 'x'),
                             ('f32', True), ('f32', None)):
        try:
            abi.check_value('v', type_name, value)
            log.append(f"    {type_name} accepted {value!r} (bad)")
            ok = False
        except ArgumentError:
            pass
    log.append("    a float, a bool and a string refused where they do not "
               "belong")

    # The rejected scalar types are refused BY NAME, with the reason.
    for type_name in ('f16', 'bf16', 'f64', 'bool', 'i8'):
        try:
            abi.check_type('v', type_name)
            log.append(f"    {type_name} was accepted as an ABI type (bad)")
            ok = False
        except SignatureError as e:
            if not e.detail.get('reason'):
                log.append(f"    {type_name} refused with no reason given")
                ok = False
    log.append("    f16/bf16/f64/bool/i8 refused by name, each with why")

    return ok, 'six types against an independent byte-level oracle'


def abi_version(log):
    """Version acceptance: same major, minor no greater than ours."""
    ok = True
    major, minor = abi.ABI_VERSION
    for v, want in (((major, minor), True),
                    ((major, 0), True),
                    ((major, minor + 1), False),
                    ((major + 1, 0), False),
                    ((major - 1, 0), False)):
        got = abi.supports(v)
        if got != want:
            log.append(f"    supports({v}) = {got}, contract says {want}")
            ok = False
    log.append(f"    ABI {major}.{minor}: accepts {major}.0..{major}.{minor}, "
               f"refuses {major}.{minor + 1} and {major + 1}.0")
    return ok, 'a higher minor is refused, not only a higher major'


def _runtime_fixture():
    """A session with the AXPY artifact loaded, for the cases that execute."""
    import tempfile
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    from tnca import aot
    from tnca.runtime import Session
    import kernels_triton as K
    sig = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
           'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'}
    d = tempfile.mkdtemp(prefix='r2-abi-')
    res = aot.compile_aot(K.axpy_kernel, signature=sig,
                          constexprs={'BLOCK': 16}, profile='core-sim',
                          output=os.path.join(d, 'axpy.tnca'))
    s = Session(workdir=os.path.join(d, 'dev'), deadline_cycles=200000,
                host_timeout=120)
    return s, s.load(res.path), res


def _have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def _f32s(values):
    return b''.join(struct.pack('<f', v) for v in values)


def abi02(log):
    """One artifact, three sets of addresses and values, one compile."""
    if not _have_triton():
        return False, 'needs the locked Triton'
    from tnca import artifact
    s, mod, res = _runtime_fixture()
    ok = True
    try:
        before = artifact.load(res.path)[0]['code']['segments'][0]['sha256']
        N = 16
        seen = []
        for i, (alpha, n) in enumerate(((2.0, 16), (-3.5, 8), (0.5, 1))):
            x, y, o = (s.allocate(N * 4, name=f"{c}{i}") for c in 'xyo')
            xs = [float(j + i) for j in range(N)]
            ys = [float(100 * i + j) for j in range(N)]
            s.upload(x, _f32s(xs))
            s.upload(y, _f32s(ys))
            s.upload(o, b'\xef\xbe\xad\xde' * N)
            ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                'n': n, 'alpha': alpha})
            s.wait(ev)
            got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
            want = [xs[j] * alpha + ys[j] if j < n else None
                    for j in range(N)]
            head_ok = all(got[j] == want[j] for j in range(n))
            # Past `n` the mask must have left the sentinel alone.
            tail_ok = all(struct.pack('<f', got[j]) == b'\xef\xbe\xad\xde'
                          for j in range(n, N))
            seen.append((alpha, n, x.addr, head_ok, tail_ok))
            log.append(f"    alpha={alpha:<5} n={n:<3} at {x.addr:#07x}: "
                       f"first {n} correct={head_ok}, tail untouched="
                       f"{tail_ok}")
            ok &= head_ok and tail_ok
        after = artifact.load(res.path)[0]['code']['segments'][0]['sha256']
        log.append(f"    the code hash across all three: "
                   f"{'unchanged' if before == after else 'CHANGED (bad)'}")
        log.append(f"    distinct buffer addresses used: "
                   f"{len({t[2] for t in seen})}")
        ok &= before == after and len({t[2] for t in seen}) == 3
    finally:
        s.poisoned = True
        s.close()
    return ok, 'three address sets, three scalars, one unchanged code page'


def abi03(log):
    """Every way an argument can be wrong, refused before anything runs."""
    if not _have_triton():
        return False, 'needs the locked Triton'
    from tnca.errors import ArgumentError, BufferError, TncaError
    s, mod, _res = _runtime_fixture()
    ok = True
    try:
        N = 16
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        small = s.allocate(8, name='small')
        good = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N, 'alpha': 1.0}

        def refused(label, args):
            nonlocal ok
            before = len(s._events)
            try:
                s.submit(mod, args)
                log.append(f"    {label:<34s} -> ACCEPTED (bad)")
                ok = False
            except TncaError as e:
                started = len(s._events) > before
                log.append(f"    {label:<34s} -> [{e.code}]"
                           + ('  AND QUEUED A TASK (bad)' if started else ''))
                ok &= not started

        refused('a missing argument',
                {k: v for k, v in good.items() if k != 'n'})
        refused('an extra argument', dict(good, extra=1))
        refused('an integer past i32', dict(good, n=2 ** 31))
        refused('a float where i32 is declared', dict(good, n=1.5))
        refused('a host integer as a pointer', dict(good, x_ptr=0x1000))
        refused('a byte offset past the buffer',
                dict(good, x_ptr=(x, N * 4 + 8)))
        refused('a negative byte offset', dict(good, x_ptr=(x, -8)))

        # A freed handle, and one from another session.
        s.free(small)
        refused('a freed buffer', dict(good, x_ptr=small))
        other = __import__('tnca.runtime', fromlist=['Session']).Session(
            workdir=None, deadline_cycles=200000, host_timeout=120)
        try:
            foreign = other.allocate(N * 4)
            refused("another session's buffer", dict(good, x_ptr=foreign))
        finally:
            other.poisoned = True
            other.close()
    finally:
        s.poisoned = True
        s.close()
    return ok, 'nine bad arguments, none of which started anything'


def abi04(log):
    """The block is a snapshot: mutating the caller's dict changes nothing."""
    if not _have_triton():
        return False, 'needs the locked Triton'
    s, mod, _res = _runtime_fixture()
    ok = True
    try:
        N = 16
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        s.upload(x, _f32s([2.0] * N))
        s.upload(y, _f32s([0.0] * N))

        # The same dict object, submitted and then changed. The task must
        # see what it was submitted with.
        args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N, 'alpha': 3.0}
        s._stop.set()
        s._worker.join(timeout=10)
        ev = s.submit(mod, args)
        args['alpha'] = 99.0          # after submit, before it runs
        args['n'] = 1
        s._start_worker()
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        want = [2.0 * 3.0] * N
        log.append(f"    submitted alpha=3.0, then set the dict to 99.0 "
                   f"before it ran")
        log.append(f"    the task computed with alpha="
                   f"{got[0] / 2.0:.1f}: {'the snapshot' if got == want else 'THE MUTATION (bad)'}")
        ok &= got == want
    finally:
        s.poisoned = True
        s.close()
    return ok, 'a submitted task sees the snapshot, not the caller\'s dict'


TESTS = [('ABI-01', abi01), ('ABI-VERSION', abi_version),
         ('ABI-02', abi02), ('ABI-03', abi03), ('ABI-04', abi04)]


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

    print("R2-06 argument ABI v1")
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
            log.append('    ' + traceback.format_exc()[-600:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}")
        for line in log:
            print(line)
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
        print(f"  r2-abi-tests           FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-abi-tests           PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
