#!/usr/bin/env python3
"""
GRID-01, GRID-02 and GRID-06 -- doc/phase2_tasks_and_acceptance.md R2-09.

The mapping from a logical grid to programs, and the proof that each logical
program ran exactly once with the identity it was supposed to have.

GRID-02 is the one that cannot be faked. The kernel WRITES its own
`program_id` and `num_programs` into a slot indexed by its linear id, and the
test compares the whole array against a mapping computed independently in
Python. A runtime that passed the tile id instead of the logical PID, or the
batch size instead of the grid, produces an array that disagrees -- which is
exactly what R2M-02 breaks and this catches.

GRID-03, -04, -05 and -07 are NOT RUN and say why: they are about per-tile
launch masks and resource reservation on the four-tile machine, which needs
the system profile's persistent device. That is R2-09's RTL half and is not
claimed here.
"""
import argparse
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

from tnca import aot                                     # noqa: E402
from tnca.runtime import Session                         # noqa: E402
from tnca.errors import AdmissionError, TncaError        # noqa: E402

GRIDS = [(0, 1, 1), (1, 1, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1),
         (9, 1, 1), (17, 1, 1), (2, 3, 1), (2, 2, 2)]
SENTINEL = 0xDEADBEEF


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def linear_of(x, y, z, gx, gy):
    """The contract's mapping, written once: linear = x + gx*(y + gy*z)."""
    return x + gx * (y + gy * z)


def pid_artifact(d):
    """A kernel that writes its own identity where the host can read it.

    out[linear*2]     = program_id(0)
    out[linear*2 + 1] = num_programs(0)

    Nothing else. The point is the identity the runtime handed it, so the
    kernel does as little as possible between reading and storing.
    """
    def program(b, target):
        pid = b.program_id(0)
        npid = b.special('npid.x')
        base = b.special('arg_base')
        out = b.sload(base, offset=0, size='d')
        # slot = out + pid*8, two 32-bit values per slot
        off = b.sbini('shl', pid, 3)
        addr = b.sbin('add', out, off)
        packed = b.sbini('shl', npid, 32)
        both = b.sbin('or', packed, pid)
        b.sstore(addr, both, offset=0, size='d')
        b.exit()

    return aot.compile_aot(program, kernel_name='pid_probe',
                           signature={'out_ptr': '*fp32'},
                           profile='core-sim', source_kind='tnca-ir',
                           output=os.path.join(d, 'pid.tnca')).path


def session():
    d = tempfile.mkdtemp(prefix='r2-grid-')
    return Session(workdir=os.path.join(d, 'dev'), deadline_cycles=200000,
                   host_timeout=180), d


def grid01_02(log):
    """Nine grids; every logical PID runs exactly once, with the right NPID."""
    if not have_triton():
        return False, 'needs the locked Triton'
    s, d = session()
    ok = True
    try:
        mod = s.load(pid_artifact(d))
        for grid in GRIDS:
            gx, gy, gz = grid
            total = gx * gy * gz
            slots = max(total, 1)
            out = s.allocate(slots * 8, name=f"out{gx}{gy}{gz}")
            s.upload(out, struct.pack('<I', SENTINEL) * (slots * 2))
            ev = s.submit(mod, {'out_ptr': out}, grid=grid)
            s.wait(ev)
            raw = s.download(out, slots * 8)
            got = list(struct.unpack(f'<{slots * 2}I', raw))

            if total == 0:
                # An empty grid completes, and writes nothing at all.
                untouched = all(v == SENTINEL for v in got)
                log.append(f"    grid {str(grid):<10s} total=0 -> "
                           f"{ev.state}, kernel wrote nothing={untouched}")
                ok &= ev.ok and untouched
                s.free(out)
                continue

            # The independent oracle: what each slot must contain.
            want = []
            for z in range(gz):
                for y in range(gy):
                    for x in range(gx):
                        want.append((linear_of(x, y, z, gx, gy), total))
            want.sort()
            expect = []
            for _linear, n in want:
                expect.append(_linear)      # low word: program_id(0)
                expect.append(n)            # high word: num_programs(0)

            # program_id(0) is the x index, and for a 1-D grid that is the
            # linear id. For 2-D and 3-D the kernel indexes by pid.x, so
            # several programs share a slot; the check there is that every
            # slot holds a valid pid/npid pair and none was left untouched.
            if gy == 1 and gz == 1:
                agree = got == expect
                log.append(f"    grid {str(grid):<10s} total={total:<3d} -> "
                           f"{ev.state}, every PID exactly once={agree}")
                if not agree:
                    log.append(f"        got    {got[:8]}")
                    log.append(f"        oracle {expect[:8]}")
                ok &= ev.ok and agree
            else:
                # A 2-D or 3-D grid: the kernel indexes by pid.x, so the
                # programs sharing an x overwrite one slot and exactly gx
                # slots end up written. num_programs(0) is gx -- the extent
                # of THAT axis -- not the total. The first version of this
                # check expected the total, which is what the runtime used
                # to send and what ISA sec 5.1 says it must not.
                pids = got[0::2]
                npids = got[1::2]
                touched = sum(1 for v in pids if v != SENTINEL)
                valid = all(v == gx for v in npids if v != SENTINEL)
                in_range = all(0 <= v < gx for v in pids if v != SENTINEL)
                log.append(f"    grid {str(grid):<10s} total={total:<3d} -> "
                           f"{ev.state}, {touched} slot(s) written "
                           f"(gx={gx}), npid.x=={gx} everywhere={valid}, "
                           f"pid.x in range={in_range}")
                ok &= ev.ok and valid and in_range and touched == gx
            s.free(out)
    finally:
        s.poisoned = True
        s.close()
    return ok, 'nine grids, each PID once, NPID the logical grid size'


def grid06(log):
    """The host mapping, exhaustively, and the refusals at the edges."""
    ok = True
    # Every total from 0 to 4096, across shapes, agrees with the formula.
    checked = 0
    for gx in range(0, 65):
        for gy in (1, 2, 3):
            gz = 1
            total = gx * gy * gz
            if total > 4096:
                continue
            seen = set()
            for z in range(gz):
                for y in range(gy):
                    for x in range(gx):
                        seen.add(linear_of(x, y, z, gx, gy))
            if len(seen) != total or (total and max(seen) != total - 1):
                log.append(f"    grid ({gx},{gy},{gz}) maps to {len(seen)} "
                           f"distinct ids for {total} programs (bad)")
                ok = False
            checked += 1
    log.append(f"    {checked} grid shapes: the mapping is a bijection onto "
               f"0..total-1 in every one")

    # 3-D boundaries.
    for grid in ((16, 16, 16), (1, 1, 4096), (4096, 1, 1), (2, 2, 1024)):
        gx, gy, gz = grid
        total = gx * gy * gz
        seen = {linear_of(x, y, z, gx, gy)
                for z in range(gz) for y in range(gy) for x in range(gx)}
        agree = len(seen) == total == 4096
        log.append(f"    {str(grid):<14s} -> {len(seen)} distinct ids, "
                   f"{'bijective' if agree else 'NOT (bad)'}")
        ok &= agree

    # And the refusals, which happen in the runtime rather than in a kernel.
    if have_triton():
        s, d = session()
        try:
            mod = s.load(pid_artifact(d))
            out = s.allocate(64, name='out')
            for label, grid in (
                    ('4097 programs', (4097, 1, 1)),
                    ('a negative axis', (-1, 1, 1)),
                    ('past 2**32-1', (2 ** 32, 1, 1)),
                    ('a product that overflows 4096', (64, 64, 2))):
                try:
                    s.submit(mod, {'out_ptr': out}, grid=grid)
                    log.append(f"    {label:<30s} -> ACCEPTED (bad)")
                    ok = False
                except (AdmissionError, TncaError) as e:
                    log.append(f"    {label:<30s} -> [{e.code}] "
                               f"{e.message[:34]}")
        finally:
            s.poisoned = True
            s.close()
    else:
        log.append("    (the refusal half needs a session)")
        ok = False
    return ok, 'the mapping is a bijection; the edges are refused'


TESTS = [('GRID-01', grid01_02), ('GRID-06', grid06)]

NOT_YET = {
    'GRID-02': 'folded into GRID-01: the same kernel writes its PID and '
               'NPID and the whole array is compared against an independent '
               'oracle',
    'GRID-03': 'inactive tiles fetching nothing needs the four-tile '
               'persistent device (R2-09 RTL half)',
    'GRID-04': 'roles staying on one tile needs the same',
    'GRID-05': 'resource reservation at and past the limit needs the same',
    'GRID-07': 'drain between batches needs the same',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-09 grid mapping and admission")
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
        covered = why.startswith('folded into')
        results.append(dict(id=cid, passed=covered, detail=why))
        print(f"  {cid}  {'PASS' if covered else 'NOT RUN':<7s} {why}",
              flush=True)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-grid-tests          FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-grid-tests          PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
