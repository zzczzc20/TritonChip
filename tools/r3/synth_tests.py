#!/usr/bin/env python3
"""
SYN-01 .. SYN-05 -- doc/phase3_tasks_and_acceptance.md R3-10.

flow/synth/mem_synth.py runs the structural flow; the judging is here.

The one thing worth saying up front is what is NOT claimed. There is no
PDK and no FPGA RAM primitive, so every memory here is "memory-inferred"
and nothing more. SYN-04 permits exactly that and forbids the rest, and
SYN-05 requires Fmax, power and physical area to read NOT MEASURED rather
than be converted from a gate depth. SYN-04 is therefore a test that the
report does not overclaim, which is an odd-looking test until you notice
how easy the overclaim would be.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

FLOW = os.path.join(ROOT, 'flow', 'synth', 'mem_synth.py')

# What the contract says these files are. RF-01 already checks the RTL
# adds up to this; here it is checked against what synthesis actually
# built, which is a different claim.
DECLARED = {'tnca_vrf_banked': 8 * 1024 * 8,      # 8 KiB of VRF
            'tnca_acc_rf': 16 * 1024 * 8}          # 16 KiB of ARF


def synth(tmp):
    out = os.path.join(tmp, 'mem_synth.json')
    r = subprocess.run([sys.executable, FLOW, '--json', out],
                       capture_output=True, text=True, timeout=3600,
                       cwd=ROOT)
    if not os.path.exists(out):
        return None, r
    return json.load(open(out)), r


def syn01(log, ctx):
    """The data bits survive as memory objects, and add up."""
    rep = ctx['report']
    ok = True
    for top, want_bits in DECLARED.items():
        t = rep['tops'].get(top)
        if not t or not t['ok']:
            log.append(f"    {top}: did not synthesise")
            ok = False
            continue
        got = t['memory_bits']
        frac = got / want_bits if want_bits else 0
        log.append(f"    {top:<22s} {got} bit(s) in "
                   f"{len(t['data_memories'])} memory object(s) "
                   f"= {frac * 100:.0f}% of the declared "
                   f"{want_bits // 8192} KiB")
        if frac < 0.90:
            log.append(f"        under the 90% SYN-01 requires")
            ok = False
        if got != want_bits:
            log.append(f"        and the capacity is not exactly the "
                       f"declared {want_bits}")
            ok = False

    # The bits that are NOT data memory are listed rather than absorbed.
    oc = rep['tops'].get('tnca_operand_collector')
    if oc and oc['ok']:
        log.append(f"    collector: {oc['ff_bits']} flop bit(s) + "
                   f"{oc['small_bits']} bit(s) in "
                   f"{len(oc['small_arrays'])} control array(s), "
                   f"accounted separately and not added to any capacity")
        ok &= oc['memory_bits'] == 0
    return ok, 'the declared capacity survives as memory, exactly'


def syn02(log, ctx):
    """Real ports, a synchronous read, and no data-array reset."""
    rep = ctx['report']
    ok = True
    t = rep['tops']['tnca_vrf_banked']
    for m in t['data_memories']:
        good = (m['rd_ports'] == 1 and m['wr_ports'] == 1 and m['rd_sync'])
        log.append(f"    {m['name']}: {m['rd_ports']}R/{m['wr_ports']}W, "
                   f"{'synchronous' if m['rd_sync'] else 'ASYNCHRONOUS'} "
                   f"read, {'transparent' if m['transparent'] else 'read-first'}"
                   f"  {'ok' if good else 'NOT the declared port count'}")
        ok &= good
    log.append(f"    {len(t['data_memories'])} bank(s), each one read port "
               f"and one write port, which is what the arbitration in "
               f"tnca_vrf_banked exists to ration")

    # The old file, for contrast. This is what "不存在异步read替代" is
    # about: it is not a hypothetical, it is the module this one replaces.
    old = rep['tops']['tnca_vector_rf']
    for m in old['data_memories']:
        log.append(f"    for contrast, tnca_vector_rf: {m['rd_ports']}R/"
                   f"{m['wr_ports']}W, "
                   f"{'synchronous' if m['rd_sync'] else 'ASYNCHRONOUS'} "
                   f"read -- the structure this replaces")
        # The contrast has to be real, or the new module proves nothing.
        if m['rd_sync'] and m['rd_ports'] <= 1:
            log.append("        the old file looks the same as the new "
                       "one, so nothing was changed")
            ok = False

    # No reset of the data array: a memory whose contents clear on reset is
    # a flop array wearing a memory's name, and it is the single biggest
    # way to make this report look better than the design.
    import re as _re
    src = open(os.path.join(ROOT, 'rtl', 'core', 'tnca_rf_bank.v')).read()
    # Every clocked block, and whether one that assigns the array also has
    # a reset branch. The first version of this searched 400 characters
    # after the first `if (!rst_n)` and found `s1_data <= mem[raddr]` --
    # a READ of the array, reported as a reset OF it.
    blocks = _re.split(r'always @\(', src)[1:]
    writes_array = [b for b in blocks if _re.search(r'\bmem\[[^\]]*\]'
                                                    r'(\[[^\]]*\])?\s*<=', b)]
    resetting = [b for b in writes_array if 'rst_n' in b]
    log.append(f"    {len(blocks)} clocked block(s); {len(writes_array)} "
               f"assign(s) the data array, and {len(resetting)} of those "
               f"mention reset")
    log.append(f"    the data array is not cleared on reset: "
               f"{not resetting}")
    ok &= not resetting and len(writes_array) == 1
    return ok, 'one read port, one write port, a clocked read, no array reset'


def syn03(log, ctx):
    """No latches, no loops, nothing unexplained, every changed module in."""
    rep = ctx['report']
    ok = True
    changed = ['tnca_rf_bank', 'tnca_vrf_banked', 'tnca_operand_collector']
    for top in changed:
        t = rep['tops'].get(top)
        if not t or not t['ok']:
            log.append(f"    {top}: NOT INCLUDED ({t and t.get('error')})")
            ok = False
            continue
        problems = t.get('problems') or []
        real = [p for p in problems if 'found 0 problems' not in p]
        warns = t.get('warnings') or []
        expl = t.get('explained_warnings') or []
        log.append(f"    {top:<24s} {t['latches']} latch(es), "
                   f"{len(real)} unresolved finding(s), "
                   f"{len(warns)} unexplained warning(s), "
                   f"{len(expl)} explained")
        for w in warns[:3]:
            log.append(f"        UNEXPLAINED: {w[:70]}")
        for w in expl[:2]:
            log.append(f"        explained: {w[9:79]}")
        if warns:
            ok = False
        if t['latches']:
            ok = False
        if real:
            for p in real[:3]:
                log.append(f"        {p[:74]}")
            ok = False
    log.append(f"    all {len(changed)} module(s) changed by R3-03 and "
               f"R3-04 are in the run; none is recorded as zero gates "
               f"because it was missing")
    return ok, 'no latches, no unresolved findings, nothing left out'


def syn04(log, ctx):
    """Memory-inferred, and the report says nothing stronger."""
    rep = ctx['report']
    ok = True
    log.append(f"    mapping: {rep['mapping']}")
    ok &= 'inferred' in rep['mapping'] and 'no PDK' in rep['mapping']

    # The forbidden claims. SYN-04 allows "memory-inferred" and forbids
    # saying an SRAM area or an Fmax has been met; SYN-05 forbids deriving
    # a frequency from a gate depth. So the report is searched for anyone
    # having said so.
    text = json.dumps(rep).lower()
    for claim in ('sram area', 'fmax met', 'ghz', 'mhz', 'timing closed',
                  'meets timing'):
        if claim in text:
            log.append(f"    the report claims {claim!r}, which it cannot "
                       f"without a library")
            ok = False
    log.append(f"    it claims no SRAM area, no frequency and no timing "
               f"closure: {ok}")

    # And the wrapper's contract with whatever RAM eventually implements
    # it: the read latency, the mask granularity and the RDW rule have to
    # be stated, because those are what a macro either provides or does not.
    bank = open(os.path.join(ROOT, 'rtl', 'core', 'tnca_rf_bank.v')).read()
    for term, what in (('READ_LATENCY', 'the read latency'),
                       ('RDW_BYPASS', 'the read-during-write rule'),
                       ('wmask', 'the write mask granularity')):
        present = term in bank
        log.append(f"    the wrapper states {what}: {present}")
        ok &= present
    return ok, 'memory-inferred, stated as such, with an interface contract'


def syn05(log, ctx):
    """Capacity, gates, flops and STA listed separately."""
    rep = ctx['report']
    ok = True
    log.append(f"    {'module':<24s} {'data memory':>14s} "
               f"{'flop bits':>10s} {'logic cells':>12s} {'STA':>14s}")
    for top in sorted(rep['tops']):
        t = rep['tops'][top]
        if not t['ok']:
            continue
        log.append(f"    {top:<24s} {t['memory_bits'] / 8192:>10.1f} KiB "
                   f"{t['ff_bits']:>10d} {t['logic_cells']:>12d} "
                   f"{'NOT MEASURED':>14s}")
    for field in ('fmax', 'area', 'power'):
        log.append(f"    {field}: {rep[field]}")
        ok &= rep[field] == 'NOT MEASURED'
    log.append(f"    the model clock is the simulation's, not a timing "
               f"result; the structural cap is {rep['cap_seconds']}s per "
               f"top and nothing timed out")
    for top, t in rep['tops'].items():
        if not t['ok'] and 'cap' in str(t.get('error', '')):
            log.append(f"    {top} timed out and is NOT DONE, not zero")
            ok = False
    return ok, 'four columns, and the ones with no library say so'


TESTS = [('SYN-01', syn01), ('SYN-02', syn02), ('SYN-03', syn03),
         ('SYN-04', syn04), ('SYN-05', syn05)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3-10 structural synthesis of the storage")
    import tempfile
    tmp = tempfile.mkdtemp(prefix='r3-syn-')
    rep, r = synth(tmp)
    if rep is None:
        print(f"  the flow produced no report: {(r.stderr or r.stdout)[-300:]}")
        return 1
    ctx = {'report': rep}
    results = []
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
    if bad:
        print(f"  r3-synth-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3-synth-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
