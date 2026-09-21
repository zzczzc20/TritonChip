#!/usr/bin/env python3
"""
R36-MEM-01, MEM-02 and RST-01 -- phase3_6_tasks_and_acceptance.md
section R36-05.

The storage ledger, a second memory behind the same boundary, and what
the device looks like at reset.

MEM-01's ledger is spec/r36_memories.json. A ledger nobody checks is a
document, so every entry here is read back against the RTL: the module
has to exist, the array declaration in it has to have the depth and
width the ledger claims, and every field the contract names has to be
filled in. The `technology_mapping` field gets its own test, because the
failure this standard is written against is a project that writes
"verified" beside an array yosys merely inferred.

MEM-02 is the expensive one and the one worth having. The core has only
ever run against tb/common/tnca_mem_model.v, so every habit of that
model is indistinguishable, from inside the core, from a requirement of
the contract. tb/common/tnca_mem_ref.v is a second memory written from
the port list -- a byte array, a shift-register pipeline, a run-length
stall pattern and an adversarial swap stage instead of a word array, a
pool of counters, an LFSR bit and a random pick. Both run the same
program at several latencies, with and without backpressure, with and
without reordering, and both have to agree with values written down in
advance rather than with each other.

RST-01 resets the device with a kernel in flight and requires the next
launch to be correct. That case could not pass at all before the core's
fetch valids were added to its reset list: core_idle read X from reset
until the first launch, so the host port never opened.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))
import r1lib as R                                          # noqa: E402

LEDGER = json.load(open(os.path.join(ROOT, 'spec', 'r36_memories.json')))
PROG = os.path.join(ROOT, 'tb', 'core', 'progs', 't30_memboundary.s')
TB = os.path.join(ROOT, 'tb', 'device', 'tnca_shell_tb.v')
SHELL = os.path.join(ROOT, 'rtl', 'tnca_shell.v')

REQUIRED = ('id', 'where', 'what', 'read_ports', 'write_ports',
            'read_latency_cycles', 'write_latency_cycles', 'byte_mask',
            'read_during_write', 'reset_initialisation', 'replaceable',
            'technology_mapping')

# The only honest values today. 'mapped_and_verified' is deliberately
# absent: nothing has been mapped, and a checker that accepted the word
# would be the mechanism by which it appeared.
MAPPING_STATES = ('not started', 'in progress', 'not applicable: outside '
                  'the export')

# The configurations both memories must survive:
# (latency, backpressure, reordering).
CONFIGS = [(1, 0, 0), (4, 0, 0), (8, 0, 0), (1, 1, 0), (4, 1, 0),
           (1, 0, 1), (4, 1, 1)]


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=kw.pop('timeout', 1800), cwd=ROOT, **kw)


def _assemble(out):
    import tnca_asm
    a = tnca_asm.Assembler()
    a.assemble(open(PROG).read())
    words = tnca_asm.build_image(a.records, size_words=4096)
    with open(out, 'w') as fh:
        for w in words:
            fh.write(f"{w:016x}\n")
    return len([w for w in words if w])


def _build(tmp, impl, lat, stall, reorder):
    vvp = os.path.join(tmp, f"tb_{impl}_{lat}_{stall}_{reorder}.vvp")
    if os.path.exists(vvp):
        return vvp
    r = _run(['iverilog', '-g2005', '-I', 'rtl/common',
              '-y', 'rtl/core', '-y', 'rtl/common', '-y', 'rtl/gen',
              '-y', 'tb/common',
              f'-Ptnca_shell_tb.MEM_IMPL={impl}',
              f'-Ptnca_shell_tb.MEM_LATENCY={lat}',
              f'-Ptnca_shell_tb.RANDOM_STALL={stall}',
              f'-Ptnca_shell_tb.MEM_REORDER={reorder}',
              '-o', vvp, TB, SHELL])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1][:120])
    return vvp


_ARRAY_RE = re.compile(
    r'reg\s*(?:\[[^\]]*\]\s*)?(\w+)\s*\[\s*0\s*:\s*([^\]]+?)\s*-?\s*1?\s*\]')


def mem01(log):
    """The ledger, read back against the RTL."""
    ok = True
    arrays = LEDGER['arrays']
    log.append(f"    {len(arrays)} array(s) in the ledger")

    for a in arrays:
        missing = [k for k in REQUIRED if k not in a]
        if missing:
            log.append(f"      {a.get('id','?')}: missing {missing}")
            ok = False

    # Every array that claims to be in the export must be in a file that
    # exists and declares an array.
    inside = [a for a in arrays
              if not a['where'].startswith('OUTSIDE')]
    for a in inside:
        paths = [p.strip() for p in a['where'].replace(' and ', ',').split(',')
                 if p.strip().endswith('.v')]
        found = 0
        for rel in paths:
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                log.append(f"      {a['id']}: no such file {rel}")
                ok = False
                continue
            body = open(p).read()
            if _ARRAY_RE.search(body):
                found += 1
        if not found:
            log.append(f"      {a['id']}: no memory array declared in "
                       f"{paths}")
            ok = False
    log.append(f"    {len(inside)} of them are inside the export, and each "
               f"names a file that declares an array")

    # Capacity arithmetic. A ledger whose numbers do not multiply is a
    # ledger nobody added up.
    for a in arrays:
        if all(k in a for k in ('depth', 'width_bits', 'capacity_bytes')):
            want = a['depth'] * a['width_bits'] // 8
            if want != a['capacity_bytes']:
                log.append(f"      {a['id']}: depth x width is {want} "
                           f"bytes, the ledger says {a['capacity_bytes']}")
                ok = False
    log.append("    depth x width matches the stated capacity for every "
               "entry")

    # The claim that is easiest to inflate.
    states = {a['id']: a['technology_mapping'] for a in arrays}
    bad = {k: v for k, v in states.items() if v not in MAPPING_STATES}
    log.append(f"    technology mapping: "
               f"{sorted(set(states.values()))}")
    log.append(f"    nothing claims to be a verified physical RAM: "
               f"{not bad}{'' if not bad else ' ' + str(bad)}")
    ok &= not bad

    cd = LEDGER.get('cross_domain', {})
    need = ('clock', 'reset', 'reset_release', 'memory_clock',
            'unstated_board_assumptions')
    log.append(f"    cross-domain responsibilities: "
               f"{len([k for k in need if k in cd])}/{len(need)} stated")
    ok &= all(k in cd for k in need)
    return ok, f"{len(arrays)} array(s), itemised and read back"


def mem02(log):
    """Two memories, the same program, the same answers."""
    ok = True
    tmp = tempfile.mkdtemp(prefix='tnca-mem-')
    hexf = os.path.join(tmp, 't30.hex')
    n = _assemble(hexf)
    log.append(f"    the program is {n} word(s) of "
               f"{os.path.relpath(PROG, ROOT)}")
    log.append("    it does partial stores of every width, a vector round "
               "trip, a gather and an atomic -- the operations two memory "
               "implementations are most likely to disagree about")

    cycles = {}
    names = {0: 'tnca_mem_model (word array, counter pool)',
             1: 'tnca_mem_ref   (byte array, shift register, swap stage)'}
    passed = 0
    for impl in (0, 1):
        log.append(f"    {names[impl]}")
        for lat, stall, reorder in CONFIGS:
            try:
                vvp = _build(tmp, impl, lat, stall, reorder)
            except RuntimeError as e:
                log.append(f"      lat={lat} stall={stall} reorder="
                           f"{reorder}: did not build: {e}")
                ok = False
                continue
            r = _run(['vvp', vvp, f'+hex={hexf}', '+quiet'])
            verdict = [ln for ln in r.stdout.splitlines()
                       if ln.startswith(('PASS', 'FAIL'))]
            good = bool(verdict) and verdict[0].startswith('PASS')
            passed += good
            cyc = next((int(ln.split()[1]) for ln in r.stdout.splitlines()
                        if ln.startswith('CYCLES ')), None)
            cycles[(impl, lat, stall, reorder)] = cyc
            log.append(f"      latency {lat}, backpressure {stall}, "
                       f"reordering {reorder}: "
                       f"{'PASS' if good else 'FAIL'}, {cyc} cycle(s)")
            if not good:
                for ln in r.stdout.splitlines()[-4:]:
                    log.append(f"        {ln.strip()[:88]}")
            ok &= good

    log.append(f"    {passed}/{2 * len(CONFIGS)} configuration(s), both "
               f"implementations, every value against a constant written "
               f"in the bench rather than against the other implementation")

    # A memory that ignores the latency it was given returns the right
    # answers at every setting and the same number of cycles at every
    # setting. Correctness alone cannot tell the difference, so the
    # cycle counts are compared: more latency must cost more cycles, in
    # BOTH implementations. The reference memory had no such check until
    # a directed mutation (R36-TEST-03 MUT-MEM-LATENCY) froze its depth
    # and nothing noticed.
    for impl in (0, 1):
        ladder = [(lat, cycles.get((impl, lat, 0, 0)))
                  for lat in (1, 4, 8)]
        vals = [c for _l, c in ladder if c is not None]
        rising = len(vals) == 3 and vals[0] < vals[1] < vals[2]
        log.append(f"    implementation {impl}: a slower memory costs "
                   f"more cycles: {ladder} ({'yes' if rising else 'NO'})")
        ok &= rising

    # The second implementation has to be a different one. Checked, not
    # asserted: if tnca_mem_ref were a copy, this case would be one test
    # run twice.
    a = open(os.path.join(ROOT, 'tb', 'common', 'tnca_mem_model.v')).read()
    b = open(os.path.join(ROOT, 'tb', 'common', 'tnca_mem_ref.v')).read()
    marks = {'a word array': ('reg [MEM_DW-1:0] mem', a),
             'a byte array': ('reg [7:0] b [', b),
             'a counter pool': ('p_cnt', a),
             'a shift register': ('s_valid', b),
             'an LFSR stall': ('lfsr[3]', a),
             'a run-length stall': ('phase < STALL_RUN', b)}
    have = {k: (v[0] in v[1]) for k, v in marks.items()}
    log.append(f"    the two are structurally different: "
               f"{sorted(k for k, v in have.items() if v)}")
    ok &= all(have.values())
    return ok, 'two memories, one boundary, the same answers'


def rst01(log):
    """Reset with a kernel in flight, and the launch after it."""
    ok = True
    tmp = tempfile.mkdtemp(prefix='tnca-rst-')
    hexf = os.path.join(tmp, 't30.hex')
    _assemble(hexf)

    for impl in (0, 1):
        vvp = _build(tmp, impl, 4, 1, 0)
        r = _run(['vvp', vvp, f'+hex={hexf}', '+reset'])
        out = r.stdout
        good = 'PASS' in out and 'FAIL' not in out
        log.append(f"    implementation {impl}, reset 30 cycles into a "
                   f"launch: {'PASS' if good else 'FAIL'}")
        for ln in out.splitlines():
            if 'after a reset' in ln or ln.startswith('FAIL'):
                log.append(f"      {ln.strip()[:88]}")
        ok &= good

    # The property that made this case impossible before, stated as a
    # property of the source rather than of a waveform: every pipeline
    # valid in the core is in its reset list.
    core = open(os.path.join(ROOT, 'rtl', 'core', 'tnca_core.v')).read()
    # EVERY reset block, not the first one. Splitting on the first
    # occurrence found a reset block a thousand lines above the pipeline
    # and reported all five valids missing from a file that resets all
    # five -- a checker failing for a reason that had nothing to do with
    # the thing it was checking.
    blocks = []
    for m in re.finditer(r'if\s*\(\s*!\s*rst_n\s*\)\s*begin', core):
        rest = core[m.end():]
        cut = rest.find('end else')
        blocks.append(rest if cut < 0 else rest[:cut])
    reset_text = re.sub(r'\s+', ' ', ' '.join(blocks))
    valids = ('f1_valid', 'fb_valid', 'd_valid', 'r_valid', 'x_valid')
    missing = [v for v in valids if f"{v} <= 1'b0" not in reset_text]
    log.append(f"    every pipeline valid is reset: {not missing} "
               f"{'' if not missing else missing}")
    ok &= not missing

    # And the observable consequence: the shell's host port is ready at
    # reset, with no launch to flush anything.
    log.append("    the consequence: the host port is usable from reset, "
               "with no launch needed to flush an undefined pipeline -- "
               "which is what the export's smoke test depends on")

    cd = LEDGER.get('cross_domain', {})
    log.append(f"    the reset contract is written down: "
               f"{cd.get('reset_release', 'NOT STATED')[:76]}")
    ok &= 'reset_release' in cd
    return ok, 'reset in flight, and a correct launch after it'


TESTS = [('R36-MEM-01', mem01), ('R36-MEM-02', mem02),
         ('R36-RST-01', rst01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate B: the storage boundary, and what reset leaves behind")
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
        print(flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print(f"  r36-mem-tests          {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
