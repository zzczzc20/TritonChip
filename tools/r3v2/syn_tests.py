#!/usr/bin/env python3
"""
R3V2-SYN-01, -02 and REG-01 -- doc/phase3_tasks_and_acceptance.md G6.

What the candidate's storage looks like structurally, whether the
integration connects up, and whether everything that used to pass still
does.

The memory-preserving flow is flow/synth/mem_synth.py: `memory -nomap`,
so an array a real flow would give to an SRAM stays a memory object
instead of becoming a few hundred thousand flops. SYN-01 is about the
VRF's own bits; SYN-02 adds the check that the whole core, elaborated with
that file in it, connects up and has nothing unexplained in it.

Nothing here claims an area or a frequency. There is no PDK and no FPGA
primitive, so the mapping is memory-inferred and Fmax, power and physical
area read NOT MEASURED -- and SYN-02 greps the report to make sure nobody
said otherwise.
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
import r1lib as R                                        # noqa: E402

FLOW = os.path.join(ROOT, 'flow', 'synth', 'mem_synth.py')
CONTRACT = json.load(open(os.path.join(ROOT, 'spec',
                                       'r3v2_contract.json')))
INVENTORY = json.load(open(os.path.join(ROOT, 'spec',
                                        'r3v2_inventory.json')))
# The modules this stage changed or added. SYN-02 asks that all of them be
# in the run, so they are named rather than globbed: a module that stopped
# being included would report nothing and look clean.
CHANGED = ['tnca_rf_bank', 'tnca_vrf_banked', 'tnca_operand_collector']


def rtl_files():
    out = []
    for sub in ('core', 'common', 'gen'):
        d = os.path.join(ROOT, 'rtl', sub)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith('.v')]
    return out


def synth(tmp):
    out = os.path.join(tmp, 'mem_synth.json')
    subprocess.run([sys.executable, FLOW, '--json', out],
                   capture_output=True, text=True, timeout=3600, cwd=ROOT)
    return json.load(open(out)) if os.path.exists(out) else None


def integration_check(banked):
    """Elaborate the whole core and let yosys check the connections."""
    script = (f"read_verilog -DSYNTHESIS -I "
              f"{os.path.join(ROOT, 'rtl', 'common')} "
              + ' '.join(rtl_files()) +
              f"; chparam -set USE_BANKED_VRF {1 if banked else 0} tnca_core"
              f"; hierarchy -top tnca_core -check; check -noinit; ls")
    r = subprocess.run(['yosys', '-p', script], capture_output=True,
                       text=True, timeout=1800, cwd=ROOT)
    warns = [l.strip() for l in r.stdout.splitlines()
             if l.startswith('Warning:')]
    explained = [w for w in warns if 'with list of registers' in w]
    unexplained = [w for w in warns if w not in explained]
    problems = [l.strip() for l in r.stdout.splitlines()
                if re.search(r'found \d+ problems|combinational loop|'
                             r'multiple drivers|is used but has no driver',
                             l) and 'found 0 problems' not in l]
    mods = set()
    seen = False
    for line in r.stdout.splitlines():
        if re.match(r'^\d+ modules:', line):
            seen = True
            continue
        if seen:
            m = re.match(r'^\s+(?:\$paramod\S*\\)?(tnca_\w+)\s*$', line)
            if m:
                mods.add(m.group(1))
    return {'ok': r.returncode == 0, 'unexplained': unexplained,
            'explained': len(explained), 'problems': problems,
            'modules': mods}


def syn01(log, ctx):
    """The candidate's VRF: exactly 8 KiB, in synchronous memory."""
    rep = ctx['synth']
    ok = True
    t = rep['tops']['tnca_vrf_banked']
    want = CONTRACT['vrf']['data_bits']
    got = t['memory_bits']
    log.append(f"    {got} bit(s) in {len(t['data_memories'])} memory "
               f"object(s); the contract declares {want}")
    ok &= got == want == 65536
    frac = got / want
    log.append(f"    {frac * 100:.0f}% of the declared data is in "
               f"synchronous memory objects (SYN-01 wants 90%)")
    ok &= frac >= 0.90

    p = CONTRACT['ports']
    for m in t['data_memories']:
        good = (m['rd_ports'] == p['accepted_reads_per_bank_per_cycle']
                and m['wr_ports'] == p['accepted_writes_per_bank_per_cycle']
                and m['rd_sync'])
        log.append(f"      {m['name']}: {m['rd_ports']}R/{m['wr_ports']}W, "
                   f"{'synchronous' if m['rd_sync'] else 'ASYNCHRONOUS'}"
                   f"  {'matches the profile' if good else 'DOES NOT'}")
        ok &= good

    # The control state is listed on its own. Folding it into the capacity
    # is how a structure disappears; folding the old ARF's bits into the
    # ratio would be worse, so neither is done.
    log.append(f"    control state, counted separately: {t['ff_bits']} "
               f"flop bit(s) + {t['small_bits']} bit(s) in "
               f"{len(t['small_arrays'])} array(s) (the per-lane valid "
               f"bits, 512 per bank)")
    arf = rep['tops'].get('tnca_acc_rf', {})
    log.append(f"    the ARF is {arf.get('memory_bits', 0) // 8192} KiB and "
               f"is NOT counted toward that ratio: it is still "
               f"{CONTRACT['arf']['implementation']}")
    ok &= CONTRACT['arf']['synchronous'] is False
    return ok, 'exactly 8 KiB, synchronous, with the control state listed'


def syn02(log, ctx):
    """Nothing unexplained, and the integration connects."""
    rep = ctx['synth']
    ok = True
    for top in CHANGED:
        t = rep['tops'].get(top)
        if not t or not t['ok']:
            log.append(f"    {top}: NOT IN THE RUN")
            ok = False
            continue
        warns = t.get('warnings') or []
        log.append(f"    {top:<24s} {t['latches']} latch(es), "
                   f"{len(t.get('problems') or [])} finding(s), "
                   f"{len(warns)} unexplained warning(s)")
        ok &= not t['latches'] and not warns

    integ = ctx['integration']
    log.append(f"    the whole core with the banked file elaborated and "
               f"connection-checked: {integ['ok']}")
    log.append(f"      {len(integ['problems'])} unresolved finding(s), "
               f"{len(integ['unexplained'])} unexplained warning(s), "
               f"{integ['explained']} explained")
    for w in integ['unexplained'][:3]:
        log.append(f"        UNEXPLAINED: {w[:74]}")
    ok &= integ['ok'] and not integ['problems'] and not integ['unexplained']

    have = integ['modules']
    log.append(f"    the integrated hierarchy contains "
               f"{'tnca_vrf_banked' if 'tnca_vrf_banked' in have else 'NOT the banked file'}"
               f" and {'still tnca_vector_rf' if 'tnca_vector_rf' in have else 'no behavioural file'}")
    ok &= 'tnca_vrf_banked' in have and 'tnca_vector_rf' not in have

    log.append(f"    mapping: {rep['mapping']}")
    text = json.dumps(rep).lower()
    for claim in ('sram area', 'fmax met', 'ghz', 'mhz', 'timing closed'):
        if claim in text:
            log.append(f"    the report claims {claim!r} without a library")
            ok = False
    log.append(f"    Fmax {rep['fmax']}, area {rep['area']}, power "
               f"{rep['power']}")
    ok &= all(rep[k] == 'NOT MEASURED' for k in ('fmax', 'area', 'power'))
    return ok, 'no latches, the integration connects, nothing overclaimed'


def reg01(log, ctx):
    """Everything G0 froze as inherited still passes."""
    ok = True
    for entry in INVENTORY['inherited']:
        cmd = entry['command'].split()
        r = subprocess.run(['make'] + cmd[1:], cwd=ROOT,
                           capture_output=True, text=True, timeout=7200)
        passed = r.returncode == 0
        # A skip is not a pass. The contract says so: no missing-dependency
        # skip, no xfail, no xpass.
        skipped = len(re.findall(r'\bSKIP\b', r.stdout))
        log.append(f"    {entry['id']:<22s} {entry['command']:<34s} "
                   f"{'PASS' if passed else 'FAIL'}"
                   + (f"  ({skipped} SKIP)" if skipped else ''))
        ok &= passed and skipped == 0
    log.append(f"    {len(INVENTORY['inherited'])} inherited regression(s), "
               f"each frozen in the inventory with a reason")
    log.append(f"    the old default profile is untouched: `make core` is "
               f"the reference and is in that list")
    return ok, 'every inherited regression, no skips'


TESTS = [('R3V2-SYN-01', syn01), ('R3V2-SYN-02', syn02),
         ('R3V2-REG-01', reg01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G6: the storage structurally, and everything that used "
          "to pass")
    tmp = tempfile.mkdtemp(prefix='r3v2-syn-')
    ctx = {'tmp': tmp}
    need = not args.only or any(c.startswith('R3V2-SYN') for c in args.only)
    if need:
        ctx['synth'] = synth(tmp)
        if ctx['synth'] is None:
            print("  the synthesis flow produced no report")
            return 1
        ctx['integration'] = integration_check(banked=True)
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
            log.append('    ' + traceback.format_exc()[-900:].replace(
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
        print(f"  r3v2-syn-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-syn-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
