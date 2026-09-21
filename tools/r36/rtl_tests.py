#!/usr/bin/env python3
"""
R36-RTL-01, RTL-02 and RTL-03 -- phase3_6_tasks_and_acceptance.md
section R36-04.

The export package, the public top's timing contract, and the claim that
the synthesizable path has no simulator in it.

RTL-01 is checked by leaving. The package is copied to a directory with
no relationship to this repository, the checker changes into it, and
everything after that uses only what is in the package: no -y search of
ours, no include path of ours, no file of ours. A package that only
worked from inside the tree would pass a check performed inside the
tree, which is the check not being performed.

RTL-02 is about the ports being enough. The old device's behaviour lived
in a testbench, so "can a platform drive this?" had no answer: the
platform would have had to reimplement the sequencing. The smoke test in
the package drives the public ports and nothing else, and the backpressure
cases here raise the memory latency and turn on random stalling to show
the contract holds when the timing changes rather than only at the
latency it was written against.

RTL-03 is mechanical, on purpose. Every `$`-task, every delay and every
plusarg in the synthesizable set is a finding, and the manifest's
declared list of simulation-only behaviour has to account for each one.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, HERE)
import r1lib as R                                          # noqa: E402
import export as E                                         # noqa: E402

TIMEOUT = 1800


def _run(cmd, cwd, timeout=TIMEOUT, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def _package(log):
    """Build the export and put it somewhere unrelated to this repository."""
    staging = tempfile.mkdtemp(prefix='tnca-export-build-')
    src = os.path.join(staging, 'pkg')
    words = E.build_program()
    m = E.export(src, program_words=words, result_addr=0x800,
                 expected=0x0DEFACED)
    # Somewhere else entirely, and NOT under the repository.
    away = tempfile.mkdtemp(prefix='tnca-elsewhere-')
    dst = os.path.join(away, 'hardware')
    shutil.copytree(src, dst)
    shutil.rmtree(staging)
    log.append(f"    built {len(m['rtl'])} synthesizable file(s), "
               f"{len(m['sim'])} simulation file(s)")
    log.append(f"    copied to {dst}, which is not under {ROOT}")
    return dst, m


def rtl01(log):
    """The package elaborates and runs a program, on its own."""
    ok = True
    pkg, m = _package(log)

    # Nothing in the package may DEPEND on a path back into the
    # repository. A comment that names one is documentation -- the
    # shell's header explains what the testbench it replaced used to do,
    # and the first version of this check called that a leak. So
    # comments come out of the Verilog first, and what is left is
    # include directives, string literals and build-script arguments:
    # the things that would actually make the package need our tree.
    leaks = []
    for base, _dirs, names in os.walk(pkg):
        for n in names:
            p = os.path.join(base, n)
            rel = os.path.relpath(p, pkg)
            if n == 'MANIFEST.json' or n == 'README.md':
                # The manifest records provenance on purpose, and the
                # README explains the layout. Neither is compiled.
                continue
            try:
                text = open(p, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            if n.endswith(('.v', '.vh')):
                text = E._strip_comments(text)
            for pat in (ROOT, '../', 'rtl/core/', 'rtl/common/',
                        'rtl/gen/', 'tb/common/', 'tb/device/'):
                if pat in text:
                    leaks.append((rel, pat))
    log.append(f"    paths leading out of the package, in code rather "
               f"than in comments: "
               f"{len(leaks)}{'' if not leaks else ' ' + str(leaks[:3])}")
    ok &= not leaks

    # And the package must not be reachable from ours by accident: no
    # file in it may be a link to somewhere else.
    links = [os.path.relpath(os.path.join(b, n), pkg)
             for b, _d, ns in os.walk(pkg) for n in ns
             if os.path.islink(os.path.join(b, n))]
    log.append(f"    symlinks out of the package: {len(links)}")
    ok &= not links

    r = _run(['sh', './elaborate.sh'], pkg)
    log.append(f"    elaborate.sh: {r.stdout.strip() or r.stderr.strip()[:90]}")
    ok &= r.returncode == 0

    r = _run(['sh', './run_smoke.sh'], pkg)
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    for ln in tail[-3:]:
        log.append(f"    smoke: {ln.strip()[:88]}")
    ok &= r.returncode == 0 and 'SMOKE PASS' in r.stdout

    r = _run(['sh', './build_binding.sh'], pkg)
    log.append(f"    binding: {(r.stdout or r.stderr).strip().splitlines()[-1][:88]}")
    ok &= r.returncode == 0

    # Completeness of the record itself.
    need = ('synthesizable_top', 'parameters', 'tools', 'rtl', 'sim',
            'includes', 'unresolved', 'simulation_only_behaviour',
            'claim_level')
    missing = [k for k in need if k not in m]
    log.append(f"    manifest fields: {len(need) - len(missing)}/{len(need)}, "
               f"{len(m['parameters'])} parameter(s), "
               f"{len(m['tools'])} tool version(s)")
    ok &= not missing
    log.append(f"    undeclared blackboxes: {len(m['unresolved'])}")
    ok &= not m['unresolved']
    log.append(f"    the claim is {m['claim_level']!r}, and it is not "
               f"platform-verified: {m['claim_level'] != 'platform-verified'}")
    ok &= m['claim_level'] != 'platform-verified'

    # Every file the manifest lists is present, and every file present is
    # listed. A package that shipped something it did not declare would
    # be exactly as hard to review as one that declared something it did
    # not ship.
    listed = {e['file'] for e in m['rtl'] + m['sim'] + m['includes']}
    present = set()
    for sub in ('rtl', 'sim'):
        for n in os.listdir(os.path.join(pkg, sub)):
            present.add(f"{sub}/{n}")
    log.append(f"    listed {len(listed)}, present {len(present)}, "
               f"undeclared {sorted(present - listed)}, "
               f"missing {sorted(listed - present)}")
    ok &= listed == present
    return ok, 'the package elaborates and runs a program from elsewhere'


def rtl02(log):
    """The public top's timing contract, under conditions it was not
    written against."""
    ok = True
    pkg, _m = _package(log)

    # The smoke test, but with the memory made slow and then made
    # hostile. Nothing in the bench assumes a latency, so if the contract
    # is real these pass unchanged and only the cycle count moves.
    out = tempfile.mkdtemp(prefix='tnca-smoke-')
    rtl = sorted(os.listdir(os.path.join(pkg, 'rtl')))
    srcs = [f"rtl/{n}" for n in rtl if n.endswith('.v')]
    srcs += [f"sim/{n}" for n in sorted(os.listdir(os.path.join(pkg, 'sim')))
             if n.endswith('.v') and 'device' not in n]

    # Latency, backpressure and reordering -- the three things the
    # memory contract allows and that a public top has to survive. This
    # used to vary latency alone while its own docstring claimed it
    # turned on random stalling; the smoke bench hardwired the stall and
    # reorder knobs to zero, so there was nothing to turn on. An audit
    # of these checkers found the docstring describing code that was not
    # there.
    seen = {}
    for label, params in (
            ('latency 1', []),
            ('latency 4', ['-Ptnca_smoke.MEM_LATENCY=4']),
            ('latency 1, backpressure',
             ['-Ptnca_smoke.MEM_STALL=1']),
            ('latency 4, backpressure, reordering',
             ['-Ptnca_smoke.MEM_LATENCY=4', '-Ptnca_smoke.MEM_STALL=1',
              '-Ptnca_smoke.MEM_REORDER=1'])):
        vvp = os.path.join(out, f"s{len(seen)}.vvp")
        r = _run(['iverilog', '-g2005', '-I', 'rtl', '-I', 'sim',
                  '-s', 'tnca_smoke'] + params + ['-o', vvp,
                 'tnca_smoke.v'] + srcs, pkg)
        if r.returncode != 0:
            log.append(f"    {label}: did not build: "
                       f"{r.stderr.strip().splitlines()[-1][:70]}")
            ok = False
            continue
        r = _run(['vvp', vvp], pkg)
        passed = 'SMOKE PASS' in r.stdout
        cyc = re.search(r'ran in (\d+) cycle', r.stdout)
        seen[label] = int(cyc.group(1)) if cyc else None
        log.append(f"    {label}: {'PASS' if passed else 'FAIL'}, "
                   f"{seen[label]} cycle(s) to run the program")
        ok &= passed
    shutil.rmtree(out, ignore_errors=True)

    # A slower memory must actually be slower, and a hostile one must
    # cost something too. Identical numbers would mean the parameters
    # reached nothing and the cases were decoration.
    a, b = seen.get('latency 1'), seen.get('latency 4')
    c = seen.get('latency 1, backpressure')
    log.append(f"    a slower memory really is slower: "
               f"{a} -> {b} ({'yes' if (a and b and b > a) else 'NO'})")
    ok &= bool(a and b and b > a)
    log.append(f"    backpressure really costs cycles: "
               f"{a} -> {c} ({'yes' if (a and c and c > a) else 'NO'})")
    ok &= bool(a and c and c > a)
    log.append("    and the answers are the same in every one of them, "
               "which is the property: the timing is not assumed")
    return ok, 'the ports carry the contract under latency, backpressure '\
               'and reordering'


# The system functions that ARE hardware: they describe a width or a
# signedness and every synthesis tool implements them. Listed by name
# rather than matched by a pattern, so adding one is a decision somebody
# makes rather than a regex quietly widening.
_SYNTHESIZABLE_SYSTEM = {
    '$signed', '$unsigned', '$clog2', '$bits', '$size', '$left',
    '$right', '$high', '$low', '$increment', '$countones', '$onehot',
    '$onehot0',
}

# Anything that cannot exist in hardware.
_SIMISMS = [
    (re.compile(r'\$[a-z_]\w*', re.I), 'a system task'),
    (re.compile(r'(?<![\w.])#\s*\d'), 'a delay'),
    (re.compile(r'\binitial\b'), 'an initial block'),
    (re.compile(r'\bforce\b|\brelease\b'), 'a force/release'),
]


def rtl03(log):
    """No simulator in the synthesizable path, and one core implementation."""
    ok = True
    pkg, m = _package(log)

    findings = []
    for entry in m['rtl']:
        p = os.path.join(pkg, entry['file'])
        body = E._strip_comments(open(p).read())
        # `ifdef-guarded simulation assertions are a different thing: they
        # are compiled out, and the structural check below runs with
        # SYNTHESIS defined, which is how the synthesis flow reads them.
        body = re.sub(r'`ifndef\s+SYNTHESIS.*?`endif', '', body, flags=re.S)
        body = re.sub(r'`ifdef\s+SIM.*?`endif', '', body, flags=re.S)
        for rx, what in _SIMISMS:
            for mm in rx.finditer(body):
                tok = mm.group(0)
                if what == 'a system task' and \
                        tok.lower() in _SYNTHESIZABLE_SYSTEM:
                    continue
                findings.append((entry['file'], what, tok[:20]))
    log.append(f"    simulator-only constructs in the synthesizable set: "
               f"{len(findings)}")
    for f in findings[:6]:
        log.append(f"      {f[0]} {f[1]} {f[2]!r}")
    ok &= not findings

    declared = {d['file'] for d in m['simulation_only_behaviour']}
    simfiles = {e['file'] for e in m['sim']}
    log.append(f"    simulation files: {sorted(simfiles)}")
    log.append(f"    each one declared with a reason: "
               f"{simfiles <= declared} ({sorted(simfiles - declared)})")
    ok &= simfiles <= declared

    # The structural check, in two passes with different costs.
    #
    # The whole hierarchy gets the connection check: elaborate everything
    # from the exported files alone and let yosys object to a port that
    # does not line up, a net with two drivers or a width that does not
    # match. `proc` is deliberately NOT run over the whole core here --
    # it takes upward of a quarter of an hour on the generated XLS units
    # and R3 v2's accepted r3v2-syn-tests already covers the core's
    # internals. Repeating it would buy a longer gate and no new fact.
    #
    # The SHELL is the code this stage added, so it gets the full
    # treatment: proc, then `check -assert`, which is where a latch or a
    # combinational loop is actually reported.
    files = ' '.join(e['file'] for e in m['rtl'])
    incs = ' '.join(sorted({f"-I {os.path.dirname(e['file']) or '.'}"
                            for e in m['includes']}))
    top = m['synthesizable_top']

    def yosys(script, label):
        ys = tempfile.NamedTemporaryFile('w', suffix='.ys', delete=False)
        ys.write(script)
        ys.close()
        r = _run(['yosys', '-q', '-s', ys.name], pkg, timeout=900)
        os.unlink(ys.name)
        if r.returncode != 0:
            bad = [ln for ln in (r.stdout + r.stderr).splitlines()
                   if ln.strip() and ('Warning' in ln or 'ERROR' in ln)]
            for ln in bad[:6]:
                log.append(f"      {ln.strip()[:90]}")
            log.append(f"    {label}: FAILED")
            return False
        log.append(f"    {label}: clean")
        return True

    ok &= yosys(f"read_verilog -DSYNTHESIS {incs} {files}\n"
                f"hierarchy -top {top} -check\ncheck -noinit\n",
                "the whole exported hierarchy elaborates and connects")
    ok &= yosys(f"read_verilog -DSYNTHESIS {incs} {files}\n"
                f"hierarchy -top {top} -check\n"
                f"select {top}\nproc\ncheck -assert\n",
                f"`proc; check -assert` on {top}: no latch, no "
                f"combinational loop, no multiple driver, no width "
                f"problem")

    # The simulation binding and any future platform shell instantiate
    # the SAME top. Checked, not asserted: the binding's source must name
    # it, and the package must contain exactly one definition of it.
    top = m['synthesizable_top']
    binding = open(os.path.join(pkg, 'sim', 'tnca_device.v')).read()
    defs = 0
    for entry in m['rtl'] + m['sim']:
        body = E._strip_comments(open(os.path.join(pkg, entry['file'])).read())
        defs += len(re.findall(rf'^\s*module\s+{top}\b', body, re.M))
    log.append(f"    the binding instantiates {top}: {top in binding}; "
               f"the package defines it {defs} time(s)")
    ok &= (top in binding) and defs == 1
    return ok, 'no simulator in the synthesizable path, one implementation'


TESTS = [('R36-RTL-01', rtl01), ('R36-RTL-02', rtl02),
         ('R36-RTL-03', rtl03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate B: the public RTL, and an export that stands on its own")
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
    print(f"  r36-rtl-tests          {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
