#!/usr/bin/env python3
"""
R36-FPGA-01 to FPGA-04 -- the board wrapper, gated where nothing else reaches.

WHY A FILE OF ITS OWN

R36-RTL-03 checks the synthesizable set. The synthesizable set is whatever
tools/r36/export.py can reach by walking the module graph outward from
`tnca_shell`, and rtl/fpga/tnca_fpga_top.v is the module that instantiates
the shell -- it sits ABOVE that root, not below it, so the walk never
arrives, SEARCH never names the directory, and the manifest has no entry
for any of the three files.

That is the shape of a wrapper rather than a hole somebody chose, and the
consequence was real anyway: tnca_fpga_top.v, tnca_mmio.v and
tnca_fpga_mem.v were written and not one of them had ever been compiled.
`make lint` had the same blind spot from the same cause until RTL_FPGA was
added to RTL_ALL, and `make lint` is not the answer on its own either -- it
finds a syntax error and nothing that follows one.

WHAT A WRAPPER GETS WRONG

Three things, and all three were found here rather than imagined:

  * A port that does not connect, or connects at the wrong width.
    `hierarchy -check` is what objects.
  * A latch or a combinational loop. `proc; check -assert` is where those
    are reported, and a wrapper is unusually prone to them because its
    handshakes are combinational by design -- this one shipped with
    `host_req_valid` driven from a register against a `host_req_ready`
    that did not wait for it, which cost a false handshake in the first
    cycle of every request and looked, from the host, like a hang.
  * A memory that infers as a register file. This is worse than the other
    two because it does not fail: yosys says "Replacing memory with list
    of registers", the design elaborates, and what comes out is a great
    many flops and a timing closure that never happens. rtl/fpga/
    tnca_fpga_mem.v exists because a board cannot `$readmemh` its way to
    an initial image, so a `$mem` cell is the thing to look for and its
    absence is the finding.

WHAT THIS DOES NOT ESTABLISH

Not that anything runs on a board, and not that the design meets timing at
any frequency. R36 keeps `platform-verified` reserved for hardware;
python/tnca/backends/u50.py is the file that would drive an Alveo U50 and
it has never touched one.

WHY EACH MODULE IS ELABORATED AS ITS OWN TOP

`hierarchy` rewrites a parameterised module's name to
`$paramod$<hash>\tnca_fpga_mem`, so a later `select tnca_fpga_mem` matches
NOTHING -- silently, and with a successful exit status. A check written as
`select <module>; proc; check -assert` therefore reports the same thing
whether or not it ran, and a first version of this file did exactly that.
Elaborating each module at the root removes the parameterisation and the
ambiguity with it, and `_processes_left` below asserts that the process
count reached zero so that a `proc` which did nothing cannot be mistaken
for a `proc` which found nothing.
"""
import argparse
import glob
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
# Taken from the gate that already owns the table rather than copied out.
# A second copy is a second thing to keep in step, and the copy nobody
# updates is the one that quietly stops finding anything.
from rtl_tests import _SIMISMS, _SYNTHESIZABLE_SYSTEM      # noqa: E402

TIMEOUT = 900
TOP = 'tnca_fpga_top'
FPGA_DIR = os.path.join('rtl', 'fpga')
INCS = ['-I rtl/common', '-I rtl/noc']
RTLLIB = ['common', 'core', 'gen', 'nuca', 'noc', 'tile']

# The modules that carry the wrapper's own logic, each elaborated at the
# root. The wrapper itself is checked separately because it needs the
# whole library behind it.
STANDALONE = ['tnca_mmio', 'tnca_fpga_mem']

_PROBLEMS = re.compile(r'Found and reported (\d+) problems')
_STAT_NAME = re.compile(r'^=== (\S+) ===\s*$', re.M)
_MEM_CELL = re.compile(r'^\s+\$(mem|mem_v2|memrd|memwr)\s+(\d+)\s*$', re.M)


def _rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, '/')


def _glob(*parts):
    return sorted(_rel(p) for p in glob.glob(os.path.join(ROOT, *parts)))


def _fpga_files():
    return _glob(FPGA_DIR, '*.v')


def _library_files():
    """The whole library, for elaborating the wrapper.

    Written out rather than passed as one pattern because yosys's
    `read_verilog` has no library search: iverilog's `-y` has no
    equivalent here, and a file that is not named is a module that is not
    in the design. rtl/*.v is last for readability only -- `hierarchy`
    resolves after every file has been read, so the order carries no
    meaning.
    """
    out = list(_fpga_files())
    for d in RTLLIB:
        out += _glob('rtl', d, '*.v')
    out += _glob('rtl', '*.v')
    return out


def _run_yosys(script, log, what):
    """Run a yosys script from the repository root. Returns (ok, text).

    A missing yosys is a failure and not a skip. R1-02: a case that cannot
    run has not passed, and every claim this file makes is a claim about
    what a synthesiser found. Reported rather than raised, so that the
    reason reaches the log instead of a subprocess traceback.
    """
    if shutil.which('yosys') is None:
        log.append('    yosys is not installed, so this check did NOT RUN')
        log.append('    that is a failure and not a skip: connection, '
                   'latch, combinational loop and memory inference are all '
                   'statements about a synthesiser, and not one of them has '
                   'been established here')
        return False, ''

    fd, path = tempfile.mkstemp(suffix='.ys', prefix='tnca-fpga-')
    with os.fdopen(fd, 'w') as fh:
        fh.write(script)
    try:
        r = subprocess.run(['yosys', '-s', path], cwd=ROOT,
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        log.append(f'    {what}: yosys did not finish in {TIMEOUT}s')
        log.append('    a synthesiser that never returns has established '
                   'nothing; it has only stopped saying')
        return False, ''
    finally:
        os.unlink(path)
    return r.returncode == 0, (r.stdout or '') + (r.stderr or '')


def _problem_count(text):
    """What the last `check` pass reported, or None if no pass ran."""
    found = None
    for m in _PROBLEMS.finditer(text):
        found = int(m.group(1))
    return found


def _processes_left(text):
    """Processes remaining across the stat blocks, or None if none.

    None is itself a finding: it means the selection was empty and every
    pass after it ran on nothing.
    """
    hits = [int(m.group(1)) for m in
            re.finditer(r'^\s*Number of processes:\s+(\d+)\s*$', text, re.M)]
    return None if not hits else sum(hits)


def _stat_names(text):
    return sorted(set(_STAT_NAME.findall(text)))


def _mem_cells(text):
    return sum(int(m.group(2)) for m in _MEM_CELL.finditer(text))


def _interesting(text, n=6):
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if s and ('Error' in s or 'ERROR' in s):
            keep.append(s[:110])
    for line in text.splitlines():
        s = line.strip()
        if len(keep) >= n:
            break
        if s and 'Warning' in s and s not in keep:
            keep.append(s[:110])
    return keep[:n]


def _verdict(log, label, text, want_name, procs_must_be_zero=True):
    """The two things every structural check here has to say out loud."""
    n = _problem_count(text)
    procs = _processes_left(text)
    names = _stat_names(text)
    ok = True

    log.append(f'    {label}: {n} problem(s) reported by check')
    if n is None:
        log.append('    no check pass ran, so nothing was checked')
        ok = False
    elif n != 0:
        for line in _interesting(text):
            log.append('      ' + line)
        ok = False

    log.append(f'    modules in the final stat: {names or "none"}')
    if want_name is not None and want_name not in names:
        log.append(f'    {want_name} is not among them: the selection '
                   f'matched nothing and the passes above ran on an empty '
                   f'design, which exits successfully and proves nothing')
        ok = False

    log.append(f'    processes left after proc: {procs}')
    if procs_must_be_zero and procs != 0:
        log.append('    proc did not finish, so no claim about latches or '
                   'combinational loops is available')
        ok = False
    return ok


def fpga01(log):
    """No simulator in the board wrapper."""
    files = _fpga_files()
    if not files:
        log.append(f'    no .v files under {FPGA_DIR}/')
        log.append('    an empty set is not a clean set: this check would '
                   'otherwise pass over nothing at all')
        return False, f'nothing under {FPGA_DIR}/ to check'

    findings = []
    for p in files:
        body = E._strip_comments(open(os.path.join(ROOT, p)).read())
        body = re.sub(r'`ifndef\s+SYNTHESIS.*?`endif', '', body, flags=re.S)
        body = re.sub(r'`ifdef\s+SIM.*?`endif', '', body, flags=re.S)
        for rx, what in _SIMISMS:
            for mm in rx.finditer(body):
                tok = mm.group(0)
                if what == 'a system task' and \
                        tok.lower() in _SYNTHESIZABLE_SYSTEM:
                    continue
                findings.append((p, what, tok[:20]))

    log.append(f'    files: {files}')
    log.append(f'    simulator-only constructs: {len(findings)}')
    for f in findings[:6]:
        log.append(f'      {f[0]} {f[1]} {f[2]!r}')
    if findings:
        return False, f'{len(findings)} simulator-only construct(s) in ' \
                      f'{FPGA_DIR}/'
    return True, 'no simulator in the board wrapper'


def fpga02(log):
    """The wrapper elaborates, and its ports connect."""
    files = ' '.join(_library_files())
    ok, text = _run_yosys(
        f'read_verilog -DSYNTHESIS {" ".join(INCS)} {files}\n'
        f'hierarchy -top {TOP} -check\n'
        f'check -noinit\n'
        # So that the verdict below can say the top is in the design
        # rather than only that nothing objected. Without it there is no
        # stat block, and a check that reads one reports the same thing
        # for "clean" and for "ran on nothing".
        f'stat {TOP}\n', log, f'elaborating {TOP}')
    if not ok:
        for line in _interesting(text):
            log.append('      ' + line)
        log.append(f'    {TOP} does not elaborate and connect')
        return False, f'{TOP} does not elaborate and connect'
    ok = _verdict(log, f'hierarchy -top {TOP} -check then check -noinit',
                  text, TOP, procs_must_be_zero=False)
    return ok, ('the board wrapper elaborates and connects' if ok
                else f'{TOP} does not elaborate cleanly')


def fpga03(log):
    """proc, then check -assert, on the wrapper."""
    files = ' '.join(_library_files())
    ok, text = _run_yosys(
        f'read_verilog -DSYNTHESIS {" ".join(INCS)} {files}\n'
        f'hierarchy -top {TOP} -check\n'
        f'select {TOP}\n'
        f'proc\n'
        f'check -assert\n'
        f'stat\n', log, f'proc on {TOP}')
    if not ok:
        for line in _interesting(text):
            log.append('      ' + line)
        return False, f'proc or check -assert failed on {TOP}'
    ok = _verdict(log, f'proc; check -assert on {TOP}', text, TOP)
    return ok, ('no latch, no combinational loop, no multiple driver, no '
                'width problem in the board wrapper' if ok
                else f'check -assert is not clean on {TOP}')


def fpga04(log):
    """Every module in the wrapper, elaborated at the root."""
    ok = True
    for name in STANDALONE:
        src = f'{FPGA_DIR}/{name}.v'
        if not os.path.exists(os.path.join(ROOT, src)):
            log.append(f'    {src} is missing')
            ok = False
            continue
        got, text = _run_yosys(
            f'read_verilog -DSYNTHESIS {" ".join(INCS)} {src}\n'
            f'hierarchy -top {name}\n'
            f'proc\n'
            f'check -assert\n'
            f'memory -nomap\n'
            f'stat\n', log, f'proc on {name}')
        if not got:
            for line in _interesting(text):
                log.append('      ' + line)
            ok = False
            continue
        got = _verdict(log, f'proc; check -assert on {name}', text, name)
        if name == 'tnca_fpga_mem':
            mems = _mem_cells(text)
            log.append(f'    memory cells inferred: {mems}')
            if mems == 0:
                log.append('      this is the failure that does not fail: '
                           'the memory became a list of registers, the '
                           'design elaborates, and the board gets flops '
                           'where block RAM was the point')
                got = False
        ok &= got

    if ok:
        return True, 'the wrapper\'s modules hold at the root, and the ' \
                     'memory infers as a memory'
    return False, 'a module of the board wrapper does not hold on its own'


TESTS = [('R36-FPGA-01', fpga01), ('R36-FPGA-02', fpga02),
         ('R36-FPGA-03', fpga03), ('R36-FPGA-04', fpga04)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print('R3.6 Gate B: the board wrapper, which the export walk cannot '
          'reach')
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f'{type(e).__name__}: {e}'
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
        print(flush=True)
    if args.only and not results:
        print(f'  the --only filter {args.only} matched nothing')
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print(f"  r36-fpga-tests         {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f'{len(results)} check(s)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
