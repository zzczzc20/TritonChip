#!/usr/bin/env python3
"""
R36-FPGA-01 to FPGA-07 -- the board wrapper, gated where nothing else reaches.

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

WHERE THE STRUCTURAL CHECKS STOP, AND WHAT GOES PAST THEM

Everything above is a statement a synthesiser makes, and a synthesiser is
blind to the one thing rtl/fpga/tnca_axi_lite.v is: a two-clock adapter
whose whole job is that the registers on either side of the crossing never
see the same edge. `proc; check -assert` will happily pass a design whose
handshake loses every other request, and the Alveo's bus clock and the
device clock are not synchronised and never will be -- so the property that
matters is not structural and cannot be reached from yosys.

R36-FPGA-05 is therefore a simulation rather than an elaboration:
tb/fpga/tb_tnca_axi_lite.v against eleven clock ratios, each of which must
say PASS. The bench is self-checking -- it counts strobes, watches for a
`reg_we` that lasts longer than a cycle, and asserts that the read/write
collision it exists to arbitrate actually happened -- so PASS is the whole
verdict and this file does not re-derive it.

  The ratio that matters is 1:8. The platform clock is 300 MHz and the
  device clock is taken from it through a divide-by-eight, so the bench
  at 2 ns and 16 ns is the pair the board would run. The other ten exist
  because a crossing that is correct at one ratio is often correct only
  there: an integer ratio hides a metastable window that a 4:5 ratio
  exposes, and a self-limiting handshake that stalls when the slow side
  is a hundred times slower has not been shown to work, only to work
  here. The bench holds the bus clock faster than the device clock
  throughout, which is not a gap in the sweep but the design's
  precondition -- the toggle it crosses on is only guaranteed to be
  observed if the sampling side is the faster one.

WHY R36-FPGA-06 EXISTS: A PASSING BENCH IS NOT YET EVIDENCE

The bench at R36-FPGA-05 passed the first time it was run, and the
arbitration it was written to test could be deleted from the design
without turning it red. The reason was that the sweep put its delay on the
write side, and the write has a fixed address-to-toggle latency, so the
read and the write never landed in the same device cycle no matter where
the phase was placed: the case being tested was not reachable from the
test. Nothing about the PASS line said so.

R36-FPGA-06 breaks six things in rtl/fpga/tnca_axi_lite.v, one at a time,
in a copy, and requires the bench to fail on each. A mutation the bench
does not notice is reported as a failure of R36-FPGA-06 rather than of
the mutation, because a bench that has stopped watching a thing is the
more dangerous of the two. It also rebuilds the unmutated file in the same
pass and requires PASS, since a mutation result means nothing without the
control beside it -- and the mutations are written as (old, new) string
pairs that must each match exactly once, so that a rename in the design
makes this gate red instead of silently mutating nothing.

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
# Shorter, because the benches have a watchdog of their own that ends them
# in simulation time. This is the backstop for the other kind of hang: a
# design that spins in zero simulated time never reaches the watchdog, and
# iverilog 10.3 was measured producing exactly that. Multiplied by the
# seventeen runs across R36-FPGA-05 and -06 it is still bounded.
TB_TIMEOUT = 120
TOP = 'tnca_fpga_top'
FPGA_DIR = os.path.join('rtl', 'fpga')
INCS = ['-I rtl/common', '-I rtl/noc']
RTLLIB = ['common', 'core', 'gen', 'nuca', 'noc', 'tile']

# The modules that carry the wrapper's own logic, each elaborated at the
# root. The wrapper itself is checked separately because it needs the
# whole library behind it.
STANDALONE = ['tnca_mmio', 'tnca_fpga_mem', 'tnca_axi_lite']

# The bench that drives the adapter from the bus side, and the file it
# drives. Named here rather than globbed: a second bench under tb/fpga/
# is not a reason for this check to start claiming something about it.
AXIL_TB = os.path.join('tb', 'fpga', 'tb_tnca_axi_lite.v')
AXIL_DUT = os.path.join(FPGA_DIR, 'tnca_axi_lite.v')

# (FAST_HALF, SLOW_HALF) in nanoseconds -- the two half-periods the bench
# elaborates with, so the clock ratio is SLOW_HALF/FAST_HALF. 1:8 is the
# board's; the rest are the awkward ones around it.
AXIL_RATIOS = [(1, 1), (1, 2), (1, 3), (1, 8), (3, 24), (5, 8),
               (2, 3), (1, 13), (1, 25), (4, 5), (1, 40)]
AXIL_BOARD_RATIO = (1, 8)

# The bench that drives the whole top -- adapter, join, second clock, reset
# ordering -- with the real device behind it, and the file it drives. The
# adapter bench above stops at a stub register file; tnca_fpga_top's bench
# never crosses a clock. This is the layer between them, and it had no
# bench at all until this check existed.
U50TOP_TB  = os.path.join('tb', 'fpga', 'tb_tnca_u50_top.v')
U50TOP_DUT = os.path.join(FPGA_DIR, 'tnca_u50_top.v')

# Two ratios. 1:8 is the board's -- ap_clk divided by BUFGCE_DIV's 8 -- and is
# where the suite runs. 1:1 is here because it is not a reduction of that but a
# different corner: with equal half-periods the two clocks are not merely
# nearby, they are the same edge, which is the degenerate case a divided clock
# approaches at the limit and the one a handshake that needs phase margin
# would fail at. Both must pass.
U50TOP_RATIOS = [(1, 8), (1, 1)]

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
    """The lines worth putting in the log when something said no.

    Matched without regard to case because the two tools disagree about
    it: yosys writes `ERROR:` and iverilog writes `error:`, and a filter
    that knew only the first would report a build failure as an empty
    list of reasons.
    """
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if s and 'error' in s.lower():
            keep.append(s[:110])
    for line in text.splitlines():
        s = line.strip()
        if len(keep) >= n:
            break
        if s and 'warning' in s.lower() and s not in keep:
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


#------------------------------------------------------------------------------
# The two that simulate
#------------------------------------------------------------------------------

# Outside the f-strings below, and not inline in them: an f-string that
# nests the same quote character is a syntax error before Python 3.12,
# and the Makefile asks for 3.9.
_VERDICT = {True: 'PASS', False: 'FAIL', None: 'NO VERDICT'}


def _require_iverilog(log):
    """Whether there is a simulator, with the reason in the log if not.

    A missing iverilog is a failure and not a skip, for the reason
    _run_yosys gives about yosys: the claim the next two checks make is a
    claim about what a simulation did.
    """
    if shutil.which('iverilog') is None or shutil.which('vvp') is None:
        log.append('    iverilog is not installed, so this check did NOT RUN')
        log.append('    that is a failure and not a skip: whether a request '
                   'survives the clock crossing, or a strobe is emitted '
                   'once, or the read data that comes back is the read data '
                   'that left, is a statement about a simulation, and '
                   'nothing has established any of it here')
        return False
    return True


def _run_axil(tmp, dut, tag, fast_half, slow_half):
    """Build and run the adapter's bench. Returns (verdict, detail).

    verdict is True, False, or None -- and None is a third answer on
    purpose. A bench that neither passed nor failed, or that ran out of
    wall clock, has not reached a conclusion; folding that into False
    would let a bench which stopped early be reported as a bench which
    found something.

    The verdict is the summary line and not the first line to contain
    "FAIL": the bench prints one "FAIL" per failing check as it goes, and
    those are what the count in the summary is made of. Matching on the
    prefix is what separates the two.
    """
    vvp = os.path.join(tmp, f'{tag}_{fast_half}_{slow_half}.vvp')
    r = subprocess.run(
        ['iverilog', '-g2005', '-I', 'rtl/common',
         f'-Ptb_tnca_axi_lite.FAST_HALF={fast_half}',
         f'-Ptb_tnca_axi_lite.SLOW_HALF={slow_half}',
         '-o', vvp, AXIL_TB, dut],
        cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        return None, ('it does not build: '
                      + ' | '.join(_interesting(r.stdout + r.stderr, 2)))

    try:
        r = subprocess.run(['vvp', vvp], cwd=ROOT, capture_output=True,
                           text=True, timeout=TB_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, (f'the simulation did not finish in {TB_TIMEOUT}s, '
                      f'which is a hang and not a result')
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith('FAIL tb_tnca_axi_lite'):
            return False, line.strip()
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith('PASS tb_tnca_axi_lite'):
            return True, line.strip()
    return None, ('the bench ended without a verdict: '
                  + ' | '.join(_interesting(r.stdout + r.stderr, 2)))


def _axil_missing(log):
    for p in (AXIL_TB, AXIL_DUT):
        if not os.path.exists(os.path.join(ROOT, p)):
            log.append(f'    {p} is missing')
            log.append('    an absent bench is not a passing one')
            return p
    return None


def _run_u50top(tmp, tag, fast_half, slow_half, dut=None):
    """Build and run the whole-top bench. Returns (verdict, detail).

    The file list is the whole library, which is the same set R36-FPGA-04
    elaborates -- tnca_u50_top needs the adapter, the device, and everything
    the device instantiates behind it. `dut` is a replacement for
    rtl/fpga/tnca_u50_top.v, which is dropped from the list rather than passed
    alongside it: two files defining one module is a build error, and a
    mutation test that fails to build looks exactly like a mutation that was
    caught.
    """
    vvp = os.path.join(tmp, f'{tag}_{fast_half}_{slow_half}.vvp')
    if dut is None:
        files = _library_files()
    else:
        files = [f for f in _library_files() if f != U50TOP_DUT]
        files.append(dut)
    r = subprocess.run(
        ['iverilog', '-g2005', '-I', 'rtl/common', '-I', 'rtl/noc',
         f'-Ptb_tnca_u50_top.FAST_HALF={fast_half}',
         f'-Ptb_tnca_u50_top.SLOW_HALF={slow_half}',
         '-o', vvp, U50TOP_TB] + files,
        cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        return None, ('it does not build: '
                      + ' | '.join(_interesting(r.stdout + r.stderr, 2)))

    try:
        r = subprocess.run(['vvp', vvp], cwd=ROOT, capture_output=True,
                           text=True, timeout=TB_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, (f'the simulation did not finish in {TB_TIMEOUT}s, '
                      f'which is a hang and not a result')
    lines = (r.stdout + r.stderr).splitlines()

    # The per-check failures, which are NOT the verdict. tb_tnca_axi_lite.v
    # prints "FAIL <what>: got ..." as it goes and one summary line at the end,
    # and `_run_axil` above matches on the prefix for exactly that reason. The
    # same split is used here, but the check lines are kept rather than dropped,
    # because a bench that fails three checks and then hangs on the fourth
    # reports the hang as its last line -- and a mutation whose evidence is
    # "the payload window read back a value with its halves swapped" should not
    # be recorded as "the watchdog fired", which is a weaker and less useful
    # statement about the same design.
    checks = [l.strip() for l in lines
              if l.startswith('FAIL ') and not l.startswith('FAIL tb_')]

    for line in lines:
        if line.startswith('FAIL tb_tnca_u50_top'):
            if not checks:
                return False, line.strip()
            return False, (f'{checks[0]} -- and the bench then ended with '
                           f'"{line.strip()}" ({len(checks)} check(s) failed)')
    for line in lines:
        if line.startswith('PASS tb_tnca_u50_top'):
            return True, line.strip()
    return None, ('the bench ended without a verdict: '
                  + ' | '.join(_interesting(r.stdout + r.stderr, 2)))


def fpga05(log):
    """Eleven clock ratios, each of which must pass."""
    if not _require_iverilog(log):
        return False, 'iverilog is not installed, so nothing was simulated'
    gone = _axil_missing(log)
    if gone:
        return False, f'{gone} is missing'

    bad = []
    log.append('    bus/device   bus clock      device clock   verdict')
    with tempfile.TemporaryDirectory(prefix='tnca-axil-') as tmp:
        for fast_half, slow_half in AXIL_RATIOS:
            ok, why = _run_axil(tmp, AXIL_DUT, 'dut',
                                fast_half, slow_half)
            board = '' if (fast_half, slow_half) != AXIL_BOARD_RATIO \
                else '  <- the board\'s ratio'
            log.append(
                f'    {fast_half:>3d}:{slow_half:<3d}   '
                f'{1000.0 / (2 * fast_half):8.3f} MHz   '
                f'{1000.0 / (2 * slow_half):8.3f} MHz   '
                f'{_VERDICT[ok]}{board}')
            if ok is not True:
                log.append(f'        {why}')
                bad.append(f'{fast_half}:{slow_half}')

    log.append(f'    ratios passed: {len(AXIL_RATIOS) - len(bad)} '
               f'of {len(AXIL_RATIOS)}')
    log.append('    these are ratios and not frequencies: the bench runs at '
               'nanosecond half-periods, so the absolute rate differs from '
               'the board\'s by the same factor on both sides')
    if bad:
        return False, f'the adapter fails at {len(bad)} clock ratio(s): ' \
                      f'{", ".join(bad)}'
    return True, f'the adapter holds at {len(AXIL_RATIOS)} clock ratios, ' \
                 f'including the board\'s 1:8'


# The six, each as (name, old, new). Every `old` must occur exactly once
# in rtl/fpga/tnca_axi_lite.v: a site that has been renamed makes this
# gate red rather than silently mutating nothing, which is the failure
# mode a mutation table has and the reason `patch` in mutation_tests.py
# is written the way it is.
MUTANTS = [
    ('the read and the write are not arbitrated',
     'wire rd_take = rd_pending && !reg_we_q;',
     'wire rd_take = rd_pending;'),
    ('reg_we is held rather than pulsed',
     'reg_we_q <= wr_pulse;',
     "if (wr_pulse) reg_we_q <= 1'b1;"),
    ('every bus write completes a register write',
     'if (awaddr_q[2]) begin',
     "if (1'b1) begin"),
    ('the low half is not carried into the high one',
     'wr_data_f <= {wdata_q, shadow_lo};',
     'wr_data_f <= {wdata_q, wdata_q};'),
    ('the read address never crosses',
     'rd_addr_q <= rd_addr_f;',
     'rd_addr_q <= rd_addr_q;'),
    # The mux that picks the half moved to the device's side of the crossing
    # (see the declaration of `rd_data_s`), so this one now holds the register
    # that actually crosses. It is the same mutation it always was.
    ('the read data never crosses',
     'rd_data_s  <= reg_rdata_sel;',
     'rd_data_s  <= rd_data_s;'),
]


def fpga06(log):
    """Break the adapter six ways; the bench must notice each one."""
    if not _require_iverilog(log):
        return False, 'iverilog is not installed, so nothing was simulated'
    gone = _axil_missing(log)
    if gone:
        return False, f'{gone} is missing'

    text = open(os.path.join(ROOT, AXIL_DUT)).read()
    uncaught = []
    with tempfile.TemporaryDirectory(prefix='tnca-axil-mut-') as tmp:
        # The control, run here rather than assumed to have been run by
        # R36-FPGA-05: `--only` exists, and a mutation result means
        # nothing without the pass it is a departure from.
        ok, why = _run_axil(tmp, AXIL_DUT, 'real', *AXIL_BOARD_RATIO)
        log.append(f'    unmutated {AXIL_DUT} at '
                   f'{AXIL_BOARD_RATIO[0]}:{AXIL_BOARD_RATIO[1]}: '
                   f'{_VERDICT[ok]}  {why}')
        if ok is not True:
            log.append('    without that pass nothing below can be read as '
                       'a catch: a bench that fails on everything catches '
                       'nothing, and it is the more likely of the two')
            return False, 'the bench does not pass on the unmutated file'

        for i, (name, old, new) in enumerate(MUTANTS):
            n = text.count(old)
            if n != 1:
                log.append(f'    [{name}] the site occurs {n} time(s), not '
                           f'once, so the mutation did not apply')
                log.append('      nothing was broken and so nothing was '
                           'caught, which is not the same as a catch')
                uncaught.append(name)
                continue
            path = os.path.join(tmp, f'mut{i}.v')
            with open(path, 'w') as fh:
                fh.write(text.replace(old, new, 1))
            ok, why = _run_axil(tmp, path, f'mut{i}', *AXIL_BOARD_RATIO)
            if ok is False:
                log.append(f'    [{name}] caught: {why}')
            elif ok is True:
                log.append(f'    [{name}] NOT CAUGHT: the bench passed on '
                           f'a design where {name}')
                log.append('      the bench has stopped watching this, '
                           'which is a failure of the bench and not of '
                           'the mutation')
                uncaught.append(name)
            else:
                log.append(f'    [{name}] no verdict: {why}')
                uncaught.append(name)

    if uncaught:
        return False, f'{len(uncaught)} of {len(MUTANTS)} mutation(s) ' \
                      f'went uncaught'
    return True, f'all {len(MUTANTS)} mutations of the adapter are caught'


# Four, each as (name, old, new), all against rtl/fpga/tnca_u50_top.v. Every
# `old` must occur exactly once in that file -- and `.rst_n_async(...)` is the
# anchor that makes the last two unique, because rtl/fpga/tnca_u50_top.v
# connects the same five register wires to both of the blocks it joins, and the
# device instance is the only one preceded by that line.
#
# The first two are the ordering this file exists to establish, attacked from
# each end: one makes the bus side lead, the other shortens the device side's
# head start so that the ordering still holds between the two reset wires while
# the device itself is not yet running. The second is the one a check written
# against dev_rst_n alone would miss, which is why the bench watches
# u_dut.u_device.rst_n as well.
U50TOP_MUTANTS = [
    ('the bus side is released straight from the asynchronous reset',
     'dev_ready_axi <= {dev_ready_axi[0], dev_rst_n};',
     'dev_ready_axi <= {dev_ready_axi[0], rst_n_async};'),
    ('the device side leads by one clk_dev edge instead of four',
     'wire dev_rst_n = dev_rst_sync[3];',
     'wire dev_rst_n = dev_rst_sync[0];'),
    ('the write data crosses the join with its halves swapped',
     '        .rst_n_async(rst_n_async),\n\n'
     '        .reg_addr(reg_addr),\n'
     '        .reg_wdata(reg_wdata),',
     '        .rst_n_async(rst_n_async),\n\n'
     '        .reg_addr(reg_addr),\n'
     '        .reg_wdata({reg_wdata[31:0], reg_wdata[63:32]}),'),
    ('the device is addressed at a shifted address',
     '        .rst_n_async(rst_n_async),\n\n'
     '        .reg_addr(reg_addr),',
     '        .rst_n_async(rst_n_async),\n\n'
     '        .reg_addr(reg_addr >> 1),'),
]


def fpga07(log):
    """The whole top, with the real device behind it, and four breakages."""
    if not _require_iverilog(log):
        return False, 'iverilog is not installed, so nothing was simulated'
    for p in (U50TOP_TB, U50TOP_DUT):
        if not os.path.exists(os.path.join(ROOT, p)):
            log.append(f'    {p} is missing')
            log.append('    an absent bench is not a passing one')
            return False, f'{p} is missing'

    text = open(os.path.join(ROOT, U50TOP_DUT)).read()
    with tempfile.TemporaryDirectory(prefix='tnca-u50top-') as tmp:
        # The control, run here rather than assumed to have been run: every
        # mutation result below is a departure from it, and a bench that fails
        # on everything catches nothing.
        for fast_half, slow_half in U50TOP_RATIOS:
            ok, why = _run_u50top(tmp, 'real', fast_half, slow_half)
            board = '  <- the board\'s ratio' \
                if (fast_half, slow_half) == AXIL_BOARD_RATIO else ''
            log.append(f'    unmutated at {fast_half}:{slow_half}: '
                       f'{_VERDICT[ok]}{board}')
            log.append(f'      {why}')
            if ok is not True:
                log.append('    without that pass nothing below can be read '
                           'as a catch')
                return False, f'the top does not pass at ' \
                              f'{fast_half}:{slow_half}'

        uncaught = []
        for i, (name, old, new) in enumerate(U50TOP_MUTANTS):
            n = text.count(old)
            if n != 1:
                log.append(f'    [{name}] the site occurs {n} time(s), not '
                           f'once, so the mutation did not apply')
                log.append('      nothing was broken and so nothing was '
                           'caught, which is not the same as a catch')
                uncaught.append(name)
                continue
            path = os.path.join(tmp, f'mut{i}.v')
            with open(path, 'w') as fh:
                fh.write(text.replace(old, new, 1))
            ok, why = _run_u50top(tmp, f'mut{i}', *AXIL_BOARD_RATIO,
                                  dut=path)
            if ok is False:
                log.append(f'    [{name}] caught: {why}')
            elif ok is True:
                log.append(f'    [{name}] NOT CAUGHT: the bench passed on '
                           f'a design where {name}')
                log.append('      the bench has stopped watching this, '
                           'which is a failure of the bench and not of '
                           'the mutation')
                uncaught.append(name)
            else:
                log.append(f'    [{name}] no verdict: {why}')
                uncaught.append(name)

    if uncaught:
        return False, f'{len(uncaught)} of {len(U50TOP_MUTANTS)} ' \
                      f'mutation(s) went uncaught'
    return True, f'the whole top holds at ' \
                 f'{len(U50TOP_RATIOS)} ratios and all ' \
                 f'{len(U50TOP_MUTANTS)} of its breakages are caught'


TESTS = [('R36-FPGA-01', fpga01), ('R36-FPGA-02', fpga02),
         ('R36-FPGA-03', fpga03), ('R36-FPGA-04', fpga04),
         ('R36-FPGA-05', fpga05), ('R36-FPGA-06', fpga06),
         ('R36-FPGA-07', fpga07)]


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
