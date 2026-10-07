"""The U50 registers, driven on the real RTL by a simulator.

WHY THIS IS A FILE OF ITS OWN

python/tnca/backends/u50.py talks to an FPGA and must not name a
simulator; tools/r36/backend_tests.py scans the runtime and the compiler
for exactly that word, and the four files allowed to carry it are listed
by name in BINDINGS with a reason beside each. This is a fifth, and the
reason is the same shape as the other four's: the stand-in for a board
is not part of the general code, it is part of a binding.

Keeping it apart also has a practical effect beyond the scan. `u50.py`
stays importable where no simulator exists -- on a host that has an
XRT install and no RTL -- and this module is only imported when
somebody asks for `fpga-u50-sim` by name.

WHAT IT DRIVES

tb/fpga/tnca_fpga_regs.v, which instantiates rtl/fpga/tnca_fpga_top.v
and applies register reads and writes from two files. That is the same
top and the same port the board presents, so the Python above it --
u50.U50Backend, which is what runs on the board -- is exercised rather
than described.

Section 3.4's definition of reference-available is "an implementation
runs here, through this interface, on real hardware or a protocol loop
to it". This is the protocol loop to it. The board is the same backend
with a different link, which is why `fpga-u50` can be reference-
available and `platform-verified` stays reserved for an actual board.
"""
import os
import re
import shutil
import subprocess
import time

from ..errors import DependencyFailed, DeviceLost, TncaError, Timeout
from .u50 import DEFAULT_WORDS, R_MAGIC, U50Backend, U50Link

#: The compiler this repository is pinned to. The Makefile says it in a
#: fingerprint and `device.py` says it in a remedy; neither checks, and
#: this is the binding that cannot afford not to.
PINNED_IVERILOG = 12

#: Where the harness and the RTL sit, relative to the repository root.
_BENCH = os.path.join('tb', 'fpga', 'tnca_fpga_regs.v')
_RTL = [
    os.path.join('rtl', 'fpga', 'tnca_fpga_top.v'),
    os.path.join('rtl', 'fpga', 'tnca_mmio.v'),
    os.path.join('rtl', 'fpga', 'tnca_fpga_mem.v'),
    os.path.join('rtl', 'tnca_shell.v'),
]
_LIBDIRS = [os.path.join('rtl', d)
            for d in ('common', 'core', 'gen', 'nuca')]


def _repo_root():
    """The checkout this package was imported from.

    Three levels up from python/tnca/backends/. Not a search: a stand-in
    that hunted for RTL could find somebody else's, and the failure that
    produces is a device that runs a different program from the one the
    caller compiled.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', '..', '..'))


def _tools():
    """The compiler and the runtime to build and run the harness with.

    Resolved and checked together rather than each being taken from PATH
    at the moment it is needed, because being wrong here does not
    announce itself.

    Every other binding in this repository reads a device that
    `device.prepare()` built once and cached. This one rebuilds its
    harness on EVERY open -- there is no cache -- so it re-reads
    whatever `iverilog` is first on PATH every single time it runs. On a
    shell that had not sourced the environment putting the conda
    toolchain first, that was /usr/bin/iverilog 10.3. It accepted this
    design, warned about nothing, and emitted a binary that never
    advanced simulation time: the host saw a device that had started,
    that answered no transaction, and it reported a thirty-second wait.
    That is exactly what a slow machine looks like, so the toolchain was
    the last thing to suspect. Refused here instead, naming the path and
    the version, because nothing downstream can tell that binary from a
    working one.

    The runtime comes from the same installation as the compiler: a
    `.vvp` file is version-specific bytecode, and a mismatched pair is
    the same silent failure one layer along.
    """
    exe = shutil.which('iverilog')
    if exe is None:
        raise DependencyFailed(
            'iverilog is not on PATH, so the harness cannot be built',
            remedy='the pinned version is 12.0; see the Makefile. It '
                   'is not substituted from elsewhere')
    try:
        probe = subprocess.run([exe, '-V'], capture_output=True, text=True,
                               timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise DependencyFailed(
            f'{exe} would not report its version: {e}',
            remedy='the pinned version is 12.0; see the Makefile') from e
    said = ((probe.stdout or '') + (probe.stderr or '')).splitlines()
    m = re.search(r'version\s+(\d+)\.(\d+)', said[0]) if said else None
    if m is None:
        raise DependencyFailed(
            f'{exe} did not say which version it is: '
            f'{said[0][:80] if said else "no output"}',
            remedy='the pinned version is 12.0; see the Makefile')
    if int(m.group(1)) < PINNED_IVERILOG:
        raise DependencyFailed(
            f'{exe} is version {m.group(1)}.{m.group(2)}, and this '
            f'repository is pinned to {PINNED_IVERILOG}.0',
            detail={'found': exe, 'version': m.group(0)},
            remedy='this binding reads `iverilog` from PATH on every '
                   'open, so an older compiler earlier on PATH is the '
                   'one that gets used. Put the pinned toolchain first, '
                   'as the Makefile\'s targets expect. An older one '
                   'accepts this design and emits a binary that never '
                   'advances simulation time, which reaches a caller as '
                   'a device that started and answers nothing')
    vvp = os.path.join(os.path.dirname(exe), 'vvp')
    if not os.path.exists(vvp):
        vvp = shutil.which('vvp') or 'vvp'
    return exe, vvp


class SimLink(U50Link):
    """One harness process, one transaction at a time, over two files.

    The batching is the whole design. A register write does not go out when
    it is called; it is appended, and the next read flushes the lot in one
    round trip. `U50Backend` writes a descriptor and then reads a response,
    and writes a payload word and then reads a status, so the batches are
    exactly the groups it meant to be atomic -- and a medium that did one
    round trip per register write would spend its whole life waiting for a
    poll interval it did not need.
    """

    def __init__(self, harness=None, *, words=DEFAULT_WORDS, workdir=None,
                 pollcycles=20, intercycle=8, watchdog=200_000_000,
                 host_timeout=30.0, trace=False):
        self.harness = harness
        self.words = words
        self.dir = workdir or os.path.join(_repo_root(), 'sim', 'build',
                                           'u50-sim')
        os.makedirs(self.dir, exist_ok=True)
        self.cmd = os.path.join(self.dir, 'regs.cmd')
        self.rsp = os.path.join(self.dir, 'regs.rsp')
        self.log = os.path.join(self.dir, 'regs.log')
        self.pollcycles = pollcycles
        self.intercycle = intercycle
        self.watchdog = watchdog
        self.host_timeout = host_timeout
        self.trace = trace
        self.proc = None
        self.tid = 0
        self._pending = []
        # Set by `_build` to the runtime belonging to the compiler that
        # produced the harness. A harness the caller supplied is run by
        # whatever `vvp` is on PATH, which is the caller's choice to make
        # -- they named the binary.
        self._vvp = 'vvp'

    # ---- lifetime ---------------------------------------------------------
    def open(self):
        if self.proc is not None:
            raise TncaError('this link already has a device', code='INTERNAL')
        for p in (self.cmd, self.rsp):
            if os.path.exists(p):
                os.unlink(p)
        bench = self.harness or self._build()
        argv = [self._vvp, bench,
                f'+cmd={self.cmd}', f'+rsp={self.rsp}',
                f'+pollcycles={self.pollcycles}',
                f'+intercycle={self.intercycle}',
                f'+watchdog={self.watchdog}']
        if self.trace:
            argv.append('+trace')
        with open(self.log, 'w') as lg:
            self.proc = subprocess.Popen(argv, stdout=lg, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self._await_up()
        return self

    def close(self):
        # A write that was batched and never flushed was a write the caller
        # asked for, so it goes out before the process does.
        if self._pending and self.alive():
            try:
                self._round_trip([])
            except TncaError:
                pass
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def describe(self):
        return 'a protocol loop to the RTL: the same register port on ' \
               'rtl/fpga/tnca_fpga_top.v, driven by a simulator'

    def log_tail(self, n=6):
        try:
            with open(self.log) as fh:
                return fh.read().strip().splitlines()[-n:]
        except OSError:
            return []

    # ---- the registers ----------------------------------------------------
    def read_reg(self, offset):
        # The pending writes go with it: a read is the point at which the
        # caller expects everything it has asked for so far to have happened.
        cmds = self._pending + [f'r {offset}']
        self._pending = []
        vals = self._round_trip(cmds)
        if len(vals) != 1:
            raise DeviceLost(
                f'the harness answered {len(vals)} value(s) to one read',
                detail={'offset': offset, 'epoch': self.proc and self.proc.pid},
                remedy='the harness and this link disagree about the '
                       'transaction format; the response file is the '
                       'place to look')
        return vals[0]

    def write_reg(self, offset, value):
        self._pending.append(f'w {offset} {int(value)}')

    # ---- the round trip ---------------------------------------------------
    def _round_trip(self, cmds):
        self.tid += 1
        tid = self.tid
        tmp = self.cmd + '.tmp'
        with open(tmp, 'w') as fh:
            fh.write(f'{tid}\n')
            for c in cmds:
                fh.write(c + '\n')
        # Written whole and then renamed: the harness would otherwise read
        # half a batch and apply a prefix of it, and a prefix of a
        # descriptor is a request that was never made.
        os.replace(tmp, self.cmd)
        return self._await_tid(tid)

    def _await_tid(self, tid):
        deadline = time.time() + self.host_timeout
        while True:
            got = self._read_rsp()
            if got is not None:
                got_tid, vals = got
                if got_tid == tid:
                    return vals
                if got_tid > tid:
                    raise DeviceLost(
                        f'the harness answered transaction {got_tid} while '
                        f'{tid} was outstanding',
                        detail={'log': self.log, 'tail': self.log_tail()})
                # Older than the one outstanding: the previous answer, not
                # yet overwritten. Same rule as `Transport._await`.
            if not self.alive():
                raise DeviceLost(
                    'the device process is not running',
                    detail={'exit': self.proc.returncode, 'log': self.log,
                            'tail': self.log_tail()},
                    remedy='call Session.recover() to start a new epoch')
            if time.time() > deadline:
                raise Timeout(
                    f'no answer to transaction {tid} after '
                    f'{self.host_timeout}s',
                    detail={'log': self.log, 'tail': self.log_tail()},
                    remedy='this is a HOST wait; the device may still be '
                           'working. It is not a verdict on the task')
            time.sleep(0.001)

    def _read_rsp(self):
        try:
            with open(self.rsp) as fh:
                toks = fh.read().split()
        except OSError:
            return None
        # `end` is what makes a partial write impossible to believe. The
        # harness writes the id, the values and then the marker, so a file
        # that ends without it is a file in progress.
        if len(toks) < 2 or toks[-1] != 'end':
            return None
        try:
            return int(toks[0]), [int(t, 16) for t in toks[1:-1]]
        except ValueError:
            return None

    # ---- bringing the harness up -----------------------------------------
    def _await_up(self):
        """Ask the harness a question, and take the answer as "up".

        Not by reading its log, which was the first attempt and did not
        work. A simulator with its stdout redirected to a file buffers it:
        the line saying the harness was up sat in a buffer while this
        waited for it, so a device that had been answering for thirty
        seconds looked like one that never started. `pgrep` is what settled
        it -- the process was there and the log was empty.

        Asking is also the stronger claim. A process that started is not a
        device that answers, and the register read below is the first thing
        the caller was going to do anyway.
        """
        self.read_reg(R_MAGIC)

    def _build(self):
        """Elaborate the harness against the repository's RTL.

        The file list is written out rather than globbed: a glob would pick
        up a file somebody added under rtl/fpga for another purpose and
        silently change what this device is.
        """
        root = _repo_root()
        for rel in [_BENCH] + _RTL:
            if not os.path.exists(os.path.join(root, rel)):
                raise DependencyFailed(
                    f'{rel} is not in {root}, so there is no RTL to drive',
                    detail={'root': root},
                    remedy='this binding drives the repository\'s RTL. Run '
                           'it from a checkout, or pass harness= a '
                           'prebuilt .vvp')
        iverilog, vvp = _tools()
        self._vvp = vvp
        out = os.path.join(self.dir, 'regs.vvp')
        argv = [iverilog, '-g2005', '-I', os.path.join('rtl', 'common')]
        for d in _LIBDIRS:
            argv += ['-y', d]
        argv += ['-o', out, _BENCH] + _RTL
        r = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                           timeout=1800)
        if r.returncode != 0 or not os.path.exists(out):
            raise DependencyFailed(
                'the harness would not elaborate: '
                + (r.stderr or r.stdout).strip()[-400:],
                detail={'root': root, 'argv': argv})
        return out


def stand_in(**kw):
    """The factory `fpga-u50-sim` is registered with. See backend.py.

    A different NAME from `fpga-u50` and not a fallback under it. Section
    3.4: a missing backend that quietly became the simulator would report
    a platform working that was never there. So the two are asked for by
    name and neither answers for the other, even though what differs
    between them is one constructor argument.
    """
    kw.pop('bench', None)
    link = kw.pop('link', None)
    if link is None:
        link = SimLink(harness=kw.pop('harness', None),
                       words=kw.pop('words', DEFAULT_WORDS),
                       workdir=kw.pop('workdir', None))
    return U50Backend(link, **kw)
