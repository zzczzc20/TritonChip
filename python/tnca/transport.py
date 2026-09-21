"""
The host side of the device transport -- spec/r2_transport_v1.json.

A request/response mailbox with a monotonic sequence number, because that is
the shape a hardware doorbell has: the host writes a descriptor and rings, the
device reads it, acts, and writes a completion. The medium here is two text
files; swapping them for two MMIO registers changes the medium and not the
protocol, which is what R2-10 asks for.

Two rules this module exists to enforce.

**A response carries the sequence it answers.** A response for an older
sequence is a late message -- from a request that timed out, or from a
previous device epoch -- and is discarded rather than believed. ERR-05 is
exactly this case, and a transport that returns "the last thing in the file"
cannot tell the difference.

**Restarting the device is an event, not a detail.** `Transport.start` is
called once per epoch. Nothing in the normal path restarts it: RT-02 wants
thirty-two launches in one session, and a runtime that quietly relaunched the
simulator between them would make "the intermediate result stayed in device
memory" a sentence with no content.
"""
import os
import shutil
import subprocess
import tempfile
import time

from .errors import DeviceLost, TncaError
from . import backend as _backend

OP_UPLOAD = 1
OP_DOWNLOAD = 2
OP_LAUNCH = 3
OP_QUIESCE = 4
OP_SHUTDOWN = 5
OP_PING = 6

ST_OK = 0
ST_FAULT = 1
ST_TIMEOUT = 2
ST_REFUSED = 3

OP_NAME = {OP_UPLOAD: 'UPLOAD', OP_DOWNLOAD: 'DOWNLOAD',
           OP_LAUNCH: 'LAUNCH', OP_QUIESCE: 'QUIESCE',
           OP_SHUTDOWN: 'SHUTDOWN', OP_PING: 'PING'}

STATUS_NAME = {ST_OK: 'OK', ST_FAULT: 'FAULT', ST_TIMEOUT: 'TIMEOUT',
               ST_REFUSED: 'REFUSED'}

WORD = 8


class Reply:
    def __init__(self, seq, status, r0, r1):
        self.seq = seq
        self.status = status
        self.r0 = r0
        self.r1 = r1

    @property
    def ok(self):
        return self.status == ST_OK

    def __repr__(self):
        return (f"<Reply seq={self.seq} {STATUS_NAME.get(self.status, '?')} "
                f"r0={self.r0} r1={self.r1}>")


class Transport(_backend.Backend):
    """One device process, one epoch, reached through a shared directory.

    The binding this repository has always used, now declared through
    the interface every backend answers (phase 3.6 section 4.2) rather
    than being the only thing the runtime knew how to talk to.
    """

    name = 'sim-file'
    state = _backend.REFERENCE_AVAILABLE
    #: A launch is not idempotent. Section 4.2: a timeout must not
    #: resend one, because the kernel would run twice.
    launch_is_idempotent = False

    def __init__(self, bench, *, workdir=None, poll_cycles=20,
                 deadline_cycles=2_000_000, host_timeout=600.0, trace=False,
                 image=None, epoch=0):
        self.bench = bench
        self.dir = workdir or tempfile.mkdtemp(prefix='tnca-dev-')
        os.makedirs(self.dir, exist_ok=True)
        self.req = os.path.join(self.dir, 'req.txt')
        self.rsp = os.path.join(self.dir, 'rsp.txt')
        self.up = os.path.join(self.dir, 'up.hex')
        self.down = os.path.join(self.dir, 'down.hex')
        self.log = os.path.join(self.dir, 'device.log')
        self.poll_cycles = poll_cycles
        self.deadline_cycles = deadline_cycles
        self.host_timeout = host_timeout
        self.trace = trace
        self.image = image
        self.epoch = epoch
        self.seq = 0
        self.proc = None
        self.history = []

    # ---- lifetime ---------------------------------------------------------
    def start(self):
        if self.proc is not None:
            raise TncaError('this transport already has a device',
                            code='INTERNAL')
        for p in (self.req, self.rsp, self.down):
            if os.path.exists(p):
                os.unlink(p)
        argv = ['vvp', self.bench, f'+req={self.req}', f'+rsp={self.rsp}',
                f'+up={self.up}', f'+down={self.down}',
                f'+pollcycles={self.poll_cycles}',
                f'+deadline={self.deadline_cycles}']
        if self.image:
            argv.append(f'+hex={self.image}')
        if self.trace:
            argv.append('+trace')
        self.proc = subprocess.Popen(
            argv, stdout=open(self.log, 'w'), stderr=subprocess.STDOUT,
            start_new_session=True)
        # The device writes sequence 0 when it is up. Waiting for that is the
        # difference between "slow to start" and "never started", and a
        # runtime that cannot tell them apart reports the wrong one.
        self._await(0, what='the device to come up')
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.alive():
                self.request(OP_SHUTDOWN)
        except TncaError:
            pass
        try:
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    # ---- the protocol -----------------------------------------------------
    def request(self, op, a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, *,
                timeout=None):
        if not self.alive():
            raise DeviceLost(
                'the device process is not running',
                detail={'log': self.log, 'epoch': self.epoch},
                remedy='call Session.recover() to start a new epoch')
        self.seq += 1
        seq = self.seq
        tmp = self.req + '.tmp'
        with open(tmp, 'w') as fh:
            fh.write(f"{seq} {op} {a0} {a1} {a2} {a3} {a4} {a5}\n")
        # Written whole, then replaced: a device that polls mid-write would
        # otherwise read half a request and act on it.
        os.replace(tmp, self.req)
        reply = self._await(seq, timeout=timeout)
        self.history.append(dict(epoch=self.epoch, seq=seq, op=op,
                                 a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5,
                                 status=reply.status, r0=reply.r0,
                                 r1=reply.r1))
        return reply

    def _await(self, seq, *, timeout=None, what=None):
        deadline = time.time() + (timeout or self.host_timeout)
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                raise DeviceLost(
                    f"the device exited while waiting for "
                    f"{what or f'sequence {seq}'}",
                    detail={'exit': self.proc.returncode, 'log': self.log,
                            'tail': self._log_tail()})
            got = self._read_reply()
            if got is not None:
                if got.seq == seq:
                    return got
                if got.seq > seq:
                    # Impossible unless something else is writing the
                    # mailbox: a response ahead of the request it answers.
                    raise DeviceLost(
                        f"the device answered sequence {got.seq} while "
                        f"{seq} was outstanding",
                        detail={'log': self.log})
                # got.seq < seq: the previous answer, not yet overwritten.
            if time.time() > deadline:
                raise TncaError(
                    f"no answer to sequence {seq} after "
                    f"{timeout or self.host_timeout}s",
                    code='WAIT_TIMEOUT',
                    detail={'log': self.log, 'tail': self._log_tail()},
                    remedy='this is a HOST wait; the device may still be '
                           'working. It is not a verdict on the task')
            time.sleep(0.002)

    def _read_reply(self):
        try:
            with open(self.rsp) as fh:
                parts = fh.read().split()
        except OSError:
            return None
        if len(parts) < 4:
            return None            # a partial write; look again
        try:
            return Reply(*[int(p) for p in parts[:4]])
        except ValueError:
            return None

    def _log_tail(self, n=6):
        try:
            with open(self.log) as fh:
                return fh.read().strip().splitlines()[-n:]
        except OSError:
            return []

    # ---- operations -------------------------------------------------------
    def upload(self, addr, data):
        """Bytes into device memory, through the device's memory system."""
        if addr % WORD:
            raise TncaError(f"upload address {addr:#x} is not 8-byte aligned",
                            code='ARGUMENT_INVALID')
        padded = bytes(data) + b'\x00' * (-len(data) % WORD)
        words = len(padded) // WORD
        with open(self.up, 'w') as fh:
            for i in range(words):
                fh.write(f"{int.from_bytes(padded[i * WORD:(i + 1) * WORD], 'little'):016x}\n")
        return self.request(OP_UPLOAD, addr, words)

    def download(self, addr, nbytes):
        if addr % WORD:
            raise TncaError(f"download address {addr:#x} is not 8-byte "
                            "aligned", code='ARGUMENT_INVALID')
        words = (nbytes + WORD - 1) // WORD
        reply = self.request(OP_DOWNLOAD, addr, words)
        if not reply.ok:
            return reply, b''
        # The device said OK, so the file it was told to write must be
        # there and must be readable. When it is not, something answered
        # for a request it did not serve -- and the user gets a stable code
        # for that, not a FileNotFoundError from inside the transport.
        out = bytearray()
        try:
            with open(self.down) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out += int(line, 16).to_bytes(WORD, 'little')
        except (OSError, ValueError) as e:
            raise DeviceLost(
                'the device reported a successful download but produced no '
                'readable data',
                detail={'file': self.down, 'error': str(e),
                        'seq': reply.seq, 'log': self.log},
                remedy='the device and the host are out of step; call '
                       'Session.recover() to start a new epoch') from e
        if len(out) < words * WORD:
            raise DeviceLost(
                f"the device returned {len(out)} bytes for a "
                f"{words * WORD}-byte download",
                detail={'file': self.down, 'seq': reply.seq},
                remedy='call Session.recover() to start a new epoch')
        return reply, bytes(out[:nbytes])

    def launch(self, pc, arg_base, role_mask=1, pid=(0, 0, 0),
               npid=(1, 1, 1), timeout=None):
        """All six grid values.

        ISA sec 5.1 makes program_id(axis) the logical index on that axis.
        Sending only x meant a 2x3 grid handed every program the LINEAR id
        under the name of the x index, and a kernel cannot tell.
        """
        px, py, pz = pid
        nx, ny, nz = npid
        a3 = (px & 0xFFFF) | ((py & 0xFFFF) << 16)
        a4 = (pz & 0xFFFF) | ((nx & 0xFFFF) << 16)
        a5 = (ny & 0xFFFF) | ((nz & 0xFFFF) << 16)
        return self.request(OP_LAUNCH, pc, arg_base, role_mask, a3, a4, a5,
                            timeout=timeout)

    def quiesce(self):
        return self.request(OP_QUIESCE)

    def capabilities(self):
        out = super().capabilities()
        out.update({'medium': 'a shared directory', 'bench': self.bench,
                    'reaches': 'the RTL directly, in a child process'})
        return out

    def ping(self):
        return self.request(OP_PING)
