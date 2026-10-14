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
import ctypes
import functools
import os
import operator
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from .errors import DeviceLost, TncaError
from . import backend as _backend


def serialized_io(fn):
    """A mailbox operation owns its sequence and payload through the reply.

    A request-only lock is insufficient: a second upload can overwrite the
    first payload before its request, or a download can consume another
    request's payload. RLock permits upload/download to call request.
    """
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        with self._io_lock:
            return fn(self, *args, **kwargs)
    return call

#: PR_SET_PDEATHSIG, from <linux/prctl.h>: "signal this process when its
#: parent dies". 1 is the option number, not a flag.
_PR_SET_PDEATHSIG = 1


def _arm_parent_death():
    """Ask the kernel to terminate this process when its parent dies.

    Called in the child between fork and exec, so the request is in place
    before the device starts. It survives the exec: the parent-death signal
    is cleared only for set-user-ID, set-group-ID and capability-bearing
    binaries, and a simulator is none of those.

    Best-effort on purpose. A child that could not arm the signal must still
    run -- refusing to start a device because it could not promise to die
    later would trade a leak for an outage, and the leak is the smaller
    problem.
    """
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

OP_UPLOAD = 1
OP_DOWNLOAD = 2
OP_LAUNCH = 3
OP_QUIESCE = 4
OP_SHUTDOWN = 5
OP_PING = 6
# Transport 1.1: read one device counter; r0/r1 = its low/high 32 bits.
OP_PERF_READ = 7
# R40 additive extension; legacy LAUNCH keeps its original 16-bit packing.
OP_GRID_CAP = 8
OP_LAUNCH32 = 9

ST_OK = 0
ST_FAULT = 1
ST_TIMEOUT = 2
ST_REFUSED = 3

OP_NAME = {OP_UPLOAD: 'UPLOAD', OP_DOWNLOAD: 'DOWNLOAD',
           OP_LAUNCH: 'LAUNCH', OP_QUIESCE: 'QUIESCE',
           OP_SHUTDOWN: 'SHUTDOWN', OP_PING: 'PING',
           OP_PERF_READ: 'PERF_READ', OP_GRID_CAP: 'GRID_CAP',
           OP_LAUNCH32: 'LAUNCH32'}

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
        self._io_lock = threading.RLock()

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
        # Debugging only: more plusargs for the bench (e.g. +ctxtrace,
        # +hangdump=N), from the environment, so a kernel run through the
        # whole stack can be traced without another entry point.
        argv += os.environ.get('TNCA_DEVICE_PLUSARGS', '').split()
        # The device is put in its own session so that a signal delivered to
        # the host's process group -- Ctrl-C in a terminal, a supervisor
        # stopping the job -- does not land on it mid-launch. That is right
        # while the host is alive, and it is exactly what lets the device
        # outlive it: `stop` is the only reaper, and a host killed with
        # SIGKILL never reaches `stop`. What is left behind is a simulator
        # polling a mailbox nobody will write to, forever.
        #
        # `+deadline` does not cover this. tb/device/tnca_device.v bounds a
        # single launch with it, and an orphan's problem is that no launch is
        # in progress -- there is nothing for a launch deadline to count.
        #
        # Measured before this line existed: a device with PPID 1 that had
        # been running for 14.5 hours, 8.9 CPU-hours of work, 61.6% of a core,
        # beside 581 abandoned work directories. So the parent asks the kernel
        # to end the child instead, which is the one promise SIGKILL cannot
        # break.
        #
        # `preexec_fn` is documented as unsafe in a process with threads. This
        # runtime is single-threaded host code; if that ever stops being true,
        # this is the line that needs revisiting.
        spawn = {'preexec_fn': _arm_parent_death} if hasattr(os, 'fork') else {}
        with open(self.log, 'w') as log:
            self.proc = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, **spawn)
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

    # ---- the medium -------------------------------------------------------
    #
    # The three below are what a SECOND medium replaces, and they are methods
    # rather than inlined bodies for that reason. The module's own opening
    # note says a doorbell and two MMIO registers "change the medium and not
    # the protocol"; this is where that substitution is made real, and the
    # board binding in backends/u50.py overrides exactly these.
    #
    # The protocol, the sequence bookkeeping, the stale-reply rule and the
    # a3/a4/a5 grid packing above `request` are NOT overridable, because they
    # are not the medium -- they are the protocol, and a second copy of the
    # protocol is how two bindings start answering differently.
    def _put_request(self, seq, op, a0, a1, a2, a3, a4, a5):
        """Command out. One line, written whole and then replaced: a device
        that polls mid-write would otherwise read half a request and act on
        it."""
        tmp = self.req + '.tmp'
        with open(tmp, 'w') as fh:
            fh.write(f"{seq} {op} {a0} {a1} {a2} {a3} {a4} {a5}\n")
        os.replace(tmp, self.req)

    def _put_payload(self, addr, words):
        """Payload out: `words`, to be placed at `addr`.

        The address is carried here and not remembered by the caller,
        because the two mediums address payload differently -- a file holds
        whatever was written to it, and a memory window needs to be told
        where each word goes, one access at a time. A medium that ignored
        the address would be a medium that could only ever fill location
        zero.
        """
        with open(self.up, 'w') as fh:
            for w in words:
                fh.write(f"{w:016x}\n")

    def _get_payload(self, addr, nwords):
        """Payload in: `nwords` words, read from `addr`."""
        out = bytearray()
        with open(self.down) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out += int(line, 16).to_bytes(WORD, 'little')
        return bytes(out)

    # ---- the protocol -----------------------------------------------------
    @serialized_io
    def request(self, op, a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, *,
                timeout=None):
        if not self.alive():
            raise DeviceLost(
                'the device process is not running',
                detail={'log': self.log, 'epoch': self.epoch},
                remedy='call Session.recover() to start a new epoch')
        self.seq += 1
        seq = self.seq
        self._put_request(seq, op, a0, a1, a2, a3, a4, a5)
        reply = self._await(seq, timeout=timeout)
        self.history.append(dict(epoch=self.epoch, seq=seq, op=op,
                                 a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5,
                                 status=reply.status, r0=reply.r0,
                                 r1=reply.r1))
        return reply

    def _gone_detail(self):
        """What to report when the medium stops answering.

        A child process and a board stop for different reasons and the
        useful detail differs: one has an exit status, the other has a
        register that quit moving. Kept beside `_log_tail`, which the board
        overrides as well, so that `_await` has exactly one place to ask
        whether the device is still there.
        """
        if self.proc is None:
            return {'log': self.log}
        return {'exit': self.proc.returncode, 'log': self.log}

    def _await(self, seq, *, timeout=None, what=None):
        deadline = time.time() + (timeout or self.host_timeout)
        while True:
            # `alive()` rather than `self.proc.poll()` inline. The two said
            # the same thing for this binding and the second one cannot be
            # answered by a board, which has no process to poll -- so the
            # wait loop asks the question its subclasses already implement.
            if not self.alive():
                raise DeviceLost(
                    f"the device stopped answering while waiting for "
                    f"{what or f'sequence {seq}'}",
                    detail=dict(self._gone_detail(), tail=self._log_tail()),
                    remedy='call Session.recover() to start a new epoch')
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
    @serialized_io
    def upload(self, addr, data):
        """Bytes into device memory, through the device's memory system."""
        _backend.check_transfer(addr, len(data), what='upload')
        padded = bytes(data) + b'\x00' * (-len(data) % WORD)
        n = len(padded) // WORD
        self._put_payload(addr, [int.from_bytes(padded[i * WORD:(i + 1) * WORD],
                                                'little') for i in range(n)])
        return self.request(OP_UPLOAD, addr, n)

    @serialized_io
    def download(self, addr, nbytes):
        _backend.check_transfer(addr, nbytes, what='download')
        words = (nbytes + WORD - 1) // WORD
        reply = self.request(OP_DOWNLOAD, addr, words)
        if not reply.ok:
            return reply, b''
        # The device said OK, so the payload it was told to produce must be
        # there and must be readable. When it is not, something answered
        # for a request it did not serve -- and the user gets a stable code
        # for that, not a raw OSError from inside the transport.
        try:
            out = self._get_payload(addr, words)
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
        try:
            values = tuple(operator.index(v) for v in (*pid, *npid))
        except TypeError:
            raise ValueError('launch grid values must be integers') from None
        px, py, pz, nx, ny, nz = values
        if any(v < 0 or v > 0xFFFFFFFF for v in values):
            raise ValueError('launch grid values must fit unsigned 32 bits')
        if any(v > 0xFFFF for v in values):
            if self.large_grid_limit() is None:
                raise ValueError('this image does not support 32-bit launch grids')
            return self.request(OP_LAUNCH32, pc, arg_base, role_mask,
                                px | (py << 32), pz | (nx << 32), ny | (nz << 32),
                                timeout=timeout)
        a3 = (px & 0xFFFF) | ((py & 0xFFFF) << 16)
        a4 = (pz & 0xFFFF) | ((nx & 0xFFFF) << 16)
        a5 = (ny & 0xFFFF) | ((nz & 0xFFFF) << 16)
        return self.request(OP_LAUNCH, pc, arg_base, role_mask, a3, a4, a5,
                            timeout=timeout)

    @serialized_io
    def physical_grid_capability(self):
        """Read-only opt-in negotiation; old images explicitly refuse opcode 10."""
        if not hasattr(self, '_physical_grid_capability'):
            reply = self.request(10)
            if reply.status == ST_REFUSED:
                self._physical_grid_capability = None
            elif reply.ok and 1 <= reply.r0 <= 16 and 1 <= reply.r1 <= 0xFFFFFFFF:
                self._physical_grid_capability = (reply.r0, reply.r1)
            else:
                raise DeviceLost('malformed physical grid capability response')
        return self._physical_grid_capability

    def launch_grid(self, pc, arg_base, grid, *, arg_stride=0,
                    active_cores=None, role_mask=1, group_width=1, timeout=None):
        """One hardware command for the whole grid; no host per-program loop."""
        dims = tuple(operator.index(v) for v in grid)
        if len(dims) != 3 or any(v < 0 or v > 0xFFFFFFFF for v in dims):
            raise ValueError('grid must contain three unsigned 32-bit axes')
        pc, arg_base, arg_stride, role_mask = map(operator.index,
                                                (pc, arg_base, arg_stride, role_mask))
        if any(v < 0 or v > 0xFFFFFFFFFFFFFFFF or v % 8 for v in (pc, arg_base, arg_stride)):
            raise ValueError('PC, argument base and stride must be aligned uint64 values')
        group_width=operator.index(group_width)
        if group_width not in (1,2,4):
            raise ValueError('physical program group width must be 1, 2 or 4')
        if group_width>1:
            capability_group=self.physical_chunk_capability()
            if capability_group is None or group_width>capability_group:
                raise ValueError('image does not support this physical chunk width')
        if role_mask < 0 or role_mask > 0xFFFFFFFF or role_mask & 0x0FFFFFF0:
            raise ValueError('role mask must contain only four context bits and slice flags')
        capability = self.physical_grid_capability()
        if capability is None:
            raise ValueError('image does not support physical grid dispatch')
        cores, limit = capability
        mask = (1 << cores) - 1 if active_cores is None else operator.index(active_cores)
        if mask <= 0 or mask >= 1 << cores:
            raise ValueError('active core mask is outside the advertised physical cores')
        gx, gy, gz = dims
        if gx * gy * gz > limit:
            raise ValueError('grid exceeds the advertised streaming budget')
        return self.request(11, pc, arg_base, role_mask | ((group_width.bit_length()-1)<<8),
                            gx | (gy << 32), gz | (mask << 32), arg_stride,
                            timeout=timeout)

    @serialized_io
    def physical_chunk_capability(self):
        if not hasattr(self, '_physical_chunk_capability'):
            reply=self.request(14)
            if reply.status == ST_REFUSED:
                self._physical_chunk_capability=None
            elif reply.ok and reply.r0 & 1 and reply.r1 in (1,2,4):
                self._physical_chunk_capability=reply.r1
            else:
                raise DeviceLost('invalid physical chunk capability')
        return self._physical_chunk_capability

    @serialized_io
    def memory_port_capability(self):
        if not hasattr(self, '_memory_port_capability'):
            reply=self.request(16)
            if reply.status==ST_REFUSED:self._memory_port_capability=None
            elif reply.ok and 1<=reply.r0<=16 and reply.r1==1:self._memory_port_capability=reply.r0
            else:raise DeviceLost('invalid memory outlet capability')
        return self._memory_port_capability

    @serialized_io
    def memory_port_metrics(self):
        ports=self.memory_port_capability()
        if ports is None:raise ValueError('image does not expose memory outlet counters')
        rows=[]
        for port in range(ports):
            reply=self.request(17,port)
            if not reply.ok or not (0<=reply.r0<=0xFFFFFFFFFFFFFFFF and 0<=reply.r1<=0xFFFFFFFFFFFFFFFF):
                raise DeviceLost('invalid memory outlet counters')
            rows.append({'port':port,'read_lines':reply.r0,'write_lines':reply.r1})
        return rows

    @serialized_io
    def physical_grid_metrics(self):
        """Hardware counters for the last accepted launch, retained until the next.

        Occupied cycles count accepted work awaiting core completion, including
        memory stalls; they are not instruction-retirement or utilization rates.
        """
        capability = self.physical_grid_capability()
        if capability is None:
            raise ValueError('image does not support physical grid dispatch')
        rows=[]
        peak=None
        for core in range(capability[0]):
            counts=self.request(12, core)
            activity=self.request(13, core)
            if not counts.ok or not activity.ok:
                raise DeviceLost('physical grid telemetry is unavailable')
            if not (0 <= counts.r1 <= counts.r0 <= 0xFFFFFFFF and
                    0 <= activity.r1 <= 0xFFFFFFFFFFFFFFFF and
                    0 <= activity.r0 <= capability[0]) or (peak is not None and peak != activity.r0):
                raise DeviceLost('invalid physical grid telemetry')
            peak=activity.r0
            rows.append({'core':core,'issued':counts.r0,'completed':counts.r1,
                         'occupied_cycles':activity.r1})
        return {'peak_active':peak,'cores':rows}

    @serialized_io
    def large_grid_limit(self):
        """Query once per backend epoch; old images explicitly refuse op 8."""
        if not hasattr(self, '_large_grid_limit'):
            reply = self.request(OP_GRID_CAP)
            if reply.status == ST_REFUSED:
                self._large_grid_limit = None
            elif reply.ok and reply.r0 == 32 and 4096 < reply.r1 <= 0xFFFFFFFF:
                self._large_grid_limit = reply.r1
            else:
                raise DeviceLost('invalid GRID_CAP response', detail={'reply': repr(reply)})
        return self._large_grid_limit

    def quiesce(self):
        return self.request(OP_QUIESCE)

    def capabilities(self):
        out = super().capabilities()
        out.update({'medium': 'a shared directory', 'bench': self.bench,
                    'reaches': 'the RTL directly, in a child process',
                    # The mailbox carries an opcode field, so a caller
                    # can send one the device does not have and see it
                    # refused.
                    'raw_request': True})
        return out

    def ping(self):
        return self.request(OP_PING)

    def perf_read(self, index):
        """A 64-bit counter, or None from a device without op 7.

        The bench writes r0 and r1 as signed 32-bit integers, so each half
        is masked back to its bits before they are joined.
        """
        reply = self.request(OP_PERF_READ, index)
        if reply.status == ST_REFUSED:
            return None
        if not reply.ok:
            raise TncaError(f"PERF_READ {index} answered {reply!r}",
                            code='PERF_READ')
        return (reply.r0 & 0xFFFFFFFF) | ((reply.r1 & 0xFFFFFFFF) << 32)
