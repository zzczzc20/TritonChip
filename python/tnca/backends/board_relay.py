"""The R41 board behind a Unix socket: a trusted relay owns the card, a remote
host side drives it link by link.

WHY

An agent's candidate is compiled and run by its own host side (the compiler
and this package) on another machine. Its measurements still have to come from
the shared U280, and no candidate-written code may execute on the card host.
So the card host runs only this file and the session that starts it
(tools/tritonchip_fpga_harness/board_session.py), both from the platform's own
checkout. The relay opens the board link once, holds the card's exclusive XRT
context for the whole card session, and executes link-level operations -- read
a register, write a register, open/read/write the HBM buffer -- requested over
a Unix socket, which the harness forwards over SSH. Everything above the link
(the mailbox protocol, payload windows, launches, the striped layout's checks)
is the candidate's `MulticoreMMIOBackend`, unchanged, on the far side.

    python -m tnca.backends.board_relay --xclbin X [--index N | --bdf BDF]
        --unix SOCKET --observe-dir DIR [--label-file FILE] [--stop FILE]
        [--idle-timeout S] [--deadline-seconds S] [--drain-seconds S]
        [--hbm-bytes N] [--summary FILE] [--ip-name NAME] [--words N]

It prints `listening SOCKET` once the board is open, its identity register
answers and (with --hbm-bytes) the HBM buffer is allocated and cleared, and a
short `stopped {json}` on the way out; --summary gets the full account, every
connection with its label and client label. It exits 0 when stopped as asked,
3 when the device stopped answering (DEVICE_LOST: an XRT register access or
DMA failed), 4 on anything else unexpected (INTERNAL: the relay's own fault
until shown otherwise) and 2 when it never started. It serves many connections, one
at a time, and ends on the first of: the stop file appears, `--idle-timeout`
seconds pass with no connection, `--deadline-seconds` pass, or the device is
lost. The link is closed and the image left loaded.

One connection at a time: a connection that arrives while another is being
served waits up to ten seconds for it to end (a client that has just exited
is usually only an unread EOF away), and is refused after that, as the board's
exclusive context would refuse a second local process.

THE PROTOCOL

relay.py's framing: `struct.pack('<II', len(head), len(payload)) + head +
payload`, head JSON. Replies are `{"ok": true, ...}` or `{"ok": false,
"error": ..., "code": ...}` (a tnca error code). Ops:

    hello        {label}            -> protocol, label, client_label, device, image
    read         {offset}           -> value
    host_status  {}                 -> value (the payload window's status)
    write        {}                 -> nothing (carries only `writes`)
    hbm_open     {nbytes}           -> base, size, allocation, layout
    hbm_write    {offset} + payload -> nothing
    hbm_read     {offset, nbytes}   -> payload
    bye          {}                 -> nothing; the connection ends

Any op may carry `writes: [[offset, value], ...]`, performed in order before
the op itself. `RelayLink.write_reg` returns nothing to its caller, so the
client holds its register writes and sends them with the next operation that
needs an answer: a request's ten descriptor writes and its doorbell travel with
the first poll of its response, a payload word's address, data and start with
its status read. Every operation crosses an SSH tunnel at a few milliseconds.

Register offsets must be 8-byte aligned and below 0x200: the AXI-Lite slave
decodes nine address bits (rtl/board/tnca_multicore_hbm_xo.v), so 0x248 would
be the doorbell under another name and pass by the observer.

WHOSE CONNECTION IT IS

The label a connection's launches are filed under is read from --label-file
when the connection is accepted. The harness writes that file over ssh between
point containers, when no client is connected; the hello's own label is kept
only as `client_label`, because candidate code can reconnect with any label it
likes. A missing file or one that is not [A-Za-z0-9_-]{1,32} is `unlabelled`.

When an accepted connection's label differs from the previous connection's
and the HBM buffer exists, the buffer is cleared before the connection is
served: once the doorbell register reads idle (waited for up to
--drain-seconds), the window registers are set to the whole buffer and the
device's caches hand all of it back (the backend's own cache handoff request,
which writes back and invalidates every line), then the whole buffer is
written with zeros by DMA, 16 MiB at a time. It is cleared the same
way once when it is first allocated. Nothing a point (or an earlier user of
the card) left in device memory reaches the next point. A connection whose
clearing cannot be done is refused, never served on a dirty buffer.

THE OBSERVER

What the evaluator scores is written here, by the relay, to
`<observe-dir>/<label>.log`, from rtl/fpga/tnca_multicore_mmio.v's registers:

  * the relay keeps its own watermark of `served` (0xa0), moved only by
    requests it saw admitted, its own handoffs included. Before each
    operation, around each doorbell and at connection end it reads `served`:
    an advance that matches the ring it forwarded (req_seq and req_op read as
    it was rung) is that request, and if req_op was 11 a launch is pending,
    its id = served, under the connection's label; any other advance is an
    `unaccounted served=<old>-><new>` line. A ring whose advance a read right
    after it does not show yet is settled at the next operation, not taken as
    dropped;
  * before serving each operation, before each doorbell, and at connection
    end, a pending launch whose id equals `launch_sequence` (0xf8) has
    completed, and its line is appended:

        cmd op=1 seq=<id> obs=<launch_end-launch_start> done=<rsp_r1:rsp_r0> status=<rsp_status>

  * at connection end a pending launch is waited for, polled for up to
    --drain-seconds, before `end launches=<n>` closes the connection's record.

The fields are tb/fpga/tnca_multicore_board_regs.v's simulation observer's.
There are no settle/quiet/rd/wr fields: the board has no observation of its
cores draining or of its outlets' transactions in flight, and a field the
board cannot measure is left out, not written as zero. Anything that is not a
clean observation is a line of its own, never a `cmd` line: `unfinished seq=`
(still running when the drain ran out), `refused seq= status=` (answered
without starting), `lost seq=` (its response registers no longer answer it),
`unaccounted served=`, `uncleared connection=` and
`refused-concurrent connection=`. The harness
rejects a point whose log holds any of them. The client cannot write the launch
registers (the control block has no write path to them) and cannot ring a
doorbell except through this relay, so the record is the platform's.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import select
import signal
import socket
import struct
import sys
import time

from ..errors import (ALL_CODES, ArgumentError, DependencyFailed, DeviceLost,
                      Timeout, TncaError)
from ..hbm_layout import StripedHBMLayout
from .u50 import (AXI_STAT_PRESENT, DEFAULT_WORDS, IP_NAME, R_AXI_BASE, R_AXI_SIZE,
                  R_AXI_STAT, R_DOORBELL, R_MAGIC, R_REQ_A0, R_REQ_A1, R_REQ_A2,
                  R_REQ_A3, R_REQ_A4, R_REQ_A5, R_REQ_OP, R_REQ_SEQ, R_RSP_R0,
                  R_RSP_R1, R_RSP_SEQ, R_RSP_STATUS, R_SERVED, U50Link)

PROTOCOL = 1

# rtl/fpga/tnca_multicore_mmio.v's launch timestamps (U50Backend.
# launch_timestamps reads the same three), its physical-grid launch opcode
# and its cache handoff (multicore_mmio.OP_CACHE_HANDOFF).
R_LAUNCH_VERSION = 0xE0
R_LAUNCH_START = 0xE8
R_LAUNCH_END = 0xF0
R_LAUNCH_SEQUENCE = 0xF8
OP_PHYSICAL_LAUNCH = 11
OP_CACHE_HANDOFF = 18

#: The largest HBM transfer one message carries, and the piece the relay
#: clears the buffer in. The client splits bigger transfers; the relay refuses
#: bigger ones, since it owns memory on a shared host.
CHUNK = 1 << 24
MAX_HEAD = 1 << 20
#: Nine decoded address bits (rtl/board/tnca_multicore_hbm_xo.v ADDR_BITS(9)):
#: every offset at or above this aliases one below it, the doorbell included.
MAX_OFFSET = 0x200
#: How long a connection that arrived during another may wait for it to end.
BACKLOG_SECONDS = 10.0

_LABEL = re.compile(r'[A-Za-z0-9_-]{1,32}')
_M64 = (1 << 64) - 1
_M32 = (1 << 32) - 1


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------
def _send(sock, obj, payload=b''):
    head = json.dumps(obj, separators=(',', ':')).encode()
    sock.sendall(struct.pack('<II', len(head), len(payload)) + head)
    if len(payload):
        sock.sendall(payload)


def _recv_exactly(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:])
        if not k:
            return None
        got += k
    return bytes(buf)


def _recv(sock, *, max_head=MAX_HEAD, max_payload=CHUNK):
    """(head, payload), or (None, None) on a close at a message boundary."""
    first = _recv_exactly(sock, 8)
    if first is None:
        return None, None
    hl, pl = struct.unpack('<II', first)
    if hl > max_head or pl > max_payload:
        raise ArgumentError(f'a message of {hl}+{pl} bytes exceeds {max_head}+{max_payload}')
    head = _recv_exactly(sock, hl)
    payload = _recv_exactly(sock, pl) if pl else b''
    if head is None or payload is None:
        raise ArgumentError('the connection closed inside a message')
    obj = json.loads(head.decode())
    if not isinstance(obj, dict):
        raise ArgumentError('a message head is a JSON object')
    return obj, payload


def _int(head, key, lo, hi):
    v = head.get(key)
    if type(v) is not int or not lo <= v < hi:
        raise ArgumentError(f'{key}={v!r} is not an integer in [{lo:#x}, {hi:#x})')
    return v


def _offset(value):
    if type(value) is not int or not 0 <= value < MAX_OFFSET or value % 8:
        raise ArgumentError(f'register offset {value!r} is not an aligned control-block address '
                            f'below {MAX_OFFSET:#x}')
    return value


def _label(text):
    return text if isinstance(text, str) and _LABEL.fullmatch(text) else None


# ---------------------------------------------------------------------------
# the server: trusted, on the card host
# ---------------------------------------------------------------------------
class _Connection:
    def __init__(self, sock, number, label, log):
        self.sock, self.number, self.label, self.log = sock, number, label, log
        self.client_label = None
        self.launches = self.anomalies = self.ops = 0
        self.pending = None
        self.doorbell = None             # a ring whose admission is not settled yet
        self.cleared_seconds = None
        self.started = time.monotonic()

    def line(self, text, *, anomaly=False):
        self.log.write(text + '\n')
        self.log.flush()
        self.anomalies += anomaly

    def record(self):
        return dict(number=self.number, label=self.label, client_label=self.client_label,
                    ops=self.ops, launches=self.launches, anomalies=self.anomalies,
                    cleared_seconds=self.cleared_seconds,
                    seconds=round(time.monotonic() - self.started, 1))


class BoardRelay:
    """One open board link, served to one connection at a time."""

    def __init__(self, link, observe_dir, image, *, label_file=None, drain_seconds=120.0):
        self.link = link
        self.observe_dir = observe_dir
        self.image = image
        self.label_file = label_file
        self.drain_seconds = drain_seconds
        self.hbm = None
        self.used_since_clear = False
        self.fatal = self.fatal_code = None
        self.previous_label = None
        self.connections = []
        self.refused = 0
        self.clears = []
        # The relay's own account of the device's `served`: moved only by
        # requests the relay saw admitted, its own handoffs included.
        self.served = link.read_reg(R_SERVED)

    def fail(self, error):
        """End the card session over `error`, keeping the first cause: a
        DEVICE_LOST is the device's, anything unexpected is INTERNAL."""
        if self.fatal is None:
            reply = _error_reply(error)
            self.fatal, self.fatal_code = reply['error'], reply['code']

    # -- a connection's life -------------------------------------------------
    def read_label(self):
        try:
            with open(self.label_file) as fh:
                return _label(fh.read().strip()) or 'unlabelled'
        except (OSError, TypeError, UnicodeDecodeError):
            return 'unlabelled'

    def _open_log(self, label):
        try:
            return open(os.path.join(self.observe_dir, label + '.log'), 'a', buffering=1)
        except OSError as e:
            # Nothing can be served that cannot be recorded; the card is fine.
            raise DependencyFailed(f'the observer log for {label} cannot be opened: {e}') from e

    def accept(self, sock, number):
        """The connection, under the label file's label, on a cleared buffer."""
        label = self.read_label()
        conn = _Connection(sock, number, label, self._open_log(label))
        _say(f'connection {number} label={label}')
        try:
            self.settle(conn)            # nothing may have moved while nobody was connected
            if self.hbm is not None and label != self.previous_label and self.used_since_clear:
                conn.cleared_seconds = self.clear_hbm()
        except Exception:
            conn.line(f'uncleared connection={number}', anomaly=True)
            self._close(conn, close_socket=False)     # the caller sends why
            raise
        self.previous_label, self.used_since_clear = label, True
        return conn

    def refuse(self, sock, number, why):
        """A connection that cannot be served: told why, and filed as such."""
        self.refused += 1
        label = self.read_label()
        try:
            with self._open_log(label) as log:
                log.write(f'refused-concurrent connection={number}\nend launches=0\n')
        except DependencyFailed as e:
            _say(str(e))
        try:
            sock.settimeout(5.0)
            _send(sock, dict(ok=False, code='DEPENDENCY_FAILED', error=why))
        except OSError:
            pass
        sock.close()
        _say(f'connection {number} label={label} refused: {why}')

    def end(self, conn):
        """Settle its last ring, wait out its launch, then close its record."""
        try:
            if self.fatal is None:
                self.settle(conn)
                drain = time.monotonic() + self.drain_seconds
                while not self.observe(conn) and time.monotonic() < drain:
                    time.sleep(0.05)
                    self.settle(conn)
        except Exception as e:                             # noqa: BLE001
            self.fail(e)
        if conn.pending is not None:
            conn.line(f'unfinished seq={conn.pending}', anomaly=True)
        self._close(conn)

    def _close(self, conn, *, close_socket=True):
        conn.line(f'end launches={conn.launches}')
        conn.log.close()
        if close_socket:
            try:
                conn.sock.close()
            except OSError:
                pass
        self.connections.append(conn.record())
        _say(f'connection {conn.number} ended: {json.dumps(conn.record(), sort_keys=True)}')

    # -- the observer --------------------------------------------------------
    def settle(self, conn, *, just_rung=False):
        """Account for every advance of `served` since the relay last did.

        An advance is a request the device admitted. One that matches the
        connection's unsettled ring (its req_seq, req_op read as it was rung)
        is that ring's; any other is an `unaccounted` line, and the point is
        rejected -- something admitted a request the relay did not forward.
        Right after a ring (`just_rung`), no advance yet proves nothing across
        the AXI-Lite bridge, so the ring stays unsettled; by the next
        operation a ring with no advance was dropped by the device.
        """
        served = self.link.read_reg(R_SERVED)
        if served == self.served:
            if not just_rung:
                conn.doorbell = None
            return
        rung, conn.doorbell = conn.doorbell, None
        if rung is not None and served == rung[0]:
            self._admitted(conn, served, rung[1])
        else:
            conn.line(f'unaccounted served={self.served}->{served}', anomaly=True)
        self.served = served

    def _admitted(self, conn, seq, op):
        if op != OP_PHYSICAL_LAUNCH:
            return
        if conn.pending is not None:
            # Admitted means the device was idle, so the earlier launch
            # completed unobserved and its registers now belong to this one.
            conn.line(f'lost seq={conn.pending}', anomaly=True)
        conn.pending = seq

    def observe(self, conn):
        """Resolve the connection's pending launch if the device has; True
        when nothing is left pending."""
        pending = conn.pending
        if pending is None:
            return True
        r = self.link.read_reg
        seq = r(R_LAUNCH_SEQUENCE)
        if seq == pending:
            start, end = r(R_LAUNCH_START), r(R_LAUNCH_END)
            status, r0, r1 = r(R_RSP_STATUS), r(R_RSP_R0), r(R_RSP_R1)
            if r(R_RSP_SEQ) != pending or r(R_LAUNCH_SEQUENCE) != pending:
                conn.line(f'lost seq={pending}', anomaly=True)
            else:
                conn.line(f'cmd op=1 seq={pending} obs={(end - start) & _M64} '
                          f'done={((r1 & _M32) << 32) | (r0 & _M32)} status={status & 0xF}')
                conn.launches += 1
            conn.pending = None
        elif r(R_RSP_SEQ) == pending:
            # rsp_seq shows the pending sequence when the doorbell refused the
            # launch (its flags: rsp_status 3, launch_sequence left as it was)
            # -- or when the launch completed between the two reads above, since
            # the device sets rsp_seq and launch_sequence to it in one cycle
            # (rtl/fpga/tnca_multicore_mmio.v). A second look at
            # launch_sequence tells them apart; a completion is the next
            # observe()'s to log. A refusal always carries a status: one
            # without is not taken for a refusal either.
            status = r(R_RSP_STATUS) & 0xF
            if r(R_LAUNCH_SEQUENCE) != pending and status != 0:
                conn.line(f'refused seq={pending} status={status}', anomaly=True)
                conn.pending = None
        return conn.pending is None

    def write(self, conn, offset, value):
        offset = _offset(offset)
        if type(value) is not int or not 0 <= value <= _M64:
            raise ArgumentError(f'register value {value!r} is not an unsigned 64-bit integer')
        if offset != R_DOORBELL or not value & 1:
            self.link.write_reg(offset, value)
            return
        r = self.link.read_reg
        self.settle(conn)                # every earlier advance accounted for first
        self.observe(conn)               # the registers still answer the last request
        # The descriptor as rung: the ring itself changes neither register.
        seq, op = r(R_REQ_SEQ), r(R_REQ_OP)
        self.link.write_reg(offset, value)
        conn.doorbell = (seq, op)
        # Usually settled here; a read that does not yet show the advance
        # leaves it to the next operation's settle(), never to "not admitted".
        self.settle(conn, just_rung=True)

    # -- one message ---------------------------------------------------------
    def handle(self, conn, head, payload):
        """(reply, payload, keep the connection)."""
        self.settle(conn)
        self.observe(conn)
        writes = head.get('writes', [])
        if not isinstance(writes, list) or any(not isinstance(w, list) or len(w) != 2 for w in writes):
            raise ArgumentError('writes is a list of [offset, value] pairs')
        for offset, value in writes:
            self.write(conn, offset, value)
        op = head.get('op')
        if op == 'hello':
            if conn.ops > 1:
                raise ArgumentError('hello is the first operation of a connection')
            conn.client_label = _label(head.get('label'))
            return dict(ok=True, protocol=PROTOCOL, label=conn.label, client_label=conn.client_label,
                        device=self.link.describe(), image=self.image), b'', True
        if op == 'read':
            return dict(ok=True, value=self.link.read_reg(_offset(head.get('offset')))), b'', True
        if op == 'host_status':
            return dict(ok=True, value=self.link.read_host_status()), b'', True
        if op == 'write':
            return dict(ok=True), b'', True
        if op == 'hbm_open':
            return dict(self.hbm_open(_int(head, 'nbytes', 1, 1 << 40)), ok=True), b'', True
        if op == 'hbm_write':
            self.link.hbm_write(_int(head, 'offset', 0, 1 << 40), payload)
            return dict(ok=True), b'', True
        if op == 'hbm_read':
            data = self.link.hbm_read(_int(head, 'offset', 0, 1 << 40), _int(head, 'nbytes', 0, CHUNK + 1))
            return dict(ok=True), data, True
        if op == 'bye':
            return dict(ok=True), b'', False
        raise ArgumentError(f'unknown operation {op!r}')

    # -- the HBM buffer ------------------------------------------------------
    def hbm_open(self, nbytes):
        """One buffer for the card session, handed to every connection.

        Every connection's backend asks at start. A second 8 GiB allocation
        cannot succeed while the first lives, so one is kept and handed out
        again, for the same size only; accept() clears it between labels.
        Normally it was made at startup (allocate(), --hbm-bytes). Made here,
        inside a connection, it is zeroed without the cache handoff: a request
        of the relay's own would advance `served` under the client's backend,
        whose next request would then carry a sequence the device drops.
        """
        if self.hbm is None:
            self.allocate(nbytes, handoff=False)
            self.used_since_clear = True     # by the connection that asked
        elif nbytes != self.hbm['requested']:
            raise ArgumentError(f'this card session allocated its HBM buffer for '
                                f'{self.hbm["requested"]} bytes; {nbytes} were asked for')
        return {k: self.hbm[k] for k in ('base', 'size', 'allocation', 'layout')}

    def allocate(self, nbytes, *, handoff=True):
        """The buffer, made and cleared. At startup, before any client."""
        got = self.link.hbm_open(nbytes)
        if got is None:
            raise DependencyFailed('this board link cannot allocate the AXI window')
        layout = getattr(self.link, 'hbm_layout', None)
        self.hbm = dict(requested=nbytes, base=got[0], size=got[1],
                        allocation=getattr(self.link, 'hbm_allocation', None),
                        layout=layout.name if layout is not None else None)
        self.clear_hbm(handoff=handoff)

    def clear_hbm(self, *, handoff=True):
        """The device's caches hand the buffer back, then the buffer is zeroed.
        Only between connections, where no client's backend is numbering
        requests (see hbm_open)."""
        t = time.monotonic()
        if handoff:
            self._handoff()
        # Physical BO offsets through the XRT accessor, whatever the layout:
        # every byte of the buffer, in CHUNK pieces.
        accessor = getattr(self.link, '_accessor', None)
        write = getattr(accessor, 'bo_write', None) or self.link.hbm_write
        zeros, size = bytes(CHUNK), self.hbm['size']
        for at in range(0, size, CHUNK):
            write(at, zeros if size - at >= CHUNK else zeros[:size - at])
        seconds = round(time.monotonic() - t, 3)
        self.clears.append(seconds)
        self.used_since_clear = False
        _say(f'HBM buffer of {size} bytes cleared in {seconds}s (cache handoff: {handoff})')
        return seconds

    def _handoff(self):
        """The backend's OP_CACHE_HANDOFF over the whole buffer, as a request
        of the relay's own between connections: every cached line written
        back and invalidated, so the zeros are what the device reads next.

        Always after the device reads idle (bounded by --drain-seconds, else
        the connection is refused `uncleared`), and always over the whole
        allocated buffer: the window registers are set to it first (they
        accept a write only when idle, and the next client's backend sets
        them again at its start), because the device refuses a handoff that
        reaches past its current window, and a client's window says nothing
        about what was cached before it.
        """
        r, w = self.link.read_reg, self.link.write_reg
        deadline = time.monotonic() + self.drain_seconds
        while r(R_DOORBELL):
            if time.monotonic() > deadline:
                raise DependencyFailed('the device is still busy with an earlier launch; '
                                       'its memory cannot be cleared')
            time.sleep(0.05)
        stat = r(R_AXI_STAT)
        if not stat & AXI_STAT_PRESENT:
            raise DependencyFailed('the image has no AXI window to hand back')
        base, size = self.hbm['base'], self.hbm['size']
        w(R_AXI_BASE, base)
        w(R_AXI_SIZE, size)
        if (r(R_AXI_BASE), r(R_AXI_SIZE)) != (base, size):
            raise DeviceLost('the AXI window registers did not take the whole buffer before clearing')
        seq = self.served + 1
        for offset, value in ((R_REQ_SEQ, seq), (R_REQ_OP, OP_CACHE_HANDOFF), (R_REQ_A0, (stat >> 32) << 12),
                              (R_REQ_A1, size), (R_REQ_A2, 0), (R_REQ_A3, 0), (R_REQ_A4, 0), (R_REQ_A5, 0)):
            w(offset, value)
        w(R_DOORBELL, 1)
        deadline = time.monotonic() + 60.0
        while r(R_RSP_SEQ) != seq:
            if time.monotonic() > deadline:
                raise DeviceLost('the cache handoff before clearing HBM did not complete')
            time.sleep(0.001)
        self.served = seq                # the relay's own request, accounted
        status = r(R_RSP_STATUS) & 0xF
        if status:
            raise DeviceLost(f'the cache handoff before clearing HBM answered status {status}')

    def summary(self):
        return dict(connections=len(self.connections), refused=self.refused,
                    launches=sum(c['launches'] for c in self.connections),
                    anomalies=sum(c['anomalies'] for c in self.connections),
                    labels=sorted({c['label'] for c in self.connections}),
                    clears=len(self.clears), clear_seconds=round(sum(self.clears), 3),
                    connection_records=self.connections, fatal=self.fatal, fatal_code=self.fatal_code)


def _error_reply(e):
    """A tnca error keeps its code. StripedHBMLayout refuses a range outside
    its 8 GiB with ValueError, which is the client's argument. Anything else
    nobody expected is INTERNAL: the relay's own fault until shown otherwise,
    never DEVICE_LOST, which only the link raises when the device stopped
    answering (an XRT register access refused, a DMA failed)."""
    code = getattr(e, 'code', None)
    if code not in ALL_CODES:
        code = 'ARGUMENT_INVALID' if isinstance(e, ValueError) else 'INTERNAL'
    return dict(ok=False, error=f'{type(e).__name__}: {e}', code=code)


#: A refusal the client can act on and the device is unharmed by. Anything
#: else leaves the card in a state the relay cannot vouch for, so it ends the
#: card session: DEVICE_LOST (exit 3) or INTERNAL and the rest (exit 4).
_RECOVERABLE = frozenset({'ARGUMENT_INVALID', 'DEPENDENCY_FAILED'})
EXIT_DEVICE_LOST, EXIT_INTERNAL, EXIT_STARTUP = 3, 4, 2


def _say(text):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} relay: {text}', file=sys.stderr, flush=True)


def index_for_bdf(bdf, sysfs='/sys/bus/pci/devices'):
    """The XRT index whose render node u50._loaded_xclbin_uuid reads for BDF."""
    nodes = sorted(glob.glob(os.path.join(sysfs, bdf, 'drm', 'renderD*')))
    if len(nodes) != 1:
        raise DependencyFailed(f'{bdf} has {len(nodes)} drm render node(s), not one',
                               detail={'nodes': nodes})
    return int(os.path.basename(nodes[0])[len('renderD'):]) - 128


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _serve_message(relay, conn):
    """One request and its reply; False when the connection is over."""
    try:
        head, payload = _recv(conn.sock)
        if head is None:
            return False
        conn.ops += 1
        try:
            reply, data, keep = relay.handle(conn, head, payload)
        except Exception as e:                             # noqa: BLE001
            reply, data, keep = _error_reply(e), b'', True
            if reply['code'] not in _RECOVERABLE:
                relay.fail(e)
                keep = False
        _send(conn.sock, reply, data)
        return keep
    except (OSError, ValueError, ArgumentError) as e:
        _say(f'connection {conn.number}: {type(e).__name__}: {e}')
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xclbin', required=True)
    where = ap.add_mutually_exclusive_group()
    where.add_argument('--index', type=int)
    where.add_argument('--bdf')
    ap.add_argument('--unix', required=True, help='the Unix socket to listen on')
    ap.add_argument('--observe-dir', required=True, help='where <label>.log observer records go')
    ap.add_argument('--label-file', help='the trusted label of the next connection (else "unlabelled")')
    ap.add_argument('--stop', help='stop when this file appears')
    ap.add_argument('--idle-timeout', type=float, default=1800.0,
                    help='stop after this many seconds with no connection (0: never)')
    ap.add_argument('--deadline-seconds', type=float, default=0.0,
                    help='stop this many seconds after start (0: no deadline)')
    ap.add_argument('--drain-seconds', type=float, default=120.0,
                    help='how long a connection\'s end waits for its last launch')
    ap.add_argument('--hbm-bytes', type=int, default=0,
                    help='allocate and clear the HBM buffer at startup (the backend\'s hbm_bytes)')
    ap.add_argument('--summary', help='write the full summary (every connection) here as JSON')
    ap.add_argument('--ip-name', default=IP_NAME)
    ap.add_argument('--words', type=int, default=DEFAULT_WORDS)
    args = ap.parse_args(argv)

    from .multicore_mmio import MulticoreMMIOBackend, MulticoreXDMALink
    started = time.monotonic()

    # SIGTERM has to unwind: the link's XRT context is closed in `finally`.
    def _bye(_sig, _frm):                                  # pragma: no cover
        raise SystemExit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _bye)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    reason, conn, srv, relay, link, startup_code = None, None, None, None, None, None
    try:
        index = index_for_bdf(args.bdf) if args.bdf else (args.index or 0)
        os.makedirs(args.observe_dir, exist_ok=True)
        link = MulticoreXDMALink(xclbin=args.xclbin, index=index, ip_name=args.ip_name, words=args.words,
                                 what=f'U280 {args.bdf or f"index {index}"}')
        link.open()
        magic = link.read_reg(R_MAGIC)
        if magic != MulticoreMMIOBackend.identity_magic:
            raise DeviceLost(f'the identity register reads {magic:#018x}, not the multicore '
                             f'control block {MulticoreMMIOBackend.identity_magic:#018x}')
        # Without the launch timestamps every launch would read as unfinished.
        if link.read_reg(R_LAUNCH_VERSION) != 1:
            raise DependencyFailed('the image has no version-1 launch timestamps; '
                                   'the observer could record nothing')
        # The node the link just checked the file against (u50._loaded_xclbin_uuid).
        with open(f'/sys/class/drm/renderD{128 + index}/device/xclbinuuid') as fh:
            uuid = fh.read().strip().lower()
        relay = BoardRelay(link, args.observe_dir,
                           dict(xclbin_sha256=_sha256(args.xclbin), uuid=uuid, bdf=args.bdf, index=index),
                           label_file=args.label_file, drain_seconds=args.drain_seconds)
        if args.hbm_bytes:
            relay.allocate(args.hbm_bytes)
        if os.path.exists(args.unix):
            os.unlink(args.unix)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mask = os.umask(0o077)               # the socket is born 0600
        try:
            srv.bind(args.unix)
        finally:
            os.umask(mask)
        srv.listen(8)
        srv.setblocking(False)               # accept() below never waits

        def _accept():
            try:
                sock, _ = srv.accept()
            except (BlockingIOError, InterruptedError):
                return None
            sock.setblocking(True)
            return sock
        print(f'listening {args.unix}', flush=True)
        _say(f'serving {link.describe()} image {uuid} on {args.unix}')

        deadline = started + args.deadline_seconds if args.deadline_seconds > 0 else None
        idle_since, number, backlog = time.monotonic(), 0, None
        while reason is None:
            now = time.monotonic()
            if relay.fatal:
                reason = 'device-lost' if relay.fatal_code == 'DEVICE_LOST' else 'internal-error'
            elif deadline is not None and now >= deadline:
                reason = 'deadline'
            elif args.stop and os.path.exists(args.stop):
                reason = 'stop-file'
            elif conn is None and args.idle_timeout > 0 and now - idle_since >= args.idle_timeout:
                reason = 'idle-timeout'
            if reason:
                break
            if conn is None:
                watch, wait = [srv], 1.0
            elif backlog is None:
                watch, wait = [srv, conn.sock], 1.0
            else:                            # a newcomer waits: watch only the one being served
                watch, wait = [conn.sock], max(0.0, min(1.0, backlog + BACKLOG_SECONDS - now))
            ready, _, _ = select.select(watch, [], [], wait)
            if conn is not None and conn.sock in ready and not _serve_message(relay, conn):
                relay.end(conn)
                conn, idle_since = None, time.monotonic()
            if conn is None:
                sock = _accept() if srv in ready or backlog is not None else None
                backlog = None
                if sock is not None:
                    number += 1
                    sock.settimeout(300.0)       # a message, once begun, arrives
                    try:
                        conn = relay.accept(sock, number)
                    except Exception as e:             # noqa: BLE001
                        reply = _error_reply(e)
                        _say(f'connection {number} not served: {reply["error"]}')
                        try:
                            _send(sock, reply)
                        except OSError:
                            pass
                        sock.close()
                        if reply['code'] not in _RECOVERABLE:
                            relay.fail(e)
                        idle_since = time.monotonic()
            elif srv in ready:
                backlog = time.monotonic()
            elif backlog is not None and time.monotonic() - backlog >= BACKLOG_SECONDS:
                backlog = None
                sock = _accept()
                if sock is not None:
                    number += 1
                    relay.refuse(sock, number, f'the board relay is serving connection {conn.number} '
                                               f'({conn.label}); the card takes one client at a time')
    except (KeyboardInterrupt, SystemExit):               # pragma: no cover
        reason = reason or 'signal'
    except Exception as e:                                 # noqa: BLE001
        _say(f'failed: {type(e).__name__}: {e}')
        startup_code = _error_reply(e)['code']
        if relay is not None:
            relay.fail(e)
            reason = reason or ('device-lost' if relay.fatal_code == 'DEVICE_LOST' else 'internal-error')
        else:
            reason = reason or 'failed'
    finally:
        if conn is not None:
            relay.end(conn)
        try:
            if link is not None:
                link.close()
        finally:
            if srv is not None:
                srv.close()
                if os.path.exists(args.unix):
                    os.unlink(args.unix)
    if relay is not None and relay.fatal and reason not in ('device-lost', 'internal-error'):
        reason = 'device-lost' if relay.fatal_code == 'DEVICE_LOST' else 'internal-error'
    summary = dict(reason=reason, seconds=round(time.monotonic() - started, 1))
    if relay is not None:
        summary.update(relay.summary())
    else:
        # Never listened: whether the image's control block failed to answer
        # (DEVICE_LOST) or the relay could not start for another reason.
        summary.update(fatal_code=startup_code)
    if args.summary:
        with open(args.summary + '.tmp', 'w') as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
        os.replace(args.summary + '.tmp', args.summary)
    # One short line: whoever reads it may not be reading while this exits.
    short = {k: v for k, v in summary.items() if k != 'connection_records'}
    try:
        print('stopped ' + json.dumps(short, sort_keys=True), flush=True)
    except OSError:
        pass                                # the session that read `listening` is gone
    _say(f'stopped: {reason}')
    if relay is None:
        return EXIT_STARTUP
    if relay.fatal:
        return EXIT_DEVICE_LOST if relay.fatal_code == 'DEVICE_LOST' else EXIT_INTERNAL
    return 0


# ---------------------------------------------------------------------------
# the client: untrusted, wherever the candidate's host side runs
# ---------------------------------------------------------------------------
_ERRORS = {'DEVICE_LOST': DeviceLost, 'DEPENDENCY_FAILED': DependencyFailed,
           'ARGUMENT_INVALID': ArgumentError, 'WAIT_TIMEOUT': Timeout}

#: What each relay this process reached said it serves, by socket path: the
#: image's sha256 and uuid, from its hello.
_IDENTITY = {}


class RelayLink(U50Link):
    """A board link whose registers and HBM are a relay's, over a Unix socket.

    The same methods `MulticoreXDMALink` gives the backend, and nothing the
    backend does is different above them. `xclbin` is None on purpose: the
    image is a file on the card host, chosen there. `image_sha256` and
    `image_uuid` are what the relay says it serves (its hello), which is where
    a record's image identity comes from on this backend.
    """

    xclbin = None

    def __init__(self, address, label=None, *, words=DEFAULT_WORDS, timeout=600.0):
        if label is not None and _label(label) is None:
            raise ArgumentError(f'relay label {label!r} is not [A-Za-z0-9_-]{{1,32}}')
        self.address = os.fspath(address)
        self.label = label
        self.words = words
        self.timeout = timeout
        self.relay_info = None
        self.hbm_allocation = None
        self.hbm_layout = None
        self._sock = None
        self._writes = []

    @property
    def image_sha256(self):
        return ((self.relay_info or {}).get('image') or {}).get('xclbin_sha256')

    @property
    def image_uuid(self):
        return ((self.relay_info or {}).get('image') or {}).get('uuid')

    def describe(self):
        device = (self.relay_info or {}).get('device', 'a board')
        return f'{device} through the board relay at {self.address}'

    def open(self):
        if self._sock is not None:
            return self
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            sock.connect(self.address)
        except OSError as e:
            sock.close()
            raise DependencyFailed(f'no board relay answers at {self.address}: {e}',
                                   remedy='a card session (tritonchip_fpga_harness.board_session) '
                                          'starts one on the card host; its ready.json names the '
                                          'socket, which the harness forwards here') from e
        self._sock, self._writes = sock, []
        head = {'op': 'hello', 'protocol': PROTOCOL}
        if self.label is not None:
            head['label'] = self.label
        try:
            reply, _ = self._call(head)
            if reply.get('protocol') != PROTOCOL:
                raise DependencyFailed(f'the relay speaks protocol {reply.get("protocol")}, '
                                       f'this link {PROTOCOL}')
        except BaseException:
            self._drop()                    # refused (a busy relay) or unreachable: not open
            raise
        self.relay_info = reply
        _IDENTITY[self.address] = reply.get('image')
        return self

    def close(self):
        if self._sock is None:
            return
        try:
            self._call({'op': 'bye'})
        except TncaError:
            pass
        self._drop()

    def alive(self):
        return self._sock is not None

    def read_reg(self, offset):
        return self._call({'op': 'read', 'offset': int(offset)})[0]['value']

    def read_host_status(self):
        return self._call({'op': 'host_status'})[0]['value']

    def write_reg(self, offset, value):
        if self._sock is None:
            raise DeviceLost('the board relay is not connected', detail={'medium': self.describe()})
        self._writes.append([int(offset), int(value) & _M64])
        if len(self._writes) >= 4096:
            self._call({'op': 'write'})

    def hbm_open(self, nbytes):
        reply, _ = self._call({'op': 'hbm_open', 'nbytes': int(nbytes)})
        base, size = reply['base'], reply['size']
        self.hbm_allocation = reply.get('allocation')
        self.hbm_layout = None
        if reply.get('layout') == StripedHBMLayout.name:
            # The relay's link verified the PC0..31 topology and applies the
            # layout; this object says so to the backend's own check.
            try:
                self.hbm_layout = StripedHBMLayout(base, size)
            except ValueError as e:
                raise DependencyFailed(str(e)) from e
        elif reply.get('layout') is not None:
            raise DependencyFailed(f'the relay reports an unknown HBM layout {reply["layout"]!r}')
        return base, size

    def hbm_write(self, offset, data):
        view = memoryview(data).cast('B')
        for at in range(0, len(view), CHUNK):
            self._call({'op': 'hbm_write', 'offset': int(offset) + at}, view[at:at + CHUNK])

    def hbm_read(self, offset, nbytes):
        out = bytearray()
        for at in range(0, int(nbytes), CHUNK):
            n = min(CHUNK, int(nbytes) - at)
            _, data = self._call({'op': 'hbm_read', 'offset': int(offset) + at, 'nbytes': n})
            if len(data) != n:
                raise DeviceLost(f'the relay returned {len(data)} of {n} HBM bytes')
            out += data
        return bytes(out)

    # -- the wire ------------------------------------------------------------
    def _call(self, head, payload=b''):
        if self._sock is None:
            raise DeviceLost('the board relay is not connected', detail={'medium': self.describe()})
        if self._writes:
            head = dict(head, writes=self._writes)
            self._writes = []
        try:
            _send(self._sock, head, payload)
            reply, data = _recv(self._sock, max_head=MAX_HEAD, max_payload=CHUNK)
        except (OSError, ValueError, TncaError) as e:
            self._drop()
            raise DeviceLost(f'the board relay stopped answering: {e}',
                             detail={'medium': self.describe(), 'op': head.get('op')}) from e
        if reply is None:
            self._drop()
            raise DeviceLost('the board relay closed the connection',
                             detail={'medium': self.describe(), 'op': head.get('op')})
        if not reply.get('ok'):
            code = reply.get('code')
            cls = _ERRORS.get(code)
            detail = {'medium': self.describe(), 'op': head.get('op')}
            if cls is not None:
                raise cls(reply.get('error', 'the relay refused'), detail=detail)
            raise TncaError(reply.get('error', 'the relay refused'),
                            code=code if code in ALL_CODES else 'INTERNAL', detail=detail)
        return reply, data

    def _drop(self):
        sock, self._sock = self._sock, None
        self._writes = []
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def relay_identity(address, label=None):
    """{'xclbin_sha256', 'uuid', ...} of the image the relay at `address`
    serves: from this process's earlier hello to it, or a hello of its own
    (one connection, no operation, `end launches=0` in the observer)."""
    address = os.fspath(address)
    if _IDENTITY.get(address) is None:
        link = RelayLink(address, label=label)
        try:
            link.open()
        finally:
            link.close()
    return _IDENTITY[address]


def relay(**kwargs):
    """The factory `fpga-u280-multicore-relay` is registered with.

    `address` is the relay's socket as this host sees it (the SSH-forwarded
    path); `label` is sent in the hello and recorded as the client's claim
    only (the relay files launches under the label the harness wrote). The
    rest are `multicore_mmio.board`'s backend arguments -- ready.json's
    `backend_args` -- and none of the link's: the card host chose the image
    and opened the link, so `xclbin`, `index`, `ip_name` and `accessor` are
    refused here rather than accepted and ignored.
    """
    from .multicore_mmio import board
    kwargs.pop('bench', None)
    address = kwargs.pop('address', None)
    if not address:
        raise DependencyFailed('fpga-u280-multicore-relay reaches a card through a board relay, and '
                               'no relay address was given',
                               remedy='pass address=<the forwarded relay socket> (and label=) in '
                                      'TNCA_BACKEND_ARGS')
    named = sorted(k for k in ('xclbin', 'index', 'ip_name', 'accessor') if k in kwargs)
    if named:
        raise ArgumentError(f'the relay owns the board link; {", ".join(named)} cannot be chosen here',
                            remedy='ready.json names the image the card host loaded')
    link = RelayLink(address, label=kwargs.pop('label', None),
                     words=kwargs.pop('words', DEFAULT_WORDS),
                     timeout=kwargs.pop('relay_timeout', 600.0))
    backend = board(link=link, **kwargs)
    backend.name = 'fpga-u280-multicore-relay'
    return backend


if __name__ == '__main__':
    sys.exit(main())
