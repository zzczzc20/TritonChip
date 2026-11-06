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
        --unix SOCKET --observe-dir DIR [--stop FILE] [--idle-timeout S]
        [--deadline-seconds S] [--ip-name NAME] [--words N]

It prints `listening SOCKET` once the board is open and its identity register
answers, and `stopped {json}` on the way out. It serves many connections, one
at a time (a second concurrent one is refused, as the board's exclusive context
would refuse a second local process), and ends on the first of: FILE appears,
`--idle-timeout` seconds pass with no connection, `--deadline-seconds` pass,
or the device is lost. The link is closed and the image left loaded.

THE PROTOCOL

relay.py's framing: `struct.pack('<II', len(head), len(payload)) + head +
payload`, head JSON. Replies are `{"ok": true, ...}` or `{"ok": false,
"error": ..., "code": ...}` (a tnca error code). Ops:

    hello        {label}            -> protocol, label, device, image
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
its status read. Every operation crosses an SSH tunnel at a few milliseconds,
and without this a request was eleven crossings before its first poll.

THE OBSERVER

What the evaluator scores is written here, by the relay, to
`<observe-dir>/<label>.log` (the label each connection's hello names; several
connections may share one, and append). Before forwarding any doorbell (0x48,
bit 0) and when a connection ends, the relay looks at
rtl/fpga/tnca_multicore_mmio.v's launch registers. When the launch this
connection rang (req_op 11, admitted: idle and req_seq > served) has completed
-- launch_sequence (0xf8) equals its sequence and the response registers still
answer it -- it appends

    cmd op=1 seq=<launch_sequence> obs=<launch_end-launch_start> done=<rsp_r1:rsp_r0> status=<rsp_status>

the same fields as tb/fpga/tnca_multicore_board_regs.v's simulation observer.
There are no settle/quiet/rd/wr fields: the board has no observation of its
cores draining or of its outlets' transactions in flight, and a field the
board cannot measure is left out, not written as zero. A rung launch that has
not completed when its connection ends is written as `unfinished seq=<n>`, one
the device answered without starting as `refused seq=<n> status=<s>`, one whose
ring did not take as `dropped seq=<n>`, and one whose response registers no
longer answer it as `lost seq=<n>`: none of them is a `cmd` line, so the
evaluator sees no observation rather than a made-up one. Each connection ends
with `end launches=<n>`. The client cannot write these registers (the control
block has no write path to them) and cannot ring a doorbell except through
this relay, so the record is the platform's, not the candidate's.

A client that rings a launch while another is still running is not tracked
past what the device does with it: the device drops such a ring, and in the
one race where it does not, that launch goes unrecorded rather than misfiled.
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
from .u50 import (DEFAULT_WORDS, IP_NAME, R_DOORBELL, R_MAGIC, R_REQ_OP,
                  R_REQ_SEQ, R_RSP_R0, R_RSP_R1, R_RSP_SEQ, R_RSP_STATUS,
                  R_SERVED, U50Link)

PROTOCOL = 1

# rtl/fpga/tnca_multicore_mmio.v's launch timestamps (U50Backend.
# launch_timestamps reads the same three) and its physical-grid launch opcode.
R_LAUNCH_VERSION = 0xE0
R_LAUNCH_START = 0xE8
R_LAUNCH_END = 0xF0
R_LAUNCH_SEQUENCE = 0xF8
OP_PHYSICAL_LAUNCH = 11

#: The largest HBM transfer one message carries. The client splits bigger
#: ones; the relay refuses bigger ones, since it owns memory on a shared host.
CHUNK = 1 << 24
MAX_HEAD = 1 << 20
#: The control block's AXI-Lite aperture (fpga/kernel_multicore_hbm.xml
#: range 0x1000). An offset XRT would refuse is refused here instead, since a
#: refused register access is read as a lost device and ends the card session.
MAX_OFFSET = 0x1000

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
        raise ArgumentError(f'register offset {value!r} is not an aligned control-block address')
    return value


# ---------------------------------------------------------------------------
# the server: trusted, on the card host
# ---------------------------------------------------------------------------
class _Connection:
    def __init__(self, sock, number):
        self.sock, self.number = sock, number
        self.label = None
        self.log = None
        self.launches = 0
        self.pending = None
        self.ops = 0
        self.started = time.monotonic()

    def line(self, text):
        self.log.write(text + '\n')
        self.log.flush()


class BoardRelay:
    """One open board link, served to one connection at a time."""

    def __init__(self, link, observe_dir, image):
        self.link = link
        self.observe_dir = observe_dir
        self.image = image
        self.hbm = None
        self.fatal = None
        self.connections = self.refused = self.launches = self.unfinished = 0
        self.labels = set()

    # -- a connection's life -------------------------------------------------
    def begin(self, conn, head):
        """The label comes from a first `hello`; anything else is unlabelled."""
        label = head.get('label') if head.get('op') == 'hello' else None
        conn.label = label if isinstance(label, str) and _LABEL.fullmatch(label) else 'unlabelled'
        conn.log = open(os.path.join(self.observe_dir, conn.label + '.log'), 'a', buffering=1)
        self.connections += 1
        self.labels.add(conn.label)
        _say(f'connection {conn.number} label={conn.label}')

    def end(self, conn):
        """Capture what the connection left, then close its record."""
        if conn.log is not None:
            try:
                if self.fatal is None:
                    self.capture(conn)
            except Exception as e:                         # noqa: BLE001
                self.fatal = self.fatal or f'capture at connection end: {e}'
            if conn.pending is not None:
                conn.line(f'unfinished seq={conn.pending}')
                self.unfinished += 1
            conn.line(f'end launches={conn.launches}')
            conn.log.close()
            self.launches += conn.launches
        try:
            conn.sock.close()
        except OSError:
            pass
        _say(f'connection {conn.number} label={conn.label} ended: ops={conn.ops} '
             f'launches={conn.launches} seconds={time.monotonic() - conn.started:.1f}')

    # -- the observer --------------------------------------------------------
    def capture(self, conn):
        """Resolve this connection's rung launch, if the device has."""
        if conn.pending is None:
            return
        r = self.link.read_reg
        seq = r(R_LAUNCH_SEQUENCE)
        if seq == 0:
            return                       # admitted and still running
        pending, conn.pending = conn.pending, None
        if seq != pending:
            # The launch registers were not touched: the device answered the
            # doorbell without starting (flags refused), or never took it.
            if r(R_RSP_SEQ) == pending:
                conn.line(f'refused seq={pending} status={r(R_RSP_STATUS) & 0xF}')
            else:
                conn.line(f'dropped seq={pending}')
            return
        start, end = r(R_LAUNCH_START), r(R_LAUNCH_END)
        status, r0, r1 = r(R_RSP_STATUS), r(R_RSP_R0), r(R_RSP_R1)
        if r(R_RSP_SEQ) != seq or r(R_LAUNCH_SEQUENCE) != seq:
            conn.line(f'lost seq={pending}')
            return
        conn.line(f'cmd op=1 seq={seq} obs={(end - start) & _M64} '
                  f'done={((r1 & _M32) << 32) | (r0 & _M32)} status={status & 0xF}')
        conn.launches += 1

    def write(self, conn, offset, value):
        offset = _offset(offset)
        if type(value) is not int or not 0 <= value <= _M64:
            raise ArgumentError(f'register value {value!r} is not an unsigned 64-bit integer')
        launch = None
        if offset == R_DOORBELL and value & 1:
            self.capture(conn)
            r = self.link.read_reg
            seq = r(R_REQ_SEQ)
            # The doorbell is admitted only when idle and req_seq > served
            # (the control block's own rule); anything else is dropped there.
            if r(R_REQ_OP) == OP_PHYSICAL_LAUNCH and not r(R_DOORBELL) and seq > r(R_SERVED):
                launch = seq
        self.link.write_reg(offset, value)
        if launch is not None:
            conn.pending = launch

    # -- one message ---------------------------------------------------------
    def handle(self, conn, head, payload):
        """(reply, payload, keep the connection)."""
        writes = head.get('writes', [])
        if not isinstance(writes, list) or any(not isinstance(w, list) or len(w) != 2 for w in writes):
            raise ArgumentError('writes is a list of [offset, value] pairs')
        for offset, value in writes:
            self.write(conn, offset, value)
        op = head.get('op')
        if op == 'hello':
            if conn.ops > 1:
                raise ArgumentError('hello is the first operation of a connection')
            return dict(ok=True, protocol=PROTOCOL, label=conn.label,
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

    def hbm_open(self, nbytes):
        """One buffer for the card session, made on the first request.

        Every connection's backend asks at start. A second 8 GiB allocation
        cannot succeed while the first lives, and freeing it between
        connections would hand the next one a different buffer for no gain,
        so the first is kept and handed out again, for the same size only.
        """
        if self.hbm is None:
            got = self.link.hbm_open(nbytes)
            if got is None:
                raise DependencyFailed('this board link cannot allocate the AXI window')
            layout = getattr(self.link, 'hbm_layout', None)
            self.hbm = dict(requested=nbytes, base=got[0], size=got[1],
                            allocation=getattr(self.link, 'hbm_allocation', None),
                            layout=layout.name if layout is not None else None)
        elif nbytes != self.hbm['requested']:
            raise ArgumentError(f'this card session allocated its HBM buffer for '
                                f'{self.hbm["requested"]} bytes; {nbytes} were asked for')
        return {k: self.hbm[k] for k in ('base', 'size', 'allocation', 'layout')}


def _error_reply(e):
    code = getattr(e, 'code', None)
    if code not in ALL_CODES:
        # StripedHBMLayout refuses a range outside its 8 GiB with ValueError.
        code = 'ARGUMENT_INVALID' if isinstance(e, ValueError) else 'DEVICE_LOST'
    return dict(ok=False, error=f'{type(e).__name__}: {e}', code=code)


#: A refusal the client can act on and the device is unharmed by. Anything
#: else (a register access XRT refused, an error nobody expected) leaves the
#: card in a state the relay cannot vouch for, so it ends the card session.
_RECOVERABLE = frozenset({'ARGUMENT_INVALID', 'DEPENDENCY_FAILED'})


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xclbin', required=True)
    where = ap.add_mutually_exclusive_group()
    where.add_argument('--index', type=int)
    where.add_argument('--bdf')
    ap.add_argument('--unix', required=True, help='the Unix socket to listen on')
    ap.add_argument('--observe-dir', required=True, help='where <label>.log observer records go')
    ap.add_argument('--stop', help='stop when this file appears')
    ap.add_argument('--idle-timeout', type=float, default=1800.0,
                    help='stop after this many seconds with no connection (0: never)')
    ap.add_argument('--deadline-seconds', type=float, default=0.0,
                    help='stop this many seconds after start (0: no deadline)')
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

    reason, conn, srv, relay, link = None, None, None, None, None
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
                           dict(xclbin_sha256=_sha256(args.xclbin), uuid=uuid, bdf=args.bdf, index=index))
        if os.path.exists(args.unix):
            os.unlink(args.unix)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mask = os.umask(0o077)               # the socket is born 0600
        try:
            srv.bind(args.unix)
        finally:
            os.umask(mask)
        srv.listen(8)
        print(f'listening {args.unix}', flush=True)
        _say(f'serving {link.describe()} image {uuid} on {args.unix}')

        deadline = started + args.deadline_seconds if args.deadline_seconds > 0 else None
        idle_since, number = time.monotonic(), 0
        while reason is None:
            now = time.monotonic()
            if relay.fatal:
                reason = 'device-lost'
            elif deadline is not None and now >= deadline:
                reason = 'deadline'
            elif args.stop and os.path.exists(args.stop):
                reason = 'stop-file'
            elif conn is None and args.idle_timeout > 0 and now - idle_since >= args.idle_timeout:
                reason = 'idle-timeout'
            if reason:
                break
            watch = [srv] + ([conn.sock] if conn else [])
            ready, _, _ = select.select(watch, [], [], 1.0)
            if srv in ready:
                sock, _ = srv.accept()
                number += 1
                if conn is None:
                    sock.settimeout(300.0)       # a message, once begun, arrives
                    conn = _Connection(sock, number)
                else:
                    relay.refused += 1
                    try:
                        sock.settimeout(5.0)
                        _send(sock, dict(ok=False, code='DEPENDENCY_FAILED',
                                         error=f'the board relay is serving connection {conn.number} '
                                               f'({conn.label}); the card takes one client at a time'))
                    except OSError:
                        pass
                    sock.close()
            if conn is not None and conn.sock in ready:
                keep = True
                try:
                    head, payload = _recv(conn.sock)
                    if head is None:
                        keep = False
                    else:
                        conn.ops += 1
                        if conn.label is None:
                            relay.begin(conn, head)
                        try:
                            reply, data, keep = relay.handle(conn, head, payload)
                        except Exception as e:             # noqa: BLE001
                            reply, data = _error_reply(e), b''
                            if reply['code'] not in _RECOVERABLE:
                                relay.fatal = reply['error']
                                keep = False
                        _send(conn.sock, reply, data)
                except (OSError, ValueError, ArgumentError) as e:
                    _say(f'connection {conn.number}: {type(e).__name__}: {e}')
                    keep = False
                if not keep:
                    relay.end(conn)
                    conn, idle_since = None, time.monotonic()
    except (KeyboardInterrupt, SystemExit):               # pragma: no cover
        reason = reason or 'signal'
    except Exception as e:                                 # noqa: BLE001
        _say(f'failed: {type(e).__name__}: {e}')
        reason = reason or 'failed'
        if relay is not None:
            relay.fatal = relay.fatal or f'{type(e).__name__}: {e}'
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
    summary = dict(reason=reason, seconds=round(time.monotonic() - started, 1))
    if relay is not None:
        summary.update(connections=relay.connections, refused=relay.refused, launches=relay.launches,
                       unfinished=relay.unfinished, labels=sorted(relay.labels), fatal=relay.fatal)
    try:
        print('stopped ' + json.dumps(summary, sort_keys=True), flush=True)
    except OSError:
        pass                                # the session that read `listening` is gone
    _say(f'stopped: {json.dumps(summary, sort_keys=True)}')
    if relay is None:
        return 2
    return 3 if relay.fatal else 0


# ---------------------------------------------------------------------------
# the client: untrusted, wherever the candidate's host side runs
# ---------------------------------------------------------------------------
_ERRORS = {'DEVICE_LOST': DeviceLost, 'DEPENDENCY_FAILED': DependencyFailed,
           'ARGUMENT_INVALID': ArgumentError, 'WAIT_TIMEOUT': Timeout}


class RelayLink(U50Link):
    """A board link whose registers and HBM are a relay's, over a Unix socket.

    The same methods `MulticoreXDMALink` gives the backend, and nothing the
    backend does is different above them. `xclbin` is None on purpose: the
    image is a file on the card host, chosen there, and code here that wanted
    to hash it would be hashing something else. `relay_info` is what the relay
    said it is serving (device, image sha256 and uuid).
    """

    xclbin = None

    def __init__(self, address, label=None, *, words=DEFAULT_WORDS, timeout=600.0):
        if label is not None and not (isinstance(label, str) and _LABEL.fullmatch(label)):
            raise ArgumentError(f'relay label {label!r} is not [A-Za-z0-9_-]{{1,32}}',
                                remedy='the label names the observer log; an unusable one would '
                                       'file this connection\'s launches under "unlabelled"')
        self.address = os.fspath(address)
        self.label = label
        self.words = words
        self.timeout = timeout
        self.relay_info = None
        self.hbm_allocation = None
        self.hbm_layout = None
        self._sock = None
        self._writes = []

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


def relay(**kwargs):
    """The factory `fpga-u280-multicore-relay` is registered with.

    `address` is the relay's socket as this host sees it (the SSH-forwarded
    path); `label` names this connection's observer log. The rest are
    `multicore_mmio.board`'s backend arguments -- ready.json's `backend_args`
    -- and none of the link's: the card host chose the image and opened the
    link, so `xclbin`, `index`, `ip_name` and `accessor` are refused here
    rather than accepted and ignored.
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
