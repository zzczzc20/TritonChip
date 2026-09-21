"""The same RTL, reached over a socket instead of a shared directory.

doc/phase3_6_tasks_and_acceptance.md R36-BE-02: a reference adapter that
is not the built-in binding, and that executes a numerical program
"通过真实 RTL 或与真实 RTL 相连的协议闭环" -- through real RTL, or a
protocol loop connected to it. It also says what will not do: "仅用 mock
返回正确数值不能证明执行接入成功".

So nothing here computes an answer. A relay process owns the device and
speaks the file protocol to it; this adapter reaches the relay over TCP.
Every number that comes back was produced by the same Verilog the
built-in binding drives. What changes is the medium, which is exactly
the boundary section 4.1 wants replaceable: "同一语义和实现的不同
transport binding，应能使用同一 artifact；不能只因连接介质变化要求重新
编译".

The protocol is deliberately dull -- a JSON header and a length-prefixed
payload -- because its purpose is to be reimplemented by somebody else,
over something else, without reading this file for hints.
"""
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

from ..errors import TncaError
from .. import backend as _backend
from ..backend import Backend, REFERENCE_AVAILABLE

PROTOCOL_VERSION = 1


def _send(sock, obj, payload=b''):
    head = json.dumps(obj, separators=(',', ':')).encode()
    sock.sendall(struct.pack('<II', len(head), len(payload)) + head + payload)


def _recv_exactly(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise TncaError('the relay closed the connection',
                            code='DEVICE_LOST')
        out += chunk
    return bytes(out)


def _recv(sock):
    hl, pl = struct.unpack('<II', _recv_exactly(sock, 8))
    head = json.loads(_recv_exactly(sock, hl).decode())
    return head, _recv_exactly(sock, pl)


class _Reply:
    """Shaped like the built-in transport's reply, and nothing more."""

    def __init__(self, seq, status, r0, r1):
        self.seq, self.status, self.r0, self.r1 = seq, status, r0, r1

    @property
    def ok(self):
        return self.status == 0

    def __repr__(self):
        return (f"<reply seq={self.seq} status={self.status} "
                f"r0={self.r0} r1={self.r1}>")


class SocketBackend(Backend):
    """Reaches a device through a relay, over TCP."""

    name = 'sim-socket'
    state = REFERENCE_AVAILABLE
    #: A launch is NOT idempotent here either. Section 4.2 forbids a
    #: timeout resending one, and a relay in the middle does not change
    #: that: the device would run the kernel twice.
    launch_is_idempotent = False

    def __init__(self, bench, *, workdir=None, deadline_cycles=2_000_000,
                 host_timeout=600.0, trace=False, image=None, epoch=0,
                 address=None, **_ignored):
        self.bench = bench
        self.dir = workdir or tempfile.mkdtemp(prefix='tnca-relay-')
        self.deadline_cycles = deadline_cycles
        self.host_timeout = host_timeout
        self.trace = trace
        self.image = image
        self.epoch = epoch
        self.seq = 0
        self.history = []
        self._address = address       # given: connect; absent: start one
        self._sock = None
        self._relay = None
        self.log = os.path.join(self.dir, 'relay.log')

    # -- lifetime -----------------------------------------------------------
    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        if self._address is None:
            self._relay = subprocess.Popen(
                [sys.executable, '-m', 'tnca.backends.relay', self.bench,
                 '--workdir', os.path.join(self.dir, 'dev'),
                 '--deadline', str(self.deadline_cycles)]
                + (['--image', self.image] if self.image else []),
                stdout=subprocess.PIPE, stderr=open(self.log, 'w'),
                start_new_session=True, env=_child_env())
            line = self._relay.stdout.readline().decode().strip()
            if not line.startswith('listening '):
                raise TncaError(
                    f"the relay did not come up: {line or 'no output'}",
                    code='DEVICE_LOST',
                    detail={'log': _tail(self.log)})
            self._address = ('127.0.0.1', int(line.split()[1]))
        deadline = time.time() + 30
        while True:
            try:
                self._sock = socket.create_connection(self._address,
                                                      timeout=self.host_timeout)
                break
            except OSError:
                if time.time() > deadline:
                    raise TncaError(
                        f"could not reach the relay at {self._address}",
                        code='DEVICE_LOST',
                        detail={'log': _tail(self.log)})
                time.sleep(0.05)
        head, _ = self._call({'op': 'hello', 'protocol': PROTOCOL_VERSION})
        if head.get('protocol') != PROTOCOL_VERSION:
            raise TncaError(
                f"the relay speaks protocol {head.get('protocol')} and "
                f"this adapter speaks {PROTOCOL_VERSION}",
                code='PROTOCOL_MISMATCH',
                remedy='a protocol version is not negotiated down: a '
                       'binding that guessed would be guessing about '
                       'completion semantics')
        return self

    def stop(self):
        if self._sock is not None:
            try:
                _send(self._sock, {'op': 'bye'})
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        if self._relay is not None:
            # Give it a moment to act on the `bye` above and stop its own
            # device. Terminating straight away worked, in the sense that
            # this process went away -- and left the simulator it had
            # started running with no parent.
            try:
                self._relay.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._relay.terminate()
                try:
                    self._relay.wait(timeout=10)
                except subprocess.TimeoutExpired:          # pragma: no cover
                    self._relay.kill()
            self._relay = None

    def alive(self):
        return self._sock is not None and (
            self._relay is None or self._relay.poll() is None)

    # -- requests -----------------------------------------------------------
    def _call(self, obj, payload=b''):
        if self._sock is None:
            raise TncaError('this backend is not connected',
                            code='DEVICE_LOST')
        obj = dict(obj, epoch=self.epoch)
        try:
            _send(self._sock, obj, payload)
            head, data = _recv(self._sock)
        except OSError as e:
            raise TncaError(f"the relay stopped answering: {e}",
                            code='DEVICE_LOST',
                            detail={'log': _tail(self.log)}) from e
        if head.get('error'):
            raise TncaError(head['error'], code=head.get('code', 'INTERNAL'))
        # A reply from another epoch is not this session's business.
        if 'epoch' in head and head['epoch'] != self.epoch:
            raise TncaError(
                f"a reply arrived under epoch {head['epoch']} and this "
                f"session is epoch {self.epoch}",
                code='STALE_REPLY',
                remedy='an old epoch\'s answer is discarded rather than '
                       'used: it describes a device state that no longer '
                       'exists')
        return head, data

    def upload(self, addr, data):
        # Checked HERE, before anything goes on the wire. This binding
        # used to send an unaligned address, let the device raise, and
        # lose the connection to the relay's error path -- so the same
        # illegal call was a clean ARGUMENT_INVALID on the file mailbox
        # and a dead session here. The conformance suite found it.
        _backend.check_transfer(addr, len(data), what='upload')
        head, _ = self._call({'op': 'upload', 'addr': int(addr),
                              'nbytes': len(data)}, bytes(data))
        return self._reply(head)

    def download(self, addr, nbytes):
        _backend.check_transfer(addr, nbytes, what='download')
        head, data = self._call({'op': 'download', 'addr': int(addr),
                                 'nbytes': int(nbytes)})
        if len(data) != nbytes:
            raise TncaError(
                f"a partial transfer: {len(data)} byte(s) of {nbytes}",
                code='TRANSFER_SHORT',
                remedy='a short download is an error, not a shorter '
                       'answer: the caller asked for a range')
        return self._reply(head), data

    def launch(self, pc, arg_base, role_mask=1, pid=(0, 0, 0),
               npid=(1, 1, 1)):
        head, _ = self._call({'op': 'launch', 'pc': int(pc),
                              'arg_base': int(arg_base),
                              'role_mask': int(role_mask),
                              'pid': list(pid), 'npid': list(npid)})
        return self._reply(head)

    def quiesce(self):
        head, _ = self._call({'op': 'quiesce'})
        return self._reply(head)

    def ping(self):
        head, _ = self._call({'op': 'ping'})
        return self._reply(head)

    def _reply(self, head):
        self.seq = head.get('seq', self.seq + 1)
        r = _Reply(self.seq, head.get('status', 0), head.get('r0', 0),
                   head.get('r1', 0))
        self.history.append(r)
        return r

    def capabilities(self):
        out = super().capabilities()
        out.update({'protocol': PROTOCOL_VERSION, 'medium': 'tcp',
                    'reaches': 'the same RTL as sim-file, through a relay',
                    # A structured protocol, not an opcode field: there
                    # is no way to ask for an operation that does not
                    # exist except by naming one, which the relay
                    # refuses by name.
                    'raw_request': False})
        return out


def _child_env():
    env = dict(os.environ)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                        '..'))
    env['PYTHONPATH'] = os.pathsep.join(
        [here] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return env


def _tail(path, n=8):
    try:
        with open(path) as fh:
            return ''.join(fh.readlines()[-n:])
    except OSError:
        return ''


def encode(data):
    """For a relay that prefers text. Unused here; kept for reimplementers."""
    return base64.b64encode(data).decode()
