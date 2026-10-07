"""A relay: owns a device, serves it over a socket.

doc/phase3_6_tasks_and_acceptance.md R36-BE-02.

This exists so the reference adapter is a real protocol loop rather
than a mock. It drives the same RTL the built-in binding drives, using
the same transport, and forwards what comes back. It computes nothing:
every number a client receives came out of the Verilog.

It is also the shape an external branch would write -- a small process
that owns whatever the platform is and speaks one protocol to the
runtime -- which is why it is a module of its own rather than a helper
inside the adapter.

    python -m tnca.backends.relay <bench> [--workdir D] [--deadline N]
"""
import argparse
import json
import signal
import socket
import struct
import sys


def _send(conn, obj, payload=b''):
    head = json.dumps(obj, separators=(',', ':')).encode()
    conn.sendall(struct.pack('<II', len(head), len(payload)) + head + payload)


def _recv_exactly(conn, n):
    out = bytearray()
    while len(out) < n:
        chunk = conn.recv(n - len(out))
        if not chunk:
            return None
        out += chunk
    return bytes(out)


def _recv(conn):
    head = _recv_exactly(conn, 8)
    if head is None:
        return None, None
    hl, pl = struct.unpack('<II', head)
    h = _recv_exactly(conn, hl)
    p = _recv_exactly(conn, pl)
    if h is None or p is None:
        return None, None
    return json.loads(h.decode()), p


# Errors the device can refuse and carry on from. Anything else ends
# the connection, because the relay cannot tell a live device from one
# that has stopped answering except by what it said.
RECOVERABLE = frozenset({'ARGUMENT_INVALID', 'PROTOCOL_ERROR',
                         'PROFILE_MISMATCH', 'UNSUPPORTED'})


def serve(conn, dev, epoch):
    """One client, until it says goodbye or the socket closes."""
    while True:
        head, payload = _recv(conn)
        if head is None:
            return False
        op = head.get('op')
        try:
            if op == 'hello':
                _send(conn, {'protocol': 1, 'epoch': epoch,
                             'device': dev.bench})
                continue
            if op == 'bye':
                # A CLEAN goodbye, distinguished from a socket that just
                # closed: the relay exits its accept loop on this one and
                # gets to run its cleanup.
                return True
            if op == 'upload':
                r = dev.upload(head['addr'], payload)
                _send(conn, _reply(r, epoch))
            elif op == 'download':
                r, data = dev.download(head['addr'], head['nbytes'])
                _send(conn, _reply(r, epoch), data)
            elif op == 'launch':
                r = dev.launch(head['pc'], head['arg_base'],
                               role_mask=head.get('role_mask', 1),
                               pid=tuple(head.get('pid', (0, 0, 0))),
                               npid=tuple(head.get('npid', (1, 1, 1))))
                _send(conn, _reply(r, epoch))
            elif op == 'quiesce':
                r = dev.quiesce()
                _send(conn, _reply(r, epoch))
            elif op == 'ping':
                r = dev.ping()
                _send(conn, _reply(r, epoch))
            else:
                _send(conn, {'error': f"unknown operation {op!r}",
                             'code': 'PROTOCOL_ERROR'})
        except Exception as e:                             # noqa: BLE001
            # The device's failure is reported, not hidden behind a
            # plausible answer. A relay that invented a reply here would
            # be the mock this stage forbids.
            code = getattr(e, 'code', 'DEVICE_LOST')
            _send(conn, {'error': f"{type(e).__name__}: {e}",
                         'code': code})
            # ...but a refusable error is not device loss, and closing
            # the connection made it one. An illegal argument used to
            # end the session here, so the same mistake cost a reply on
            # one backend and the device on the other. R36-IO-03 asks
            # for exactly this distinction.
            if code in RECOVERABLE:
                continue
            return False


def _reply(r, epoch):
    return {'seq': getattr(r, 'seq', 0), 'status': getattr(r, 'status', 0),
            'r0': getattr(r, 'r0', 0), 'r1': getattr(r, 'r1', 0),
            'epoch': epoch}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bench')
    ap.add_argument('--workdir')
    ap.add_argument('--deadline', type=int, default=2_000_000)
    ap.add_argument('--image')
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--port', type=int, default=0)
    # Bounds how long this process may outlive its client. Zero disables
    # it, which is only correct for a relay somebody else supervises.
    ap.add_argument('--idle-timeout', type=float, default=120.0)
    args = ap.parse_args(argv)

    from tnca import transport as T
    dev = T.Transport(args.bench, workdir=args.workdir,
                      deadline_cycles=args.deadline, image=args.image,
                      epoch=args.epoch)
    dev.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', args.port))
    srv.listen(4)
    port = srv.getsockname()[1]
    # The client waits for this line. Printing it before the device is up
    # would hand out an address that cannot answer.
    print(f"listening {port}", flush=True)

    # SIGTERM has to run the cleanup below.
    #
    # Python's default SIGTERM handling ends the process without
    # unwinding, so `finally: dev.stop()` never ran and the simulator
    # this relay started was left orphaned -- reparented to init and
    # spinning. Four of them were found on a four-core machine, taking
    # the whole machine while an acceptance gate measured RTL beside
    # them. The client terminates this process as a matter of course, so
    # that was every socket-backend session.
    def _bye(_sig, _frm):                                  # pragma: no cover
        raise SystemExit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):
            pass

    # A relay must not outlive its client by more than this.
    #
    # `accept` used to block forever. A client that died between starting
    # this process and connecting to it -- killed, or crashed on the way
    # -- left the relay waiting for a client that was never coming, with
    # the device it owns still running and still costing CPU for nobody.
    # Ten of those were found reparented to init on one machine, each
    # with its simulator still spinning. `SocketBackend.start` reaps on
    # its own failure now, which covers the ordinary way in; this covers
    # the one where the client never got to run any cleanup at all.
    #
    # A client that is genuinely coming arrives in milliseconds: it
    # connects as soon as it has read the line printed above.
    if args.idle_timeout > 0:
        srv.settimeout(args.idle_timeout)

    try:
        while True:
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                # Nobody came, or whoever was here never said goodbye.
                # Either way there is no session to serve.
                print(f"no client for {args.idle_timeout:g}s: stopping the "
                      f"device and exiting", file=sys.stderr, flush=True)
                break
            # Said rather than relied on: a socket accepted from a
            # listener that has a timeout still comes back blocking, and
            # `serve` has to block -- it is waiting on a client that is
            # thinking, not on an idle moment. A timeout inherited from
            # the listener would cut off a long launch instead.
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                said_bye = serve(conn, dev, args.epoch)
            finally:
                conn.close()
            if said_bye:
                break
    except (KeyboardInterrupt, SystemExit):                # pragma: no cover
        pass
    finally:
        dev.stop()
        srv.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
