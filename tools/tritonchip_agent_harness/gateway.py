#!/usr/bin/env python3
"""The egress gateway: an agent's only way out.

It runs in the trusted sidecar and listens on a Unix socket, which the agent's
container (no network) reaches through bridge.py. It accepts HTTP CONNECT to
port 443 of the hosts it is given, nothing else, and opens each tunnel either
directly or through the site's upstream proxy, as the host's route says.

It never terminates TLS. What it learns is a host, a port, a duration and the
bytes each way -- never content, never a credential -- and it writes one line
per tunnel or refusal to an append-only log outside the agent's reach. Those
byte counts are what a CLI's own token report is checked against.

Standard library only; runs as a script: python3 gateway.py --socket … --route host=direct …
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import signal
import socket
import sys
import time
from typing import Optional

HEAD_LIMIT = 8192
CONNECT_TIMEOUT = 30.0
PORT = 443
LOCAL_NAMES = ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback")


def internal_ip(text: str) -> Optional[bool]:
    """Whether a literal address is this machine or a private network; None if it is not an address.

    Alternative IPv4 spellings (`127.1`, `2130706433`, `0x7f000001`) count as
    addresses, because a resolver downstream would read them as one.
    """
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(text))
        except OSError:
            return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


async def internal(host: str, resolve: bool) -> bool:
    """`open` mode lets unlisted hosts out, but never back in: not loopback, not a private network.

    A name is resolved here only when the tunnel goes out directly; through an
    upstream proxy the proxy resolves it, and only the name can be checked.
    """
    if host in LOCAL_NAMES or host.endswith(".localhost"):
        return True
    literal = internal_ip(host)
    if literal is not None:
        return literal
    if not resolve:
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError:
        return False                      # it will not connect either
    return any(internal_ip(i[4][0]) for i in infos)


class Gateway:
    def __init__(self, routes: dict, upstream: Optional[tuple], log_path: str,
                 open_route: Optional[str] = None) -> None:
        self.routes = routes
        self.upstream = upstream
        # `open` mode: a host that is not listed goes out by this route
        # instead of being refused. Everything is still logged.
        self.open_route = open_route
        self.log = open(log_path, "a", buffering=1, encoding="utf-8")
        self.totals = {"tunnels": 0, "refused": 0, "failed": 0, "bytes_up": 0, "bytes_down": 0}

    def record(self, **fields) -> None:
        fields["ts"] = round(time.time(), 3)
        self.log.write(json.dumps(fields, sort_keys=True) + "\n")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        start = time.time()
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), CONNECT_TIMEOUT)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
            self.totals["refused"] += 1
            self.record(event="refused", reason="no complete request head")
            writer.close()
            return
        words = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(words) != 3 or words[0] != "CONNECT":
            await self.refuse(writer, 405, "Method Not Allowed", reason="not CONNECT",
                              method=words[0] if words else "")
            return
        host, _, port = words[1].rpartition(":")
        host = host.strip("[]").lower()
        if not port.isdigit():
            await self.refuse(writer, 400, "Bad Request", reason="no port", target=words[1][:200])
            return
        route = self.routes.get(host) or self.open_route
        ports = (PORT, 80) if self.open_route else (PORT,)
        if route is None or int(port) not in ports:
            await self.refuse(writer, 403, "Forbidden", reason="not listed", host=host[:200], port=int(port))
            return
        if host not in self.routes and await internal(host, resolve=route == "direct"):
            await self.refuse(writer, 403, "Forbidden", reason="internal address", host=host[:200], port=int(port))
            return
        try:
            up_reader, up_writer = await self.open(host, route, int(port))
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            self.totals["failed"] += 1
            self.record(event="failed", host=host, route=route, error=type(exc).__name__, detail=str(exc)[:200])
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            await self.drain_close(writer)
            return
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        counts = [0, 0]
        await asyncio.gather(self.pipe(reader, up_writer, counts, 0), self.pipe(up_reader, writer, counts, 1))
        for w in (writer, up_writer):
            await self.drain_close(w)
        self.totals["tunnels"] += 1
        self.totals["bytes_up"] += counts[0]
        self.totals["bytes_down"] += counts[1]
        self.record(event="tunnel", host=host, port=int(port), route=route, listed=host in self.routes,
                    bytes_up=counts[0], bytes_down=counts[1], seconds=round(time.time() - start, 3))

    async def refuse(self, writer: asyncio.StreamWriter, code: int, text: str, **fields) -> None:
        self.totals["refused"] += 1
        self.record(event="refused", status=code, **fields)
        writer.write(f"HTTP/1.1 {code} {text}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
        await self.drain_close(writer)

    async def open(self, host: str, route: str, port: int = PORT):
        if route == "direct":
            return await asyncio.wait_for(asyncio.open_connection(host, port), CONNECT_TIMEOUT)
        if self.upstream is None:
            raise ValueError("route is upstream but no upstream proxy was given")
        reader, writer = await asyncio.wait_for(asyncio.open_connection(*self.upstream), CONNECT_TIMEOUT)
        writer.write(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), CONNECT_TIMEOUT)
        first = head.split(b"\r\n", 1)[0]
        status = first.split()
        if len(status) < 2 or status[1] != b"200":
            writer.close()
            raise ValueError(f"upstream proxy answered {first[:80]!r}")
        return reader, writer

    @staticmethod
    async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter, counts: list, i: int) -> None:
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                counts[i] += len(data)
                dst.write(data)
                await dst.drain()
            if dst.can_write_eof():
                dst.write_eof()
        except (ConnectionError, OSError):
            pass

    @staticmethod
    async def drain_close(writer: asyncio.StreamWriter) -> None:
        try:
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        writer.close()


def parse_routes(items: list) -> dict:
    routes = {}
    for item in items:
        host, sep, route = item.partition("=")
        if not sep or route not in ("direct", "upstream"):
            raise SystemExit(f"gateway: --route takes host=direct or host=upstream, not {item!r}")
        routes[host.lower()] = route
    return routes


def parse_upstream(url: Optional[str]) -> Optional[tuple]:
    if not url:
        return None
    rest = url.split("://", 1)[-1].rstrip("/")
    host, _, port = rest.rpartition(":")
    return host, int(port)


async def serve(args) -> None:
    gw = Gateway(parse_routes(args.route), parse_upstream(args.upstream), args.log, args.open)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = await asyncio.start_unix_server(gw.handle, path=args.socket, limit=HEAD_LIMIT * 2)
    os.chmod(args.socket, 0o600)
    gw.record(event="started", routes=gw.routes, upstream=bool(gw.upstream), mode="open" if args.open else "offline")
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
    async with server:
        await stop
    gw.record(event="stopped", **gw.totals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--socket", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--route", action="append", default=[], help="host=direct or host=upstream; repeatable")
    ap.add_argument("--upstream", help="http://host:port of the upstream proxy, for routes that use it")
    ap.add_argument("--open", choices=("direct", "upstream"),
                    help="open mode: let unlisted hosts out by this route (ports 443 and 80) instead of refusing them")
    args = ap.parse_args(argv)
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
