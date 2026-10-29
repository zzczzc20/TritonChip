#!/usr/bin/env python3
"""Inside an agent's container: 127.0.0.1:<port> to the gateway's Unix socket.

The container has no network, so the CLI is told HTTPS_PROXY=http://127.0.0.1:<port>
and this forwards each connection, byte for byte, to the socket the gateway
serves. It decides nothing: what may pass is the gateway's business. An agent
that stops it only loses its own way out.

Standard library only; runs as the container's main process:
python3 bridge.py --listen 127.0.0.1:3128 --socket /run/tnca-gateway/gateway.sock
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
        if dst.can_write_eof():
            dst.write_eof()
    except (ConnectionError, OSError):
        pass


async def serve(host: str, port: int, socket_path: str) -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(socket_path)
        except OSError:
            writer.close()
            return
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
        for w in (writer, up_writer):
            w.close()

    server = await asyncio.start_server(handle, host, port)
    async with server:
        await server.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:3128")
    ap.add_argument("--socket", required=True)
    args = ap.parse_args(argv)
    host, _, port = args.listen.rpartition(":")
    asyncio.run(serve(host, int(port), args.socket))
    return 0


if __name__ == "__main__":
    sys.exit(main())
