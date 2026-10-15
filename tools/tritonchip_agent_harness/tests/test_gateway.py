"""The gateway, end to end on this machine: a fake upstream proxy stands in for the internet."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import gateway  # noqa: E402  (a script; imported as a module here)


class FakeUpstream:
    """An HTTP CONNECT proxy that answers `status` and then echoes the tunnel."""

    def __init__(self, status: bytes = b"200") -> None:
        self.status = status
        self.asked = []

    async def handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        self.asked.append(head.split(b"\r\n", 1)[0].decode())
        writer.write(b"HTTP/1.1 " + self.status + b" X\r\n\r\n")
        await writer.drain()
        if self.status == b"200":
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        writer.close()


async def exchange(sock: str, request: bytes, payload: bytes = b"") -> tuple:
    reader, writer = await asyncio.open_unix_connection(sock)
    writer.write(request)
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    echoed = b""
    if payload and b" 200 " in head.split(b"\r\n", 1)[0] + b" ":
        writer.write(payload)
        await writer.drain()
        echoed = await reader.readexactly(len(payload))
    writer.close()
    return head.split(b"\r\n", 1)[0].decode(), echoed


class GatewayTest(unittest.TestCase):
    def run_case(self, upstream_status: bytes, cases, open_route=None):
        async def body(tmp: Path):
            fake = FakeUpstream(upstream_status)
            up = await asyncio.start_server(fake.handle, "127.0.0.1", 0)
            port = up.sockets[0].getsockname()[1]
            gw = gateway.Gateway({"allowed.test": "upstream"}, ("127.0.0.1", port), str(tmp / "log.jsonl"), open_route)
            sock = str(tmp / "gw.sock")
            server = await asyncio.start_unix_server(gw.handle, path=sock)
            results = [await exchange(sock, req, payload) for req, payload in cases]
            await asyncio.sleep(0.1)
            server.close()
            up.close()
            gw.log.close()
            entries = [json.loads(l) for l in (tmp / "log.jsonl").read_text().splitlines()]
            return results, entries, fake.asked

        with tempfile.TemporaryDirectory() as d:
            return asyncio.run(body(Path(d)))

    def test_listed_host_tunnels_and_counts_bytes(self):
        results, entries, asked = self.run_case(b"200", [(b"CONNECT allowed.test:443 HTTP/1.1\r\n\r\n", b"ping!")])
        self.assertIn("200", results[0][0])
        self.assertEqual(results[0][1], b"ping!")
        self.assertEqual(asked, ["CONNECT allowed.test:443 HTTP/1.1"])
        tunnel = [e for e in entries if e["event"] == "tunnel"][0]
        self.assertEqual((tunnel["host"], tunnel["bytes_up"], tunnel["bytes_down"]), ("allowed.test", 5, 5))

    def test_refusals(self):
        results, entries, asked = self.run_case(b"200", [
            (b"CONNECT other.test:443 HTTP/1.1\r\n\r\n", b""),
            (b"CONNECT allowed.test:80 HTTP/1.1\r\n\r\n", b""),
            (b"GET http://allowed.test/ HTTP/1.1\r\n\r\n", b""),
            (b"CONNECT allowed.test HTTP/1.1\r\n\r\n", b""),
        ])
        self.assertEqual([r[0].split()[1] for r in results], ["403", "403", "405", "400"])
        self.assertEqual(asked, [], "nothing refused may reach the upstream")
        self.assertEqual(sum(1 for e in entries if e["event"] == "refused"), 4)

    def test_upstream_refusal_is_a_502_and_logged(self):
        results, entries, _ = self.run_case(b"403", [(b"CONNECT allowed.test:443 HTTP/1.1\r\n\r\n", b"")])
        self.assertIn("502", results[0][0])
        self.assertTrue(any(e["event"] == "failed" for e in entries))

    def test_open_mode_lets_unlisted_hosts_out_and_logs_them(self):
        results, entries, asked = self.run_case(b"200", [(b"CONNECT other.test:443 HTTP/1.1\r\n\r\n", b"hi")],
                                                open_route="upstream")
        self.assertIn("200", results[0][0])
        self.assertEqual(asked, ["CONNECT other.test:443 HTTP/1.1"])
        tunnel = [e for e in entries if e["event"] == "tunnel"][0]
        self.assertEqual((tunnel["host"], tunnel["listed"]), ("other.test", False))

    def test_open_mode_never_reaches_this_machine_or_its_network(self):
        targets = [b"127.0.0.1:443", b"localhost:443", b"[::1]:443", b"2130706433:443", b"127.1:443",
                   b"10.0.0.7:443", b"192.168.1.1:80", b"169.254.169.254:80", b"[::ffff:127.0.0.1]:443"]
        results, entries, asked = self.run_case(b"200", [(b"CONNECT " + t + b" HTTP/1.1\r\n\r\n", b"") for t in targets],
                                                open_route="upstream")
        self.assertEqual([r[0].split()[1] for r in results], ["403"] * len(targets))
        self.assertEqual(asked, [], "nothing internal may reach the upstream")
        self.assertEqual({e.get("reason") for e in entries if e["event"] == "refused"}, {"internal address"})

    def test_internal_addresses(self):
        self.assertTrue(gateway.internal_ip("10.1.2.3"))
        self.assertFalse(gateway.internal_ip("8.8.8.8"))
        self.assertIsNone(gateway.internal_ip("example.com"))

    def test_routes_parse(self):
        self.assertEqual(gateway.parse_routes(["A.example.com=direct"]), {"a.example.com": "direct"})
        with self.assertRaises(SystemExit):
            gateway.parse_routes(["a.example.com=sideways"])
        self.assertEqual(gateway.parse_upstream("http://127.0.0.1:7890/"), ("127.0.0.1", 7890))


if __name__ == "__main__":
    unittest.main()
