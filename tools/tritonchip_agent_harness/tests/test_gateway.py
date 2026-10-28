"""The gateway, end to end on this machine: a fake upstream proxy stands in for the internet."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    def run_case(self, upstream_status: bytes, cases, open_route=None, answers=None):
        """`answers` stands in for DNS: each name's successive answers, the last one repeating."""
        self.resolved = []

        async def resolve(host, port):
            self.resolved.append(host)
            queue = (answers or {}).get(host)
            if not queue:
                raise OSError(f"{host}: no such name in this test")
            return queue.pop(0) if len(queue) > 1 else queue[0]

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

        with tempfile.TemporaryDirectory() as d, mock.patch.object(gateway, "resolve", resolve):
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
                                                open_route="upstream", answers={"other.test": [["93.184.216.34"]]})
        self.assertIn("200", results[0][0])
        self.assertEqual(asked, ["CONNECT 93.184.216.34:443 HTTP/1.1"], "the upstream gets the address that passed")
        tunnel = [e for e in entries if e["event"] == "tunnel"][0]
        self.assertEqual((tunnel["host"], tunnel["address"], tunnel["listed"]), ("other.test", "93.184.216.34", False))

    def test_a_name_is_resolved_once_and_its_tunnel_goes_to_that_address(self):
        # DNS rebinding: a public answer to the check, loopback to any later lookup.
        def rebinding():
            return {"rebind.test": [["93.184.216.34"], ["127.0.0.1"]]}
        _, _, asked = self.run_case(b"200", [(b"CONNECT rebind.test:443 HTTP/1.1\r\n\r\n", b"")],
                                    open_route="upstream", answers=rebinding())
        self.assertEqual((asked, self.resolved), (["CONNECT 93.184.216.34:443 HTTP/1.1"], ["rebind.test"]))
        opened = []

        async def no_connection(host, port, **kw):
            opened.append((host, port))
            raise OSError("no connection in this test")
        with mock.patch.object(gateway.asyncio, "open_connection", no_connection):
            results, _, _ = self.run_case(b"200", [(b"CONNECT rebind.test:443 HTTP/1.1\r\n\r\n", b"")],
                                          open_route="direct", answers=rebinding())
        self.assertIn("502", results[0][0])
        self.assertEqual((opened, self.resolved), ([("93.184.216.34", 443)], ["rebind.test"]))

    def test_local_names_with_trailing_dots_never_reach_the_upstream(self):
        targets = [b"localhost.:443", b"LOCALHOST.:443", b"api.localhost.:443", b"localhost..:443", b".:443"]
        results, _, asked = self.run_case(b"200", [(b"CONNECT " + t + b" HTTP/1.1\r\n\r\n", b"") for t in targets],
                                          open_route="upstream")
        self.assertEqual([r[0].split()[1] for r in results], ["403"] * len(targets))
        self.assertEqual((asked, self.resolved), ([], []), "refused before any lookup, and never sent upstream")

    def test_a_name_with_any_internal_answer_is_refused(self):
        results, entries, asked = self.run_case(b"200", [(b"CONNECT mixed.test:443 HTTP/1.1\r\n\r\n", b"")],
                                                open_route="upstream",
                                                answers={"mixed.test": [["93.184.216.34", "10.0.0.5"]]})
        self.assertIn("403", results[0][0])
        self.assertEqual(asked, [])
        self.assertEqual([e.get("reason") for e in entries if e["event"] == "refused"], ["internal address"])

    def test_an_unresolvable_name_fails_without_reaching_the_upstream(self):
        results, _, asked = self.run_case(b"200", [(b"CONNECT nowhere.test:443 HTTP/1.1\r\n\r\n", b"")],
                                          open_route="upstream")
        self.assertIn("502", results[0][0])
        self.assertEqual(asked, [])

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
