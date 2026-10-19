#!/usr/bin/env python3
"""tnca-service: the agent's way to ask the evaluator for a measurement, within its budget.

    tnca-service budget                what is left
    tnca-service submit score          the hidden validation set, in simulation: an aggregate
    tnca-service submit fpga-quick     synthesis and timing of the whole device on the FPGA part
    tnca-service submit fpga-slow      full link; optional validation-set board measurement
    tnca-service status [ID]           one request, or all of them
    tnca-service result ID             a finished request's result
    tnca-service wait ID [SECONDS]     wait for a request to finish, then print its result

A request snapshots the working directory the way the final evaluation will:
only changes under the task's design paths count. Requests run in the
background; the clock of the run keeps going while they do.

Installed into the agent's container as one read-only file; it only talks to
the harness over the socket at $TNCA_SERVICE.
"""
import json
import os
import socket
import sys
import time

SOCKET = os.environ.get("TNCA_SERVICE", "/run/tnca-service/service.sock")


def call(request: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(120)
    try:
        s.connect(SOCKET)
        s.sendall((json.dumps(request) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    return json.loads(data.decode() or "{}")


def main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "budget":
        reply = call({"op": "budget"})
    elif cmd == "submit" and len(rest) == 1:
        reply = call({"op": "submit", "service": rest[0]})
    elif cmd == "status":
        reply = call({"op": "status", "id": rest[0]} if rest else {"op": "status"})
    elif cmd == "result" and len(rest) == 1:
        reply = call({"op": "result", "id": rest[0]})
    elif cmd == "wait" and rest:
        limit = time.time() + (float(rest[1]) if len(rest) > 1 else 3600.0)
        while True:
            reply = call({"op": "result", "id": rest[0]})
            if reply.get("state") not in ("queued", "running") or time.time() > limit:
                break
            time.sleep(15)
    else:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    print(json.dumps(reply, indent=2, sort_keys=True))
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
