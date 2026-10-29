"""smoke: one trivial request per account, through the same launch path a task run uses.

A smoke run passes when the CLI answers, the model that served it is one the
account names, its usage is reported, the gateway logged the traffic, and --
in offline mode -- nothing went to a host the account does not list, and an
unlisted host is refused from inside the same container.

This is the gateway's and the adapters' regression test: after any change to
either, or to a CLI version, every account's smoke run must still pass, and
its list of refused hosts is worth reading.
"""

from __future__ import annotations

import json
import secrets
import time

from . import doctor, provider, runner
from .siteconfig import Site, credential_status

PROMPT = "Reply with exactly: OK"


def run(site: Site, label: str, mode: str = "offline") -> dict:
    account = site.account(label)
    cred = credential_status(account)
    if not cred["ok"]:
        return {"label": label, "ok": False, "problems": cred["problems"]}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)
    run_dir = site.state_dir / "runs" / f"smoke-{label}-{stamp}"
    run_dir.mkdir(parents=True)
    result = {"schema": "tritonchip-agent-smoke/1", "run_id": run_dir.name,
              **runner.run_agent(site, account, PROMPT, mode, run_dir, None, timeout=300)}

    # The boundary, probed from a fresh container on the same gateway configuration.
    gw = f"tnca-gw-{run_dir.name}-probe"
    socket_dir, _ = doctor.start_gateway(site, run_dir / "probe", gw, list(account.hosts))
    try:
        shell = (f"python3 /opt/tnca-harness/bridge.py --listen {doctor.BRIDGE} --socket {doctor.SOCKET} & sleep 1; "
                 f"python3 /opt/tnca-harness/canary.py net --via {doctor.BRIDGE} --allowed {account.hosts[0]} "
                 f"--denied example.com")
        proc, _ = doctor._canary(site, f"tnca-probe-{run_dir.name}", None, {}, [], shell=shell,
                                 extra_mounts=[provider.Mount(socket_dir, doctor.SOCKET_DIR)])
        boundary = doctor._parse(proc)
    finally:
        doctor.stop_gateway(gw)

    served = [m.split("/")[-1] for m in result.get("served_models", [])]
    unlisted = [h for h, t in result["gateway"]["tunnels"].items() if not t["listed"]]
    result["checks"] = [
        {"check": "CLI answered", "ok": result.get("exit") == 0
            and (result.get("text") or "").strip().strip(".") == "OK", "detail": result.get("text")},
        {"check": "served only the account's models", "ok": bool(served) and set(served) <= set(account.models.values()),
         "detail": result.get("served_models")},
        {"check": "usage reported", "ok": bool(result.get("usage"))},
        {"check": "traffic logged by the gateway", "ok": bool(result["gateway"]["tunnels"])},
    ]
    if mode == "offline":
        result["checks"].append({"check": "no traffic to an unlisted host", "ok": not unlisted, "detail": unlisted})
    result["checks"] += [{**b, "check": "boundary: " + b["check"]} for b in boundary]
    result["ok"] = all(c["ok"] for c in result["checks"])
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["run_dir"] = str(run_dir)
    return result


__all__ = ["PROMPT", "run"]
