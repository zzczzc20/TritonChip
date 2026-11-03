"""smoke: one trivial request per account, through the same launch path a task run uses.

A smoke run passes when the CLI answers, the model that served it is one the
account names, its usage is reported, the gateway logged the traffic, and --
in offline mode -- nothing went to a host the account does not list, and an
unlisted host is refused from inside the same container.

This is the gateway's and the adapters' regression test: after any change to
either, or to a CLI version, every account's smoke run must still pass, and
its list of refused hosts is worth reading.

With `probe="skills"` the request is to list the skills the CLI has, and the
run passes only when every skill the harness provides is linked into the CLI's
skills directory and named in the answer: the CLI loaded it on its own.
"""

from __future__ import annotations

import json
import secrets
import time

from . import doctor, provider, runner
from .siteconfig import Site, credential_status

PROMPT = "Reply with exactly: OK"
PROMPTS = {"ok": PROMPT,
           "skills": "List the names of the skills available to you, one per line, exactly as they are named, "
                     "and nothing else. Do not run any of them."}


def run(site: Site, label: str, mode: str = "offline", probe: str = "ok") -> dict:
    account = site.account(label)
    cred = credential_status(account)
    if not cred["ok"]:
        return {"label": label, "ok": False, "problems": cred["problems"]}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)
    run_dir = site.state_dir / "runs" / f"smoke-{label}-{stamp}"
    run_dir.mkdir(parents=True)
    result = {"schema": "tritonchip-agent-smoke/1", "run_id": run_dir.name, "probe": probe,
              **runner.run_agent(site, account, PROMPTS[probe], mode, run_dir, None, timeout=300)}

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
    text = (result.get("text") or "").strip()
    if probe == "ok":
        answered = {"check": "CLI answered", "ok": result.get("exit") == 0 and text.strip(".") == "OK",
                    "detail": result.get("text")}
    else:
        provided = sorted(result.get("skills", {}).get("provided", {}))
        linked = sorted(result.get("skills", {}).get("linked", []))
        answered = {"check": "every provided skill linked and named by the CLI",
                    "ok": result.get("exit") == 0 and bool(provided) and linked == provided
                    and all(name in text for name in provided),
                    "detail": {"provided": provided, "linked": linked, "answer": result.get("text"),
                               "link_error": result.get("skills", {}).get("error")}}
    result["checks"] = [
        answered,
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


__all__ = ["PROMPT", "PROMPTS", "run"]
