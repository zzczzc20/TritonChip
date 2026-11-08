#!/usr/bin/env python3
"""RTL ground truth for the R41 model calibration, measured the evaluator's way.

Every item runs against its own three-core device built from the task's device
description (python -m tnca.backends.multicore_mmio_sim, the board-register
bench with the platform's observer), exactly as the agent harness's evaluator
measures a point (evaluator._measure_multicore). The cycles kept are the
observer's count at the device boundary (evaluator.observed_multicore_point).

Two kinds of item:
  point  an ArchBench point of the base, run with `python -m archbench.run --point`;
  case   a case file (make() -> module, inputs), run with the test-case skill's
         tnca_case.py against the same device; its driver-side cycles are checked
         against the observer as a point's are.

A variant is the base, or the base plus a candidate directory (candidate.json +
files/, the harness's format): its hardware files go to the device tree, all of
it to the host tree (evaluator._trees).

Records go to a store shared with other sessions (see STORE_README below): one
JSON per (base commit, hardware digest, software digest, active cores, item).
Before an item runs, the store is checked and a claim file is created with
O_EXCL, so two sessions never run the same measurement.

    python3 truth.py --harness ~/tritonchip-agent-harness-work/harness-pool \
        --site ~/.config/tritonchip-agent-harness/site-islands.json \
        --skill DIR/testcase-design --store ~/tritonchip-rtl-truth \
        --variant base [--candidate DIR] --items items.json --work DIR --session NAME [--jobs 12]

Only visible points and agent-visible cases may be measured with this tool
(the calibration may not see a hidden point); --hidden FILE refuses any item
named in that list.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

STORE_README = """# R41 RTL ground truth (shared)

Measured the agent harness evaluator's way (three-core device per item, the
platform's observer, boundary cycles). Written by
tools/performance_harness/systemc/r41cal/truth.py (branch author-model-r41).

records/<key>.json   one measurement; key = <base12>-hw<10>-sw<10>-c<cores>-<item slug>-<item digest 8>
claims/<key>.claim   a measurement in flight (session, host, pid, start); removed when it ends

Rules for every session that writes here:
  1. Before running a measurement, look for its key in records/ and claims/.
  2. Claim it by creating claims/<key>.claim with O_EXCL; remove the claim when done.
  3. Only visible ArchBench points and agent-visible cases: never a hidden-scored
     validation point or a held-out point.
  4. A record from another measurement path may be added only after a cross-check
     shows its cycles equal this path's on the same items.
"""

HARDWARE = ("rtl/", "spec/r1_profiles.json", "spec/devices/")


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_digests(candidate):
    """(hardware digest, software digest) of a candidate's taken files; 'base' for none."""
    if candidate is None:
        return "base", "base"
    report = json.loads((candidate / "candidate.json").read_text())
    hw, sw = [], []
    for c in sorted(report["taken"], key=lambda c: c["path"]):
        f = candidate / "files" / c["path"]
        body = b"<deleted>" if c["status"] == "deleted" else f.read_bytes()
        (hw if c["path"].startswith(HARDWARE) else sw).append(f"{c['path']}\0{sha(body)}")
    return (sha("\n".join(hw).encode()) if hw else "base"), (sha("\n".join(sw).encode()) if sw else "base")


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:80]


def item_digest(item):
    if item["kind"] == "case":
        return sha(Path(item["file"]).read_bytes())
    return sha(item["id"].encode())


def key_of(base, hw, sw, cores, item):
    h = (hw if hw == "base" else hw[:10])
    s = (sw if sw == "base" else sw[:10])
    return f"{base[:12]}-hw{h}-sw{s}-c{cores}-{slug(item['id'])}-{item_digest(item)[:8]}"


def claim(store, key, session):
    path = store / "claims" / f"{key}.claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"session": session, "host": socket.gethostname(), "pid": os.getpid(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True, help="the testcase-design skill directory")
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--hidden", type=Path, help="a JSON list of point ids that must never be measured")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.harness))
    from tritonchip_agent_harness import evaluator as E, provider
    from tritonchip_agent_harness.provider import AgentContainer, Mount
    from tritonchip_agent_harness.siteconfig import load_site
    from tritonchip_agent_harness.task import load_task

    site = load_site(args.site)
    task = load_task(args.harness / "tritonchip_agent_harness/tasks/r41-dp-formal/task.json")
    ev = task.evaluation
    assert ev["profile"] == "core-vrf-sync-v5-x8-g2c4-dp-board" and ev["device"] == "u280-3core-r41", ev
    repo = Path(os.path.expanduser(str(site.repository)))

    items = json.loads(args.items.read_text())
    for it in items:
        assert it["kind"] in ("point", "case"), it
        if it["kind"] == "case":
            it["file"] = str(Path(it["file"]).resolve())
    if args.hidden:
        hidden = set(json.loads(args.hidden.read_text()))
        bad = [it["id"] for it in items if it["id"] in hidden]
        if bad:
            raise SystemExit(f"refused: {len(bad)} item(s) are hidden points")

    store = args.store.expanduser()
    for d in (store, store / "records", store / "claims"):
        d.mkdir(parents=True, exist_ok=True)
    if not (store / "README.md").exists():
        (store / "README.md").write_text(STORE_README)

    candidate = args.candidate.resolve() if args.candidate else None
    hw, sw = candidate_digests(candidate)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    host, dev = E._trees(repo, task, candidate, work)
    desc = json.loads((dev / E.device_path(task)).read_text())
    cores = int(desc["platform"]["core_count"])
    problems = E.device_check(repo, task, dev)
    if problems:
        raise SystemExit("device check: " + "; ".join(problems))

    todo = []
    for it in items:
        k = key_of(task.base_commit, hw, sw, cores, it)
        if (store / "records" / f"{k}.json").exists():
            log("have", k)
            continue
        if args.dry_run:
            log("would run", k)
            continue
        if not claim(store, k, args.session):
            log("claimed elsewhere", k)
            continue
        todo.append((k, it))
    if not todo:
        log("nothing to run")
        return 0

    out = work / "measure"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cases_dir = work / "cases"
    shutil.rmtree(cases_dir, ignore_errors=True)
    cases_dir.mkdir()
    mailroot = out / "mailbox"
    for d in (mailroot, mailroot / "in", mailroot / "out"):
        d.mkdir(exist_ok=True)
        d.chmod(0o777)
    hostout = out / "host"
    hostout.mkdir()
    memory = int(ev.get("memory_bytes", E.MULTICORE_MEMORY))
    timeout = float(ev["timeout_seconds"])
    dpath = f"/work/{E.device_path(task)}"
    name = f"rtltruth-{slug(args.variant)}-{os.getpid()}"
    devices = []
    t0 = time.time()
    try:
        for i, (k, it) in enumerate(todo):
            inbox, outbox, obs = mailroot / "in" / str(i), mailroot / "out" / str(i), out / f"observer-{i}"
            for d in (inbox, outbox, obs):
                d.mkdir(exist_ok=True)
                d.chmod(0o777)
            dname = f"{name}-dev{i}"
            c = AgentContainer(name=dname, image=site.images["toolchain"], cpus=1, memory="4g", pids=256,
                               mounts=[Mount(dev, "/work"), Mount(inbox, "/mb/in"),
                                       Mount(outbox, "/mb/out", read_only=False),
                                       Mount(obs, "/obs", read_only=False)],
                               env={"PYTHONDONTWRITEBYTECODE": "1"},
                               labels={"tritonchip-rtl-truth": "device"}, workdir="/work")
            provider.remove(dname)
            devices.append(dname)
            provider.docker(*c.run_args(["/opt/torchenv/bin/python", "-m", "tnca.backends.multicore_mmio_sim",
                                         "--device", dpath, "--inbox", "/mb/in", "--outbox", "/mb/out",
                                         "--build", "/obs/build", "--observe", "/obs/observe.log",
                                         "--memory-bytes", str(memory), "--watchdog", str(E.MULTICORE_WATCHDOG)],
                                        detach=True, remove=False))
            (hostout / f"backend-{i}.json").write_text(json.dumps(
                {"board_top": True, "device": dpath, "inbox": f"/mb/in/{i}", "outbox": f"/mb/out/{i}",
                 "memory_bytes": memory, "hbm_dma": True}))
            if it["kind"] == "case":
                shutil.copy2(it["file"], cases_dir / f"{i}-{Path(it['file']).name}")
        env = (f'TNCA_BACKEND=fpga-multicore-sim TNCA_BACKEND_ARGS="$(cat /out/backend-$1.json)" '
               f'TNCA_COMPILE_CACHE_DIR=/out/caches/tnca-inductor TNCA_ACTIVE_CORES={cores} '
               f'TNCA_HOST_TIMEOUT={timeout:g}')
        lines = ["set -e", "cd /work", "mkdir -p /out/caches",
                 "make -s python/tnca/data/.staged > /out/stage.log 2>&1",
                 "run_point() {",
                 f'  {env} /opt/torchenv/bin/python -m archbench.run --platform tnca --profile {ev["profile"]} '
                 f'--point "$2" --out /out/archbench-$1.jsonl > /out/run-$1.log 2>&1 || true',
                 "}",
                 "run_case() {",
                 f'  {env} TNCA_ROOT=/work /opt/torchenv/bin/python /opt/tnca-skills/testcase-design/scripts/tnca_case.py '
                 f'"/cases/$2" --profile {ev["profile"]} --json /out/case-$1.jsonl > /out/run-$1.log 2>&1 || true',
                 "}"]
        order = sorted(range(len(todo)), key=lambda i: -float(todo[i][1].get("cost", 0)))  # longest first
        for n in range(0, len(order), args.jobs):
            for i in order[n:n + args.jobs]:
                it = todo[i][1]
                arg = it["id"] if it["kind"] == "point" else f"{i}-{Path(it['file']).name}"
                lines.append(f"run_{it['kind']} {i} '{arg}' &")
            lines.append("wait")
        (hostout / "measure.sh").write_text("\n".join(lines) + "\n")
        h = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                           pids=site.pids,
                           mounts=[Mount(host, "/work", read_only=False), Mount(hostout, "/out", read_only=False),
                                   Mount(mailroot / "in", "/mb/in", read_only=False), Mount(mailroot / "out", "/mb/out"),
                                   Mount(args.skill.resolve(), "/opt/tnca-skills/testcase-design"),
                                   Mount(cases_dir, "/cases")],
                           labels={"tritonchip-rtl-truth": "host"}, workdir="/work")
        log("running", len(todo), "items of", args.variant, "in", name)
        exit_code = E._run(h, ["sh", "/out/measure.sh"], timeout)
    finally:
        E._stop_devices(devices, out)
    wall = round(time.time() - t0)

    drain = ev.get("drain_cycles")
    written = 0
    for i, (k, it) in enumerate(todo):
        rec = None
        if it["kind"] == "point":
            path = hostout / f"archbench-{i}.jsonl"
            for line in (E._container_text(path).splitlines() if path.exists() else []):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("point_id") == it["id"]:
                    rec = r
        else:
            path = hostout / f"case-{i}.jsonl"
            for line in (E._container_text(path).splitlines() if path.exists() else []):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = {"status": r.get("status"), "time": {"device": (r.get("time") or {}).get("device_cycles")},
                       "correctness": r.get("correctness"), "fallbacks": r.get("fallbacks"),
                       "case_record": {kk: r.get(kk) for kk in ("problems", "time", "host", "bound", "eta", "tol")}}
        obs = E.read_multicore_observer(out / f"observer-{i}" / "observe.log")
        point = E.observed_multicore_point(rec, obs, None if drain is None else int(drain))
        record = {"schema": "tritonchip-rtl-truth/1", "key": k, "variant": args.variant,
                  "base_commit": task.base_commit, "hardware_digest": hw, "software_digest": sw,
                  "candidate": None if candidate is None else
                  [c["path"] for c in json.loads((candidate / "candidate.json").read_text())["taken"]],
                  "active_cores": cores, "item": {kk: it[kk] for kk in it if kk != "file"},
                  "item_digest": item_digest(it), "profile": ev["profile"], "device": ev["device"],
                  "memory_bytes": memory, "drain_cycles": drain, "platform_commit": ev["platform"]["commit"],
                  "path": "evaluator multicore: board-register bench per item, observer boundary cycles",
                  "result": point, "host_exit": exit_code, "sweep_wall_seconds": wall,
                  "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "session": args.session,
                  "host": socket.gethostname()}
        if point.get("status") is None and point.get("cycles") is None:
            log("no result", k)
        else:
            (store / "records" / f"{k}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
            written += 1
            log(point.get("status"), point.get("cycles"), k)
        (store / "claims" / f"{k}.claim").unlink(missing_ok=True)
    log("done", written, "of", len(todo), "records in", wall, "s; host exit", exit_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
