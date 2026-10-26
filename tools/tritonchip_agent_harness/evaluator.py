"""The evaluator, at CHIA's level: rebuild from source with no AI involved, measure with its own copies.

It builds a tree from the FULL base commit (nothing hidden) with the
candidate's files on top -- only files under the task's design paths, which
the gate already decided -- so the testbench, ArchBench, the tools and the
flow are the base's, never the agent's. It then runs ArchBench on the task's
validation and final points, in a container with no network and no
credential, and compares every point with the base's own cycles.

Each point is judged on its own. A point that is wrong, falls back to the
host, or produced no record is a failed point, and the other points still
count: a set reports how many points passed, the geometric-mean speedup over
the points that passed, and fast_1, the share of ALL points that passed and
beat the base. fast_1 is there because the passed-only mean alone would reward
breaking a point the candidate had made slower.

With an observer (the task names an evaluator platform), the device runs out
of the candidate's reach (README section 11, P2.1): it is built from the base
code with only the candidate's hardware on top, and each point's device runs
in a container of its own, behind a Unix socket, where the platform counts the
device's cycles and flags memory activity outside a launch. The score is that
count, not the number the measuring process reports. The candidate's code still
runs in that process (it is the compiler); P2.2 is what checks the results it
hands back.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import provider
from .provider import AgentContainer, Mount
from .siteconfig import Site
from .task import Task
from .workspace import apply_candidate, export, git

SCRIPT = """set -e
cd /work
make -s python/tnca/data/.staged > /out/stage.log 2>&1
/opt/torchenv/bin/tnca-sim prepare --profile {profile} > /out/prepare.log 2>&1
/opt/torchenv/bin/python -m archbench.run --platform tnca --profile {profile} --jobs {jobs} \\
    --point {points} --out /out/archbench.jsonl > /out/archbench.log 2>&1 || true
"""


# The candidate's paths the device is built from. Everything else in the
# device's tree is the base's, so building it runs no candidate code.
HARDWARE = ("rtl/", "spec/r1_profiles.json", "spec/devices/")
# The device's own deadline for one launch, as the runtime's default session sets it.
DEADLINE = 5_000_000
SOCKETS = "/run/tnca-device"

BUILD = """set -e
cd /work
make -s python/tnca/data/.staged > /out/stage.log 2>&1
/opt/torchenv/bin/tnca-sim prepare --profile {profile} > /out/prepare.log 2>&1
/opt/torchenv/bin/python -c "from tnca import device; print(device.require('{profile}'))" > /out/bench.txt
"""

# Candidate RTL is simulated where the observer writes its log, so it may not
# reach files or the host: no file or system tasks, no include from outside the tree.
_RTL_FORBIDDEN = re.compile(r"\$(fopen|fclose|fwrite|fdisplay|fstrobe|fmonitor|fflush|fscanf|fgets|fgetc|fread|"
                            r"ungetc|rewind|fseek|system|readmemh|readmemb|writememh|writememb|dumpfile|dumpvars)\b")
_RTL_INCLUDE = re.compile(r'`include\s+"([^"]+)"')
_RTL_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_RTL_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
# Ways RTL can make simulation and synthesis see different designs, or reach
# into the testbench that counts it. Each is refused where the candidate adds
# it; what the base already has, in the same file, is not the candidate's.
_RTL_SPLIT = (
    ("synthesis pragma", re.compile(r"(?i)\b(?:synthesis|synopsys|pragma)\s+(?:translate_(?:off|on)|full_case|parallel_case)\b"),
     "raw"),
    ("tool macro", re.compile(r"`(?:ifdef|ifndef|elsif)\s+(?:SYNTHESIS|__ICARUS__|VERILATOR|XILINX_SIMULATOR|XIL_\w*|IVERILOG)\b"),
     "code"),
    ("defparam", re.compile(r"\bdefparam\b"), "code"),
    ("hierarchical reference", re.compile(r"(?<![\w.$`'])\$?(?:root|[A-Za-z_][\w$]*)\.[A-Za-z_][\w$]*"), "code"),
)


def _rtl_findings(text: str) -> set:
    """(rule, the stripped line) for every construct of _RTL_SPLIT in one file."""
    code = _RTL_STRING.sub('""', _RTL_COMMENT.sub(" ", text))
    found = set()
    for name, pattern, where in _RTL_SPLIT:
        source = text if where == "raw" else code
        for line in source.splitlines():
            if pattern.search(line):
                found.add((name, " ".join(line.split())))
    return found


def rtl_check(candidate: Optional[Path], repo: Optional[Path] = None) -> list:
    """Why the candidate's RTL may not be built and simulated; [] when it may.

    With ``repo``, a split construct the base already has on the same line of
    the same file is the base's, not the candidate's.
    """
    if candidate is None:
        return []
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    problems = set()
    for c in report["taken"]:
        path = c["path"]
        if c["status"] == "deleted" or not path.startswith("rtl/"):
            continue
        raw = (candidate / "files" / path).read_text(encoding="utf-8", errors="replace")
        text = _RTL_COMMENT.sub(" ", raw)
        for m in _RTL_FORBIDDEN.finditer(text):
            problems.add(f"{path}: ${m.group(1)}")
        for m in _RTL_INCLUDE.finditer(text):
            if m.group(1).startswith("/") or ".." in Path(m.group(1)).parts:
                problems.add(f'{path}: `include "{m.group(1)}"')
        base = set()
        if repo is not None and c["status"] != "added":
            try:
                base = _rtl_findings(git("-C", str(repo), "show", f"{report['base_commit']}:{path}").decode(
                    "utf-8", errors="replace"))
            except Exception:  # noqa: BLE001 -- a file the base does not have adds everything
                base = set()
        for name, line in sorted(_rtl_findings(raw) - base):
            problems.add(f"{path}: {name}: {line[:80]}")
    return sorted(problems)


def _overlay_platform(repo: Path, platform: dict, tree: Path) -> None:
    """The evaluator's own platform, from the commit the task pins, over whatever the base had."""
    for path in platform["paths"]:
        target = tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("-C", str(repo), "show", f"{platform['commit']}:{path}"))


def read_observer(path: Path) -> Optional[dict]:
    """The platform's log for one device: its count of every launch, and activity outside one."""
    if not path.exists():
        return None
    cmds, end = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words:
            continue
        try:
            fields = {k: int(v) for k, v in (w.split("=", 1) for w in words[1:])}
        except ValueError:
            continue
        if words[0] == "cmd":
            cmds.append(fields)
        elif words[0] == "end":
            end = fields
    launches = [c for c in cmds if c.get("op") == 1]
    return {"launches": len(launches),
            "cycles": sum(c.get("obs", 0) for c in launches),
            "done_cycles": sum(c.get("done", 0) for c in launches),
            "failed_launches": sum(1 for c in launches if c.get("status", 0) != 0),
            "after_done": sum(c.get("after_done", 0) for c in cmds),
            "outside": sum(c.get("outside", 0) for c in cmds) + (end or {}).get("outside", 0),
            "extra": sum(c.get("extra", 0) for c in cmds) + (end or {}).get("extra", 0),
            "ended": end is not None}


def observed_point(record: Optional[dict], obs: Optional[dict]) -> dict:
    """One point as the evaluator scores it: the record's verdict, the observer's cycles."""
    rec_cycles = ((record or {}).get("time") or {}).get("device")
    status = (record or {}).get("status")
    problem = None
    if obs is None or not obs["ended"] or obs["launches"] == 0:
        problem = "no-observation"
    elif obs["outside"] or obs["extra"] or obs["after_done"]:
        problem = "activity-outside-launch"
    elif rec_cycles is not None and rec_cycles % 2**32 != obs["done_cycles"] % 2**32:
        # The measuring process reported cycles the device never did: it was tampered with.
        problem = "record-disagrees"
    if status == "pass" and problem:
        status = problem
    return {"status": status, "cycles": obs["cycles"] if obs else None, "record_cycles": rec_cycles,
            "fallbacks": (record or {}).get("fallbacks"),
            "max_rel_err": ((record or {}).get("correctness") or {}).get("max_rel_err"),
            "observer": obs, "observer_problem": problem}


# A multicore task (evaluation.device names a description in spec/devices):
# the device is the image's own top in the board-register bench, run by
# `python -m tnca.backends.multicore_mmio_sim` in a container of its own; the
# host reaches it only through a mailbox directory, and the bench's observer
# log is written where the host cannot write.
MULTICORE_MEMORY = 16 << 20
MULTICORE_WATCHDOG = 2_000_000_000


def read_multicore_observer(path: Path) -> Optional[dict]:
    """The board-register bench's log: every launch's MMIO endpoints, the
    count the device reported, and how long the device took to drain."""
    if not path.exists():
        return None
    launches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words or words[0] != "cmd":
            continue
        try:
            launches.append({k: int(v) for k, v in (w.split("=", 1) for w in words[1:])})
        except ValueError:
            continue
    launches = [c for c in launches if c.get("op") == 1]
    return {"launches": len(launches),
            "cycles": sum(c.get("obs", 0) for c in launches),
            "done_cycles": sum(c.get("done", 0) for c in launches),
            "failed_launches": sum(1 for c in launches if c.get("status", 0) != 0),
            "max_settle": max((c.get("settle", 0) for c in launches), default=0),
            "not_quiet": sum(1 for c in launches if c.get("quiet", 1) != 1),
            "ended": bool(launches)}


def observed_multicore_point(record: Optional[dict], obs: Optional[dict], drain_cycles: Optional[int]) -> dict:
    """One multicore point: the record's verdict, the endpoints' cycles."""
    rec_cycles = ((record or {}).get("time") or {}).get("device")
    status = (record or {}).get("status")
    problem = None
    if obs is None or obs["launches"] == 0:
        problem = "no-observation"
    elif obs["failed_launches"]:
        # The device faulted a launch, whatever the host made of it.
        problem = "device-fault"
    elif (status == "pass" or rec_cycles is not None) and rec_cycles != obs["done_cycles"]:
        # The host reported cycles the device never did: it was tampered with.
        problem = "record-disagrees"
    elif obs["not_quiet"] or (drain_cycles is not None and obs["max_settle"] > drain_cycles):
        # The device said done before its work had drained.
        problem = "not-drained"
    if status == "pass" and problem:
        status = problem
    return {"status": status, "cycles": obs["cycles"] if obs else None, "record_cycles": rec_cycles,
            "fallbacks": (record or {}).get("fallbacks"),
            "max_rel_err": ((record or {}).get("correctness") or {}).get("max_rel_err"),
            "observer": obs, "observer_problem": problem}


def device_path(task: Task) -> str:
    return f"spec/devices/{task.evaluation['device']}.json"


def device_check(repo: Path, task: Task, tree: Path) -> list:
    """Why the candidate's device description may not be built; [] when it may.

    The platform is the base's: the candidate may change the uncore, never the
    card's or the flow's fixed fields, nor the per-core profile the task scores.
    """
    name = device_path(task)
    base = json.loads(git("-C", str(repo), "show", f"{task.base_commit}:{name}"))
    try:
        cand = json.loads((tree / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{name}: {e}"]
    if not isinstance(cand, dict) or not isinstance(cand.get("platform"), dict):
        return [f"{name}: not a device description"]
    problems = [f"{name}: platform.{k}: {cand['platform'].get(k)!r}, fixed at {v!r}"
                for k, v in sorted(base["platform"].items()) if cand["platform"].get(k) != v]
    problems += [f"{name}: platform.{k} is not the base's" for k in sorted(set(cand["platform"]) - set(base["platform"]))]
    if cand.get("name") != base.get("name"):
        problems.append(f"{name}: name {cand.get('name')!r} is not the base's {base.get('name')!r}")
    if cand.get("per_core_profile") != task.evaluation["profile"]:
        problems.append(f"{name}: per_core_profile {cand.get('per_core_profile')!r} is not the scored "
                        f"profile {task.evaluation['profile']!r}")
    return problems


def _run(c: AgentContainer, command: list, timeout: float) -> Optional[int]:
    """A container run to its end, and its exit status; None when it ran past
    the timeout, and it is then removed rather than left to run on."""
    provider.remove(c.name)  # a container of this name left by a run that died
    try:
        return provider.docker(*c.run_args(command, detach=False, remove=True), check=False,
                               timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        provider.remove(c.name)
        return None


def _stop_devices(devices: list, out: Path) -> None:
    """Every device container's log kept, and the container removed. A device
    whose host said goodbye has stopped; one still up gets the rest of a
    minute -- a host that died never says goodbye."""
    deadline = time.time() + 60
    for dname in devices:
        while time.time() < deadline and provider.inspect_state(dname).get("Status") == "running":
            time.sleep(1)
        logs = provider.docker("logs", dname, check=False)
        (out / f"{dname.rsplit('-', 1)[-1]}.log").write_text((logs.stdout or "") + (logs.stderr or ""), encoding="utf-8")
        provider.remove(dname)


def _measure_multicore(site: Site, hosttree: Path, devtree: Path, task: Task, out: Path, name: str,
                       repo: Path) -> dict:
    """Each point against its own three-core device, in a container the candidate's code cannot reach."""
    ev = task.evaluation
    points = list(ev["validation"]) + list(ev["final"])
    out.mkdir(parents=True, exist_ok=True)
    problems = device_check(repo, task, devtree)
    if problems:
        (out / "device-check.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        return {"exit": None, "points": {}, "missing": points, "device_build": "failed",
                "device_problems": problems, "observed": True}
    base_desc = json.loads((devtree / device_path(task)).read_text(encoding="utf-8"))
    cores = int(base_desc["platform"]["core_count"])
    memory = int(ev.get("memory_bytes", MULTICORE_MEMORY))
    timeout = float(ev["timeout_seconds"])
    desc = f"/work/{device_path(task)}"
    # A mailbox per point, one directory per direction (multicore_mmio_sim):
    # the host writes only the inboxes, each device writes only its outbox.
    mailroot = out / "mailbox"
    for d in (mailroot, mailroot / "in", mailroot / "out"):
        d.mkdir(exist_ok=True)
        d.chmod(0o777)
    # The host sees its own directory, never the observers' beside it: the
    # candidate's code runs in the host container, and a count it could
    # rewrite would be no count.
    hostout = out / "host"
    hostout.mkdir(exist_ok=True)
    devices: list = []
    try:
        for i, _ in enumerate(points):
            inbox, outbox, obs = mailroot / "in" / str(i), mailroot / "out" / str(i), out / f"observer-{i}"
            for d in (inbox, outbox, obs):
                d.mkdir(exist_ok=True)
                d.chmod(0o777)
            dname = f"{name}-dev{i}"
            d = AgentContainer(name=dname, image=site.images["toolchain"], cpus=1, memory="4g", pids=256,
                               mounts=[Mount(devtree, "/work"), Mount(inbox, "/mb/in"),
                                       Mount(outbox, "/mb/out", read_only=False),
                                       Mount(obs, "/obs", read_only=False)],
                               env={"PYTHONDONTWRITEBYTECODE": "1"},
                               labels={"tritonchip-agent-harness": "evaluator-device"}, workdir="/work")
            provider.remove(dname)  # a container of this name left by a run that died
            devices.append(dname)
            provider.docker(*d.run_args(["/opt/torchenv/bin/python", "-m", "tnca.backends.multicore_mmio_sim",
                                         "--device", desc, "--inbox", "/mb/in", "--outbox", "/mb/out",
                                         "--build", "/obs/build",
                                         "--observe", "/obs/observe.log", "--memory-bytes", str(memory),
                                         "--watchdog", str(MULTICORE_WATCHDOG)],
                                        detach=True, remove=False))
            args = {"board_top": True, "device": desc, "inbox": f"/mb/in/{i}", "outbox": f"/mb/out/{i}",
                    "memory_bytes": memory, "hbm_dma": True}
            (hostout / f"backend-{i}.json").write_text(json.dumps(args), encoding="utf-8")
        lines = ["set -e", "cd /work", "make -s python/tnca/data/.staged > /out/stage.log 2>&1",
                 "run() {",
                 f'  TNCA_BACKEND=fpga-multicore-sim TNCA_BACKEND_ARGS="$(cat /out/backend-$1.json)" '
                 f'TNCA_ACTIVE_CORES={cores} TNCA_HOST_TIMEOUT={timeout:g} /opt/torchenv/bin/python -m archbench.run '
                 f'--platform tnca '
                 f'--profile {ev["profile"]} --point "$2" --out /out/archbench-$1.jsonl > /out/archbench-$1.log 2>&1 || true',
                 "}"]
        jobs = max(1, int(ev["jobs"]))
        for k in range(0, len(points), jobs):
            lines += [f"run {i} '{points[i]}' &" for i in range(k, min(k + jobs, len(points)))] + ["wait"]
        (hostout / "evaluate.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
        h = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                           pids=site.pids, mounts=[Mount(hosttree, "/work", read_only=False),
                                                   Mount(hostout, "/out", read_only=False),
                                                   Mount(mailroot / "in", "/mb/in", read_only=False),
                                                   Mount(mailroot / "out", "/mb/out")],
                           labels={"tritonchip-agent-harness": "evaluator"}, workdir="/work")
        exit_code = _run(h, ["sh", "/out/evaluate.sh"], timeout)
    finally:
        _stop_devices(devices, out)
    drain = ev.get("drain_cycles")
    records = {}
    for i, p in enumerate(points):
        rec = None
        path = hostout / f"archbench-{i}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("point_id") == p:
                    rec = r
        obs = read_multicore_observer(out / f"observer-{i}" / "observe.log")
        if rec is not None or obs is not None:
            records[p] = observed_multicore_point(rec, obs, None if drain is None else int(drain))
    return {"exit": exit_code, "points": records, "missing": [p for p in points if p not in records],
            "observed": True}


def _build_device(site: Site, devtree: Path, task: Task, out: Path, name: str) -> Optional[str]:
    """The device the points run on, built in a container from the base code and the candidate's hardware."""
    profile = task.evaluation["profile"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "build.sh").write_text(BUILD.format(profile=profile), encoding="utf-8")
    c = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                       pids=site.pids, mounts=[Mount(devtree, "/work", read_only=False), Mount(out, "/out", read_only=False)],
                       env={"TNCA_CACHE": "/work/.tnca-cache"},
                       labels={"tritonchip-agent-harness": "evaluator-device-build"}, workdir="/work")
    _run(c, ["sh", "/out/build.sh"], float(task.evaluation["timeout_seconds"]))
    bench = (out / "bench.txt").read_text(encoding="utf-8").strip() if (out / "bench.txt").exists() else ""
    return bench or None


def _measure_observed(site: Site, hosttree: Path, devtree: Path, task: Task, out: Path, name: str) -> dict:
    """Each point against its own device, in a container the candidate's code cannot reach."""
    ev = task.evaluation
    points = list(ev["validation"]) + list(ev["final"])
    out.mkdir(parents=True, exist_ok=True)
    bench = _build_device(site, devtree, task, out / "device-build", f"{name}-build")
    if bench is None:
        return {"exit": None, "points": {}, "missing": points, "device_build": "failed"}
    sockets = out / "sockets"
    sockets.mkdir(exist_ok=True)
    sockets.chmod(0o777)
    # The host sees its own directory, never the observers' beside it (as in
    # _measure_multicore).
    hostout = out / "host"
    hostout.mkdir(exist_ok=True)
    timeout = float(ev["timeout_seconds"])
    devices: list = []
    try:
        for i, _ in enumerate(points):
            obs = out / f"observer-{i}"
            obs.mkdir(exist_ok=True)
            obs.chmod(0o777)
            dname = f"{name}-dev{i}"
            d = AgentContainer(name=dname, image=site.images["toolchain"], cpus=1, memory="2g", pids=256,
                               mounts=[Mount(devtree, "/work"), Mount(sockets, SOCKETS, read_only=False),
                                       Mount(obs, "/obs", read_only=False)],
                               env={"TNCA_CACHE": "/work/.tnca-cache", "PYTHONDONTWRITEBYTECODE": "1",
                                    "TNCA_DEVICE_PLUSARGS": "+observe=/obs/observe.log"},
                               labels={"tritonchip-agent-harness": "evaluator-device"}, workdir="/work")
            provider.remove(dname)  # a container of this name left by a run that died
            devices.append(dname)
            provider.docker(*d.run_args(["/opt/torchenv/bin/python", "-m", "tnca.backends.relay", bench,
                                         "--unix", f"{SOCKETS}/{i}.sock", "--workdir", "/tmp/dev",
                                         "--deadline", str(DEADLINE), "--idle-timeout", f"{timeout:g}"],
                                        detach=True, remove=False))
        lines = ["set -e", "cd /work", "make -s python/tnca/data/.staged > /out/stage.log 2>&1",
                 "run() {",
                 f'  TNCA_BACKEND=sim-socket TNCA_BACKEND_ARGS="{{\\"address\\": \\"{SOCKETS}/$1.sock\\", '
                 f'\\"bench\\": \\"device-container\\"}}" /opt/torchenv/bin/python -m archbench.run --platform tnca '
                 f'--profile {ev["profile"]} --point "$2" --out /out/archbench-$1.jsonl > /out/archbench-$1.log 2>&1 || true',
                 "}"]
        jobs = max(1, int(ev["jobs"]))
        for k in range(0, len(points), jobs):
            lines += [f"run {i} '{points[i]}' &" for i in range(k, min(k + jobs, len(points)))] + ["wait"]
        (hostout / "evaluate.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
        h = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                           pids=site.pids, mounts=[Mount(hosttree, "/work", read_only=False),
                                                   Mount(hostout, "/out", read_only=False), Mount(sockets, SOCKETS)],
                           labels={"tritonchip-agent-harness": "evaluator"}, workdir="/work")
        exit_code = _run(h, ["sh", "/out/evaluate.sh"], timeout)
    finally:
        _stop_devices(devices, out)
    records = {}
    for i, p in enumerate(points):
        rec = None
        path = hostout / f"archbench-{i}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("point_id") == p:
                    rec = r
        obs = read_observer(out / f"observer-{i}" / "observe.log")
        if rec is not None or obs is not None:
            records[p] = observed_point(rec, obs)
    return {"exit": exit_code, "points": records, "missing": [p for p in points if p not in records],
            "observed": True}


def _measure(site: Site, tree: Path, task: Task, out: Path, name: str) -> dict:
    ev = task.evaluation
    points = list(ev["validation"]) + list(ev["final"])
    script = SCRIPT.format(profile=ev["profile"], jobs=int(ev["jobs"]), points=" ".join(points))
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluate.sh").write_text(script, encoding="utf-8")
    c = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                       pids=site.pids, mounts=[Mount(tree, "/work", read_only=False), Mount(out, "/out", read_only=False)],
                       labels={"tritonchip-agent-harness": "evaluator"}, workdir="/work")
    exit_code = _run(c, ["sh", "/out/evaluate.sh"], float(ev["timeout_seconds"]))
    records = {}
    path = out / "archbench.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[r["point_id"]] = {"status": r.get("status"), "cycles": (r.get("time") or {}).get("device"),
                                      "fallbacks": r.get("fallbacks"),
                                      "max_rel_err": (r.get("correctness") or {}).get("max_rel_err")}
    return {"exit": exit_code, "points": records,
            "missing": [p for p in points if p not in records]}


def _trees(repo: Path, task: Task, candidate: Optional[Path], work: Path) -> tuple:
    """The host's tree (base plus the candidate) and the device's (base plus its hardware only)."""
    host = work / "host-tree"
    shutil.rmtree(host, ignore_errors=True)
    if candidate is None:
        export(repo, task.base_commit, (), host)
    else:
        apply_candidate(repo, task.base_commit, candidate, host)
    platform = task.evaluation.get("platform")
    if not platform:
        return host, None
    _overlay_platform(repo, platform, host)
    dev = work / "device-tree"
    shutil.rmtree(dev, ignore_errors=True)
    if candidate is None:
        export(repo, task.base_commit, (), dev)
    else:
        apply_candidate(repo, task.base_commit, candidate, dev, only=HARDWARE)
    _overlay_platform(repo, platform, dev)
    return host, dev


def _measure_trees(site: Site, host: Path, dev: Optional[Path], task: Task, out: Path, name: str,
                   repo: Path) -> dict:
    if dev is None:
        return _measure(site, host, task, out, name)
    if task.evaluation.get("device"):
        return _measure_multicore(site, host, dev, task, out, name, repo)
    return _measure_observed(site, host, dev, task, out, name)


def baseline_key(task: Task) -> str:
    """What the base's cycles depend on: the base commit, the profile, the points, the platform --
    and for a multicore task the machine as well: the device, the memory it is given and the drain
    bound the base is judged by."""
    ev = task.evaluation
    points = hashlib.sha256(json.dumps([ev["validation"], ev["final"]]).encode()).hexdigest()[:8]
    platform = ev.get("platform")
    key = f"{task.base_commit[:12]}-{ev['profile']}-{points}" + (f"-obs{platform['commit'][:12]}" if platform else "")
    if ev.get("device"):
        machine = json.dumps([ev["device"], int(ev.get("memory_bytes", MULTICORE_MEMORY)), ev.get("drain_cycles")])
        key += f"-dev{hashlib.sha256(machine.encode()).hexdigest()[:8]}"
    return key


def baseline(site: Site, repo: Path, task: Task) -> dict:
    """The base's own cycles, measured once per baseline_key and kept."""
    key = baseline_key(task)
    cache = site.state_dir / "baselines" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    work = site.state_dir / "baselines" / key
    shutil.rmtree(work, ignore_errors=True)
    host, dev = _trees(repo, task, None, work)
    result = _measure_trees(site, host, dev, task, work / "out", f"tnca-eval-base-{key}", repo)
    shutil.rmtree(host, ignore_errors=True)
    if dev is not None:
        shutil.rmtree(dev, ignore_errors=True)
    if not result["missing"] and all(p["status"] == "pass" for p in result["points"].values()):
        cache.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def measure_validation(site: Site, repo: Path, task: Task, candidate: Optional[Path], job_dir: Path, name: str) -> dict:
    """The scoring service: the validation set alone, the final evaluator's way; an aggregate for the agent."""
    import dataclasses
    base = baseline(site, repo, task)
    host, dev = _trees(repo, task, candidate, job_dir)
    vtask = dataclasses.replace(task, evaluation={**task.evaluation, "final": []})
    got = _measure_trees(site, host, dev, vtask, job_dir / "eval", name, repo)
    shutil.rmtree(host, ignore_errors=True)
    if dev is not None:
        shutil.rmtree(dev, ignore_errors=True)
    full = score(base["points"], got["points"], task.evaluation["validation"])
    build = None
    if got.get("device_problems"):
        # A multicore description the evaluator would not build: its own reasons.
        build = list(got["device_problems"])
    elif got.get("device_build") == "failed":
        log = job_dir / "eval" / "device-build" / "prepare.log"
        build = (log.read_text(errors="replace").splitlines()[-20:] if log.exists() else ["the device did not build"])
    return {"aggregate": {k: full[k] for k in ("n", "passed", "speedup_geomean", "fast_1")},
            "device_build_errors": build, "full": full}


def _rtl_parameters(profiles_json: str, name: str) -> dict:
    p = json.loads(profiles_json)["profiles"].get(name) or {}
    params = dict(p.get("elaboration") or {})
    params.update((p.get("implementation") or {}).get("rtl_parameters") or {})
    return params


def _device_fields(text: str) -> dict:
    """A device description's fields, flattened: per_core_profile, platform.*, uncore.*."""
    try:
        d = json.loads(text)
    except ValueError:
        return {}
    out = {"per_core_profile": d.get("per_core_profile")}
    for section in ("platform", "uncore"):
        out.update({f"{section}.{k}": v for k, v in (d.get(section) or {}).items()})
    return out


def _device_change(repo: Path, task: Task, after_text: Optional[str]) -> dict:
    if not task.evaluation.get("device"):
        return {}
    before = _device_fields(git("-C", str(repo), "show", f"{task.base_commit}:{device_path(task)}").decode())
    after = _device_fields(after_text) if after_text is not None else {}
    return {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}


def machine_change(repo: Path, task: Task, tree: Path) -> dict:
    """Whether the candidate changed the machine it is measured on, and how.

    Allowed -- the tasks are cross-layer -- and recorded, because a faster
    machine and a faster program on the same machine are different results;
    the hardware's cost is not charged yet.
    """
    name = task.evaluation["profile"]
    before = _rtl_parameters(git("-C", str(repo), "show", f"{task.base_commit}:spec/r1_profiles.json").decode(), name)
    after = _rtl_parameters((tree / "spec/r1_profiles.json").read_text(encoding="utf-8"), name)
    changed = {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
    out = {"profile": name, "changed": changed}
    if task.evaluation.get("device"):
        p = tree / device_path(task)
        out["device"] = task.evaluation["device"]
        out["device_changed"] = _device_change(repo, task, p.read_text(encoding="utf-8") if p.exists() else None)
    return out


def hardware_changed(repo: Path, task: Task, candidate: Optional[Path]) -> dict:
    """Whether the candidate's machine differs from the base's: an RTL file, or the scored profile."""
    if candidate is None:
        return {"changed": False, "rtl": [], "profile": {}, "device": {}}
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    rtl = sorted(c["path"] for c in report["taken"] if c["path"].startswith("rtl/"))
    profile = {}
    taken = {c["path"]: c for c in report["taken"]}
    if "spec/r1_profiles.json" in taken:
        name = task.evaluation["profile"]
        before = _rtl_parameters(git("-C", str(repo), "show", f"{task.base_commit}:spec/r1_profiles.json").decode(), name)
        if taken["spec/r1_profiles.json"]["status"] == "deleted":
            after = {}
        else:
            after = _rtl_parameters((candidate / "files" / "spec/r1_profiles.json").read_text(encoding="utf-8"), name)
        profile = {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
    device = {}
    if task.evaluation.get("device") and device_path(task) in taken:
        c = taken[device_path(task)]
        device = _device_change(repo, task, None if c["status"] == "deleted" else
                                (candidate / "files" / device_path(task)).read_text(encoding="utf-8"))
    return {"changed": bool(rtl or profile or device), "rtl": rtl, "profile": profile, "device": device}


def _geomean(xs: list) -> Optional[float]:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs and all(x > 0 for x in xs) else None


def score(base: dict, got: dict, names) -> dict:
    """One set of points, each judged on its own against the base's cycles."""
    rows = []
    for p in names:
        b, c = base.get(p, {}), got.get(p)
        if c is None:
            failure = "missing"
        elif c.get("status") != "pass":
            failure = c.get("status") or "no-status"
        elif c.get("fallbacks"):
            failure = "fallback"          # part of it ran on the host: the chain was skipped
        elif not c.get("cycles") or not b.get("cycles"):
            failure = "no-cycles"
        else:
            failure = None
        speedup = b["cycles"] / c["cycles"] if failure is None else None
        rows.append({"point": p, "status": (c or {}).get("status"), "failure": failure,
                     "base_cycles": b.get("cycles"), "cycles": (c or {}).get("cycles"), "speedup": speedup,
                     # The device's own count, which the card reports too (certify.py compares them).
                     "done_cycles": (c or {}).get("record_cycles", (c or {}).get("cycles"))})
    passed = [r["speedup"] for r in rows if r["failure"] is None]
    n = len(rows)
    return {"points": rows, "n": n, "passed": len(passed),
            "speedup_geomean": _geomean(passed),
            "fast_1": (sum(1 for x in passed if x > 1.0) / n) if n else None}


def evaluate(site: Site, repo: Path, task: Task, candidate: Optional[Path], run_dir: Path) -> dict:
    base = baseline(site, repo, task)
    host, dev = _trees(repo, task, candidate, run_dir)
    machine = machine_change(repo, task, host)
    got = _measure_trees(site, host, dev, task, run_dir / "eval", f"tnca-eval-{run_dir.name}", repo)
    shutil.rmtree(host, ignore_errors=True)
    if dev is not None:
        shutil.rmtree(dev, ignore_errors=True)
    validation = score(base["points"], got["points"], task.evaluation["validation"])
    final = score(base["points"], got["points"], task.evaluation["final"])
    return {
        # The run means something only if the base itself measured cleanly.
        "valid": bool(base["points"]) and not base["missing"]
        and all(p.get("status") == "pass" for p in base["points"].values()),
        "validation": validation, "final": final,
        "all_passed": validation["passed"] == validation["n"] and final["passed"] == final["n"],
        "evaluator_exit": got["exit"],
        "observed": bool(got.get("observed")),
        "platform": task.evaluation.get("platform"),
        "machine": machine,
    }


__all__ = ["HARDWARE", "baseline", "baseline_key", "device_check", "evaluate", "hardware_changed", "measure_validation",
           "observed_multicore_point", "observed_point", "read_multicore_observer", "read_observer", "rtl_check",
           "score"]
