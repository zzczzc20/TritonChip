"""A snapshot's points on the shared card, with no candidate code on the card host (report section 8.9).

The card host runs only the platform's own code, from a clean checkout of the task's platform
commit (`evaluation.platform.commit`, synced the FPGA harness's way): `board_session` there waits
for the card to be continuously idle, takes the shared lease, loads the image unless its UUID is
already the loaded one (a measured image stays loaded; the next session overwrites it), and runs
the board relay (`tnca.backends.board_relay`), which owns the card and executes link operations
only. The candidate's whole host side -- its compiler and the runtime -- runs here, in the
evaluator's container with no network, and reaches the relay through an SSH-forwarded Unix
socket mounted into that container.

What is trusted, and why (the review of 2026-11-07):

- The count. The relay, not the measuring process, logs each launch the device admitted (it
  reads the control block's `served` register after every doorbell, so an aliased or unexpected
  doorbell is seen too) with the control block's own launch timestamps and done count, to a log
  on the card host that the container cannot reach. That log is the score, as the bench's
  observer is in simulation (evaluator.observed_multicore_point). A point whose log has any
  other line -- a launch that never finished, one refused or lost -- is rejected
  (`unobserved-launch`), never scored on the launches that did finish. The board has no drain
  observation, so `not-drained` stays simulation's.
- The attribution. Each point runs in a container of its own, and before it starts the harness
  writes the point's label into the session's folder over SSH; the relay files every launch of
  a connection under the label it read when it accepted that connection, whatever the client
  says. Between labels the relay zeroes the card's memory, so nothing carries from one point (or
  an earlier user of the card) to the next.
- What comes back. Everything the container wrote is the candidate's: files are opened without
  following links, as regular files of bounded size, and nothing of a board check but whether it
  passed reaches the agent.

Link latency lengthens a run, never its count.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from . import evaluator
from .provider import AgentContainer, Mount
from .siteconfig import Site
from .task import Task, board_points

MOUNT = "/run/tnca-board"
BACKEND = "fpga-u280-multicore-relay"
# The device cycles one launch may take on the card (TNCA_DEADLINE). archbench.run's own default, 2e8 (2 s at
# 100 MHz), is a simulator's: R41's board points run up to a minute of device time and a single launch of the
# larger ones (a 4096-cube matmul, an HBM-sized transpose) passes 2 s, so they ended as runtime faults at 2e8
# cycles (R41-F2's base, 2026-11-09). The value is the R41 pilot's board measurements' (board_pilot.py), on
# which the point split's "under a minute of device time" was judged; the session's own time limit still bounds
# a launch that hangs.
DEADLINE_CYCLES = 200_000_000_000
IDLE_SECONDS = 900
POLL_SECONDS = 20
MARGIN = 120              # seconds of a session kept back for the relay's own end
READ_LIMIT = 8 << 20      # the most of one container-written file the harness reads
# Session statuses (board_session.py) that are the candidate's to pay for: its image was refused,
# did not load, or lost the device. Anything else that is not `ok` is the infrastructure's.
CANDIDATE_STATUSES = ("refused-admission", "program-failed", "device-lost")


class BoardError(RuntimeError):
    """A card session that could not run: the infrastructure's, never the candidate's."""


def read_safe(path: Path, limit: int = READ_LIMIT) -> Optional[bytes]:
    """A file the candidate's container could write: no link followed, a regular file, at most `limit` bytes."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        return os.read(fd, limit)
    finally:
        os.close(fd)


def _text(path: Path, limit: int = READ_LIMIT) -> str:
    data = read_safe(path, limit)
    return data.decode("utf-8", errors="replace") if data else ""


# -- the card host -----------------------------------------------------------------------------

def _card(site: Site):
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    cfg = site.fpga_pipeline or {}
    if not cfg.get("board_host") or not cfg.get("card"):
        raise BoardError("this site names no card host for board sessions (fpga_pipeline.board_host and .card)")
    hosts = load_hosts(cfg["hosts"])
    return hosts, host_for(hosts, cfg["board_host"], "board")


def _remote(host, argv: list, *, data: Optional[bytes] = None, timeout: int = 600, check: bool = True):
    from tritonchip_fpga_harness.pipeline import remote
    return remote(host, argv, data=data, timeout=timeout, check=check)


def _cat(host, path: str) -> Optional[bytes]:
    done = _remote(host, ["cat", path], check=False, timeout=120)
    return done.stdout if done.returncode == 0 else None


def base_image(site: Site, task: Task) -> dict:
    """The base's image on the card host: the task names its hash, the site where the card host keeps it."""
    sha = ((task.services or {}).get("fpga_slow") or {}).get("base_image_sha256")
    images = ((site.fpga_pipeline or {}).get("card") or {}).get("images") or {}
    if not sha:
        raise BoardError("the task names no base image (services.fpga_slow.base_image_sha256)")
    if sha not in images:
        raise BoardError(f"the site has no path on the card host for the base image {sha[:12]}")
    return {"path": images[sha], "sha256": sha, "kind": "base"}


def put_image(site: Site, build_run: str, sha256: str, xclbin_name: str, tag: str,
              build_host: Optional[str] = None) -> dict:
    """A built image, from the build host's run to the card host, by its hash. `build_host`: the machine
    that built it (an image recorded before there were several has none: the site's first)."""
    from tritonchip_fpga_harness.hosts import host_for
    hosts, card = _card(site)
    builder = host_for(hosts, build_host or site.fpga_pipeline["build_host"], "build")
    data = _cat(builder, f"{builder.state_dir}/runs/{build_run}/artifacts/{xclbin_name}")
    if data is None:
        raise BoardError(f"the build host has no {xclbin_name} for run {build_run}")
    if hashlib.sha256(data).hexdigest() != sha256:
        raise BoardError(f"the build host's {xclbin_name} for run {build_run} is not the image the build recorded")
    folder = f"{card.state_dir}/board-images"
    path = f"{folder}/{sha256}.xclbin"
    _remote(card, ["mkdir", "-p", folder])
    _remote(card, ["dd", f"of={path}.part", "bs=1048576"], data=data, timeout=1800)
    _remote(card, ["mv", "-f", f"{path}.part", path])
    return {"path": path, "sha256": sha256, "kind": "built", "build_run": build_run}


def _forward(card, local: Path, remote_socket: str, log: Path) -> subprocess.Popen:
    """The relay's socket here: an SSH local forward of a Unix socket, connectable by the container's user."""
    from tritonchip_fpga_harness.pipeline import CONNECT_TIMEOUT, SSH_OPTIONS
    from tritonchip_fpga_harness.worker import ssh_prefix
    argv = ssh_prefix(host=card.ssh, port=card.port, identity_file=card.identity_file,
                      connect_timeout=CONNECT_TIMEOUT,
                      options=SSH_OPTIONS + ("ExitOnForwardFailure=yes", "StreamLocalBindUnlink=yes",
                                             "StreamLocalBindMask=0111"))
    cut = argv.index("--")
    argv = argv[:cut] + ["-N", "-L", f"{local}:{remote_socket}"] + argv[cut:]
    with open(log, "ab") as err:   # a file, never a pipe nobody reads: a full pipe would stall the forward
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
    deadline = time.time() + 60
    while time.time() < deadline and not local.exists():
        if proc.poll() is not None:
            raise BoardError(f"the socket forward to the card host exited: {_text(log)[-300:]}")
        time.sleep(0.5)
    if not local.exists():
        _stop(proc)
        raise BoardError("the socket forward to the card host did not come up")
    return proc


def _stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


class Session:
    """What a measurement may do with the card session it runs in."""

    def __init__(self, card, folder: str, forward: subprocess.Popen, socket_dir: Path, ready: dict, ends_at: float):
        self.card, self.folder, self.forward, self.socket_dir, self.ready = card, folder, forward, socket_dir, ready
        self.ends_at = ends_at

    def label(self, label: str) -> None:
        """The label the relay files the next connection's launches under: written here, never by the client."""
        q = shlex.quote
        _remote(self.card, ["sh", "-c", f"printf %s {q(label)} > {q(self.folder)}/label.tmp && "
                                        f"mv -f {q(self.folder)}/label.tmp {q(self.folder)}/label"])

    def left(self) -> float:
        return self.ends_at - time.time()

    def check(self) -> None:
        if self.forward.poll() is not None:
            raise BoardError("the socket forward to the card host dropped during the measurement")


def session(site: Site, repo: Path, task: Task, image: dict, tag: str, deadline_s: float, job_dir: Path,
            measure: Callable[[Session], dict], labels: list, device: Optional[bytes] = None,
            base_xclbin: Optional[str] = None) -> dict:
    """One card session: the image on the card, the relay up, `measure(session)` run here, then the card's
    own logs. `device`: the candidate's device description, uploaded for the backend's arguments.
    Raises BoardError when the card host could not give a session."""
    _hosts, card = _card(site)
    cfg = site.fpga_pipeline["card"]
    from tritonchip_fpga_harness.pipeline import sync
    platform = task.evaluation["platform"]["commit"]
    synced = sync(card, platform, repo)
    checkout = synced["checkout"]
    folder = f"{card.state_dir}/board-sessions/{tag}"
    sock = f"/tmp/tnca-relay-{tag}.sock"
    q = shlex.quote
    started = time.time()
    # The wait for an idle card is the other users' doing: it has its own cap, and the measurement gets its
    # whole `deadline_s` from the moment the card is ready (the review's N2).
    wait_cap = float(cfg.get("max_wait_seconds", 4 * 3600))
    _remote(card, ["mkdir", "-p", folder])
    extra = ""
    if device is not None:
        _remote(card, ["dd", f"of={folder}/device.json"], data=device)
        extra += f"--device {q(folder + '/device.json')} "
    if base_xclbin:
        extra += f"--base-xclbin {q(base_xclbin)} "
    # The relay's chunk store (2026-11-11): what clients write is kept on the card host by digest, so a point's
    # weights cross the SSH tunnel once, not at every launch of every session. Needs a platform that has it.
    cache_gib = ((task.services or {}).get("fpga_slow") or {}).get("relay_cache_gib")
    if cache_gib:
        extra += (f"--chunk-cache {q(str(card.state_dir) + '/relay-cache')} "
                  f"--chunk-cache-bytes {int(float(cache_gib) * (1 << 30))} ")
    # `mkdir .started` is the start's own lock: SSH's retry of a dropped connection cannot start it twice.
    script = (f"mkdir {q(folder + '/.started')} 2>/dev/null || exit 0; cd {q(checkout)} && "
              # The host's own PYTHONPATH (hosts.json `environment`: XRT's python bindings) stays behind ours.
              f"PYTHONPATH=tools:python${{PYTHONPATH:+:$PYTHONPATH}} PYTHONDONTWRITEBYTECODE=1 "
              f"nohup {q(card.python)} -m tritonchip_fpga_harness.board_session --xclbin {q(image['path'])} "
              f"--sha256 {q(image['sha256'])} --bdf {q(cfg['bdf'])} --socket {q(sock)} --out {q(folder)} "
              f"--stop {q(folder + '/stop')} --idle-seconds {int(cfg.get('idle_seconds', IDLE_SECONDS))} "
              f"--deadline-seconds {int(deadline_s + wait_cap)} {extra}"
              # One point's compile may keep the relay without a client for a while: the task's own limit.
              f"--relay-idle-seconds {int(float(task.evaluation['timeout_seconds']))} "
              f"> {q(folder + '/session.log')} 2>&1 < /dev/null &")
    _remote(card, ["sh", "-c", script])
    job_dir.mkdir(parents=True, exist_ok=True)
    record = {"host": card.name, "folder": folder, "platform_commit": platform, "sync": synced,
              "image": image, "started_at": started}
    ready = None
    while time.time() - started < wait_cap + 600:
        raw = _cat(card, f"{folder}/ready.json")
        if raw:
            ready = json.loads(raw)
            break
        ended = _cat(card, f"{folder}/session.json")
        if ended:
            record["session"] = json.loads(ended)
            record["finished_at"] = time.time()
            return record
        time.sleep(POLL_SECONDS)
    if ready is None:
        _remote(card, ["touch", f"{folder}/stop"], check=False)
        record.update(session={"status": "busy-timeout", "cause": "infrastructure",
                               "detail": "the card host gave no session in time"}, finished_at=time.time())
        return record
    record["ready"] = ready
    record["ready_at"] = time.time()
    local_dir = Path(tempfile.mkdtemp(prefix="tnca-board-"))
    local_dir.chmod(0o755)
    forward = None
    try:
        forward = _forward(card, local_dir / "relay.sock", sock, job_dir / "forward.log")
        s = Session(card, folder, forward, local_dir, ready,
                    min(record["ready_at"] + deadline_s, started + deadline_s + wait_cap) - MARGIN)
        record["measured"] = measure(s)
    finally:
        _stop(forward)
        shutil.rmtree(local_dir, ignore_errors=True)
        _remote(card, ["touch", f"{folder}/stop"], check=False)
    waited = time.time()
    while time.time() - waited < 300:
        ended = _cat(card, f"{folder}/session.json")
        if ended:
            record["session"] = json.loads(ended)
            break
        time.sleep(5)
    else:
        record["session"] = {"status": "no-session-record", "cause": "infrastructure",
                             "detail": "the card session left no record after its stop"}
    observe = job_dir / "observe"
    observe.mkdir(exist_ok=True)
    for label in labels:
        raw = _cat(card, f"{folder}/observe/{label}.log")
        if raw is not None:
            (observe / f"{label}.log").write_bytes(raw)
    record["finished_at"] = time.time()
    return record


# -- here: the candidate's host side, in the evaluator's containers ------------------------------

def _env(task: Task, cores: int, timeout: float, deadline: Optional[int] = None) -> str:
    return (f"TNCA_BACKEND={BACKEND} TNCA_BACKEND_ARGS=\"$(cat /out/backend.json)\" "
            f"TNCA_OPERATING_CLOCK_EVIDENCE=/out/clock_evidence.json "
            f"TNCA_COMPILE_CACHE_DIR=/caches/tnca-inductor TNCA_ACTIVE_CORES={cores} TNCA_HOST_TIMEOUT={timeout:g} "
            f"TNCA_DEADLINE={int(deadline or DEADLINE_CYCLES)}")


# A point's launches may take this many times the base's cycles for the whole point before the device gives
# up on one (decided 2026-11-10): a runaway program ends in seconds, not in DEADLINE_CYCLES' 2000 s.
DEADLINE_FACTOR = 20
# The health check's tolerance on the card's cycles of its known-answer point (the card repeats a program to
# within about half a percent).
HEALTH_TOLERANCE = 0.02


def health_tree(site: Site, repo: Path, task: Task) -> Path:
    """The base's own host tree, staged once and kept: the health check runs its known-answer point with the
    base's software, never a candidate's, so a candidate's compiler cannot fail the card's check."""
    key = hashlib.sha256(json.dumps([task.base_commit, (task.evaluation.get("platform") or {}).get("commit")]
                                    ).encode()).hexdigest()[:16]
    root = site.state_dir / "health-trees" / key
    host = root / "host-tree"
    if (root / "staged").exists():
        return host
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "staged").exists():
            return host
        host, _dev = evaluator._trees(repo, task, None, root)
        out, caches = root / "stage", root / "caches"
        for d in (out, caches):  # a bind mount's source must exist, or docker refuses the run (exit 125)
            d.mkdir(exist_ok=True)
            d.chmod(0o777)
        status = evaluator._run(_container(site, f"tnca-health-stage-{key}", host, out, caches, None, {}),
                                ["sh", "-c", "cd /work && make -s python/tnca/data/.staged > /out/stage.log 2>&1"],
                                1800)
        if status != 0:
            raise BoardError(f"the health tree did not stage (exit {status!r})")
        (root / "staged").write_text("ok\n")
    return host


def judge_health(record: dict, observe: Path, health: dict) -> dict:
    """Whether the card answered the known-answer point as the base does: it passed, the relay counted every
    launch, and its cycles are within HEALTH_TOLERANCE of the base's on the card."""
    rec = ((record.get("measured") or {}).get("health_record"))
    obs = read_board_observer(observe / "health.log")
    row = evaluator.observed_multicore_point(rec, obs, None)
    expect = health.get("expect")
    cycles = row.get("cycles")
    why = None
    if row.get("status") != "pass":
        why = f"the known-answer point did not pass ({row.get('status')})"
    elif obs is None or obs.get("anomalies"):
        why = "the relay did not count every launch of the known-answer point"
    elif expect and cycles and abs(cycles - expect) > HEALTH_TOLERANCE * expect:
        why = f"the known-answer point took {cycles} cycles, the base {expect}"
    return {"ok": why is None, "why": why, "point": health.get("point"), "cycles": cycles, "expect": expect}


def _container(site: Site, name: str, host_tree: Path, out: Path, caches: Path, socket_dir: Optional[Path],
               extra_mounts: dict) -> AgentContainer:
    mounts = [Mount(host_tree, "/work", read_only=False), Mount(out, "/out", read_only=False),
              Mount(caches, "/caches", read_only=False)]
    if socket_dir is not None:
        mounts.append(Mount(socket_dir, MOUNT))
    for src, dst in sorted(extra_mounts.items()):
        mounts.append(Mount(Path(src), dst))
    return AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                          pids=site.pids, mounts=mounts,
                          labels={"tritonchip-agent-harness": "evaluator-board"}, workdir="/work")


def measure_host(site: Site, task: Task, host_tree: Path, out: Path, name: str, points: list, checks: list,
                 s: Session, deadlines: Optional[dict] = None, health: Optional[dict] = None) -> dict:
    """The candidate's host side against the relay: staging once, then each point, and each board check, in
    a container of its own, under the label the harness gave the relay for it."""
    ev = task.evaluation
    out.mkdir(parents=True, exist_ok=True)
    caches = out / "caches"
    caches.mkdir(exist_ok=True)
    caches.chmod(0o777)
    args = dict(s.ready.get("backend_args") or {})
    cores = int(args.get("core_count", 3))
    timeout = float(ev["timeout_seconds"])
    mounts = (site.fpga_pipeline.get("card") or {}).get("check_mounts") or {}
    stage = out / "stage"
    stage.mkdir(exist_ok=True)
    stage.chmod(0o777)
    evaluator._run(_container(site, f"{name}-stage", host_tree, stage, caches, None, {}),
                   ["sh", "-c", "cd /work && make -s python/tnca/data/.staged > /out/stage.log 2>&1"],
                   min(timeout, max(60.0, s.left())))
    runs = [(str(i), p, None) for i, p in enumerate(points)] + [(f"check-{c['name']}", None, c) for c in checks]
    done: dict = {"records": {}, "checks": {}, "stopped": None}
    for label, point, check in runs:
        s.check()
        left = s.left()
        limit = float(check.get("timeout_seconds", 3600)) if check else timeout
        if left < 60:
            done["stopped"] = f"the card session ran out of time before {label}"
            break
        box = out / label
        box.mkdir(exist_ok=True)
        box.chmod(0o777)
        (box / "backend.json").write_text(json.dumps({**args, "address": f"{MOUNT}/relay.sock", "label": label}),
                                          encoding="utf-8")
        (box / "clock_evidence.json").write_text(json.dumps(s.ready.get("clock_evidence") or {}), encoding="utf-8")
        if check:
            extra = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted((check.get("env") or {}).items()))
            command = (f"cd /work && {_env(task, cores, limit)} {extra} "
                       f"{' '.join(shlex.quote(str(a)) for a in check['command'])} > /out/check.log 2>&1; "
                       f"echo $? > /out/check.exit")
        else:
            command = (f"cd /work && {_env(task, cores, limit, (deadlines or {}).get(point))} "
                       f"/opt/torchenv/bin/python -m archbench.run "
                       f"--platform tnca --profile {ev['profile']} --point {shlex.quote(point)} "
                       f"--out /out/archbench.jsonl > /out/archbench.log 2>&1 || true")
        s.label(label)
        code = evaluator._run(_container(site, f"{name}-{label}", host_tree, box, caches, s.socket_dir,
                                         mounts if check else {}),
                              ["sh", "-c", command], min(limit, left))
        if check:
            raw = _text(box / "check.exit", 64).strip()
            done["checks"][check["name"]] = {"exit": int(raw) if raw.isdigit() else None, "timed_out": code is None,
                                             "log_tail": _text(box / "check.log").splitlines()[-40:]}
        else:
            for line in _text(box / "archbench.jsonl").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(r, dict) and r.get("point_id") == point:
                    done["records"][point] = r
    if health and not done["stopped"] and s.left() >= 60:
        # The card's health: the base's own software, its known-answer point, on this session's card.
        box = out / "health"
        box.mkdir(exist_ok=True)
        box.chmod(0o777)
        (box / "backend.json").write_text(json.dumps({**args, "address": f"{MOUNT}/relay.sock", "label": "health"}),
                                          encoding="utf-8")
        (box / "clock_evidence.json").write_text(json.dumps(s.ready.get("clock_evidence") or {}), encoding="utf-8")
        s.check()
        s.label("health")
        evaluator._run(_container(site, f"{name}-health", Path(health["tree"]), box, caches, s.socket_dir, {}),
                       ["sh", "-c", f"cd /work && {_env(task, cores, timeout)} /opt/torchenv/bin/python -m "
                        f"archbench.run --platform tnca --profile {ev['profile']} "
                        f"--point {shlex.quote(health['point'])} --out /out/archbench.jsonl "
                        f"> /out/archbench.log 2>&1 || true"], min(timeout, s.left()))
        for line in _text(box / "archbench.jsonl").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("point_id") == health["point"]:
                done["health_record"] = r
        done["health_ran"] = True
    s.check()
    s.label("idle")
    done["stage_log"] = _text(stage / "stage.log").splitlines()[-30:]
    return done


def read_board_observer(path: Path) -> Optional[dict]:
    """The relay's log for one label: its launches, as evaluator.read_multicore_observer reads the bench's, and
    any line that is neither a finished launch nor a connection's end -- a launch the device admitted that
    the log could not count."""
    text = _text(path)
    if not text:
        return None
    obs = evaluator.read_multicore_observer(path)
    anomalies = [line.split()[0] for line in text.splitlines() if line.split() and line.split()[0] not in ("cmd", "end")]
    if obs is None:
        obs = {"launches": 0, "cycles": 0, "done_cycles": 0, "failed_launches": 0, "max_settle": 0, "not_quiet": 0,
               "ended": False}
    obs["anomalies"] = anomalies[:20]
    obs["connections"] = sum(1 for line in text.splitlines() if line.startswith("end"))
    return obs


def scored(task: Task, points: list, measured: dict, observe: Path) -> dict:
    """Each point as the evaluator scores it: the host's record for its verdict, the relay's log for its count;
    a point the relay saw launch something it could not count is rejected."""
    out = {}
    records = (measured or {}).get("records") or {}
    for i, p in enumerate(points):
        obs = read_board_observer(observe / f"{i}.log")
        rec = records.get(p)
        if rec is None and obs is None:
            continue
        row = evaluator.observed_multicore_point(rec, obs, None)
        if obs and obs["anomalies"]:
            row["observer_problem"] = "unobserved-launch"
            row["status"] = "unobserved-launch"
        out[p] = row
    return out


def measure(site: Site, repo: Path, task: Task, candidate: Optional[Path], image: dict, points: list,
            job_dir: Path, name: str, tag: str, deadline_s: float, with_checks: bool = True,
            base_xclbin: Optional[str] = None, deadlines: Optional[dict] = None,
            health: Optional[dict] = None) -> dict:
    """A snapshot's points (and the task's board checks) on `image`: the card session's record, and every
    point scored from the card's own log. Raises BoardError when the infrastructure failed it."""
    checks = list(((task.services or {}).get("fpga_slow") or {}).get("board_checks") or []) if with_checks else []
    host_tree, _dev = evaluator._trees(repo, task, candidate, job_dir / "trees")
    device = None
    if task.evaluation.get("device"):
        raw = read_safe(host_tree / evaluator.device_path(task), 1 << 20)
        device = raw if raw else None
    labels = [str(i) for i in range(len(points))] + [f"check-{c['name']}" for c in checks] + (
        ["health"] if health else [])
    try:
        record = session(site, repo, task, image, tag, deadline_s, job_dir,
                         lambda s: measure_host(site, task, host_tree, job_dir / "eval", name, points, checks, s,
                                                deadlines=deadlines, health=health),
                         labels, device=device, base_xclbin=base_xclbin)
    finally:
        shutil.rmtree(job_dir / "trees", ignore_errors=True)
    record["points"] = scored(task, points, record.get("measured") or {}, job_dir / "observe")
    if health and (record.get("measured") or {}).get("health_ran"):
        record["health"] = judge_health(record, job_dir / "observe", health)
    return record


def status(record: dict) -> str:
    """'ok', 'candidate' (its image or its host side: charged) or 'infra' (tried again, not charged).

    The card session says why it ended (`cause`, board_session.py): only the image's own failures are the
    candidate's, and only for an image the candidate built -- the base's image failing is never its fault. A
    measurement that ran out of its own time (counted from the moment the card was ready) is the candidate's."""
    sess = record.get("session") or {}
    st = sess.get("status")
    if st == "ok" and "measured" in record:
        return "candidate" if (record["measured"] or {}).get("stopped") else "ok"
    if "measured" in record and (record["measured"] or {}).get("stopped") and st in ("timeout", "ok"):
        return "candidate"
    if st in CANDIDATE_STATUSES and sess.get("cause", "image") == "image" and (record.get("image") or {}).get("kind") == "built":
        return "candidate"
    return "infra"


def ok(record: dict) -> bool:
    return status(record) == "ok"


def baseline_key(task: Task) -> str:
    """What the base's card cycles depend on: its image, the platform, the base commit, the profile, the points."""
    sha = ((task.services or {}).get("fpga_slow") or {}).get("base_image_sha256")
    return hashlib.sha256(json.dumps([sha, task.evaluation["platform"]["commit"], task.base_commit,
                                      task.evaluation["profile"], board_points(task)]).encode()).hexdigest()[:16]


def baseline(site: Site, repo: Path, task: Task) -> dict:
    """The base on its own image, the board's validation points only, once per image and platform: what an
    in-run board aggregate is against. No held-out set runs while an agent does."""
    image = base_image(site, task)
    points = board_points(task)
    key = baseline_key(task)
    root = site.state_dir / "board-baselines"
    cache = root / f"{key}.json"
    if cache.exists():
        return _base_value(cache)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            return _base_value(cache)
        work = Path(tempfile.mkdtemp(prefix="tnca-board-base-", dir=str(site.state_dir)))
        try:
            cfg = (task.services or {}).get("fpga_slow") or {}
            rec = measure(site, repo, task, None, image, points, work, f"tnca-board-base-{key}",
                          f"base-{key}-{secrets.token_hex(4)}", float(cfg.get("board_minutes", 120)) * 60,
                          with_checks=False)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        failed = [p for p in points if (rec["points"].get(p) or {}).get("status") != "pass"]
        if not ok(rec):
            raise BoardError(f"the base's card session failed: {(rec.get('session') or {}).get('status')}")
        value = {"image": image, "points": rec["points"], "session": rec.get("session")}
        if failed:
            # A base point that fails on a session that ran is the base's on this card: kept, so that no
            # later request pays for another base session; delete the file to measure again.
            value["failed"] = failed
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(cache)
        return _base_value(cache)


def _base_value(cache: Path) -> dict:
    value = json.loads(cache.read_text(encoding="utf-8"))
    if value.get("failed"):
        raise BoardError(f"{len(value['failed'])} base points fail on the card ({cache.name}); "
                         "delete it to measure the base again")
    return value


__all__ = ["BACKEND", "BoardError", "Session", "base_image", "baseline", "measure", "ok", "put_image",
           "read_board_observer", "read_safe", "scored", "session", "status"]
