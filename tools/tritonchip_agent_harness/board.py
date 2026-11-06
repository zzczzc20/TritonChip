"""A snapshot's points on the shared card, with no candidate code on the card host (report section 8.9).

The card host runs only the platform's own code, from a clean checkout of the task's platform
commit (`evaluation.platform.commit`, synced the FPGA harness's way): `board_session` there waits
for the card to be continuously idle, takes the shared lease, loads the image unless its UUID is
already the loaded one (a measured image stays loaded; the next session overwrites it), and runs
the board relay (`tnca.backends.board_relay`), which owns the card and executes link operations
only. The candidate's whole host side -- its compiler and the runtime -- runs here, in the
evaluator's container with no network, and reaches the relay through an SSH-forwarded Unix
socket mounted into that container. The relay, not the measuring process, writes each launch's
count (the control block's own launch timestamps and done count) to a log on the card host that
the container cannot reach. That log is the score, as the bench's observer log is in simulation
(evaluator.observed_multicore_point); the board has no drain observation, so `not-drained` is
simulation's alone. Link latency lengthens a run, never its count.

Each ArchBench point is one client process and one relay label (its index); a task's board checks
(FP8 bit alignment, chat) run after the points in the same container and session, each under its
own label. One card session at a time per run (the card is serial); the session's own wait for an
idle card counts in the request's time.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from . import evaluator
from .provider import AgentContainer, Mount
from .siteconfig import Site
from .task import Task

MOUNT = "/run/tnca-board"
BACKEND = "fpga-u280-multicore-relay"
IDLE_SECONDS = 900
POLL_SECONDS = 20


class BoardError(RuntimeError):
    """A card session that could not run: the infrastructure's, never the candidate's."""


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


def put_image(site: Site, build_run: str, sha256: str, xclbin_name: str, tag: str) -> dict:
    """A built image, from the build host's run to the card host, by its hash."""
    from tritonchip_fpga_harness.hosts import host_for
    hosts, card = _card(site)
    builder = host_for(hosts, site.fpga_pipeline["build_host"], "build")
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


def _forward(card, local: Path, remote_socket: str) -> subprocess.Popen:
    """The relay's socket here: an SSH local forward of a Unix socket, connectable by the container's user."""
    from tritonchip_fpga_harness.pipeline import CONNECT_TIMEOUT, SSH_OPTIONS
    from tritonchip_fpga_harness.worker import ssh_prefix
    argv = ssh_prefix(host=card.ssh, port=card.port, identity_file=card.identity_file,
                      connect_timeout=CONNECT_TIMEOUT,
                      options=SSH_OPTIONS + ("ExitOnForwardFailure=yes", "StreamLocalBindUnlink=yes",
                                             "StreamLocalBindMask=0111"))
    cut = argv.index("--")
    argv = argv[:cut] + ["-N", "-L", f"{local}:{remote_socket}"] + argv[cut:]
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            start_new_session=True)
    deadline = time.time() + 60
    while time.time() < deadline and not local.exists():
        if proc.poll() is not None:
            raise BoardError(f"the socket forward to the card host exited: "
                             f"{(proc.stderr.read() or b'').decode(errors='replace')[-300:]}")
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


def session(site: Site, task: Task, image: dict, tag: str, deadline_s: float, job_dir: Path,
            measure: Callable[[Path, dict], dict], labels: list) -> dict:
    """One card session: the image on the card, the relay up, `measure(socket_dir, ready)` run here,
    then the card's own logs. Raises BoardError when the card host could not give a session."""
    hosts, card = _card(site)
    cfg = site.fpga_pipeline["card"]
    from tritonchip_fpga_harness.pipeline import sync
    platform = task.evaluation["platform"]["commit"]
    synced = sync(card, platform, site.repository)
    checkout = synced["checkout"]
    folder = f"{card.state_dir}/board-sessions/{tag}"
    sock = f"/tmp/tnca-relay-{tag}.sock"
    q = shlex.quote
    started = time.time()
    script = (f"mkdir -p {q(folder)} && cd {q(checkout)} && PYTHONPATH=tools:python PYTHONDONTWRITEBYTECODE=1 "
              f"nohup {q(card.python)} -m tritonchip_fpga_harness.board_session --xclbin {q(image['path'])} "
              f"--sha256 {q(image['sha256'])} --bdf {q(cfg['bdf'])} --socket {q(sock)} --out {q(folder)} "
              f"--stop {q(folder + '/stop')} --idle-seconds {int(cfg.get('idle_seconds', IDLE_SECONDS))} "
              f"--deadline-seconds {int(deadline_s)} "
              # One point's compile may keep the relay without a client for a while: the task's own limit.
              f"--relay-idle-seconds {int(float(task.evaluation['timeout_seconds']))} "
              f"> {q(folder + '/session.log')} 2>&1 < /dev/null &")
    _remote(card, ["sh", "-c", script])
    job_dir.mkdir(parents=True, exist_ok=True)
    record = {"host": card.name, "folder": folder, "platform_commit": platform, "sync": synced,
              "image": image, "started_at": started}
    ready = None
    while time.time() - started < deadline_s:
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
        record.update(session={"status": "timeout", "detail": "the card host gave no session in time"},
                      finished_at=time.time())
        return record
    record["ready"] = ready
    record["ready_at"] = time.time()
    local_dir = Path(tempfile.mkdtemp(prefix="tnca-board-"))
    local_dir.chmod(0o755)
    forward = None
    try:
        forward = _forward(card, local_dir / "relay.sock", sock)
        record["measured"] = measure(local_dir, ready)
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
        record["session"] = {"status": "timeout", "detail": "the card session did not end after its stop"}
    observe = job_dir / "observe"
    observe.mkdir(exist_ok=True)
    for label in labels:
        raw = _cat(card, f"{folder}/observe/{label}.log")
        if raw is not None:
            (observe / f"{label}.log").write_bytes(raw)
    record["finished_at"] = time.time()
    return record


# -- here: the candidate's host side, in the evaluator's container ------------------------------

def _script(task: Task, points: list, checks: list, cores: int, timeout: float) -> str:
    ev = task.evaluation
    env = (f'TNCA_BACKEND={BACKEND} TNCA_OPERATING_CLOCK_EVIDENCE=/out/clock_evidence.json '
           f'TNCA_COMPILE_CACHE_DIR=/out/caches/tnca-inductor TNCA_ACTIVE_CORES={cores} '
           f'TNCA_HOST_TIMEOUT={timeout:g}')
    lines = ["cd /work", "mkdir -p /out/caches /out/checks",
             "make -s python/tnca/data/.staged > /out/stage.log 2>&1 || exit 0",
             "run() {",
             f'  TNCA_BACKEND_ARGS="$(cat /out/backend-$1.json)" {env} /opt/torchenv/bin/python -m archbench.run '
             f'--platform tnca --profile {ev["profile"]} --point "$2" --out /out/archbench-$1.jsonl '
             f'> /out/archbench-$1.log 2>&1 || true',
             "}"]
    # One card: the points run one after another.
    lines += [f"run {i} {shlex.quote(p)}" for i, p in enumerate(points)]
    for c in checks:
        extra = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted((c.get("env") or {}).items()))
        argv = " ".join(shlex.quote(str(a)) for a in c["command"])
        lines.append(f'TNCA_BACKEND_ARGS="$(cat /out/backend-check-{c["name"]}.json)" {env} {extra} {argv} '
                     f'> /out/check-{c["name"]}.log 2>&1; echo $? > /out/check-{c["name"]}.exit')
    return "\n".join(lines) + "\n"


def measure_host(site: Site, task: Task, host_tree: Path, out: Path, name: str, points: list, checks: list,
                 socket_dir: Path, ready: dict) -> dict:
    """The candidate's host side against the relay, each point under its own label."""
    ev = task.evaluation
    out.mkdir(parents=True, exist_ok=True)
    hostout = out / "host"
    hostout.mkdir(exist_ok=True)
    args = dict(ready.get("backend_args") or {})
    for i, _ in enumerate(points):
        (hostout / f"backend-{i}.json").write_text(json.dumps({**args, "address": f"{MOUNT}/relay.sock",
                                                               "label": str(i)}), encoding="utf-8")
    for c in checks:
        (hostout / f"backend-check-{c['name']}.json").write_text(
            json.dumps({**args, "address": f"{MOUNT}/relay.sock", "label": f"check-{c['name']}"}), encoding="utf-8")
    (hostout / "clock_evidence.json").write_text(json.dumps(ready.get("clock_evidence") or {}), encoding="utf-8")
    cores = int(args.get("core_count", 3))
    timeout = float(ev["timeout_seconds"])
    (hostout / "evaluate.sh").write_text(_script(task, points, checks, cores, timeout), encoding="utf-8")
    mounts = [Mount(host_tree, "/work", read_only=False), Mount(hostout, "/out", read_only=False),
              Mount(socket_dir, MOUNT)]
    for src, dst in sorted(((site.fpga_pipeline.get("card") or {}).get("check_mounts") or {}).items()):
        mounts.append(Mount(Path(src), dst))
    h = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                       pids=site.pids, mounts=mounts,
                       labels={"tritonchip-agent-harness": "evaluator-board"}, workdir="/work")
    budget = timeout * max(1, len(points)) + sum(float(c.get("timeout_seconds", 3600)) for c in checks)
    exit_code = evaluator._run(h, ["sh", "/out/evaluate.sh"], budget)
    records = {}
    for i, p in enumerate(points):
        path = hostout / f"archbench-{i}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("point_id") == p:
                    records[p] = r
    ran = {}
    for c in checks:
        code = hostout / f"check-{c['name']}.exit"
        log = hostout / f"check-{c['name']}.log"
        ran[c["name"]] = {"exit": int(code.read_text().strip()) if code.exists() and code.read_text().strip()
                          else None,
                          "log_tail": evaluator._read_lines(log)[-40:] if log.exists() else []}
    return {"exit": exit_code, "records": records, "checks": ran,
            "stage_log": evaluator._read_lines(hostout / "stage.log")[-30:]}


def scored(task: Task, points: list, measured: dict, observe: Path) -> dict:
    """Each point as the evaluator scores it: the host's record for its verdict, the relay's log for its count."""
    out = {}
    records = (measured or {}).get("records") or {}
    for i, p in enumerate(points):
        obs = evaluator.read_multicore_observer(observe / f"{i}.log")
        rec = records.get(p)
        if rec is not None or obs is not None:
            out[p] = evaluator.observed_multicore_point(rec, obs, None)
    return out


def measure(site: Site, repo: Path, task: Task, candidate: Optional[Path], image: dict, points: list,
            job_dir: Path, name: str, tag: str, deadline_s: float, with_checks: bool = True) -> dict:
    """A snapshot's points (and the task's board checks) on `image`: the card session's record, and every
    point scored from the card's own log."""
    checks = list(((task.services or {}).get("fpga_slow") or {}).get("board_checks") or []) if with_checks else []
    host_tree, dev = evaluator._trees(repo, task, candidate, job_dir / "trees")
    labels = [str(i) for i in range(len(points))]
    try:
        record = session(site, task, image, tag, deadline_s, job_dir,
                         lambda socket_dir, ready: measure_host(site, task, host_tree, job_dir / "eval", name, points,
                                                                checks, socket_dir, ready),
                         labels)
    finally:
        shutil.rmtree(job_dir / "trees", ignore_errors=True)
    record["points"] = scored(task, points, record.get("measured") or {}, job_dir / "observe")
    return record


def ok(record: dict) -> bool:
    return (record.get("session") or {}).get("status") == "ok" and "measured" in record


def baseline(site: Site, repo: Path, task: Task) -> dict:
    """The base on its own image, once per image and platform: what a board aggregate is against."""
    image = base_image(site, task)
    points = [*task.evaluation["validation"], *task.evaluation["final"]]
    key = hashlib.sha256(json.dumps([image["sha256"], task.evaluation["platform"]["commit"], task.base_commit,
                                     task.evaluation["profile"], points]).encode()).hexdigest()[:16]
    root = site.state_dir / "board-baselines"
    cache = root / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        work = Path(tempfile.mkdtemp(prefix="tnca-board-base-", dir=str(site.state_dir)))
        try:
            cfg = (task.services or {}).get("fpga_slow") or {}
            rec = measure(site, repo, task, None, image, points, work, f"tnca-board-base-{key}",
                          f"base-{key}", float(cfg.get("board_minutes", 120)) * 60, with_checks=False)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not ok(rec):
            raise BoardError(f"the base did not run on its image: {(rec.get('session') or {}).get('status')}")
        value = {"image": image, "points": rec["points"], "session": rec.get("session")}
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(cache)
        return value


__all__ = ["BACKEND", "BoardError", "base_image", "baseline", "measure", "ok", "put_image", "scored", "session"]
