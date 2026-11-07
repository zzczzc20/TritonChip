"""Board certification: a final candidate's machine, built at the target's clock and run on the card.

The score is simulated device cycles on the task's profile, and at the
target's fixed clock (the U280 at 100 MHz) cycles are latency. A candidate
keeps its score only once it is shown on the card. One that changed the
machine -- an RTL file, or its profile's parameters -- is built and run:

1. Its tree is the base commit with nothing hidden, every file the gate took
   from the candidate on top (compiler and hardware alike: the code that was
   scored), then the platform -- the FPGA flow, the board wrapper and top, the
   FPGA harness -- laid over whole from the commit the task pins
   (`evaluation.certify`). The tree is committed, so both servers check out
   exactly it.
2. The FPGA harness's `pipeline` links it on the build server at the target's
   clock, with the scored profile as the image, and runs the task's points on
   the card server.
3. Every point that passed in simulation must pass on the card, with no
   fallback, and the device's own count (`time.device`) equal to the one
   simulation recorded for it. A task whose simulation is not cycle-exact with
   its card sets `evaluation.certify.board_cycles` to "record" (R41, whose
   simulation's HBM is a stand-in: the card counts 1-20 % fewer cycles, the
   base included; user's decision of 2026-11-07): the card's counts are then
   kept beside simulation's and reported, never compared, and the points
   passing on the card with no fallback and the board checks are what is
   certified. "equal", the default, compares them.

One that kept the base's machine (it changed only software: the compiler, the
runtime) is not built again: its points run on the base's own image, which the
base's certification left on the card server, with the candidate's tree
driving the card (`--base-image`, the base's certification record; step 2
becomes the FPGA harness's measure-only pipeline). Step 3 is the same.

    not-needed       an older record's verdict, from before software-only
                     candidates were run on the card: the score stood
    certified        1-3 held: the score stands
    infeasible       the build failed -- it does not fit or does not close the
                     clock -- or the profile cannot be built as the image
    board-fail       the image did not load or the board stage did not reach the points
    board-disagrees  a point failed on the card, or counted other cycles
                     (where the task compares them)
    no-verdict       the pipeline or a server failed: run it again

Every other verdict fails every point of the candidate's score.

The card server compiles the points with the candidate's compiler. Until its
sandbox is built (README section 12.5) that is code the operator runs having
read it: certify lists the candidate's files that run there, and builds
nothing without --yes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from . import evaluator, occupancy
from .task import Task
from .workspace import apply_candidate, base_tree, export, git, under

STANDS = ("not-needed", "certified")   # not-needed: records written before 2026-11-06 only
# The card backend of each FPGA harness target a task may certify on.
BOARD_BACKEND = {"u280": "fpga-u280", "u280-multicore-r41": "fpga-u280-multicore"}
# The FPGA harness's statuses for a build that ran and failed, as opposed to one that could not run.
INFEASIBLE = ("BUILD_FAIL", "TIMING_FAIL")


class CertifyError(RuntimeError):
    pass


def config(task: Task) -> dict:
    cert = task.evaluation.get("certify")
    if not isinstance(cert, dict):
        raise CertifyError(f"task {task.task_id} has no evaluation.certify: nothing names the machine's platform")
    missing = [k for k in ("commit", "paths", "target", "clock_mhz") if k not in cert]
    if missing:
        raise CertifyError(f"evaluation.certify is missing {', '.join(missing)}")
    if cert["target"] not in BOARD_BACKEND:
        raise CertifyError(f"evaluation.certify.target {cert['target']!r}: certify knows {sorted(BOARD_BACKEND)}")
    board_cycles(task)
    return cert


BOARD_CYCLES = ("equal", "record")


def board_cycles(task: Task) -> str:
    """How the card's cycle counts are judged: "equal" (the default), each must be simulation's exactly;
    "record", they are kept beside simulation's and reported but never fail a point (module docstring, 3)."""
    rule = (task.evaluation.get("certify") or {}).get("board_cycles", "equal")
    if rule not in BOARD_CYCLES:
        raise CertifyError(f"evaluation.certify.board_cycles {rule!r}: one of {', '.join(BOARD_CYCLES)}")
    return rule


def points(task: Task) -> list:
    return list(dict.fromkeys([*task.evaluation["validation"], *task.evaluation["final"]]))


def simulated_from_run(run: dict, names) -> dict:
    """point -> how simulation scored it, from a run record."""
    rows = {}
    for set_name in ("validation", "final"):
        for row in ((run.get("evaluation") or {}).get(set_name) or {}).get("points", []):
            rows[row["point"]] = row
    out = {}
    for p in names:
        row = rows.get(p) or {}
        if row.get("failure") is None and "done_cycles" not in row:
            raise CertifyError(f"{p}: the run record has no done_cycles (scored before they were recorded); "
                               "re-score the candidate")
        out[p] = {"passed": bool(row) and row.get("failure") is None, "done_cycles": row.get("done_cycles")}
    return out


def simulated_from_baseline(base: dict, names) -> dict:
    """point -> how simulation measured the base, from the evaluator's baseline cache."""
    got = base.get("points") or {}
    return {p: {"passed": (got.get(p) or {}).get("status") == "pass",
                "done_cycles": (got.get(p) or {}).get("record_cycles")} for p in names}


def build_tree(repo: Path, task: Task, candidate: Optional[Path], dest: Path) -> str:
    """The candidate's machine as one commit: the base, the candidate, the pinned platform. Its hash."""
    cert = config(task)
    shutil.rmtree(dest, ignore_errors=True)
    export(repo, task.base_commit, (), dest)
    if candidate is not None:
        apply_candidate(repo, task.base_commit, candidate, dest)
    platform = base_tree(repo, cert["commit"])
    for prefix in cert["paths"]:
        # Whole: whatever the tree had there goes, and only the pinned commit's files are there after.
        target = dest / prefix
        if prefix.endswith("/"):
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_symlink() or target.exists():
            target.unlink()
        wanted = [p for p in platform if under(p, prefix)]
        if not wanted:
            raise CertifyError(f"evaluation.certify.paths: {prefix!r} matches nothing in {cert['commit'][:12]}")
        for path in wanted:
            mode, oid = platform[path]
            data = git("-C", str(repo), "cat-file", "blob", oid)
            out = dest / path
            out.parent.mkdir(parents=True, exist_ok=True)
            if mode == "120000":
                os.symlink(os.fsdecode(data), out)
            else:
                out.write_bytes(data)
                os.chmod(out, 0o755 if mode == "100755" else 0o644)
    git("-C", str(dest), "add", "-A", "-f")
    git("-C", str(dest), "-c", "commit.gpgsign=false", "-c", "core.logAllRefUpdates=false",
        "commit", "-q", "--no-verify", "--allow-empty", "-m", f"certify {task.task_id}: platform {cert['commit'][:12]}")
    return git("-C", str(dest), "rev-parse", "HEAD").decode().strip()


def manifest(task: Task, label: str, commit: str) -> dict:
    """The FPGA harness's manifest: the scored profile as the image, the task's points on the card."""
    cert = config(task)
    profile = task.evaluation["profile"]
    return {
        "candidate_id": f"certify-{task.task_id}-{label}"[:120], "target": cert["target"], "mode": "full",
        "clock_mhz": cert["clock_mhz"], "source_commit": commit,
        "hypothesis": f"{label} of {task.task_id}: its machine ({profile}) fits the {cert['target']}, closes "
                      f"{cert['clock_mhz']} MHz, and counts on the card the cycles simulation counted.",
        # A multicore machine is its device description (spec/devices), whose per-core profile is the scored one.
        "image": {"device": task.evaluation["device"]} if task.evaluation.get("device") else {"profile": profile},
        "board": {"backend": BOARD_BACKEND[cert["target"]], "profile": profile, "program": True, "conformance": True,
                  "archbench": {"points": points(task), "jobs": 1, "counters": True, "timeout_s": 7200}},
    }


def xclbin_name(repo: Path, task: Task) -> str:
    """The image file the target's build leaves, as the pinned platform's target names it."""
    cert = config(task)
    target = json.loads(git("-C", str(repo), "show",
                            f"{cert['commit']}:tools/tritonchip_fpga_harness/targets/{cert['target']}.json"))
    return str(target.get("xclbin_name") or next(a for a in target["expected_artifacts"] if a.endswith(".xclbin")))


def compare(sim: dict, board: list, rule: str = "equal") -> list:
    """Every point simulation passed, against its board record (`rule`: board_cycles)."""
    got = {r.get("point_id"): r for r in board}
    rows = []
    for p, s in sim.items():
        if not s["passed"]:
            # The score fails it already, whatever the card says.
            rows.append({"point": p, "sim_cycles": s["done_cycles"], "board_cycles": None, "problem": None,
                         "note": "failed in simulation"})
            continue
        r = got.get(p)
        cycles = ((r or {}).get("time") or {}).get("device")
        if r is None:
            problem = "no board record"
        elif r.get("status") != "pass":
            problem = f"board status {r.get('status')}"
        elif r.get("fallbacks"):
            problem = "part of it ran on the host"
        elif rule == "equal" and cycles != s["done_cycles"]:
            problem = "cycles differ"
        else:
            problem = None
        rows.append({"point": p, "sim_cycles": s["done_cycles"], "board_cycles": cycles, "problem": problem})
    return rows


def verdict(record: dict, rows: list, board: list) -> tuple:
    """(verdict, why) from the pipeline's record and the per-point comparison."""
    build = record.get("build") or {}
    if not build:
        return "no-verdict", record.get("error") or "the pipeline stopped before the build finished"
    if build.get("status") in INFEASIBLE:
        return "infeasible", f"{build['status']}: {build.get('error') or 'see the build log'}"
    if build.get("status") != "PASS":
        return "no-verdict", f"build {build.get('status')}: {build.get('error')}"
    stage = record.get("board")
    if not stage:
        # The board stage may still be running on the card server.
        return "no-verdict", record.get("error") or "the board stage did not report"
    # Point records count only when the board stage and the whole pipeline passed: a failed
    # ArchBench run, or a retry that failed before replacing an earlier attempt's output,
    # can leave records behind that match.
    if stage.get("status") in ("ENV_FAIL", "ORPHANED", "PIPELINE_FAIL", None):
        return "no-verdict", f"board {stage.get('status')}: {stage.get('error')}"
    if stage.get("status") != "PASS":
        return "board-fail", f"board {stage.get('status')} at {stage.get('failed_stage')}: {stage.get('error')}"
    if record.get("status") != "PASS":
        return "no-verdict", f"pipeline {record.get('status')}: {record.get('error')}"
    if not board:
        return "no-verdict", "the board stage passed but left no point records"
    bad = [r for r in rows if r["problem"]]
    if not bad:
        return "certified", None
    return "board-disagrees", "; ".join(f"{r['point']}: {r['problem']}" for r in bad[:5])


def certified_score(evaluation: dict, outcome: str) -> dict:
    """The run's score after certification: as simulated, or every point failed."""
    out = {}
    for name in ("validation", "final"):
        s = evaluation[name]
        if outcome in STANDS:
            out[name] = s
            continue
        rows = [dict(r, failure=r.get("failure") or f"hardware-{outcome}", speedup=None) for r in s["points"]]
        out[name] = {"points": rows, "n": s["n"], "passed": 0, "speedup_geomean": None,
                     "fast_1": 0.0 if s["n"] else None}
    return out


def run_pipeline(tree: Path, manifest_path: Path, record_path: Path, hosts: Path, build_host: str,
                 board_host: str, build_run: Optional[str] = None) -> dict:
    """The pinned FPGA harness's `pipeline`, from the tree it is to build."""
    argv = [sys.executable, str(tree / "tools/tritonchip_fpga_harness/cli.py"), "pipeline", str(manifest_path),
            "--hosts", str(hosts), "--build-host", build_host, "--board-host", board_host,
            "--output", str(record_path)] + (["--build-run", build_run] if build_run else [])
    done = subprocess.run(argv, cwd=tree, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                          stdout=subprocess.DEVNULL)
    if not record_path.is_file():
        return {"status": "PIPELINE_FAIL", "error": f"the pipeline exited {done.returncode} and wrote no record"}
    return json.loads(record_path.read_text(encoding="utf-8"))


def measure_on_image(tree: Path, manifest: dict, record_path: Path, hosts: Path, board_host: str,
                     run_id: str, xclbin_sha256: str) -> dict:
    """The board stage of the tree on an image already on the card server (the FPGA harness's
    measure-only pipeline). The controller is this checkout's; the card server runs the tree's own
    harness, as a full pipeline's would."""
    from tritonchip_fpga_harness.hosts import load_hosts
    from tritonchip_fpga_harness.pipeline import run_pipeline as fpga_pipeline
    record = fpga_pipeline(local_repo=tree, hosts=load_hosts(hosts), manifest=manifest, build_host=None,
                           board_host=board_host, measure_run=run_id, expect_xclbin_sha256=xclbin_sha256)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def base_image(path: Path, task: Task, cert: dict) -> dict:
    """The base's certification record, as the image a software-only candidate runs on: certified, for
    this task's base and platform, with the run and the image hash it was measured with."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    pipe = rec.get("pipeline") or {}
    image = {"record": str(path), "run_id": pipe.get("run_id"),
             "xclbin_sha256": (pipe.get("build") or {}).get("xclbin_sha256"),
             "board_host": (pipe.get("board") or {}).get("host"),
             "candidate_id": (rec.get("manifest") or {}).get("candidate_id"),
             "image_parameters": rec.get("image_parameters")}
    problems = []
    if "baseline" not in rec:
        problems.append("it is not the base's certification (certify --base)")
    if rec.get("verdict") != "certified":
        problems.append(f"its verdict is {rec.get('verdict')!r}, not certified")
    if (rec.get("task") or {}).get("base_commit") != task.base_commit:
        problems.append("it certified another base commit")
    for key, want in (("platform_commit", cert["commit"]), ("target", cert["target"]),
                      ("clock_mhz", cert["clock_mhz"]), ("profile", task.evaluation["profile"])):
        if rec.get(key) != want:
            problems.append(f"its {key} is {rec.get(key)!r}, not {want!r}")
    problems += [f"it records no {k}" for k in ("run_id", "xclbin_sha256", "board_host", "candidate_id",
                                                   "image_parameters") if not image[k]]
    if problems:
        raise CertifyError(f"{path} cannot be the base's image: " + "; ".join(problems))
    return image


def fetch_board_records(hosts: Path, board_host: str, run_id: str) -> list:
    """The board stage's ArchBench records, read from the card server's run."""
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    from tritonchip_fpga_harness.pipeline import remote
    card = host_for(load_hosts(hosts), board_host, "board")
    done = remote(card, ["cat", f"{card.state_dir}/runs/{run_id}/board/archbench.jsonl"], check=False)
    if done.returncode != 0:
        return []
    return [json.loads(line) for line in done.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]


def certify(repo: Path, task: Task, *, run_dir: Optional[Path] = None, baseline: Optional[Path] = None,
            hosts: Optional[Path] = None, build_host: Optional[str] = None, board_host: Optional[str] = None,
            work: Optional[Path] = None, out: Optional[Path] = None, yes: bool = False,
            build_run: Optional[str] = None, base_image_record: Optional[Path] = None,
            pipeline: Callable[..., dict] = run_pipeline,
            measure: Callable[..., dict] = measure_on_image,
            fetch: Callable[[Path, str, str], list] = fetch_board_records) -> dict:
    """Certify one run's candidate (`run_dir`: its run.json and candidate/) or the base (`baseline`).
    A candidate that kept the base's machine runs on the base's image: `base_image_record` is the
    base's certification record."""
    if (run_dir is None) == (baseline is None):
        raise CertifyError("certify a run (its directory) or the base (the evaluator's baseline), not both")
    cert = config(task)
    names = points(task)
    record: dict = {"schema": "tritonchip-agent-certification/1",
                    "task": {"id": task.task_id, "sha256": task.sha256, "base_commit": task.base_commit},
                    "platform_commit": cert["commit"], "target": cert["target"], "clock_mhz": cert["clock_mhz"],
                    "profile": task.evaluation["profile"], "started": time.time(), "occupancy": []}
    if run_dir is not None:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (run.get("task") or {}).get("sha256") != task.sha256:
            raise CertifyError("the run was scored under another version of the task")
        if run.get("validity") != "valid":
            raise CertifyError(f"the run is {run.get('validity')}: there is no score to certify")
        candidate = run_dir / "candidate" if (run.get("gate") or {}).get("taken") else None
        label, sim = run["run_id"], simulated_from_run(run, names)
        record["run_id"] = run["run_id"]
        record["hardware"] = hw = evaluator.hardware_changed(repo, task, candidate)
        out = out or run_dir / "certification.json"
    else:
        base = json.loads(Path(baseline).read_text(encoding="utf-8"))
        candidate, label, sim, run = None, "base", simulated_from_baseline(base, names), None
        record["baseline"] = str(baseline)
        hw = {"changed": True}   # the base is certified on request, as the reference machine
        out = out or Path(baseline).with_name(Path(baseline).stem + ".certification.json")
    record["output"] = str(out)

    def finish(outcome: str, why: Optional[str] = None) -> dict:
        record.update(verdict=outcome, why=why, finished=time.time())
        if run is not None:
            record["score"] = certified_score(run["evaluation"], outcome)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    image = None
    if not hw["changed"]:
        if base_image_record is None:
            raise CertifyError("this candidate kept the base's machine, so it runs on the base's image: certify "
                               "the base first (--base), then name that record with --base-image")
        image = base_image(base_image_record, task, cert)
        record["image"] = {k: image[k] for k in ("record", "run_id", "xclbin_sha256")}
    problems = evaluator.rtl_check(candidate, repo)
    if problems:
        raise CertifyError("the candidate's RTL does not pass the gate's check: " + "; ".join(problems[:3]))
    work = work or out.parent / f"certify-{label}"
    tree = work / "tree"
    commit = build_tree(repo, task, candidate, tree)
    m = manifest(task, label, commit)
    if image is not None:
        # The card server measures the base's run, so the manifest names that run's candidate.
        m = dict(m, candidate_id=image["candidate_id"],
                 hypothesis=f"{label} of {task.task_id}: its software, on the base's image "
                            f"({image['xclbin_sha256'][:12]}), counts on the card the cycles simulation counted.")
    work.mkdir(parents=True, exist_ok=True)
    (work / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    taken = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))["taken"] if candidate else []
    record.update(tree_commit=commit, manifest=m,
                  card_runs_candidate_code=sorted(c["path"] for c in taken if c["status"] != "deleted"
                                                  and not c["path"].startswith(evaluator.HARDWARE)))
    from tritonchip_fpga_harness.image import ImageError, image_parameters
    try:
        record["image_parameters"] = image_parameters(tree, task.evaluation["profile"])
    except ImageError as exc:
        # The candidate's profile asks for a machine the card's top level cannot build.
        return finish("infeasible", f"the profile cannot be built as the image: {exc}")
    except (OSError, KeyError, ValueError) as exc:
        raise CertifyError(f"the tree has no image build to certify with: {exc}") from exc
    if image is not None and record["image_parameters"] != image["image_parameters"]:
        raise CertifyError("the candidate's image parameters are not the base's, though its machine was judged "
                           "unchanged: it cannot run on the base's image")
    if not yes:
        record["dry_run"] = True
        return record
    if image is not None:
        if not (hosts and board_host):
            raise CertifyError("name the FPGA hosts file and the board host")
        if board_host != image["board_host"]:
            raise CertifyError(f"the base's image is on {image['board_host']!r}, not {board_host!r}")
        build_host = None
    elif not (hosts and build_host and board_host):
        raise CertifyError("name the FPGA hosts file, the build host and the board host")
    started = time.time()
    if image is not None:
        pipe = measure(tree, m, work / "pipeline.json", hosts, board_host, image["run_id"], image["xclbin_sha256"])
    else:
        pipe = pipeline(tree, work / "manifest.json", work / "pipeline.json", hosts, build_host, board_host, build_run)
    record["occupancy"] = occupancy.stages(pipe, build_host, board_host, ("certify-link", "certify-board")) or [
        occupancy.entry("certify", None, started, time.time(), covers=occupancy.WHOLE_CERTIFY)]
    record["pipeline"] = {"status": pipe.get("status"), "run_id": pipe.get("run_id"), "error": pipe.get("error"),
                          "build": pipe.get("build"), "board": pipe.get("board"), "record": str(work / "pipeline.json")}
    board = fetch(hosts, board_host, pipe["run_id"]) if pipe.get("board") and pipe.get("run_id") else []
    rows = compare(sim, board, board_cycles(task))
    record["points"] = rows
    return finish(*verdict(pipe, rows, board))


__all__ = ["CertifyError", "base_image", "board_cycles", "build_tree", "certified_score", "certify", "compare", "config",
           "manifest",
           "measure_on_image",
           "simulated_from_baseline", "simulated_from_run", "verdict"]


def _board_rows(sim: dict, got: dict, rule: str = "equal") -> list:
    """Every point simulation passed, against the card's count of it (board.py: the relay's log; `rule`:
    board_cycles)."""
    rows = []
    for p, s in sim.items():
        if not s["passed"]:
            rows.append({"point": p, "sim_cycles": s["done_cycles"], "board_cycles": None, "problem": None,
                         "note": "failed in simulation"})
            continue
        r = got.get(p)
        cycles = ((r or {}).get("observer") or {}).get("done_cycles")
        if r is None:
            problem = "no board record"
        elif r.get("status") != "pass":
            problem = f"board status {r.get('status')}"
        elif r.get("fallbacks"):
            problem = "part of it ran on the host"
        elif rule == "equal" and cycles != s["done_cycles"]:
            problem = "cycles differ"
        else:
            problem = None
        rows.append({"point": p, "sim_cycles": s["done_cycles"], "board_cycles": cycles, "problem": problem})
    return rows


def after_run(site, repo: Path, task: Task, record: dict, run_dir: Path, broker=None) -> dict:
    """The run's candidate on the card, by the harness, once the run is over (report section 8.9: B1, B10).

    Selection is the run's: its final candidate (a run with islands will hand the top 3 by validation score
    here, in order). Its machine is built unless this run already built exactly this hardware (an in-run slow
    request's image is reused) or it kept the base's (the base's image). An in-run board record of exactly
    this snapshot stands for the validation points and the board checks; the rest -- the final points, or all
    of them -- run through the relay (board.py). Verdicts: certified, infeasible, board-fail (the card did not
    take the image), board-disagrees (a point or a board check), no-verdict (the infrastructure: run again)."""
    import secrets
    from . import board, fpga_slow
    cfg = (task.services or {}).get("fpga_slow") or {}
    images = broker.images if broker is not None else {}
    jobs = list(broker.jobs.values()) if broker is not None else []
    candidate = run_dir / "candidate" if record["gate"]["taken"] else None
    names = points(task)
    sim = simulated_from_run(record, names)
    job = {"id": "certify", "remote_id": f"tnca-{secrets.token_hex(8)}",
           "containers": f"tnca-cert-{secrets.token_hex(6)}"}
    work = run_dir / "certification"
    out: dict = {"started": time.time(), "hardware_changed": record["hardware"]["changed"]}
    digest = fpga_slow.snapshot_digest(candidate)
    reused = next((j for j in jobs if j["service"] == "fpga-slow" and j.get("digest") == digest
                   and j.get("state") == "done" and ((j.get("full") or {}).get("card") or {}).get("outcome") == "ok"),
                  None)
    if record["hardware"]["changed"]:
        built = images.get(fpga_slow.hardware_digest(candidate))
        if built:
            out["image_from"] = "an in-run slow request"
            image = built.get("card") or board.put_image(site, built["build_run"], built["sha256"], built["xclbin"],
                                                         job["remote_id"])
        else:
            b = fpga_slow.build(site, repo, task, candidate, work, job, int(float(cfg.get("flow_minutes", 480)) * 60))
            out["build"] = {k: v for k, v in b.items() if k != "visible"}
            if b["outcome"] != "ok":
                out.update(verdict="infeasible" if b["outcome"] == "design" else "no-verdict",
                           why=f"build {b['pipeline'].get('status')}: {(b['pipeline'].get('build') or {}).get('error')}",
                           finished=time.time())
                return out
            image = board.put_image(site, b["run_id"], b["pipeline"]["build"]["xclbin_sha256"],
                                    xclbin_name(repo, task), job["remote_id"])
    else:
        image = board.base_image(site, task)
    out["image"] = image
    got, checks = {}, {}
    todo = names
    if reused:
        card = reused["full"]["card"]["record"]
        got.update({p: r for p, r in (card.get("points") or {}).items() if p in task.evaluation["validation"]})
        checks.update(((card.get("measured") or {}).get("checks")) or {})
        todo = [p for p in names if p not in got]
        out["reused"] = {"request": reused["id"], "points": sorted(got)}
    if todo:
        rec = board.measure(site, repo, task, candidate, image, todo, work / "card", job["containers"],
                            job["remote_id"], float(cfg.get("board_minutes", 120)) * 60, with_checks=not reused,
                            base_xclbin=fpga_slow.base_xclbin(site, repo, task, candidate, image))
        out["card"] = rec
        state = board.status(rec)
        if state != "ok":
            why = (rec.get("session") or {}).get("detail") or (rec.get("session") or {}).get("reason")
            out.update(verdict="board-fail" if state == "candidate" and not (rec.get("measured") or {}).get("stopped")
                       else "no-verdict", why=why or f"card session {(rec.get('session') or {}).get('status')}",
                       finished=time.time())
            return out
        got.update(rec["points"])
        checks.update(((rec.get("measured") or {}).get("checks")) or {})
    rows = _board_rows(sim, got, board_cycles(task))
    failed_checks = sorted(n for n, c in checks.items() if c.get("exit") != 0)
    bad = [r for r in rows if r["problem"]]
    out["rows"] = rows
    if bad or failed_checks:
        why = [f"{r['point']}: {r['problem']}" for r in bad[:5]] + [f"board check {n} did not pass" for n in failed_checks]
        out.update(verdict="board-disagrees", why="; ".join(why))
    else:
        out.update(verdict="certified", why=None)
    out["finished"] = time.time()
    return out
