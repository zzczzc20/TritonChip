"""Board certification: a final candidate's machine, built at the target's clock and run on the card.

The score is simulated device cycles on the task's profile, and at the
target's fixed clock (the U280 at 100 MHz) cycles are latency. A candidate
that changed the machine -- an RTL file, or its profile's parameters -- keeps
its score only once that machine is shown to be real:

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
   simulation recorded for it.

    not-needed       the machine is the base's: the score stands
    certified        1-3 held: the score stands
    infeasible       the build failed -- it does not fit or does not close the
                     clock -- or the profile cannot be built as the image
    board-fail       the image did not load or the board stage did not reach the points
    board-disagrees  a point failed on the card, or counted other cycles
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

STANDS = ("not-needed", "certified")
BOARD_BACKEND = {"u280": "fpga-u280"}
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
    return cert


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
        "image": {"profile": profile},
        "board": {"backend": BOARD_BACKEND[cert["target"]], "profile": profile, "program": True, "conformance": True,
                  "archbench": {"points": points(task), "jobs": 1, "counters": True, "timeout_s": 7200}},
    }


def compare(sim: dict, board: list) -> list:
    """Every point simulation passed, against its board record."""
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
        elif cycles != s["done_cycles"]:
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
            build_run: Optional[str] = None,
            pipeline: Callable[..., dict] = run_pipeline,
            fetch: Callable[[Path, str, str], list] = fetch_board_records) -> dict:
    """Certify one run's candidate (`run_dir`: its run.json and candidate/) or the base (`baseline`)."""
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

    if not hw["changed"]:
        return finish("not-needed")
    problems = evaluator.rtl_check(candidate, repo)
    if problems:
        raise CertifyError("the candidate's RTL does not pass the gate's check: " + "; ".join(problems[:3]))
    work = work or out.parent / f"certify-{label}"
    tree = work / "tree"
    commit = build_tree(repo, task, candidate, tree)
    m = manifest(task, label, commit)
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
    if not yes:
        record["dry_run"] = True
        return record
    if not (hosts and build_host and board_host):
        raise CertifyError("name the FPGA hosts file, the build host and the board host")
    started = time.time()
    pipe = pipeline(tree, work / "manifest.json", work / "pipeline.json", hosts, build_host, board_host, build_run)
    record["occupancy"] = occupancy.stages(pipe, build_host, board_host, ("certify-link", "certify-board")) or [
        occupancy.entry("certify", None, started, time.time(), covers=occupancy.WHOLE_CERTIFY)]
    record["pipeline"] = {"status": pipe.get("status"), "run_id": pipe.get("run_id"), "error": pipe.get("error"),
                          "build": pipe.get("build"), "board": pipe.get("board"), "record": str(work / "pipeline.json")}
    board = fetch(hosts, board_host, pipe["run_id"]) if pipe.get("board") and pipe.get("run_id") else []
    rows = compare(sim, board)
    record["points"] = rows
    return finish(*verdict(pipe, rows, board))


__all__ = ["CertifyError", "build_tree", "certified_score", "certify", "compare", "config", "manifest",
           "simulated_from_baseline", "simulated_from_run", "verdict"]
