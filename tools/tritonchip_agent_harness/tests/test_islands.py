"""Islands (report section 8): who sees whom, the layer grid, migration, and when an island stops."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import islands  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402


def task(count=2, topology="ring", interval=3) -> Task:
    cfg = {"count": count, "topology": topology, "migration_interval": interval, "migration_rate": 0.3,
           "features": ["layer"], "max_idle_turns": 3, "max_restarts": 2, "target": {"speedup_geomean": 2.0}}
    return Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("compiler/", "rtl/"),
                evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=600,
                evaluation={"profile": "p", "validation": ["v"], "final": ["f"]},
                services={"score": {"requests": 5}}, path=HERE, sha256="", islands=cfg)


def candidate(root: Path, name: str, path: str = "compiler/a.py") -> Path:
    c = root / name
    (c / "files" / Path(path).parent).mkdir(parents=True, exist_ok=True)
    (c / "files" / path).write_text(name)
    (c / "candidate.json").write_text(json.dumps({"base_commit": "a" * 40,
                                                  "taken": [{"path": path, "status": "modified"}]}))
    return c


def commit(archive, island, root, name, speedup, fast=0.5):
    job = {"id": name, "message": f"try {name}"}
    return archive.add(island, job, candidate(root, name), {"aggregate": {"n": 1, "passed": 1, "fast_1": fast,
                                                                       "speedup_geomean": speedup}})


class Islands(unittest.TestCase):
    def test_one_function_says_who_sees_whom(self):
        self.assertEqual(islands.visible(task(3).islands, 0), [2])
        self.assertEqual(islands.visible(task(1).islands, 0), [])
        self.assertEqual(islands.visible(task(3, "all").islands, 1), [0, 2])
        self.assertEqual(islands.visible(task(3, "none").islands, 1), [])
        self.assertEqual(islands.visible(task(3, {"0": [2], "1": [], "2": [0, 1]}).islands, 2), [0, 1])

    def test_the_layer_grid_has_three_cells(self):
        self.assertEqual(islands.layer_cell(["compiler"]), "compiler-only")
        self.assertEqual(islands.layer_cell(["profile", "rtl"]), "hardware-only")
        self.assertEqual(islands.layer_cell(["compiler", "rtl"]), "cross-layer")
        self.assertEqual(islands.layer_cell(["isa"]), "cross-layer")
        self.assertIsNone(islands.layer_cell([]))

    def test_commits_fill_the_grid_and_migrate_once_every_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, a = Path(tmp), islands.Archive(Path(tmp), task(2))
            e0 = commit(a, 0, root, "x0", 1.2)
            self.assertEqual((e0["id"], e0["cell"], e0["labels"]), ("0-1", "compiler-only", ["compiler"]))
            replies = [a.reply(1, commit(a, 1, root, f"y{i}", 1.0 + i / 10)) for i in range(3)]
            self.assertNotIn("migration", replies[0])
            self.assertEqual([m["id"] for m in replies[2]["migration"]], ["0-1"])
            self.assertEqual(replies[2]["grid"]["compiler-only"]["id"], "1-3")
            commit(a, 1, root, "y3", 1.0)
            commit(a, 1, root, "y4", 1.0)
            # Island 0's elite was given to island 1 once; it is not given again.
            self.assertEqual(a.reply(1, commit(a, 1, root, "y5", 1.0))["migration"], [])
            self.assertFalse(a.target_hit.is_set())
            commit(a, 0, root, "x1", 2.5)
            self.assertTrue(a.target_hit.is_set())
            self.assertEqual(a.ranked()[0]["id"], "0-2")

    def test_read_commit_refuses_what_the_island_cannot_see(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sealed = islands.Archive(root, task(2, "none"))
            commit(sealed, 1, root, "y0", 1.1)
            self.assertIn("error", sealed.read(0, "1-1"))
            self.assertEqual(sealed.view(0)["commits"], [])
            self.assertEqual(len(sealed.view(1)["commits"]), 1)

    def test_a_read_commit_patch_applies_with_git(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            run = lambda *a, cwd=repo: subprocess.run(a, cwd=cwd, check=True, capture_output=True)  # noqa: E731
            repo.mkdir()
            run("git", "init", "-q")
            (repo / "nonl.txt").write_text("x\ny")
            (repo / "gone.txt").write_text("keep\n")
            (repo / "run.sh").write_text("#!/bin/sh\n")
            run("git", "add", "-A")
            run("git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-qm", "b")
            base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
            c = root / "cand"
            (c / "files" / "sub").mkdir(parents=True)
            (c / "files" / "nonl.txt").write_text("x\nY")
            (c / "files" / "sub" / "added.py").write_text("new\n")
            (c / "files" / "run.sh").write_text("#!/bin/sh\necho hi\n")
            os.chmod(c / "files" / "run.sh", 0o755)
            (c / "candidate.json").write_text(json.dumps({"base_commit": base, "taken": [
                {"path": "nonl.txt", "status": "modified"}, {"path": "sub/added.py", "status": "added"},
                {"path": "gone.txt", "status": "deleted"}, {"path": "run.sh", "status": "modified"}]}))
            (root / "p.diff").write_text(islands.overlay_patch(repo, base, c))
            run("git", "clone", "-q", str(repo), str(root / "ws"), cwd=root)
            run("git", "apply", str(root / "p.diff"), cwd=root / "ws")
            ws = root / "ws"
            self.assertEqual((ws / "nonl.txt").read_text(), "x\nY")
            self.assertEqual((ws / "sub" / "added.py").read_text(), "new\n")
            self.assertFalse((ws / "gone.txt").exists())
            self.assertTrue(os.access(ws / "run.sh", os.X_OK))

    def test_an_island_stops_on_idle_turns_and_restarts_a_crash_at_most_twice(self):
        broker = types.SimpleNamespace(lock=threading.Lock(), jobs={}, score_left=3,
                                       budget=lambda: {"score_requests_left": 3})
        a = islands.Archive(Path("."), task(1))
        t = islands.Turns(0, task(1), broker, a, "PROMPT")
        self.assertEqual(t({"exit": 0, "tool_calls": 2, "session": "s"}, 500), t._next(False, 500))
        self.assertIsNotNone(t({"exit": 0, "tool_calls": 0, "session": "s"}, 500))
        self.assertIsNotNone(t({"exit": 0, "tool_calls": 0, "session": "s"}, 500))
        self.assertIsNone(t({"exit": 0, "tool_calls": 0, "session": "s"}, 500))
        self.assertEqual(t.stop_reason, "idle")
        c = islands.Turns(0, task(1), broker, a, "PROMPT")
        self.assertIsInstance(c({"exit": 1, "tool_calls": 0, "session": "s"}, 500), str)   # crash 1: resume
        # That resume failed before resuming ever worked: new sessions from now on, and no crash counted.
        self.assertEqual(c({"exit": 1, "tool_calls": 0, "session": "s"}, 500)[1], True)
        self.assertEqual((c.mode, c.restarts), ("new-session", 1))
        self.assertEqual(c({"exit": 0, "tool_calls": 3, "session": "s2"}, 500)[1], True)  # stays new-session
        self.assertEqual(c({"exit": 1, "tool_calls": 0, "session": "s3"}, 500)[1], True)  # crash 2
        self.assertIsNone(c({"exit": 1, "tool_calls": 0, "session": "s4"}, 500))          # crash 3: the end
        self.assertEqual((c.stop_reason, c.restarts), ("crashed", 3))
        broker.score_left = 0
        d = islands.Turns(0, task(1), broker, a, "PROMPT")
        self.assertIsNone(d({"exit": 0, "tool_calls": 4, "session": "s"}, 500))
        self.assertEqual(d.stop_reason, "commits-exhausted")


    def test_the_resume_line_gives_the_minutes_and_every_service_left(self):
        import dataclasses
        broker = types.SimpleNamespace(lock=threading.Lock(), jobs={}, score_left=7, budget=lambda: {
            "score_requests_left": 7, "fpga_minutes_left": 30.0,
            "fpga_slow_flows_left": 1, "fpga_slow_board_runs_left": 9})
        full = dataclasses.replace(task(1), services={"score": {"requests": 20}, "fpga_quick": {"minutes": 45},
                                                      "fpga_slow": {"mode": "board", "flows": 2}})
        t = islands.Turns(0, full, broker, islands.Archive(Path("."), full), "PROMPT")
        nxt = t({"exit": 0, "tool_calls": 2, "session": "s"}, 25261)
        self.assertEqual(nxt, "Continue improving. Left: about 421 minutes of your time, 7 commits, "
                              "30.0 minutes of fpga-quick, 1 fpga-slow flows and 9 board runs. "
                              "`tnca-service archive` lists the commits you can see.")

    def test_a_plain_run_with_resume_continues_until_idle(self):
        import dataclasses
        plain = dataclasses.replace(task(1), islands=None, resume={"max_idle_turns": 2, "max_restarts": 1})
        t = islands.Turns(None, plain, None, None, "PROMPT", cfg=plain.resume)
        nxt = t({"exit": 0, "tool_calls": 5, "session": "s"}, 900)
        self.assertTrue(nxt.startswith("Continue improving.") and "submission" in nxt)
        self.assertIsNotNone(t({"exit": 0, "tool_calls": 0, "session": "s"}, 900))
        self.assertIsNone(t({"exit": 0, "tool_calls": 0, "session": "s"}, 900))
        self.assertEqual(t.stop_reason, "idle")
        self.assertIsNone(islands.Turns(None, plain, None, None, "P", cfg=plain.resume)({"stop_reason": "timeout"}, 10))

    def test_the_resume_option_changes_the_plain_prompt(self):
        import dataclasses
        from tritonchip_agent_harness.task import render
        plain = dataclasses.replace(task(1), islands=None, services={})
        self.assertIn("When you are done, stop", render(plain, "offline"))
        resumed = dataclasses.replace(plain, resume={"max_idle_turns": 3, "max_restarts": 2})
        text = render(resumed, "offline")
        self.assertIn("Keep improving until the time runs out", text)
        self.assertNotIn("When you are done, stop", text)


if __name__ == "__main__":
    unittest.main()


ROLES = [{"name": "software", "authors": ["software"], "flows": 0, "board_runs": 4, "fpga_quick_minutes": 0},
         {"name": "hardware", "authors": ["hardware"], "flows": 2, "board_runs": 3, "fpga_quick_minutes": 90},
         {"name": "both", "authors": ["hardware", "isa", "software"], "flows": 2, "board_runs": 3,
          "fpga_quick_minutes": 90}]


def roles_task(rate=1.0) -> Task:
    t = task(3, "all", 1)
    t.islands["roles"] = ROLES
    t.islands["migration_rate"] = rate
    return t


def snapshot(root: Path, name: str, files: dict) -> Path:
    c = root / name
    for path, text in files.items():
        (c / "files" / Path(path).parent).mkdir(parents=True, exist_ok=True)
        (c / "files" / path).write_text(text)
    (c / "candidate.json").write_text(json.dumps({"base_commit": "a" * 40, "taken": [
        {"path": p, "status": "modified"} for p in sorted(files)]}))
    return c


def enter(archive, island, cand, speedup=1.1):
    who = archive.authorship(island, cand)
    return archive.add(island, {"id": cand.name, "message": cand.name, "authorship": who}, cand,
                       {"aggregate": {"n": 1, "passed": 1, "fast_1": 1.0, "speedup_geomean": speedup}})


class Roles(unittest.TestCase):
    """2026-11-09: islands write their own layers; another layer only taken whole from a commit they can see."""

    def test_an_island_writes_only_its_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, a = Path(tmp), islands.Archive(Path(tmp), roles_task())
            hw = snapshot(root, "hw", {"compiler/a.py": "tile = 2\n", "rtl/core/x.v": "module x; wire w; endmodule\n"})
            refused = a.authorship(1, hw)
            self.assertIn("island 1 writes hardware only", refused["refused"])
            self.assertEqual(refused["files"], ["compiler/a.py"])
            self.assertIn("refused", a.authorship(0, hw))          # software writes no hardware either
            ok = a.authorship(2, hw)                              # island 2 writes every layer
            self.assertEqual((ok["authored"], ok["adopted"]), (["hardware", "software"], {}))
            isa = snapshot(root, "isa", {"isa/ops.json": "{}\n"})
            self.assertIn("refused", a.authorship(1, isa))        # the ISA is island 2's

    def test_a_layer_taken_whole_from_a_visible_commit_is_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, a = Path(tmp), islands.Archive(Path(tmp), roles_task())
            rtl = {"rtl/core/x.v": "module x; wire w; endmodule\n", "spec/devices/d.json": "{\"sets\": 128}\n"}
            e = enter(a, 1, snapshot(root, "h1", rtl))
            self.assertEqual((e["id"], e["authored"], e["adopted"]), ("1-1", ["hardware"], {}))
            both = snapshot(root, "s1", {"compiler/a.py": "tile = 4\n", **rtl})
            ok = a.authorship(0, both)
            self.assertEqual((ok["authored"], ok["adopted"]), (["software"], {"hardware": "1-1"}))
            e0 = enter(a, 0, both)
            self.assertEqual(e0["adopted"], {"hardware": "1-1"})
            self.assertEqual(a.view(2)["commits"][-1]["adopted"], {"hardware": "1-1"})
            # Part of it is not the layer taken whole: that is writing hardware.
            part = snapshot(root, "s2", {"compiler/a.py": "tile = 4\n", "rtl/core/x.v": rtl["rtl/core/x.v"]})
            self.assertEqual(a.authorship(0, part)["files"], ["rtl/core/x.v"])
            # Nor is a layer from a commit the island cannot see.
            sealed = islands.Archive(root, roles_task())
            sealed.cfg["topology"] = "none"
            enter(sealed, 1, snapshot(root, "h2", rtl))
            self.assertIn("refused", sealed.authorship(0, both))

    def test_migration_gives_every_elite_not_given_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Rate 0.3 over two elites: one per commit, the best not given yet first -- never only the top one.
            a = islands.Archive(root, roles_task(rate=0.3))
            enter(a, 0, snapshot(root, "s", {"compiler/a.py": "x\n"}), 1.5)
            enter(a, 1, snapshot(root, "h", {"rtl/core/x.v": "module y; endmodule\n"}), 1.2)
            got = [a.reply(2, enter(a, 2, snapshot(root, f"b{i}", {"compiler/a.py": f"{i}\n"}), 1.0))["migration"]
                   for i in range(3)]
            self.assertEqual([[m["id"] for m in g] for g in got], [["0-1"], ["1-1"], []])
            # Rate 1.0: all of them at once.
            b = islands.Archive(root, roles_task(rate=1.0))
            enter(b, 0, snapshot(root, "s", {"compiler/a.py": "x\n"}), 1.5)
            enter(b, 1, snapshot(root, "h", {"rtl/core/x.v": "module y; endmodule\n"}), 1.2)
            first = b.reply(2, enter(b, 2, snapshot(root, "c", {"compiler/a.py": "c\n"}), 1.0))["migration"]
            self.assertEqual([m["id"] for m in first], ["0-1", "1-1"])

    def test_roles_are_checked_and_said_in_each_islands_text(self):
        from tritonchip_agent_harness.task import TaskError, _check_roles, render
        slow = {"fpga_slow": {"flows": 4, "board_runs": 10}}
        self.assertEqual(len(_check_roles(Path("t"), ROLES, 3, slow)), 3)
        with self.assertRaises(TaskError):
            _check_roles(Path("t"), ROLES[:2], 3, slow)
        with self.assertRaises(TaskError):
            _check_roles(Path("t"), ROLES, 3, {"fpga_slow": {"flows": 3, "board_runs": 10}})
        with self.assertRaises(TaskError):
            _check_roles(Path("t"), [{"authors": ["firmware"]}] * 3, 3, {})
        import dataclasses
        t = dataclasses.replace(roles_task(), services={
            "score": {"requests": 40}, "fpga_quick": {"minutes": 90, "job_minutes": 30},
            "fpga_slow": {"mode": "board", "flows": 4, "board_runs": 10, "flow_minutes": 480,
                          "board_minutes": 240, "board_faster_than": 1.01}})
        texts = [render(t, "open", island=i) for i in range(3)]
        self.assertIn("Your island writes software (`compiler/`, `python/tnca/`): a commit may change only that.",
                      texts[0])
        self.assertIn("Your island writes hardware (`rtl/`", texts[1])
        self.assertIn("Your island may write every layer.", texts[2])
        self.assertIn("4 such board runs, your island's own", texts[0])
        self.assertIn("0 flows in this run, your island's own", texts[0])
        self.assertIn("Your island has no FPGA time for it.", texts[0])
        self.assertIn("You have 90 minutes of FPGA time", texts[1])
        self.assertIn("the run's result is the best commit of all islands", texts[1])
        self.assertIn("more than 1% faster than the base", texts[2])
        self.assertIn("`ends_within_minutes`", texts[2])
        self.assertIn("with every commit the elites of connected islands you have not been given yet", texts[2])
        self.assertNotIn("shared by every island", "".join(texts))
