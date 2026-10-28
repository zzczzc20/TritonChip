"""Board certification, on a throwaway repository: the tree, the comparison, the verdicts, the score."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import certify  # noqa: E402
from tritonchip_agent_harness.evaluator import hardware_changed, rtl_check, score  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402
from tritonchip_agent_harness.workspace import export, gate, git  # noqa: E402

try:
    import tritonchip_fpga_harness  # noqa: F401,E402 -- certify runs where both harnesses are
    FPGA_HARNESS = True
except ImportError:
    FPGA_HARNESS = False
needs_fpga_harness = unittest.skipUnless(FPGA_HARNESS, "certify runs on the controller, beside the FPGA harness")

WRAPPER = ("module tnca_u50_xo #(\n    parameter NUM_RCTX = 2,\n    parameter HAS_SFU = 1,\n"
           "    parameter WORDS = 4096\n) (\n    input wire ap_clk\n);\nendmodule\n")
PROFILES = {"profiles": {"v5": {"elaboration": {"NUM_RCTX": 4}}, "other": {"elaboration": {"NUM_RCTX": 8}}}}
BASE = {
    "compiler/a.py": "a = 1\n",
    "rtl/core/x.v": "module x;\n// synthesis translate_off\ninitial t = 0;\n// synthesis translate_on\nendmodule\n",
    "rtl/common/tnca_uarch_defaults.vh": "`define TNCA_UARCH_NUM_RCTX 2\n`define TNCA_UARCH_HAS_SFU 1\n",
    "rtl/board/tnca_u50_xo.v": "module tnca_u50_xo #(parameter WORDS = 4096) (input wire ap_clk); endmodule\n",
    "rtl/fpga/top.v": "module top; endmodule\n",
    "flow/fpga/package_xo.tcl": "# old\n",
    "flow/fpga/old_only.sh": "echo old\n",
    "spec/r1_profiles.json": json.dumps(PROFILES),
}
POINTS = {"validation": ["a", "b"], "final": ["c"]}


def write(root: Path, files: dict) -> None:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


def commit(root: Path, message: str) -> str:
    git("-C", str(root), "add", "-A")
    git("-C", str(root), "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
    return git("-C", str(root), "rev-parse", "HEAD").decode().strip()


def make_repo() -> tuple:
    root = Path(tempfile.mkdtemp())
    write(root, BASE)
    git("init", "-q", str(root))
    base = commit(root, "base")
    # The platform commit: the image build, a file the base had gone, the FPGA harness.
    write(root, {"rtl/board/tnca_u50_xo.v": WRAPPER, "rtl/fpga/top.v": "module top; /* P */ endmodule\n",
                 "flow/fpga/package_xo.tcl": "if {[info exists ::env(TNCA_IMAGE_CONFIG)]} {}\n",
                 "tools/tritonchip_fpga_harness/cli.py": "print('harness')\n"})
    (root / "flow/fpga/old_only.sh").unlink()
    platform = commit(root, "platform")
    return root, base, platform


def make_task(base: str, platform: str) -> Task:
    evaluation = {"profile": "v5", **POINTS, "jobs": 1, "timeout_seconds": 60,
                  "certify": {"commit": platform, "paths": ["flow/fpga/", "rtl/board/", "rtl/fpga/",
                                                            "tools/tritonchip_fpga_harness/"],
                              "target": "u280", "clock_mhz": 100}}
    return Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/", "spec/r1_profiles.json"),
                evaluator_owned=("rtl/board/", "rtl/fpga/"), hide=(), instruction="", reference=None,
                wall_seconds=60, evaluation=evaluation, services={}, path=Path("/t"), sha256="t" * 64)


def make_candidate(repo: Path, task: Task, changes: dict, run_dir: Path) -> dict:
    ws = Path(tempfile.mkdtemp()) / "ws"
    export(repo, task.base_commit, task.hide, ws)
    write(ws, changes)
    return gate(repo, task.base_commit, task.hide, task.design, ws, out=run_dir / "candidate",
                evaluator_owned=task.evaluator_owned)


SIM = {"a": 100, "b": 200, "c": 300}


def make_run(repo: Path, task: Task, changes: dict) -> Path:
    run_dir = Path(tempfile.mkdtemp())
    g = make_candidate(repo, task, changes, run_dir)
    base = {p: {"status": "pass", "cycles": 2 * c} for p, c in SIM.items()}
    got = {p: {"status": "pass", "cycles": c + 7, "record_cycles": c} for p, c in SIM.items()}
    run = {"run_id": "r1", "task": {"sha256": task.sha256}, "validity": "valid",
           "gate": {"taken": [c["path"] for c in g["taken"]]},
           "evaluation": {name: score(base, got, POINTS[name]) for name in ("validation", "final")}}
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir


def board(cycles: dict, **extra) -> list:
    return [{"point_id": p, "status": "pass", "fallbacks": [], "time": {"device": c}, **extra} for p, c in cycles.items()]


PASSED = {"status": "PASS", "run_id": "b1", "build": {"status": "PASS"}, "board": {"status": "PASS"}}


class Tree(unittest.TestCase):
    def setUp(self):
        self.repo, self.base, self.platform = make_repo()
        self.task = make_task(self.base, self.platform)

    def test_the_candidate_on_the_full_base_with_the_platform_laid_over_whole(self):
        run_dir = Path(tempfile.mkdtemp())
        g = make_candidate(self.repo, self.task, {"compiler/a.py": "a = 2\n", "rtl/core/x.v": "module x; endmodule\n",
                                                  "rtl/fpga/top.v": "module top; /* mine */ endmodule\n"}, run_dir)
        self.assertEqual([c["path"] for c in g["taken"]], ["compiler/a.py", "rtl/core/x.v"])
        tree = Path(tempfile.mkdtemp()) / "tree"
        first = certify.build_tree(self.repo, self.task, run_dir / "candidate", tree)
        self.assertEqual((tree / "compiler/a.py").read_text(), "a = 2\n")
        self.assertEqual((tree / "rtl/core/x.v").read_text(), "module x; endmodule\n")
        self.assertIn("/* P */", (tree / "rtl/fpga/top.v").read_text())
        self.assertIn("TNCA_IMAGE_CONFIG", (tree / "flow/fpga/package_xo.tcl").read_text())
        self.assertFalse((tree / "flow/fpga/old_only.sh").exists())
        self.assertTrue((tree / "tools/tritonchip_fpga_harness/cli.py").is_file())
        self.assertEqual(git("-C", str(tree), "status", "--porcelain").decode(), "")
        # The same candidate is the same commit, so both servers check out one tree.
        self.assertEqual(certify.build_tree(self.repo, self.task, run_dir / "candidate", tree), first)

    def test_a_platform_path_the_pinned_commit_lacks_is_refused(self):
        task = make_task(self.base, self.base)
        with self.assertRaisesRegex(certify.CertifyError, "matches nothing"):
            certify.build_tree(self.repo, task, None, Path(tempfile.mkdtemp()) / "tree")

    @needs_fpga_harness
    def test_the_manifest_builds_the_scored_profile_and_runs_every_point(self):
        m = certify.manifest(self.task, "r1", "c" * 40)
        self.assertEqual((m["target"], m["clock_mhz"], m["image"]), ("u280", 100, {"profile": "v5"}))
        self.assertEqual(m["board"]["profile"], "v5")
        self.assertEqual(m["board"]["archbench"]["points"], ["a", "b", "c"])
        from tritonchip_fpga_harness.schema import validate_manifest
        validate_manifest(m)


class Hardware(unittest.TestCase):
    def setUp(self):
        self.repo, self.base, self.platform = make_repo()
        self.task = make_task(self.base, self.platform)

    def changed(self, changes: dict) -> dict:
        run_dir = Path(tempfile.mkdtemp())
        make_candidate(self.repo, self.task, changes, run_dir)
        return hardware_changed(self.repo, self.task, run_dir / "candidate")

    def test_rtl_or_the_scored_profile_is_a_machine_change(self):
        self.assertFalse(self.changed({"compiler/a.py": "a = 2\n"})["changed"])
        self.assertEqual(self.changed({"rtl/core/x.v": "module x; endmodule\n"})["rtl"], ["rtl/core/x.v"])
        other = json.loads(json.dumps(PROFILES))
        other["profiles"]["other"]["elaboration"]["NUM_RCTX"] = 16
        self.assertFalse(self.changed({"spec/r1_profiles.json": json.dumps(other)})["changed"])
        scored = json.loads(json.dumps(PROFILES))
        scored["profiles"]["v5"]["elaboration"]["NUM_RCTX"] = 8
        self.assertEqual(self.changed({"spec/r1_profiles.json": json.dumps(scored)})["profile"], {"NUM_RCTX": [4, 8]})


class SplitRtl(unittest.TestCase):
    def setUp(self):
        self.repo, self.base, self.platform = make_repo()
        self.task = make_task(self.base, self.platform)

    def check(self, text: str) -> list:
        run_dir = Path(tempfile.mkdtemp())
        make_candidate(self.repo, self.task, {"rtl/core/x.v": text}, run_dir)
        return rtl_check(run_dir / "candidate", self.repo)

    def test_what_the_base_already_had_is_not_the_candidates(self):
        self.assertEqual(self.check(BASE["rtl/core/x.v"] + "wire extra;\n"), [])

    def test_constructs_that_split_simulation_from_synthesis_are_refused(self):
        found = self.check("module x;\n// synthesis translate_off\ninitial t = 0;\n// synthesis translate_on\n"
                           "`ifdef SYNTHESIS\nwire s;\n`endif\ndefparam u.W = 2;\nassign y = tb.dut.count;\n"
                           "// pragma translate_off\nendmodule\n")
        self.assertEqual([f.split(": ")[1] for f in found],
                         ["defparam", "hierarchical reference", "hierarchical reference", "synthesis pragma",
                          "tool macro"])

    def test_names_in_strings_comments_and_numbers_are_not_references(self):
        self.assertEqual(self.check('module x;\ninitial $display("a.b");\n// see tb.dut\nreal r = 1.5;\n'
                                    '`ifdef TNCA_FAST\n`endif\nendmodule\n'), [])


@needs_fpga_harness
class Certify(unittest.TestCase):
    def setUp(self):
        self.repo, self.base, self.platform = make_repo()
        self.task = make_task(self.base, self.platform)
        self.calls = []

    def pipeline(self, value: dict):
        def run(tree, manifest_path, record_path, hosts, build_host, board_host, build_run):
            self.calls.append(json.loads(Path(manifest_path).read_text()))
            return value
        return run

    def certify(self, run_dir=None, pipe=PASSED, records=None, yes=True, **kw):
        return certify.certify(self.repo, self.task, run_dir=run_dir, hosts=Path("/hosts.json"), build_host="b",
                               board_host="c", yes=yes, pipeline=self.pipeline(pipe),
                               fetch=lambda hosts, host, run_id: board(SIM) if records is None else records, **kw)

    def test_the_base_machine_needs_nothing(self):
        run_dir = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n"})
        value = self.certify(run_dir)
        self.assertEqual(value["verdict"], "not-needed")
        self.assertEqual(self.calls, [])
        saved = json.loads((run_dir / "certification.json").read_text())
        self.assertEqual(saved["score"]["final"]["passed"], 1)

    def test_without_yes_the_tree_is_built_and_nothing_runs(self):
        run_dir = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n", "rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, yes=False)
        self.assertTrue(value["dry_run"])
        self.assertEqual(value["card_runs_candidate_code"], ["compiler/a.py"])
        self.assertEqual(value["image_parameters"], {"HAS_SFU": 1, "NUM_RCTX": 4})
        self.assertEqual(self.calls, [])
        self.assertFalse((run_dir / "certification.json").exists())

    def test_equal_cycles_on_the_card_certify_and_the_score_stands(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir)
        self.assertEqual(value["verdict"], "certified")
        self.assertEqual(self.calls[0]["board"]["archbench"]["points"], ["a", "b", "c"])
        self.assertEqual(value["score"]["validation"]["passed"], 2)
        self.assertEqual([r["problem"] for r in value["points"]], [None, None, None])

    def test_one_point_counting_other_cycles_fails_every_point(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, records=board({"a": 100, "b": 201, "c": 300}))
        self.assertEqual(value["verdict"], "board-disagrees")
        self.assertIn("b: cycles differ", value["why"])
        for name in ("validation", "final"):
            s = value["score"][name]
            self.assertEqual((s["passed"], s["speedup_geomean"], s["fast_1"]), (0, None, 0.0))
            self.assertEqual({r["failure"] for r in s["points"]}, {"hardware-board-disagrees"})

    def test_a_fallback_or_a_failed_point_on_the_card_disagrees(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        records = board(SIM)
        records[0]["fallbacks"] = ["aten.mm"]
        records[1]["status"] = "fail-numerics"
        value = self.certify(run_dir, records=records)
        self.assertEqual([r["problem"] for r in value["points"]],
                         ["part of it ran on the host", "board status fail-numerics", None])

    def test_a_build_that_does_not_close_is_infeasible(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, pipe={"status": "TIMING_FAIL", "run_id": "b1",
                                            "build": {"status": "TIMING_FAIL", "error": None}})
        self.assertEqual(value["verdict"], "infeasible")
        self.assertEqual(value["score"]["final"]["points"][0]["failure"], "hardware-infeasible")

    def test_a_profile_the_image_cannot_carry_is_infeasible_before_any_build(self):
        scored = json.loads(json.dumps(PROFILES))
        scored["profiles"]["v5"]["elaboration"]["NEW_SWITCH"] = 1
        run_dir = make_run(self.repo, self.task, {"spec/r1_profiles.json": json.dumps(scored)})
        value = self.certify(run_dir)
        self.assertEqual(value["verdict"], "infeasible")
        self.assertIn("NEW_SWITCH", value["why"])
        self.assertEqual(self.calls, [])

    def test_a_board_stage_that_never_reported_has_no_verdict(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, pipe={"status": "PIPELINE_FAIL", "run_id": "b1", "build": {"status": "PASS"},
                                            "error": "ssh: connection timed out"})
        self.assertEqual(value["verdict"], "no-verdict")
        self.assertIn("timed out", value["why"])

    def test_an_image_that_did_not_load_is_a_board_failure(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, records=[], pipe={
            "status": "BOARD_FAIL", "run_id": "b1", "build": {"status": "PASS"},
            "board": {"status": "BOARD_FAIL", "failed_stage": "program", "error": "xbutil program failed"}})
        self.assertEqual(value["verdict"], "board-fail")

    def test_records_left_by_a_failed_board_stage_or_pipeline_do_not_certify(self):
        # A failed ArchBench run, or a retry that failed before replacing an earlier
        # attempt's output, can leave point records behind that match simulation.
        cases = [({"status": "BOARD_FAIL", "run_id": "b1", "build": {"status": "PASS"},
                   "board": {"status": "ARCHBENCH_FAIL", "failed_stage": "archbench", "error": "exit 1"}}, "board-fail"),
                 ({"status": "PIPELINE_FAIL", "run_id": "b1", "build": {"status": "PASS"},
                   "board": {"status": "ENV_FAIL", "error": "ssh: connection timed out"}}, "no-verdict"),
                 ({**PASSED, "status": "PIPELINE_FAIL", "error": "the packet did not import"}, "no-verdict")]
        for pipe, expected in cases:
            with self.subTest(board=pipe["board"]["status"], pipeline=pipe["status"]):
                run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
                value = self.certify(run_dir, pipe=pipe)        # the default records match simulation
                self.assertEqual(value["verdict"], expected)

    def test_the_base_is_certified_against_the_evaluators_baseline(self):
        cache = Path(tempfile.mkdtemp()) / "base.json"
        cache.write_text(json.dumps({"points": {p: {"status": "pass", "cycles": c + 7, "record_cycles": c}
                                                for p, c in SIM.items()}, "missing": []}))
        value = certify.certify(self.repo, self.task, baseline=cache, hosts=Path("/h"), build_host="b",
                                board_host="c", yes=True, pipeline=self.pipeline(PASSED),
                                fetch=lambda *a: board(SIM))
        self.assertEqual(value["verdict"], "certified")
        self.assertNotIn("score", value)
        self.assertTrue(cache.with_name("base.certification.json").is_file())

    def test_a_run_of_another_task_version_is_refused(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        run = json.loads((run_dir / "run.json").read_text())
        run["task"]["sha256"] = "0" * 64
        (run_dir / "run.json").write_text(json.dumps(run))
        with self.assertRaisesRegex(certify.CertifyError, "another version"):
            self.certify(run_dir)


if __name__ == "__main__":
    unittest.main()
