"""Board certification, on a throwaway repository: the tree, the comparison, the verdicts, the score."""
from __future__ import annotations

import dataclasses
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

    def certified_base(self, pipe=None) -> Path:
        """The base certified on card server c, as a software-only candidate's image."""
        cache = Path(tempfile.mkdtemp()) / "base.json"
        cache.write_text(json.dumps({"points": {p: {"status": "pass", "cycles": c + 7, "record_cycles": c}
                                                for p, c in SIM.items()}, "missing": []}))
        built = pipe or {"status": "PASS", "run_id": "base-run", "build": {"status": "PASS", "xclbin_sha256": "f" * 64},
                         "board": {"status": "PASS", "host": "c"}}
        certify.certify(self.repo, self.task, baseline=cache, hosts=Path("/h"), build_host="b", board_host="c",
                        yes=True, pipeline=self.pipeline(built), fetch=lambda *a: board(SIM))
        return cache.with_name("base.certification.json")

    def measured(self, value: dict):
        def run(tree, manifest, record_path, hosts, board_host, run_id, xclbin):
            self.calls.append({"manifest": manifest, "board_host": board_host, "run_id": run_id, "xclbin": xclbin})
            return value
        return run

    def test_software_alone_still_runs_on_the_card_on_the_bases_image(self):
        # A compiler-only candidate is not built again, but its points run on the card:
        # the base's image, its own compiler driving it (user, 2026-11-06).
        image = self.certified_base()
        self.calls.clear()
        run_dir = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n"})
        value = self.certify(run_dir, base_image_record=image, measure=self.measured(
            {"status": "PASS", "run_id": "base-run", "build": {"status": "PASS", "reused": True},
             "board": {"status": "PASS", "host": "c"}}))
        self.assertEqual(value["verdict"], "certified")
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual((call["run_id"], call["xclbin"], call["board_host"]), ("base-run", "f" * 64, "c"))
        self.assertEqual(call["manifest"]["candidate_id"], "certify-t-base")
        self.assertEqual(call["manifest"]["board"]["archbench"]["points"], ["a", "b", "c"])
        self.assertEqual(value["image"], {"record": str(image), "run_id": "base-run", "xclbin_sha256": "f" * 64})
        self.assertEqual(value["card_runs_candidate_code"], ["compiler/a.py"])
        self.assertEqual(value["score"]["final"]["passed"], 1)

    def test_software_alone_without_the_bases_image_is_refused(self):
        run_dir = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n"})
        with self.assertRaisesRegex(certify.CertifyError, "--base-image"):
            self.certify(run_dir)
        self.assertFalse((run_dir / "certification.json").exists())

    def test_on_the_bases_image_cycles_that_differ_still_fail_every_point(self):
        image = self.certified_base()
        run_dir = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n"})
        value = self.certify(run_dir, base_image_record=image, records=board({"a": 100, "b": 201, "c": 300}),
                             measure=self.measured({"status": "PASS", "run_id": "base-run",
                                                    "build": {"status": "PASS"}, "board": {"status": "PASS"}}))
        self.assertEqual(value["verdict"], "board-disagrees")
        self.assertEqual(value["score"]["validation"]["passed"], 0)

    def test_only_a_certified_base_on_this_card_is_an_image(self):
        run_dir = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n"})
        failed = self.certified_base({"status": "PASS", "run_id": "base-run",
                                      "build": {"status": "TIMING_FAIL", "xclbin_sha256": "f" * 64},
                                      "board": {"status": "PASS", "host": "c"}})
        with self.assertRaisesRegex(certify.CertifyError, "not certified"):
            self.certify(run_dir, base_image_record=failed, measure=self.measured({}))
        image = self.certified_base()
        with self.assertRaisesRegex(certify.CertifyError, "is on 'c', not 'elsewhere'"):
            certify.certify(self.repo, self.task, run_dir=run_dir, hosts=Path("/h"), board_host="elsewhere",
                            yes=True, base_image_record=image, measure=self.measured({}))

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

    def test_a_task_that_records_board_cycles_certifies_on_passing_points(self):
        # R41: simulation's HBM is a stand-in and the card counts fewer cycles (user's decision of 2026-11-07).
        self.task.evaluation["certify"]["board_cycles"] = "record"
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, records=board({"a": 90, "b": 201, "c": 250}))
        self.assertEqual(value["verdict"], "certified")
        self.assertEqual([r["problem"] for r in value["points"]], [None, None, None])
        self.assertEqual([r["board_cycles"] for r in value["points"]], [90, 201, 250])

    def test_recording_board_cycles_still_fails_a_fallback(self):
        self.task.evaluation["certify"]["board_cycles"] = "record"
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, records=board({"a": 90, "b": 201, "c": 250}, fallbacks=["host"]))
        self.assertEqual(value["verdict"], "board-disagrees")
        self.assertIn("part of it ran on the host", value["why"])

    def test_an_unknown_board_cycles_rule_is_refused(self):
        self.task.evaluation["certify"]["board_cycles"] = "close"
        with self.assertRaisesRegex(certify.CertifyError, "board_cycles"):
            certify.config(self.task)

    def test_the_card_sessions_rows_follow_the_same_rule(self):
        sim = {"p": {"passed": True, "done_cycles": 100}}
        got = {"p": {"status": "pass", "fallbacks": [], "observer": {"done_cycles": 90}}}
        self.assertEqual(certify._board_rows(sim, got)[0]["problem"], "cycles differ")
        row = certify._board_rows(sim, got, "record")[0]
        self.assertEqual((row["problem"], row["sim_cycles"], row["board_cycles"]), (None, 100, 90))

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

    def test_the_link_and_the_board_stage_are_timed_from_the_pipeline_record(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        names = ("sync:b", "sync:c", "doctor:b", "doctor:c", "build:submit", "build", "export", "transfer",
                 "board:measure", "board")
        steps = [{"name": n, "started_at": 1000.0 + 10 * i, "ok": True, "elapsed_s": 5.0} for i, n in enumerate(names)]
        self.certify(run_dir, pipe={**PASSED, "steps": steps})
        saved = json.loads((run_dir / "certification.json").read_text())
        self.assertEqual([(e["role"], e["host"], e["started"], e["finished"], e["seconds"], e["reserved"])
                          for e in saved["occupancy"]],
                         [("certify-link", "b", 1040.0, 1055.0, 15.0, None),
                          ("certify-board", "c", 1080.0, 1095.0, 15.0, {"cards": 1})])

    def test_without_step_timing_the_pipeline_is_one_entry_and_without_a_pipeline_there_is_none(self):
        run_dir = make_run(self.repo, self.task, {"rtl/core/x.v": "module x; endmodule\n"})
        value = self.certify(run_dir, pipe={"status": "PIPELINE_FAIL", "error": "the pipeline exited 1 and wrote no record"})
        (e,) = value["occupancy"]
        self.assertEqual((e["role"], e["host"], e["reserved"]), ("certify", None, None))
        self.assertIn("link and board stage together", e["covers"])
        self.assertTrue(value["started"] <= e["started"] <= e["finished"] <= value["finished"])
        # Software alone runs on the base's image: the card is held, no build server is.
        image = self.certified_base()
        unchanged = make_run(self.repo, self.task, {"compiler/a.py": "a = 2\n"})
        steps = [{"name": n, "started_at": 2000.0 + 10 * i, "ok": True, "elapsed_s": 5.0}
                 for i, n in enumerate(("sync:c", "doctor:c", "board:measure", "board"))]
        value = self.certify(unchanged, base_image_record=image, measure=self.measured(
            {"status": "PASS", "run_id": "base-run", "build": {"status": "PASS"}, "board": {"status": "PASS"},
             "steps": steps}))
        self.assertEqual([(e["role"], e["host"], e["started"], e["finished"]) for e in value["occupancy"]],
                         [("certify-board", "c", 2020.0, 2035.0)])

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


def stored(problems: list, checks=None, **extra) -> dict:
    """An after-run certification as a run record holds it: rows of sim 100 against the card's 90."""
    rows = [{"point": f"p{i}", "sim_cycles": 100, "board_cycles": 90, "problem": p} for i, p in enumerate(problems)]
    cert = {"verdict": "board-disagrees", "why": "...", "rows": rows,
            "card": {"measured": {"checks": checks if checks is not None else {"fp8": {"exit": 0}}}}}
    return {**cert, **extra}


def island_run(*certs, requests=()) -> dict:
    return {"run_id": "r", "task": {"id": "t", "sha256": "o" * 64},
            "islands": {"islands": [{"agent_result": {"services": {"requests": list(requests)}}}]},
            "champions": [{"rank": i, "id": f"c{i}", "certification": c} for i, c in enumerate(certs, 1)]}


class Again(unittest.TestCase):
    """A finished run's certifications judged again from what they stored (the user's decision of 2026-11-07)."""

    def test_cycles_that_only_differ_certify_under_record(self):
        value = certify.rejudge(island_run(stored(["cycles differ", "cycles differ"])), "record")
        c = value["certifications"][0]
        self.assertEqual((c["was"], c["verdict"], c["judged_again"]), ("board-disagrees", "certified", True))
        self.assertEqual(c["cycles_on_card"], {"points": 2, "board_over_simulation_geomean": 0.9,
                                               "board_over_simulation_range": [0.9, 0.9]})
        self.assertEqual(value["result"], {"rank": 1, "id": "c1"})

    def test_a_fallback_or_a_failed_board_check_still_disagrees(self):
        value = certify.rejudge(island_run(stored(["cycles differ", "part of it ran on the host"]),
                                           stored(["cycles differ"], checks={"fp8": {"exit": 1}})), "record")
        self.assertEqual([c["verdict"] for c in value["certifications"]], ["board-disagrees", "board-disagrees"])
        self.assertIn("part of it ran on the host", value["certifications"][0]["why"])
        self.assertIn("board check fp8 did not pass", value["certifications"][1]["why"])
        self.assertIsNone(value["result"])

    def test_under_equal_differing_cycles_still_disagree(self):
        value = certify.rejudge(island_run(stored([None])), "equal")
        self.assertEqual(value["certifications"][0]["verdict"], "board-disagrees")

    def test_a_point_failed_in_simulation_stays_as_it_was(self):
        cert = stored(["cycles differ"])
        cert["rows"].append({"point": "q", "sim_cycles": None, "board_cycles": None, "problem": None,
                             "note": "failed in simulation"})
        c = certify.rejudge(island_run(cert), "record")["certifications"][0]
        self.assertEqual(c["verdict"], "certified")
        self.assertEqual(c["cycles_on_card"]["points"], 1)

    def test_a_verdict_before_the_card_ran_is_kept_and_the_next_champion_can_be_the_result(self):
        value = certify.rejudge(island_run({"verdict": "infeasible", "why": "build TIMING_FAIL"},
                                           stored(["cycles differ"])), "record")
        first = value["certifications"][0]
        self.assertEqual((first["verdict"], first["judged_again"]), ("infeasible", False))
        self.assertEqual(value["result"], {"rank": 2, "id": "c2"})

    def test_a_reused_requests_board_checks_come_from_the_run_record(self):
        cert = stored(["cycles differ"], checks={}, reused={"request": "l13", "points": ["p0"]})
        ok = {"id": "l13", "full": {"card": {"outcome": "ok", "record": {"measured": {"checks": {"fp8": {"exit": 1}}}}}}}
        c = certify.rejudge(island_run(cert, requests=[ok]), "record")["certifications"][0]
        self.assertEqual(c["verdict"], "board-disagrees")
        self.assertEqual(c["checks"], {"fp8": {"exit": 1}})
        c = certify.rejudge(island_run(cert, requests=[]), "record")["certifications"][0]
        self.assertEqual((c["verdict"], c["judged_again"]), ("board-disagrees", False))
        self.assertIn("l13", c["note"])

    def test_checks_the_record_kept_are_used_as_they_are(self):
        cert = stored(["cycles differ"], checks={"fp8": {"exit": 1}})
        cert["checks"] = {"fp8": {"exit": 0}}
        self.assertEqual(certify.rejudge(island_run(cert), "record")["certifications"][0]["verdict"], "certified")

    def test_a_single_run_is_judged_too_and_a_pending_one_is_refused(self):
        run = {"run_id": "r", "task": {"id": "t"}, "certification": stored(["cycles differ"])}
        self.assertEqual(certify.rejudge(run, "record")["result"], {"rank": 1, "id": "r"})
        with self.assertRaisesRegex(certify.CertifyError, "nothing to judge again"):
            certify.rejudge({"run_id": "r", "certification": "pending"}, "record")

    def test_the_run_dir_gets_its_own_record_and_run_json_is_left_as_it_was(self):
        task = make_task("b", "p")
        task.evaluation["certify"]["board_cycles"] = "record"
        run_dir = Path(tempfile.mkdtemp())
        raw = json.dumps(island_run(stored(["cycles differ"]))).encode()
        (run_dir / "run.json").write_bytes(raw)
        value = certify.rejudge_run(task, run_dir)
        self.assertEqual((run_dir / "run.json").read_bytes(), raw)
        written = json.loads((run_dir / "certification-again.json").read_text())
        self.assertEqual((written["board_cycles"], written["result"]), ("record", {"rank": 1, "id": "c1"}))
        self.assertEqual(written["task_at_run"]["sha256"], "o" * 64)
        self.assertEqual(value["task"]["sha256"], task.sha256)
        other = dataclasses.replace(task, task_id="u")
        with self.assertRaisesRegex(certify.CertifyError, "not 'u'"):
            certify.rejudge_run(other, run_dir)


if __name__ == "__main__":
    unittest.main()
