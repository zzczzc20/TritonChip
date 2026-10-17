"""The board stage end to end, on a machine with no board.

Every vendor step is a manifest-supplied command, so these tests exercise the
real ``submit``/``measure`` paths -- the lease, the verdicts, the records left
behind -- without Vivado, XRT or a card.  The tests that read the surrounding
checkout (the flow scripts, the profile table, upstream's recorded board
results) skip when it does not carry them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.core import (
    HARNESS_TREE, HarnessError, _implementation_verdict, doctor, load_profile, measure_existing,
    plan, reconcile, software_identity, submit,
)
from tritonchip_fpga_harness.measurement import (
    MeasurementError, _default_archbench, _default_conformance, archbench_python, board_preflight,
    image_profile_contract, measure_board, summarize_archbench,
)

PY = sys.executable
RUN = "import json,os,pathlib; run=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']); (run/'board').mkdir(parents=True,exist_ok=True); "
BOARD_RECORD = {
    "schema": "archbench-torch-record/1", "point_id": "op.add/tiny/fp32-n1", "family": "add", "tier": "tiny",
    "status": "pass", "evidence_level": "measured-board", "platform": {"backend": "fpga-u50", "kind": "tnca"},
    "time": {"unit": "cycles", "device": 123, "clock_hz": 100_000_000, "launches": 1},
    "counters": {"done_cycles": 123, "done_retired": 9, "perf_cycles": 120}, "problems": [],
}


def conformance(passed: bool = True) -> list[str]:
    return [PY, "-c", RUN + f"(run/'board'/'conformance.json').write_text(json.dumps([{{'id':'R36-TEST-01','passed':{passed}}}]))"]


def archbench(record: dict[str, object] | None = None) -> list[str]:
    return [PY, "-c", RUN + "(run/'board'/'archbench.jsonl').write_text(json.dumps(" + repr(record or BOARD_RECORD) + ")+'\\n')"]


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=harness-test", "-c", "user.email=harness-test@example.invalid", *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


class BoardStageCase(unittest.TestCase):
    """A committed candidate in a throwaway repository, and a state directory beside it."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tritonchip-harness-board-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "candidate"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        self.commit("rtl/core.v", "module core; endmodule\n")
        self.state = self.tmp / "state"

    def commit(self, relative: str, text: str) -> str:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", f"touch {relative}")
        return git(self.repo, "rev-parse", "HEAD")

    def manifest(self, **board: object) -> dict[str, object]:
        value: dict[str, object] = {"candidate_id": "board-test", "target": "u50", "mode": "full"}
        if board:
            value["board"] = {"backend": "fpga-u50", "profile": "core-vrf-sync-v1", "program": False, **board}
        return value

    def run_dir(self, result: dict[str, object]) -> Path:
        return self.state / "runs" / str(result["run_id"])


class SubmitBoardVerdicts(BoardStageCase):
    def test_board_pass_records_software_and_one_lease_covers_both_stages(self) -> None:
        result = submit(HERE, self.repo, self.manifest(conformance={"command": conformance()},
                                                       archbench={"command": archbench()}), self.state, "mock")
        self.assertEqual(result["status"], "PASS", result.get("measurement_error"))
        self.assertEqual([s["name"] for s in result["stages"]], ["package", "link", "conformance", "archbench"])
        software = result["measurement"]["software"]
        self.assertEqual(software["relation"], "same-commit")
        self.assertEqual(software["commit"], result["source_commit"])
        self.assertFalse(software["dirty"])

    def test_invalid_archbench_evidence_is_archbench_fail_and_keeps_the_record(self) -> None:
        simulated = dict(BOARD_RECORD, evidence_level="measured-rtl")
        result = submit(HERE, self.repo, self.manifest(conformance={"command": conformance()},
                                                       archbench={"command": archbench(simulated)}), self.state, "mock")
        self.assertEqual(result["status"], "ARCHBENCH_FAIL")
        self.assertEqual(result["measurement"]["failed_stage"], "archbench")
        self.assertEqual(result["measurement"]["archbench"]["failed"], 1)
        # Conformance did pass, and the record still says so.
        self.assertTrue(result["measurement"]["conformance"]["passed"])
        written = json.loads((self.run_dir(result) / "board_measurement.json").read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "ARCHBENCH_FAIL")

    def test_missing_archbench_output_is_archbench_fail_not_a_crash(self) -> None:
        silent = [PY, "-c", "pass"]
        result = submit(HERE, self.repo, self.manifest(conformance={"command": conformance()},
                                                       archbench={"command": silent}), self.state, "mock")
        self.assertEqual(result["status"], "ARCHBENCH_FAIL")
        self.assertIn("missing ArchBench JSONL", result["measurement_error"])

    def test_failed_conformance_is_board_fail_and_archbench_never_runs(self) -> None:
        result = submit(HERE, self.repo, self.manifest(conformance={"command": conformance(False)},
                                                       archbench={"command": archbench()}), self.state, "mock")
        self.assertEqual(result["status"], "BOARD_FAIL")
        self.assertEqual(result["measurement"]["failed_stage"], "conformance")
        self.assertEqual([s["name"] for s in result["measurement"]["stages"]], ["conformance"])
        self.assertFalse((self.run_dir(result) / "board" / "archbench.jsonl").exists())

    def test_failed_preflight_never_reaches_the_programmer(self) -> None:
        marker = self.tmp / "programmed"
        go = [PY, "-c", f"import pathlib; pathlib.Path({str(marker)!r}).write_text('loaded')"]
        refuse = [PY, "-c", "raise SystemExit(1)"]
        result = submit(HERE, self.repo, self.manifest(program={"check_command": refuse, "command": go},
                                                       conformance={"command": conformance()}), self.state, "mock")
        self.assertEqual(result["status"], "BOARD_FAIL")
        self.assertEqual(result["measurement"]["failed_stage"], "program_check")
        self.assertFalse(marker.exists())

    def test_bdf_reaches_the_flow_programmer_under_the_name_it_reads(self) -> None:
        seen = self.tmp / "env.json"
        go = [PY, "-c", f"import json,os,pathlib; pathlib.Path({str(seen)!r}).write_text(json.dumps({{k: os.environ.get(k) for k in ('BDF','TNCA_BDF','TNCA_BOARD')}}))"]
        result = submit(HERE, self.repo, self.manifest(bdf="0000:65:00.1", program={"command": go}, conformance=False),
                        self.state, "mock")
        self.assertEqual(result["status"], "PASS", result.get("measurement_error"))
        self.assertEqual(json.loads(seen.read_text(encoding="utf-8")),
                         {"BDF": "0000:65:00.1", "TNCA_BDF": "0000:65:00.1", "TNCA_BOARD": "u50"})
        self.assertEqual(result["measurement"]["bdf"], "0000:65:00.1")

    def test_dirty_candidate_is_refused_and_recorded_when_allowed(self) -> None:
        (self.repo / "rtl" / "core.v").write_text("module core; wire x; endmodule\n", encoding="utf-8")
        with self.assertRaises(HarnessError):
            submit(HERE, self.repo, self.manifest(), self.state, "mock")
        result = submit(HERE, self.repo, self.manifest(), self.state, "mock", allow_dirty=True)
        self.assertEqual(result["status"], "PASS")
        self.assertIn("wire x", (self.run_dir(result) / "source.diff").read_text(encoding="utf-8"))


class DeferredMeasure(BoardStageCase):
    def board(self) -> dict[str, object]:
        return {"conformance": {"command": conformance()}, "archbench": {"command": archbench()}}

    def test_skip_board_then_measure(self) -> None:
        built = submit(HERE, self.repo, self.manifest(**self.board()), self.state, "mock", skip_board=True)
        self.assertEqual(built["status"], "PASS")
        self.assertTrue(built["board_deferred"])
        self.assertIsNone(built["measurement"])
        measured = measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        self.assertEqual(measured["status"], "PASS", measured.get("measurement_error"))
        self.assertFalse(measured["board_deferred"])
        self.assertEqual(measured["board_config_source"], "submit")
        self.assertEqual(measured["measurement"]["archbench"]["cycles_total"], 123)
        self.assertEqual([s["name"] for s in measured["stages"]], ["package", "link", "conformance", "archbench"])

    def test_a_second_measure_replaces_the_first_attempts_board_stages(self) -> None:
        built = submit(HERE, self.repo, self.manifest(conformance={"command": conformance(False)}), self.state, "mock",
                       skip_board=True)
        failed = measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        self.assertEqual(failed["status"], "BOARD_FAIL")
        # The link is still a link that passed.
        self.assertEqual(failed["implementation_status"], "PASS")
        again = measure_existing(HERE, self.repo, self.state, str(built["run_id"]),
                                 board_manifest=self.manifest(conformance={"command": conformance()}))
        self.assertEqual(again["status"], "PASS", again.get("measurement_error"))
        self.assertEqual([s["name"] for s in again["stages"]], ["package", "link", "conformance"])
        self.assertIsNone(again["measurement_error"])

    def test_build_only_run_takes_its_board_block_at_measure_time(self) -> None:
        built = submit(HERE, self.repo, self.manifest(), self.state, "mock")
        with self.assertRaisesRegex(HarnessError, "no board measurement configuration"):
            measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        late = dict(self.manifest(**self.board()), source_commit=built["source_commit"])
        measured = measure_existing(HERE, self.repo, self.state, str(built["run_id"]), board_manifest=late)
        self.assertEqual(measured["status"], "PASS", measured.get("measurement_error"))
        self.assertEqual(measured["board_config_source"], "measure")
        self.assertTrue((self.run_dir(built) / "board_manifest.json").is_file())
        # What the run was submitted with is not rewritten.
        self.assertNotIn("board", json.loads((self.run_dir(built) / "manifest.json").read_text(encoding="utf-8")))

    def test_measure_manifest_must_be_the_same_candidate(self) -> None:
        built = submit(HERE, self.repo, self.manifest(), self.state, "mock")
        other = dict(self.manifest(**self.board()), candidate_id="another-candidate")
        with self.assertRaisesRegex(HarnessError, "candidate_id"):
            measure_existing(HERE, self.repo, self.state, str(built["run_id"]), board_manifest=other)
        elsewhere = dict(self.manifest(**self.board()), source_commit="0" * 40)
        with self.assertRaisesRegex(HarnessError, "source_commit"):
            measure_existing(HERE, self.repo, self.state, str(built["run_id"]), board_manifest=elsewhere)

    def test_measure_refuses_software_that_moved_on_outside_the_harness(self) -> None:
        built = submit(HERE, self.repo, self.manifest(**self.board()), self.state, "mock", skip_board=True)
        self.commit("python/tnca/runtime.py", "# a different runtime\n")
        with self.assertRaisesRegex(HarnessError, "allow-source-mismatch"):
            measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        measured = measure_existing(HERE, self.repo, self.state, str(built["run_id"]), allow_source_mismatch=True)
        self.assertEqual(measured["status"], "PASS", measured.get("measurement_error"))
        software = measured["measurement"]["software"]
        self.assertEqual(software["relation"], "differs")
        self.assertEqual(software["implemented_commit"], built["source_commit"])
        self.assertEqual(software["changed_paths"], ["python/tnca/runtime.py"])

    def test_measure_accepts_a_checkout_that_only_changed_the_harness(self) -> None:
        built = submit(HERE, self.repo, self.manifest(**self.board()), self.state, "mock", skip_board=True)
        self.commit(f"{HARNESS_TREE}/core.py", "# a fixed harness\n")
        measured = measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        self.assertEqual(measured["status"], "PASS", measured.get("measurement_error"))
        self.assertEqual(measured["measurement"]["software"]["relation"], "harness-only-diff")

    def test_measure_refuses_a_dirty_checkout(self) -> None:
        built = submit(HERE, self.repo, self.manifest(**self.board()), self.state, "mock", skip_board=True)
        (self.repo / "scratch.py").write_text("x = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(HarnessError, "dirty"):
            measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        measured = measure_existing(HERE, self.repo, self.state, str(built["run_id"]), allow_dirty=True)
        self.assertTrue(measured["measurement"]["software"]["dirty"])

    def test_software_identity_tells_a_merge_of_the_same_tree_from_a_change(self) -> None:
        first = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "commit", "-q", "--allow-empty", "-m", "same tree, new commit")
        self.assertEqual(software_identity(self.repo, first)["relation"], "same-tree")
        self.assertEqual(software_identity(self.repo, "0" * 40)["relation"], "unknown")
        self.assertEqual(software_identity(self.repo, None)["relation"], "unknown")


class UnprogrammedMeasurement(BoardStageCase):
    """program: false measures whatever the card holds, so it says what that was."""

    def programmer(self, body: str) -> None:
        self.commit("flow/fpga/program.sh", "#!/bin/sh\n" + body + "\n")
        (self.repo / "flow/fpga/program.sh").chmod(0o755)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "programmer is executable")

    def test_the_loaded_image_is_put_on_record(self) -> None:
        self.programmer('[ "$1" = --status ] || exit 9\n'
                        'echo "device   0000:04:00.1"; echo "loaded   78F04922-980e-d3cb-ee78-cf732d6ec13f"; echo "kdsstat"')
        result = submit(HERE, self.repo, self.manifest(conformance={"command": conformance()}), self.state, "mock")
        self.assertEqual(result["status"], "PASS", result.get("measurement_error"))
        measurement = result["measurement"]
        self.assertFalse(measurement["programmed"])
        self.assertEqual(measurement["loaded_xclbin_uuid"], "78f04922-980e-d3cb-ee78-cf732d6ec13f")
        self.assertEqual([s["name"] for s in measurement["stages"]], ["loaded_image", "conformance"])

    def test_an_unreadable_report_is_recorded_as_unknown_not_guessed(self) -> None:
        self.programmer('echo "loaded   <none>"')
        result = submit(HERE, self.repo, self.manifest(conformance={"command": conformance()}), self.state, "mock")
        self.assertEqual(result["status"], "PASS", result.get("measurement_error"))
        self.assertIsNone(result["measurement"]["loaded_xclbin_uuid"])

    def loader(self, replacing: str, now: str) -> list[str]:
        # What flow/fpga/program.sh --go prints around the load.
        return [PY, "-c", f"print('  REPLACING={replacing}'); print('  INFO: xbutil program succeeded on 0000:04:00.1'); "
                          f"print('program.sh: OK -- device now reports xclbin {now}')"]

    def test_a_load_that_changed_the_image_is_a_reconfiguration(self) -> None:
        go = self.loader("c4637d43-d9de-4f0d-1bde-4e3c335093d2", "78f04922-980e-d3cb-ee78-cf732d6ec13f")
        measurement = submit(HERE, self.repo, self.manifest(program={"command": go}, conformance=False),
                             self.state, "mock")["measurement"]
        self.assertTrue(measurement["reconfigured"])
        self.assertEqual(measurement["replaced_xclbin_uuid"], "c4637d43-d9de-4f0d-1bde-4e3c335093d2")
        self.assertEqual(measurement["loaded_xclbin_uuid"], "78f04922-980e-d3cb-ee78-cf732d6ec13f")

    def test_a_load_of_the_image_already_there_is_recorded_as_no_reconfiguration(self) -> None:
        # XRT returns success and downloads nothing; the device keeps its state.
        same = "78f04922-980e-d3cb-ee78-cf732d6ec13f"
        result = submit(HERE, self.repo, self.manifest(program={"command": self.loader(same, same)}, conformance=False),
                        self.state, "mock")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["measurement"]["programmed"])
        self.assertIs(result["measurement"]["reconfigured"], False)

    def test_a_first_load_onto_an_empty_region_is_a_reconfiguration(self) -> None:
        go = self.loader("<none>", "78f04922-980e-d3cb-ee78-cf732d6ec13f")
        measurement = submit(HERE, self.repo, self.manifest(program={"command": go}, conformance=False),
                             self.state, "mock")["measurement"]
        self.assertTrue(measurement["reconfigured"])
        self.assertIsNone(measurement["replaced_xclbin_uuid"])

    def test_a_programmed_run_asks_for_no_report(self) -> None:
        go = [PY, "-c", "pass"]
        result = submit(HERE, self.repo, self.manifest(program={"command": go}, conformance=False), self.state, "mock")
        self.assertTrue(result["measurement"]["programmed"])
        self.assertIsNone(result["measurement"]["loaded_xclbin_uuid"])
        # A programmer that does not say what it replaced leaves the question open.
        self.assertIsNone(result["measurement"]["reconfigured"])
        self.assertEqual([s["name"] for s in result["measurement"]["stages"]], ["program"])


class ImageProfile(BoardStageCase):
    """The profile a manifest names against the image the checkout builds."""

    def setUp(self) -> None:
        super().setUp()
        base = {"NUM_RCTX": 2, "HAS_SFU": 1}
        profiles = {
            "base-v1": {"elaboration": base},
            "base-v2": {"elaboration": base, "implementation": {"rtl_parameters": {"VRF_BYPASS": 1}}},
            "wide-v4": {"elaboration": dict(base, NUM_RCTX=4), "implementation": {"rtl_parameters": {"VRF_BYPASS": 1}}},
        }
        self.commit("spec/r1_profiles.json", json.dumps({"profiles": profiles}))
        # The wrapper's NUM_RCTX is the one that reaches the device; the top's is overridden.
        self.commit("rtl/board/tnca_u50_xo.v", "module tnca_u50_xo #(\n    parameter NUM_RCTX = 2\n) ();\nendmodule\n")
        self.commit("rtl/fpga/tnca_fpga_top.v",
                    "module tnca_fpga_top #(\n    parameter NUM_RCTX = 8,\n    parameter HAS_SFU = 1,\n"
                    "    parameter VRF_BYPASS = 1\n) ();\nendmodule\n")

    def test_matching_profile_has_no_findings(self) -> None:
        contract = image_profile_contract(self.repo, "base-v2")
        self.assertEqual(contract["mismatches"], [])
        self.assertEqual(contract["more_specific"], [])
        self.assertEqual(contract["checked"], {"HAS_SFU": 1, "NUM_RCTX": 2, "VRF_BYPASS": 1})

    def test_older_profile_name_on_a_newer_image_is_reported(self) -> None:
        contract = image_profile_contract(self.repo, "base-v1")
        self.assertEqual(contract["mismatches"], [])
        self.assertEqual(contract["more_specific"], ["base-v2"])
        self.assertEqual(plan(HERE, self.repo, self.manifest(profile="base-v1"), "mock")
                         ["board_measurement"]["image_profile"]["more_specific"], ["base-v2"])

    def test_contradicted_profile_stops_before_the_card_is_touched(self) -> None:
        marker = self.tmp / "programmed"
        go = [PY, "-c", f"import pathlib; pathlib.Path({str(marker)!r}).write_text('loaded')"]
        result = submit(HERE, self.repo, self.manifest(profile="wide-v4", program={"command": go}, conformance=False),
                        self.state, "mock")
        self.assertEqual(result["status"], "BOARD_FAIL")
        self.assertEqual(result["measurement"]["failed_stage"], "image_profile")
        self.assertIn("NUM_RCTX: profile 4, RTL default 2", result["measurement_error"])
        self.assertFalse(marker.exists())

    def test_unknown_profile_or_checkout_without_a_table_is_unknown_not_passed(self) -> None:
        self.assertIsNone(image_profile_contract(self.repo, "no-such-profile"))
        self.assertIsNone(image_profile_contract(self.tmp, "base-v1"))


class BoardPreflight(BoardStageCase):
    """What the board stage needs, asked before the link instead of after it."""

    def interpreter(self, name: str, body: str) -> str:
        path = self.tmp / name
        path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def without_pytorch(self) -> str:
        return self.interpreter("python-bare", "echo \"ModuleNotFoundError: No module named 'torch'\" >&2; exit 1")

    def with_pytorch(self, tnca_backend: bool = True) -> str:
        seen = json.dumps({"torch": "2.14.0", "triton": "3.8.0", "tnca_backend": tnca_backend})
        return self.interpreter(f"python-torch-{tnca_backend}", f"echo 'HARNESS-PROBE {seen}'")

    def failed(self, checks: list[dict[str, object]]) -> dict[str, str]:
        return {str(c["name"]): str(c["detail"]) for c in checks if not c["ok"]}

    def test_interpreter_without_pytorch_is_env_fail_before_any_build(self) -> None:
        manifest = self.manifest(conformance=False, archbench={"tiers": ["tiny"], "python": self.without_pytorch()})
        result = submit(HERE, self.repo, manifest, self.state, "mock")
        self.assertEqual(result["status"], "ENV_FAIL")
        self.assertEqual(result["stages"], [])
        self.assertIn("No module named 'torch'", self.failed(result["board_preflight"])["board_archbench_python"])
        self.assertIn("board_archbench_python", result["error"])
        # The implementation never started: there is nothing to mistake for a build.
        self.assertFalse((self.run_dir(result) / "artifacts").exists())

    def test_pytorch_without_the_tnca_backend_is_not_ready(self) -> None:
        manifest = self.manifest(conformance=False, archbench={"python": self.with_pytorch(tnca_backend=False)})
        failed = self.failed(board_preflight(root=self.repo, profile=load_profile(HERE, "u50"), manifest=manifest))
        self.assertIn("tnca package is not installed", failed["board_archbench_python"])

    def test_a_ready_host_passes_every_check(self) -> None:
        xrt = self.tmp / "xrt"
        for relative in ("bin/xbutil", "bin/xclbinutil", "lib/libxrt_core.so"):
            (xrt / relative).parent.mkdir(parents=True, exist_ok=True)
            (xrt / relative).write_text("", encoding="utf-8")
            (xrt / relative).chmod(0o755)
        for relative in ("flow/fpga/program.sh", "tools/r36/conformance.py", "archbench/run.py"):
            (self.repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / relative).write_text("", encoding="utf-8")
            (self.repo / relative).chmod(0o755)
        manifest = self.manifest(program=True, archbench={"python": self.with_pytorch()})
        with mock.patch.dict(os.environ, {"XILINX_XRT": str(xrt)}):
            checks = board_preflight(root=self.repo, profile=load_profile(HERE, "u50"), manifest=manifest)
        self.assertEqual(self.failed(checks), {})
        self.assertEqual({c["name"] for c in checks}, {
            "board_program_script", "board_xbutil", "board_xclbinutil", "board_conformance_tool",
            "board_xrt_library", "board_archbench_runner", "board_archbench_python",
        })

    def test_steps_the_manifest_replaces_are_not_checked(self) -> None:
        manifest = self.manifest(program={"command": [PY, "-c", "pass"]}, conformance={"command": conformance()},
                                 archbench={"command": archbench()})
        self.assertEqual(board_preflight(root=self.repo, profile=load_profile(HERE, "u50"), manifest=manifest), [])
        self.assertEqual(board_preflight(root=self.repo, profile=load_profile(HERE, "u50"),
                                         manifest={"candidate_id": "c", "target": "u50", "mode": "full"}), [])

    def test_measure_reports_an_unready_host_and_writes_nothing_against_the_run(self) -> None:
        manifest = self.manifest(conformance=False, archbench={"python": self.without_pytorch()})
        built = submit(HERE, self.repo, manifest, self.state, "mock", skip_board=True)
        self.assertEqual(built["status"], "PASS")
        before = (self.run_dir(built) / "result.json").read_text(encoding="utf-8")
        with self.assertRaisesRegex(HarnessError, "cannot start on this host.*board_archbench_python"):
            measure_existing(HERE, self.repo, self.state, str(built["run_id"]))
        self.assertEqual((self.run_dir(built) / "result.json").read_text(encoding="utf-8"), before)
        self.assertFalse((self.run_dir(built) / "board_measurement.json").exists())

    def test_doctor_checks_a_manifests_board_stage_when_given_one(self) -> None:
        manifest = self.manifest(conformance=False, archbench={"python": self.without_pytorch()})
        self.assertTrue(doctor(HERE, self.repo, "u50", "mock")["ok"])
        report = doctor(HERE, self.repo, "u50", "mock", manifest)
        self.assertFalse(report["ok"])
        self.assertIn("board_archbench_python", self.failed(report["checks"]))

    def test_archbench_runs_under_the_interpreter_the_manifest_or_torch_python_names(self) -> None:
        with mock.patch.dict(os.environ, {"TORCH_PYTHON": "/env/torch/bin/python"}):
            self.assertEqual(archbench_python({"archbench": {"python": "/explicit/python"}}), "/explicit/python")
            self.assertEqual(archbench_python({"archbench": {"tiers": ["tiny"]}}), "/env/torch/bin/python")
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("TORCH_PYTHON", None)
            self.assertEqual(archbench_python({"archbench": {}}), sys.executable)
        argv = _default_archbench(root=self.repo, python="/env/torch/bin/python", backend="fpga-u280",
                                  profile="core-vrf-sync-v2", xclbin=Path("/run/a.xclbin"), tiers=["tiny"],
                                  output=Path("/run/board/archbench.jsonl"), jobs=1, counters=True)
        self.assertEqual(argv[:3], ["/env/torch/bin/python", "-m", "archbench.run"])
        self.assertEqual(json.loads(argv[argv.index("--backend-args") + 1]), {"xclbin": "/run/a.xclbin"})

    def test_named_points_replace_the_tiers(self) -> None:
        from tritonchip_fpga_harness.schema import SchemaError, validate_manifest
        argv = _default_archbench(root=self.repo, python="py", backend="fpga-u280", profile="core-vrf-sync-v5",
                                  xclbin=Path("/run/a.xclbin"), tiers=["tiny"], output=Path("/run/out.jsonl"),
                                  jobs=1, counters=False, points=["op.a/tiny/x", "op.b/tiny/y"])
        self.assertEqual(argv[argv.index("--point") + 1:argv.index("--jobs")], ["op.a/tiny/x", "op.b/tiny/y"])
        self.assertNotIn("--tier", argv)
        base = {"candidate_id": "c", "target": "u280", "mode": "full"}
        board = {"backend": "fpga-u280", "profile": "core-vrf-sync-v5"}
        validate_manifest(dict(base, board=dict(board, archbench={"points": ["op.a/tiny/x"]})))
        with self.assertRaisesRegex(SchemaError, "tiers or points, not both"):
            validate_manifest(dict(base, board=dict(board, archbench={"points": ["op.a/tiny/x"], "tiers": ["tiny"]})))
        with self.assertRaisesRegex(SchemaError, "non-empty list of point ids"):
            validate_manifest(dict(base, board=dict(board, archbench={"points": []})))


class RefusedTargets(BoardStageCase):
    """A target the checkout or the host cannot run is not a failed build."""

    def test_a_checkout_without_the_flow_is_env_fail_and_says_which_checks(self) -> None:
        # The throwaway repository has no flow/fpga at all.
        result = submit(HERE, self.repo, {"candidate_id": "c", "target": "u280", "mode": "full"},
                        self.state, "u280")
        self.assertEqual(result["status"], "ENV_FAIL")
        self.assertEqual(result["stages"], [])
        refused = {c["name"] for c in result["preflight_failed"]}
        self.assertLessEqual({"package_script", "link_script", "flow_clock_contract", "device_divider_contract"}, refused)
        self.assertIn("flow_clock_contract", result["error"])
        self.assertTrue((self.run_dir(result) / "preflight.json").is_file())
        self.assertFalse((self.run_dir(result) / "artifacts" / "tnca_u50.xo").exists())


class Verdicts(unittest.TestCase):
    def verdict(self, checks: list[dict[str, object]], returncode: int) -> str:
        profile = load_profile(HERE, "u50")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "artifacts").mkdir()
            for name in profile.data["expected_artifacts"]:
                (run_dir / "artifacts" / name).write_bytes(b"x")
            (run_dir / "artifact_validation.json").write_text(
                json.dumps({"ok": all(c["ok"] for c in checks), "checks": checks}), encoding="utf-8")
            return _implementation_verdict(run_dir, profile, {"mode": "full"},
                                           [{"name": "link", "returncode": returncode}])["status"]

    def test_a_link_that_failed_on_timing_is_a_timing_failure(self) -> None:
        # v++ exits non-zero when the design does not close; the routed report is still there.
        self.assertEqual(self.verdict([{"name": "vpp_rc", "ok": False}, {"name": "timing_clean", "ok": False}], 1),
                         "TIMING_FAIL")

    def test_a_link_that_failed_before_routing_is_a_build_failure(self) -> None:
        self.assertEqual(self.verdict([{"name": "vpp_rc", "ok": False}, {"name": "routed_timing_report", "ok": False}], 1),
                         "BUILD_FAIL")

    def test_a_clean_link_passes(self) -> None:
        self.assertEqual(self.verdict([{"name": "timing_clean", "ok": True}], 0), "PASS")


class Reconcile(unittest.TestCase):
    def test_finds_the_process_record_a_board_stage_leaves_under_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            run_dir = state / "runs" / "r"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps({"run_id": "r", "status": "RUNNING"}), encoding="utf-8")
            (run_dir / "logs" / "active_process.json").write_text(
                json.dumps({"pid": 999999, "pgid": 999999, "pid_starttime": "1", "boot_id": "gone"}), encoding="utf-8")
            value = reconcile(state, "r")
        self.assertEqual(value["active_process"]["pid"], 999999)
        self.assertFalse(value["process_alive"])
        self.assertEqual(value["status"], "ORPHANED")


class MeasureBoardWithoutACard(unittest.TestCase):
    def measure(self, root: Path, **board: object) -> dict[str, object]:
        profile = load_profile(HERE, "u280")
        run_dir = root / "run"
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "artifacts" / "tnca_u280.xclbin").write_bytes(b"xclbin")
        return measure_board(root=root, run_dir=run_dir, profile=profile,
                             manifest={"board": {"backend": "fpga-u280", "profile": "core-vrf-sync-v2", **board}}, timeout_s=5)

    def test_missing_program_script_fails_closed(self) -> None:
        # A checkout without the flow's programmer: nothing to exec, so no load.
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MeasurementError) as caught:
                self.measure(Path(td), program=True, conformance=False)
        self.assertEqual(caught.exception.partial["failed_stage"], "program_check")
        self.assertEqual(caught.exception.status, "BOARD_FAIL")

    def test_backend_must_be_the_profiles_board(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(MeasurementError, "does not match profile board"):
                self.measure(Path(td), backend="fpga-u50", program=False, conformance=False)

    def test_board_profile_is_never_defaulted(self) -> None:
        from tritonchip_fpga_harness.schema import SchemaError, validate_manifest
        with self.assertRaisesRegex(SchemaError, "board.profile is required"):
            validate_manifest({"candidate_id": "c", "target": "u280", "mode": "full",
                               "board": {"backend": "fpga-u280"}})
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(MeasurementError, "board.profile is required"):
                self.measure(Path(td), profile="", program=False, conformance=False)

    def test_default_conformance_names_the_manifests_profile_and_the_image(self) -> None:
        argv = _default_conformance(root=Path("/repo"), run_dir=Path("/run"), backend="fpga-u280",
                                    profile="core-vrf-sync-v2", xclbin=Path("/run/artifacts/tnca_u280.xclbin"),
                                    result_file=Path("/run/board/conformance.json"))
        self.assertEqual(argv[1], "/repo/tools/r36/conformance.py")
        self.assertEqual(argv[argv.index("--profile") + 1], "core-vrf-sync-v2")
        self.assertEqual(argv[argv.index("--xclbin") + 1], "/run/artifacts/tnca_u280.xclbin")
        self.assertIn("--require-measured", argv)


class ThisCheckout(unittest.TestCase):
    """The harness against the flow and records of the checkout it is committed in."""

    def test_profiles_name_scripts_this_checkout_has(self) -> None:
        boards = REPO / "flow" / "fpga" / "boards.sh"
        if not boards.is_file() or "u280)" not in boards.read_text(encoding="utf-8"):
            self.skipTest("checkout predates the board-selectable flow")
        for target in ("u280", "u50"):
            data = load_profile(HERE, target).data
            for key in ("package_script", "link_script", "program_script"):
                self.assertTrue((REPO / data[key]).is_file(), f"{target}: {key} {data[key]} is not in this checkout")
        # The flow's own programmer is board-aware; the harness adds none of its own.
        self.assertIn("boards.sh", (REPO / "flow/fpga/program.sh").read_text(encoding="utf-8"))

    def test_default_commands_use_options_the_tools_accept(self) -> None:
        tool = REPO / "tools" / "r36" / "conformance.py"
        runner = REPO / "archbench" / "run.py"
        if not tool.is_file() or not runner.is_file():
            self.skipTest("checkout has no conformance suite or ArchBench")
        conformance_text = tool.read_text(encoding="utf-8")
        for option in ("--backends", "--profile", "--xclbin", "--require-measured", "--json"):
            self.assertIn(f"'{option}'", conformance_text)
        runner_text = runner.read_text(encoding="utf-8")
        for option in ("--platform", "--backend", "--profile", "--backend-args", "--tier", "--jobs", "--out", "--counters"):
            self.assertIn(f"'{option}'", runner_text)

    def test_example_board_profile_is_the_image_this_checkout_builds(self) -> None:
        example = json.loads((HERE / "examples" / "candidate-u280.json").read_text(encoding="utf-8"))
        contract = image_profile_contract(REPO, example["board"]["profile"])
        if contract is None:
            self.skipTest("checkout has no profile table or board RTL")
        self.assertEqual(contract["mismatches"], [])
        self.assertEqual(contract["more_specific"], [])

    def test_upstreams_recorded_board_results_summarize_clean(self) -> None:
        recorded = REPO / "spec" / "r39" / "baseline_archbench.jsonl"
        if not recorded.is_file():
            self.skipTest("checkout has no recorded R39 board baseline")
        summary = summarize_archbench(recorded, backend="fpga-u280", expected_clock_hz=100e6, require_counters=True)
        self.assertEqual(summary["status"], "PASS", summary["failures"][:3])
        self.assertEqual(summary["records"], summary["passed"])
        self.assertEqual(set(summary["cycles_by_tier"]), {"tiny", "small"})
        self.assertEqual(summary["cycles_total"], summary["done_cycles_total"])


if __name__ == "__main__":
    unittest.main()
