from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.core import HarnessError, Profile, doctor, export_evidence, load_profile, plan, reconcile, status, submit, validate_profile


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[3]
        self.state = Path(tempfile.mkdtemp(prefix="tritonchip-harness-test-"))

    def manifest(self, mode: str = "full") -> dict[str, object]:
        return {"candidate_id": "test-candidate", "target": "u50-legacy", "mode": mode, "clock_mhz": 150}

    def test_mock_lifecycle_and_export(self) -> None:
        value = plan(HERE, self.root, self.manifest(), "mock")
        self.assertEqual(value["clock_mhz"], 150)
        result = submit(HERE, self.root, self.manifest(), self.state, "mock", allow_dirty=True)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(set(result["artifacts"]), {"artifacts/tnca_u50.xo", "artifacts/tnca_u50.xclbin", "artifacts/timing_summary.json"})
        current = status(self.state, result["run_id"])
        self.assertEqual(current["status"], "PASS")
        evidence = export_evidence(self.state, result["run_id"], self.state / "evidence")
        self.assertTrue((evidence / "result.json").is_file())
        self.assertTrue((evidence / "environment.json").is_file())
        self.assertTrue((evidence / "evidence-manifest.json").is_file())
        self.assertTrue((evidence / "artifacts" / "tnca_u50.xclbin").is_file())

    def test_mock_ooc_does_not_fabricate_xclbin(self) -> None:
        result = submit(HERE, self.root, self.manifest("ooc"), self.state, "mock", allow_dirty=True)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(set(result["artifacts"]), {"artifacts/tnca_u50.xo"})
        self.assertNotIn("artifacts/tnca_u50.xclbin", result["artifacts"])

    def test_invalid_clock_is_rejected(self) -> None:
        value = self.manifest()
        value["clock_mhz"] = 37.5
        with self.assertRaises(ValueError):
            plan(HERE, self.root, value, "mock")

    def test_doctor_has_structured_checks(self) -> None:
        result = doctor(HERE, self.root, "u50-legacy", "mock")
        self.assertTrue(result["ok"])
        self.assertEqual(result["checks"][0]["name"], "mock-adapter")

    def test_pr5_profile_records_kernel_and_device_contract(self) -> None:
        profile = load_profile(HERE, "u50-r38-pr5")
        self.assertEqual(profile.expected_clock, 100)
        self.assertEqual(profile.expected_device_divider, 1)
        self.assertEqual(profile.data["device_clock_mhz"], 100)

    def test_u280_mock_lifecycle_uses_u280_artifact_name(self) -> None:
        value = {
            "candidate_id": "pr5-u280-mock",
            "target": "u280-r38-pr5",
            "mode": "full",
            "clock_mhz": 100,
        }
        result = submit(HERE, self.root, value, self.state, "mock", allow_dirty=True)
        self.assertEqual(result["status"], "PASS")
        self.assertIn("artifacts/tnca_u280.xclbin", result["artifacts"])

    def test_vendor_adapter_must_match_target(self) -> None:
        with self.assertRaises(ValueError):
            plan(HERE, self.root, {
                "candidate_id": "bad-target-adapter",
                "target": "u280-r38-pr5",
                "mode": "full",
                "clock_mhz": 100,
            }, "u50")

    def test_profile_clock_contract_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            validate_profile(Profile("bad", {
                "board": "u50", "part": "part", "platform": "platform",
                "kernel_clock_mhz": 100, "device_clock_mhz": 37.5,
                "full_slots": 1, "ooc_slots": 2,
                "expected_artifacts": ["a.xo", "a.xclbin"],
            }))

    def test_reconcile_marks_dead_controller_orphaned(self) -> None:
        run_dir = self.state / "runs" / "orphan"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "orphan", "status": "RUNNING",
            "controller": {"pid": 999999, "pid_starttime": "1", "boot_id": "wrong"},
        }), encoding="utf-8")
        value = reconcile(self.state, "orphan")
        self.assertEqual(value["status"], "ORPHANED")
        self.assertTrue(status(self.state, "orphan")["recovery_required"])

    def test_user_configured_ooc_runs_without_profile_slots(self) -> None:
        command = [
            sys.executable, "-c",
            "import json,os,pathlib; p=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']); "
            "(p/'artifacts').mkdir(); (p/'artifacts'/'fma_ooc.dcp').write_bytes(b'dcp'); "
            "(p/'ooc_result.json').write_text(json.dumps({'status':'PASS'}))",
        ]
        manifest = {
            "candidate_id": "fma-ooc",
            "target": "u50-legacy",
            "mode": "ooc",
            "clock_mhz": 150,
            "ooc": {"top": "u_mac", "command": command, "artifacts": ["artifacts/fma_ooc.dcp"]},
        }
        result = submit(HERE, self.root, manifest, self.state, "u50", allow_dirty=True)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["stages"][0]["name"], "ooc")


if __name__ == "__main__":
    unittest.main()
