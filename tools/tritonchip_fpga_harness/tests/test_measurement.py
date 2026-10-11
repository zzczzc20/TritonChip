from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.core import load_profile
from tritonchip_fpga_harness.measurement import measure_board, summarize_archbench


class MeasurementTests(unittest.TestCase):
    def _record(self) -> dict[str, object]:
        return {
            "schema": "archbench-torch-record/1",
            "point_id": "op.add/tiny/fp32-n1",
            "family": "add",
            "tier": "tiny",
            "status": "pass",
            "evidence_level": "measured-board",
            "platform": {"backend": "fpga-u50", "kind": "tnca"},
            "time": {
                "unit": "cycles", "device": 123, "clock_hz": 37_500_000,
                "device_seconds": 123 / 37_500_000,
                "timing": "serial-grid", "launches": 1,
                "per_launch": [{"kernel": "k", "cycles": 123}],
            },
            "counters": {"done_cycles": 123, "done_retired": 9, "perf_cycles": 120,
                         "perf_x_lsu": 3},
            "problems": [],
        }

    def test_summary_requires_board_evidence_and_preserves_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "archbench.jsonl"
            path.write_text(json.dumps(self._record()) + "\n", encoding="utf-8")
            result = summarize_archbench(path, backend="fpga-u50",
                                         expected_clock_hz=37_500_000,
                                         require_counters=True)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cycles_total"], 123)
        self.assertEqual(result["done_cycles_total"], 123)
        self.assertEqual(result["done_retired_total"], 9)
        self.assertEqual(result["counter_totals"]["perf_x_lsu"], 3)

    def test_summary_rejects_rtl_evidence_when_board_was_requested(self) -> None:
        row = self._record()
        row["evidence_level"] = "measured-rtl"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "archbench.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = summarize_archbench(path, backend="fpga-u50",
                                         expected_clock_hz=37_500_000,
                                         require_counters=True)
        self.assertEqual(result["status"], "ARCHBENCH_FAIL")
        self.assertTrue(result["failures"])

    def test_summary_rejects_completion_counter_mismatch(self) -> None:
        row = self._record()
        row["counters"] = {"done_cycles": 124, "done_retired": 9}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "archbench.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = summarize_archbench(path, backend="fpga-u50",
                                         expected_clock_hz=37_500_000,
                                         require_counters=True)
        self.assertEqual(result["status"], "ARCHBENCH_FAIL")
        self.assertIn("done_cycles 124 != time.device 123",
                      result["failures"][0]["problems"])

    def test_board_stage_writes_completion_and_archbench_evidence(self) -> None:
        profile = load_profile(HERE, "u50-legacy")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            (run_dir / "artifacts").mkdir(parents=True)
            (run_dir / "artifacts" / "tnca_u50.xclbin").write_bytes(b"xclbin")
            py = sys.executable
            conf = [py, "-c", (
                "import json,os,pathlib; p=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR'])/'board'/'conformance.json'; "
                "p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps([{'passed':True}]))")]
            bench = [py, "-c", (
                "import json,os,pathlib; p=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR'])/'board'/'archbench.jsonl'; "
                "p.parent.mkdir(parents=True,exist_ok=True); r=" + repr(self._record()) + "; "
                "p.write_text(json.dumps(r)+'\\n')")]
            result = measure_board(
                root=REPO, run_dir=run_dir, profile=profile,
                manifest={"board": {"backend": "fpga-u50", "profile": "core-vrf-sync-v1", "program": False,
                                    "conformance": {"command": conf},
                                    "archbench": {"command": bench, "counters": True}}},
                timeout_s=30,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["archbench"]["cycles_total"], 123)
            self.assertTrue((run_dir / "board_measurement.json").is_file())

    def test_board_stage_programs_then_reads_cycles(self) -> None:
        profile = load_profile(HERE, "u50-legacy")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            (run_dir / "artifacts").mkdir(parents=True)
            (run_dir / "artifacts" / "tnca_u50.xclbin").write_bytes(b"xclbin")
            py = sys.executable
            check = [py, "-c", "import os,pathlib; pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']).joinpath('board','checked').parent.mkdir(parents=True,exist_ok=True); pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']).joinpath('board','checked').write_text('ok')"]
            go = [py, "-c", "import os,pathlib; pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']).joinpath('board','programmed').write_text('ok')"]
            conf = [py, "-c", "import json,os,pathlib; p=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR'])/'board'/'conformance.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps([{'passed':True}]))"]
            bench = [py, "-c", "import json,os,pathlib; p=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR'])/'board'/'archbench.jsonl'; r=" + repr(self._record()) + "; p.write_text(json.dumps(r)+'\\n')"]
            result = measure_board(
                root=REPO, run_dir=run_dir, profile=profile,
                manifest={"board": {"backend": "fpga-u50", "profile": "core-vrf-sync-v1",
                                    "program": {"check_command": check, "command": go},
                                    "conformance": {"command": conf},
                                    "archbench": {"command": bench, "counters": True}}},
                timeout_s=30,
            )
            self.assertTrue((run_dir / "board" / "checked").is_file())
            self.assertTrue((run_dir / "board" / "programmed").is_file())
            self.assertEqual(result["archbench"]["done_cycles_total"], 123)
            self.assertEqual([s["name"] for s in result["stages"]][:2], ["program_check", "program"])


if __name__ == "__main__":
    unittest.main()
