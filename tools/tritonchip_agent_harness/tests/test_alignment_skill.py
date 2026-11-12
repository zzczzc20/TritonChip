"""The cycle-model-alignment skill (ported to R41 for R41-F3, 2026-11-10): an agent skill in the catalog, its
script the R41 board-top bench's, and what it points the agent at visible and free of every held-out point."""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import skills  # noqa: E402
from tritonchip_agent_harness.task import load_task, render  # noqa: E402
from tritonchip_agent_harness.workspace import hidden as _hidden  # noqa: E402

NAME = "cycle-model-alignment"
SKILL = HERE / "skills" / NAME
ALIGN = ROOT / "tools" / "performance_harness" / "align"


class AlignmentSkill(unittest.TestCase):
    def test_it_is_an_agent_skill_of_the_catalog(self):
        self.assertIn(NAME, skills.catalog())
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith(f"---\nname: {NAME}\ndescription: "))
        ast.parse((SKILL / "scripts" / "align_case.py").read_text(encoding="utf-8"))

    def test_the_script_records_the_described_board_bench_under_the_evaluators_environment(self):
        src = (SKILL / "scripts" / "align_case.py").read_text(encoding="utf-8")
        for needle in ("build_described", "'fpga-multicore-sim'", "configs/r41-board-cycle.cfg",
                       "spec/devices/u280-3core-r41.json"):
            self.assertIn(needle, src)
        for gone in ("core-vrf-sync-v5'", "TNCA_GEMM_TILE", "r40-sim-cycle.cfg"):
            self.assertNotIn(gone, src)

    def test_the_formal_task_gives_it_and_shows_the_gate(self):
        task = load_task(HERE / "tasks" / "r41-dp-formal")
        self.assertIn(NAME, task.skills)
        self.assertIn(f"/opt/tnca-skills/{NAME}/SKILL.md", render(task, "offline", skills=task.skills))
        for path in ("tools/performance_harness/align/README.md", "tools/performance_harness/align/tracediff.py",
                     "tools/performance_harness/systemc/configs/r41-board-cycle.cfg"):
            self.assertFalse(_hidden(path, task.hide), path)
        self.assertTrue(_hidden("tools/performance_harness/fp8_cases.py", task.hide))

    def test_no_held_out_or_scored_point_is_named_in_what_it_shows(self):
        task = load_task(HERE / "tasks" / "r41-dp-formal")
        ev = task.evaluation
        files = [p for p in list(ALIGN.rglob("*")) + list(SKILL.rglob("*")) if p.is_file()]
        self.assertTrue(files)
        text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
        for pid in ev["final"] + ev.get("board_final", []) + ev["validation"] + ev.get("board_validation", []):
            self.assertNotIn(pid, text, pid)
            params = pid.rsplit("/", 1)[1].split("-", 1)[-1]
            self.assertNotIn(params, text, pid)


if __name__ == "__main__":
    unittest.main()
