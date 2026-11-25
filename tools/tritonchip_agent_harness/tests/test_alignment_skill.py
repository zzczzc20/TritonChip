"""The cycle-model-alignment skill (ported to R41 for R41-F3, 2026-11-10): an agent skill in the catalog, its
script the R41 board-top bench's, and what it points the agent at visible and free of every held-out point."""
from __future__ import annotations

import ast
import json
import runpy
import sys
import tempfile
import types
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


SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class CoverYourChange(unittest.TestCase):
    """2026-11-23 (user: "按设计做"): the agent finds what its change moves (touched.py), what a case reaches
    (mechanisms.py) and the smallest case that still reaches it (shrink_case.py), on the CA model's counters."""

    def test_the_mechanisms_are_calibration_json_s(self):
        import mechanisms as M
        from tritonchip_agent_harness import dev_cases
        self.assertEqual({k: M.RULES[k][0] for k in dev_cases.MECHANISMS}, dev_cases.MECHANISMS)
        self.assertEqual(set(M.RULES) - set(dev_cases.MECHANISMS),
                         {"barriers", "memory port queueing", "home engine queueing"})
        given = json.loads((HERE / "tasks" / "r41-dp-dir" / "dev_cases" / "CALIBRATION.json").read_text())
        self.assertLessEqual(set(given["mechanisms"]), set(M.RULES))

    def test_a_mechanism_is_reached_when_its_counters_are_not_zero(self):
        import mechanisms as M
        c = {"core_0_instructions": 5, "core_1_instructions": 3, "core_2_instructions": 0, "core_0_stall_mtx": 2,
             "core_1_mma_commands": 0, "uncore_writebacks": 4, "chip_grid_launches": 1, "chip_single_launches": 0,
             "home_0_fill_queue_cycles": 7, "uncore_hits": 0}
        self.assertEqual(M.reached(c), {"matrix engine waits": 2, "dirty writebacks": 4, "home engine queueing": 7})
        c.update(core_2_instructions=1, chip_single_launches=2)
        self.assertEqual(M.reached(c)["all three cores"], 1)
        self.assertEqual(M.reached(c)["several launches"], 2)

    def test_matrix_instructions_are_counted_from_the_compiled_code(self):
        import mechanisms as M
        with tempfile.TemporaryDirectory() as tmp:
            k = Path(tmp) / "triton_tem_fused_mm_0"
            k.mkdir()
            (k / "kernel.s").write_text("  m.mma.dp.async.k16.e4m3 a0, v1, v2\n  @p0 m.mma.async.f16 a1, v3, v4\n"
                                        "  m.mma.dp.async.k16.e4m3 a0, v5, v6\n  v.add v1, v2, v3\n")
            self.assertEqual(M.static_matrix(Path(tmp)), {"m.mma.dp.async.k16.e4m3": 2, "m.mma.async.f16": 1})

    def test_a_case_is_built_at_another_size_by_its_keyword(self):
        import mechanisms as M
        dev = types.SimpleNamespace(_point_make=lambda seed, k=64, n=32: ("dev", seed, k, n), make=lambda: ("made",))
        build, defaults = M.sized(dev, {"k": 16})
        self.assertEqual((build(), defaults), (("dev", 0, 16, 32), {"k": 64, "n": 32}))
        # Without dims it is the case's own make(): a case shrink_case.py wrote redefines it (found on Myserver).
        build, defaults = M.sized(dev, {})
        self.assertEqual((build(), defaults), (("made",), {"k": 64, "n": 32}))
        own = types.SimpleNamespace(make=lambda n=128: ("own", n))
        self.assertEqual(M.sized(own, {"n": 8})[0](), ("own", 8))
        with self.assertRaises(SystemExit):
            M.sized(own, {"k": 8})

    def test_the_smallest_size_is_found_by_bisection(self):
        import shrink_case as S
        sizes = list(range(8, 129, 8))
        seen = []
        size, probes = S.bisect(sizes, lambda s: seen.append(s) or s >= 40)
        self.assertEqual(size, 40)
        self.assertLessEqual(len(probes), 5)
        self.assertEqual([p for p, _ in probes], seen)
        self.assertEqual([S.default_step(n) for n in (512, 1000, 96, 7)], [64, 8, 32, 1])

    def test_the_case_kept_is_the_cheapest_that_reaches(self):
        """Measured on Myserver: a copy at n=32768 took 6140 model cycles, the smallest size reaching dirty
        writebacks (n=30784) 14777: a size that is not a power of two compiled to a slower program."""
        import shrink_case as S
        runs = {32768: {"status": "pass", "mechanisms": {"dirty writebacks": 1}, "device_cycles": 6140},
                30784: {"status": "pass", "mechanisms": {"dirty writebacks": 1}, "device_cycles": 14777},
                16384: {"status": "pass", "mechanisms": {}, "device_cycles": 3374},
                24576: {"status": "unsupported", "mechanisms": {"dirty writebacks": 1}, "device_cycles": 10}}
        self.assertEqual(S.cheapest(runs, {"dirty writebacks"}), 32768)

    def test_the_smaller_case_is_the_original_with_make_redefined(self):
        import shrink_case as S
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "dev.py", Path(tmp) / "dev_k16.py"
            src.write_text('POINT = "op.x/tiny/fp16-k64"\nDEMAND = 1\n\n\ndef _point_make(seed, k=64, n=32):\n'
                           '    return ("dev", seed, k, n)\n\n\ndef make():\n    return _point_make(0)\n')
            S.write_case(src, out, "k", 16, 64, True, ["a note"])
            ns = runpy.run_path(str(out))
            self.assertEqual((ns["make"](), ns["POINT"], ns["DEMAND"]), (("dev", 0, 16, 32), None, None))
            own, small = Path(tmp) / "own.py", Path(tmp) / "own_n32.py"
            own.write_text("def make(n=128):\n    return ('own', n)\n")
            S.write_case(own, small, "n", 32, 128, False, [])
            self.assertEqual(runpy.run_path(str(small))["make"](), ("own", 32))

    def test_touched_lists_what_moved_first(self):
        import touched as T
        base = {"case": "c.py", "status": "pass", "device_cycles": 100, "static": {"m.mma.async.f16": 2},
                "mechanisms": {"a": 10, "b": 5}, "counters": {"x": 1, "y": 2}}
        mine = {"case": "c.py", "status": "pass", "device_cycles": 50, "static": {},
                "mechanisms": {"a": 10, "b": 10, "c": 3}, "counters": {"x": 1, "y": 4, "z": 1}}
        d = T.compare(base, mine)
        self.assertEqual([m["mechanism"] for m in d["mechanisms"]], ["c", "b", "a"])
        self.assertEqual([m["change"] for m in d["mechanisms"]], ["new", "+100.0%", "+0.0%"])
        self.assertEqual(d["touched"], ["c", "b"])
        self.assertEqual([r["counter"] for r in d["counters_moved"]], ["z", "y"])
        self.assertEqual(d["cycles"]["speedup_on_model"], 2.0)
        self.assertIn("matrix instructions (compiled code): base", T.show(d))

    def test_the_scripts_are_named_in_the_skill(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        for script in ("touched.py", "mechanisms.py", "shrink_case.py"):
            self.assertIn(f"/opt/tnca-skills/{NAME}/scripts/{script}", text)
            ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
