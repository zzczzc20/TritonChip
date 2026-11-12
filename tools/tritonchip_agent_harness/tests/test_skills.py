"""Which skills a run gets: the catalog, the task's default set, a run's override, and what the record says."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import adapters, loop, skills  # noqa: E402
from tritonchip_agent_harness.task import TaskError, load_task, render  # noqa: E402
from tritonchip_agent_harness.tests.test_occupancy import SITE, make_repo, make_task  # noqa: E402


def write_skill(root: Path, name: str, description: str = "What it is for, in one sentence of more than forty chars.",
                extra: str = "") -> Path:
    d = root / name
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n{extra}", encoding="utf-8")
    (d / "scripts" / "tool.py").write_text("print('hi')\n", encoding="utf-8")
    return d


class Catalog(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_skill(self.tmp, "alpha", "Alpha does the first thing, which takes more than forty characters.")
        write_skill(self.tmp, "beta", "Beta does the second thing, which also takes more than forty chars.")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_skill_is_self_describing(self):
        cat = skills.catalog(self.tmp)
        self.assertEqual(list(cat), ["alpha", "beta"])
        self.assertTrue(cat["alpha"].description.startswith("Alpha does"))
        self.assertEqual(cat["alpha"].mounted, f"{adapters.SKILLS_DIR}/alpha")
        self.assertEqual(len(cat["alpha"].sha256), 64)

    def test_the_version_follows_every_byte(self):
        before = skills.version(self.tmp)
        (self.tmp / "beta" / "scripts" / "tool.py").write_text("print('changed')\n")
        after = skills.version(self.tmp)
        self.assertNotEqual(before, after)
        self.assertNotEqual(skills.catalog(self.tmp)["beta"].sha256, skills.catalog(self.tmp)["alpha"].sha256)

    def test_a_misnamed_or_bare_skill_is_refused(self):
        (self.tmp / "gamma").mkdir()
        (self.tmp / "gamma" / "SKILL.md").write_text("---\nname: delta\ndescription: long enough description here\n---\n")
        with self.assertRaisesRegex(skills.SkillError, "names it 'delta'"):
            skills.catalog(self.tmp)
        (self.tmp / "gamma" / "SKILL.md").write_text("# no front matter\n")
        with self.assertRaisesRegex(skills.SkillError, "no front matter"):
            skills.catalog(self.tmp)

    def test_select_keeps_the_order_and_refuses_the_unknown(self):
        self.assertEqual([s.name for s in skills.select(["beta", "alpha"], self.tmp)], ["beta", "alpha"])
        self.assertEqual(skills.select([], self.tmp), [])
        with self.assertRaisesRegex(skills.SkillError, "no such skill: gamma"):
            skills.select(["alpha", "gamma"], self.tmp)
        with self.assertRaisesRegex(skills.SkillError, "named twice"):
            skills.select(["alpha", "alpha"], self.tmp)

    def test_stage_copies_the_selected_and_nothing_else(self):
        dest = self.tmp / "run" / "skills"
        (self.tmp / "alpha" / "scripts" / "__pycache__").mkdir()
        (self.tmp / "alpha" / "scripts" / "__pycache__" / "x.pyc").write_bytes(b"\0")
        staged = skills.stage(skills.select(["alpha"], self.tmp), dest)
        self.assertEqual(sorted(p.name for p in staged.iterdir()), ["alpha"])
        self.assertTrue((staged / "alpha" / "scripts" / "tool.py").is_file())
        self.assertFalse((staged / "alpha" / "scripts" / "__pycache__").exists())
        staged = skills.stage(skills.select(["beta"], self.tmp), dest)        # fresh each time
        self.assertEqual(sorted(p.name for p in staged.iterdir()), ["beta"])

    def test_prompt_lines_come_from_the_front_matter(self):
        text = skills.prompt_lines(skills.select(["alpha"], self.tmp))
        self.assertEqual(text, f"- The `alpha` skill (`{adapters.SKILLS_DIR}/alpha/SKILL.md`): "
                               f"Alpha does the first thing, which takes more than forty characters.\n")


class TheHarnessCatalog(unittest.TestCase):
    def test_the_shipped_skills_load(self):
        cat = skills.catalog()
        self.assertIn("testcase-design", cat)
        for s in cat.values():
            self.assertTrue(len(s.description) > 40, s.name)
            self.assertTrue(skills.NAME.match(s.name), s.name)

    def test_every_task_declares_a_set_the_catalog_has(self):
        for task_dir in sorted((HERE / "tasks").iterdir()):
            task = load_task(task_dir)
            skills.select(task.skills)                                 # raises on an unknown name


class TaskDefault(unittest.TestCase):
    def test_an_unknown_skill_in_a_task_is_refused_at_load(self):
        value = json.loads((HERE / "tasks" / "replay-mzero" / "task.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instruction.md").write_text("Do it.\n")
            path = root / "task.json"
            path.write_text(json.dumps(dict(value, skills=["no-such-skill"])))
            with self.assertRaisesRegex(TaskError, "no such skill"):
                load_task(path)
            path.write_text(json.dumps(dict(value, skills="testcase-design")))
            with self.assertRaisesRegex(TaskError, "list of skill names"):
                load_task(path)
            path.write_text(json.dumps({k: v for k, v in value.items() if k != "skills"}))
            self.assertEqual(load_task(path).skills, ())

    def test_render_names_only_the_selected(self):
        task = load_task(HERE / "tasks" / "replay-mzero")
        self.assertNotIn("/opt/tnca-skills", render(task, "offline"))
        self.assertNotIn("/opt/tnca-skills", render(task, "offline", skills=[]))
        with_skill = render(task, "offline", skills=["testcase-design"])
        self.assertIn(f"`{adapters.SKILLS_DIR}/testcase-design/SKILL.md`", with_skill)
        with self.assertRaises(skills.SkillError):
            render(task, "offline", skills=["no-such-skill"])


class RunRecord(unittest.TestCase):
    """The record names the set a run got and where it came from."""

    def setUp(self):
        self.repo, base = make_repo()
        self.site = types.SimpleNamespace(**vars(SITE), repository=self.repo, state_dir=Path(tempfile.mkdtemp()))
        self.base = base

    def run_task(self, task, **kw) -> dict:
        rows = {"points": [], "n": 0, "passed": 0, "speedup_geomean": None, "fast_1": None}
        with mock.patch.object(loop.evaluator, "evaluate",
                               return_value={"valid": True, "validation": rows, "final": rows}):
            record = loop.run_task(self.site, task, "noop", **kw)
        return json.loads((Path(record["run_dir"]) / "run.json").read_text())

    def test_the_task_default_then_the_run_override(self):
        task = make_task(self.base)
        self.assertEqual(task.skills, ())
        saved = self.run_task(task)
        self.assertEqual((saved["skills"]["selected"], saved["skills"]["source"], saved["skills"]["hashes"]),
                         ([], "task", {}))
        self.assertEqual(saved["skills"]["catalog"], skills.version())
        saved = self.run_task(task, skills=["testcase-design"])
        self.assertEqual((saved["skills"]["selected"], saved["skills"]["source"]), (["testcase-design"], "run"))
        self.assertEqual(saved["skills"]["hashes"], skills.manifest(skills.select(["testcase-design"])))
        self.assertIn("/opt/tnca-skills/testcase-design", (Path(saved["run_id"]) and
                                                           (self.site.state_dir / "runs" / saved["run_id"] / "prompt.md")
                                                           .read_text()))
        saved = self.run_task(task, skills=[])
        self.assertEqual((saved["skills"]["selected"], saved["skills"]["source"]), ([], "run"))

    def test_an_unknown_override_is_refused_before_anything_runs(self):
        task = make_task(self.base)
        with self.assertRaises(skills.SkillError):
            self.run_task(task, skills=["no-such-skill"])

    def test_a_named_run_carries_its_name(self):
        task = make_task(self.base)
        saved = self.run_task(task, name="R41-F2")
        self.assertEqual(saved["name"], "R41-F2")
        self.assertRegex(saved["run_id"], r"^R41-F2-\d{8}T\d{6}Z-[0-9a-f]{4}$")
        self.assertTrue((self.site.state_dir / "runs" / saved["run_id"] / "run.json").is_file())
        unnamed = self.run_task(task)
        self.assertIsNone(unnamed["name"])
        self.assertTrue(unnamed["run_id"].startswith(f"{task.task_id}-noop-"))

    def test_a_name_that_is_no_safe_id_is_refused_before_anything_runs(self):
        task = make_task(self.base)
        for bad in ("", "-x", "R41 F2", "a/b", "x" * 41):
            with self.assertRaises(ValueError, msg=bad):
                self.run_task(task, name=bad)
        self.assertFalse((self.site.state_dir / "runs").exists() and any((self.site.state_dir / "runs").iterdir()))


if __name__ == "__main__":
    unittest.main()
