"""Export and gate, on a throwaway repository shaped like the real one."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.workspace import WorkspaceError, apply_candidate, export, gate, git  # noqa: E402

FILES = {
    "compiler/a.py": "a = 1\n", "compiler/c.py": "c = 3\n", "tb/t.v": "module t; endmodule\n",
    "doc/r39_report.md": "the answer\n", "doc/other.md": "fine\n", "CLAUDE.md": "do this\n",
    "compiler/AGENTS.md": "and this\n", "evidence/e.json": "{}\n",
    "tools/tritonchip_agent_harness/x.py": "harness\n", ".gitignore": "build/\n",
    "compiler/device.py": "measures\n",
}
DESIGN = ["compiler/"]
HIDE = ["doc/r39_report.md"]


def make_repo() -> tuple:
    root = Path(tempfile.mkdtemp())
    for path, text in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git("init", "-q", str(root))
    git("-C", str(root), "add", "-A")
    git("-C", str(root), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    return root, git("-C", str(root), "rev-parse", "HEAD").decode().strip()


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.repo, self.base = make_repo()
        self.ws = Path(tempfile.mkdtemp()) / "ws"
        export(self.repo, self.base, HIDE, self.ws)

    def gate(self, out=None):
        return gate(self.repo, self.base, HIDE, DESIGN, self.ws, out)

    def test_export_hides_and_has_one_commit(self):
        for gone in ("doc/r39_report.md", "CLAUDE.md", "compiler/AGENTS.md", "evidence/e.json",
                     "tools/tritonchip_agent_harness/x.py"):
            self.assertFalse((self.ws / gone).exists(), gone)
        self.assertTrue((self.ws / "doc/other.md").exists())
        self.assertEqual(git("-C", str(self.ws), "rev-list", "--count", "--all").decode().strip(), "1")
        self.assertEqual(git("-C", str(self.ws), "remote").decode().strip(), "")
        again = Path(tempfile.mkdtemp()) / "ws"
        a = export(self.repo, self.base, HIDE, again)
        self.assertEqual(a["workspace_commit"], git("-C", str(self.ws), "rev-parse", "HEAD").decode().strip())

    def test_the_export_keeps_no_reflog(self):
        self.assertFalse((self.ws / ".git" / "logs").exists())

    def test_a_hide_entry_that_hides_nothing_is_refused(self):
        for entry, hint in (("doc/r39_reprot.md", ""), ("evidence", "trailing slash")):
            with self.assertRaises(WorkspaceError) as e:
                export(self.repo, self.base, [entry], Path(tempfile.mkdtemp()) / "ws")
            self.assertIn(hint, str(e.exception))

    def test_untouched_is_an_empty_pass(self):
        r = self.gate(Path(tempfile.mkdtemp()) / "not-yet-created")
        self.assertEqual((r["verdict"], r["taken"], r["dropped"]), ("PASS", [], []))

    def test_design_changes_are_taken_others_dropped(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        (self.ws / "compiler/b.py").write_text("b = 2\n")
        (self.ws / "compiler/c.py").unlink()
        (self.ws / "tb/t.v").write_text("module t; // changed\nendmodule\n")
        (self.ws / "build").mkdir()
        (self.ws / "build/out.o").write_text("x")
        r = self.gate()
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual({(c["path"], c["status"]) for c in r["taken"]},
                         {("compiler/a.py", "modified"), ("compiler/b.py", "added"), ("compiler/c.py", "deleted")})
        self.assertEqual([c["path"] for c in r["dropped"]], ["tb/t.v"])

    def test_evaluator_owned_files_are_dropped_even_under_a_design_path(self):
        (self.ws / "compiler/device.py").write_text("measures differently\n")
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        r = gate(self.repo, self.base, HIDE, DESIGN, self.ws, evaluator_owned=["compiler/device.py"])
        self.assertEqual([c["path"] for c in r["taken"]], ["compiler/a.py"])
        self.assertEqual([(c["path"], c["reason"]) for c in r["dropped"]], [("compiler/device.py", "evaluator-owned")])

    def test_a_symlink_in_the_design_rejects(self):
        os.symlink("/etc/passwd", self.ws / "compiler/link.py")
        self.assertEqual(self.gate()["verdict"], "REJECT")

    def test_rewriting_the_workspace_history_hides_nothing(self):
        (self.ws / "compiler/a.py").write_text("a = 9\n")
        git("-C", str(self.ws), "add", "-A")
        git("-C", str(self.ws), "-c", "commit.gpgsign=false", "commit", "-q", "--amend", "-m", "base")
        self.assertEqual([c["path"] for c in self.gate()["taken"]], ["compiler/a.py"])

    def test_the_evaluator_keeps_its_own_testbench(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        (self.ws / "tb/t.v").write_text("module t; // tampered\nendmodule\n")
        cand = Path(tempfile.mkdtemp())
        self.assertEqual(self.gate(cand)["verdict"], "PASS")
        tree = Path(tempfile.mkdtemp()) / "eval"
        apply_candidate(self.repo, self.base, cand, tree)
        self.assertEqual((tree / "compiler/a.py").read_text(), "a = 2\n")
        self.assertEqual((tree / "tb/t.v").read_text(), FILES["tb/t.v"])
        self.assertEqual((tree / "doc/r39_report.md").read_text(), FILES["doc/r39_report.md"],
                         "hidden from the agent, but the evaluator builds from the full base")


if __name__ == "__main__":
    unittest.main()
