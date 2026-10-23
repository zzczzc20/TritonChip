"""Tests for tools/docmaint with a scripted fake backend (no real agent).

    python -m pytest tools/docmaint/test_docmaint.py      (or: python -m unittest)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.docmaint import graph, maintain
from tools.docmaint.backends import Backend


def sh(cwd: Path, *args: str) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


class Scripted(Backend):
    """Runs a Python snippet in the scratch copy instead of an agent."""
    name = "scripted"

    def __init__(self, scripts: dict[str, str], rc: int = 0):
        self.scripts, self.rc, self.calls = scripts, rc, []

    def available(self) -> bool:
        return True

    def command(self, prompt, workdir, timeout):
        doc = prompt.split("TARGET DOCUMENT: ", 1)[1].split("\n", 1)[0]
        self.calls.append(doc)
        code = self.scripts.get(doc, "") + f"\nimport sys; sys.exit({self.rc})"
        return [sys.executable, "-c", code]


MANIFEST = """
[settings]
backend = "agy"

[[doc]]
path = "doc/a.md"
sources = ["src/"]

[[doc]]
path = "doc/b.md"
depends_on = ["doc/a.md"]
sources = ["tools/x.py"]
"""


class Repo:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.cache = Path(self.tmp.name) / "cache"
        r = self.root
        r.mkdir()
        (r / "src").mkdir(); (r / "doc").mkdir(); (r / "tools").mkdir()
        (r / "src/core.v").write_text("module core; endmodule\n")
        (r / "tools/x.py").write_text("X = 1\n")
        (r / "doc/a.md").write_text("# A\nold\n")
        (r / "doc/b.md").write_text("# B\nsee [a](a.md)\n")
        (r / "doc/docmap.toml").write_text(MANIFEST)
        sh(r, "git", "init", "-q")
        sh(r, "git", "add", "-A")
        sh(r, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")

    def commit(self, msg="c"):
        sh(self.root, "git", "add", "-A")
        sh(self.root, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)

    def manifest(self):
        return graph.load_manifest(self.root)

    def opts(self, **kw):
        return maintain.RunOptions(cache=self.cache, **kw)


class GraphTests(unittest.TestCase):
    def test_order_and_cycle(self):
        m = graph.Manifest(docs={
            "c.md": graph.Doc("c.md", depends_on=["b.md"]),
            "b.md": graph.Doc("b.md", depends_on=["a.md"]),
            "a.md": graph.Doc("a.md")})
        self.assertEqual(graph.topo_order(m), ["a.md", "b.md", "c.md"])
        m.docs["a.md"].depends_on = ["c.md"]
        with self.assertRaisesRegex(graph.GraphError, "cycle"):
            graph.topo_order(m)

    def test_non_markdown_and_escape_refused(self):
        rp = Repo()
        (rp.root / "doc/docmap.toml").write_text('[[doc]]\npath = "tools/x.py"\n')
        with self.assertRaisesRegex(graph.GraphError, "Markdown"):
            rp.manifest()
        (rp.root / "doc/docmap.toml").write_text('[[doc]]\npath = "../x.md"\n')
        with self.assertRaisesRegex(graph.GraphError, "inside"):
            rp.manifest()

    def test_sources(self):
        files = ["src/a.v", "src/sub/b.v", "srcx/c.v", "doc/x.md"]
        self.assertEqual(graph.expand_sources(["src/"], files), ["src/a.v", "src/sub/b.v"])
        self.assertEqual(graph.expand_sources(["*.md", "doc/*.md"], files), ["doc/x.md"])

    def test_discover_is_valid(self):
        rp = Repo()
        text = graph.discover(rp.root)
        (rp.root / "doc/docmap.toml").write_text(text)
        m = rp.manifest()
        self.assertEqual(m.docs["doc/b.md"].depends_on, ["doc/a.md"])


class RunTests(unittest.TestCase):
    def test_update_then_fresh(self):
        rp = Repo()
        be = Scripted({"doc/a.md": "open('doc/a.md','w').write('# A\\nnew\\n')"})
        res = maintain.run(rp.root, rp.manifest(), be, rp.opts())
        self.assertEqual([(r.path, r.status) for r in res],
                         [("doc/a.md", "updated"), ("doc/b.md", "unchanged")])
        self.assertEqual((rp.root / "doc/a.md").read_text(), "# A\nnew\n")
        self.assertEqual(be.calls, ["doc/a.md", "doc/b.md"])
        rp.commit()
        res = maintain.run(rp.root, rp.manifest(), None, rp.opts(dry_run=True))
        self.assertEqual({r.status for r in res}, {"fresh"})

    def test_code_edit_rejected_and_repo_untouched(self):
        rp = Repo()
        be = Scripted({"doc/a.md": "open('doc/a.md','w').write('# A\\nx\\n'); open('src/core.v','w').write('hacked')"})
        res = maintain.run(rp.root, rp.manifest(), be, rp.opts())
        self.assertEqual(res[0].status, "rejected")
        self.assertEqual(res[0].violations, ["src/core.v"])
        self.assertEqual((rp.root / "src/core.v").read_text(), "module core; endmodule\n")
        self.assertEqual((rp.root / "doc/a.md").read_text(), "# A\nold\n")

    def test_new_file_and_commit_rejected(self):
        rp = Repo()
        be = Scripted({
            "doc/a.md": "open('notes.md','w').write('x')",
            "doc/b.md": textwrap.dedent("""
                import subprocess
                open('doc/b.md','a').write('more\\n')
                subprocess.run(['git','-c','user.name=a','-c','user.email=a@a','commit','-qam','x'])
            """)})
        res = maintain.run(rp.root, rp.manifest(), be, rp.opts())
        self.assertEqual([r.status for r in res], ["rejected", "rejected"])
        self.assertIn("notes.md", res[0].violations)
        self.assertFalse((rp.root / "notes.md").exists())
        self.assertEqual((rp.root / "doc/b.md").read_text(), "# B\nsee [a](a.md)\n")

    def test_staleness_propagates(self):
        rp = Repo()
        maintain.baseline(rp.root, rp.manifest())
        rp.commit("baseline")
        (rp.root / "src/core.v").write_text("module core2; endmodule\n")
        rp.commit("rtl change")
        plan = {r.path: r.status for r in maintain.run(rp.root, rp.manifest(), None, rp.opts(dry_run=True))}
        self.assertEqual(plan, {"doc/a.md": "planned", "doc/b.md": "fresh"})
        be = Scripted({"doc/a.md": "open('doc/a.md','a').write('core2\\n')"})
        res = maintain.run(rp.root, rp.manifest(), be, rp.opts())
        # a changed in this run, so b (which depends on a) is maintained too
        self.assertEqual([(r.path, r.status) for r in res],
                         [("doc/a.md", "updated"), ("doc/b.md", "unchanged")])
        prompt = next(rp.cache.rglob("doc__a.md.prompt.txt")).read_text()
        self.assertIn("src/core.v", prompt)
        self.assertIn("professional technical English", prompt)

    def test_local_edits_never_overwritten(self):
        rp = Repo()
        (rp.root / "doc/a.md").write_text("# A\nmine\n")
        be = Scripted({"doc/a.md": "open('doc/a.md','w').write('# A\\ntheirs\\n')"})
        res = maintain.run(rp.root, rp.manifest(), be, rp.opts())
        self.assertEqual(res[0].status, "skipped")
        self.assertEqual((rp.root / "doc/a.md").read_text(), "# A\nmine\n")

    def test_backend_failure_discards(self):
        rp = Repo()
        be = Scripted({"doc/a.md": "open('doc/a.md','w').write('# A\\nhalf\\n')"}, rc=3)
        res = maintain.run(rp.root, rp.manifest(), be, rp.opts(only=["doc/a.md"]))
        self.assertEqual(res[0].status, "failed")
        self.assertEqual((rp.root / "doc/a.md").read_text(), "# A\nold\n")

    def test_commit_option(self):
        rp = Repo()
        be = Scripted({"doc/a.md": "open('doc/a.md','w').write('# A\\nnew\\n')"})
        maintain.run(rp.root, rp.manifest(), be, rp.opts(commit=True))
        self.assertEqual(sh(rp.root, "git", "status", "--porcelain"), "")
        self.assertIn("docmaint", sh(rp.root, "git", "log", "-1", "--format=%s"))


if __name__ == "__main__":
    unittest.main()
