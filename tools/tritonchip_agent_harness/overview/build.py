"""Build the shareable overview page from the README and the branch's history.

    python3 tools/tritonchip_agent_harness/overview/build.py OUT.html [--base origin/main] [--remote origin/fpga-harness]

The README is the one document that is kept up to date. This page is a view of
it: the decisions (section 16), the progress table (section 17) and pilot 1
(section 19) are read from the README's own tables, and the commits from git,
so the page never says anything the README and the history do not. The
diagrams and the role and boundary text in template.html mirror README
sections 3, 5, 6.4 and 12.3; change them together.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE.parent / "README.md"
TEMPLATE = HERE / "template.html"


def section(text: str, number: int) -> str:
    m = re.search(rf"^## {number}\. .*?$(.*?)(?=^## \d+\. |\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"README has no section {number}")
    return m.group(1)


def tables(block: str) -> list:
    """Every Markdown table in a block, as a list of row dicts keyed by header."""
    out, lines = [], block.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            head = [c.strip() for c in lines[i].strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                rows.append(dict(zip(head, cells)))
                i += 1
            out.append(rows)
        else:
            i += 1
    return out


def inline(md: str) -> str:
    """The little Markdown the tables use: `code`, **bold**, and arrows. Escaped first."""
    s = html.escape(md, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=HERE).stdout


def commits(base: str, remote: str) -> list:
    pushed = set(git("rev-list", f"{base}..{remote}").split()) if remote else set()
    rows = []
    for line in git("log", "--format=%H%x1f%h%x1f%ad%x1f%s", "--date=short", f"{base}..HEAD").splitlines():
        full, short, date, subject = line.split("\x1f")
        rows.append({"hash": short, "date": date, "subject": inline(subject), "pushed": full in pushed})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--remote", default="origin/fpga-harness")
    a = ap.parse_args(argv)
    text = README.read_text(encoding="utf-8")

    decided, open_ = tables(section(text, 16))[:2]
    progress = tables(section(text, 17))[0]
    pilot = [t for t in tables(section(text, 19)) if t and "Configuration" in t[0]]
    status_line = re.search(r"^Status: (.*?)$", section(text, 17), re.M)

    data = {
        "decided": [{"date": r["Date"], "text": inline(r["Decision"])} for r in decided],
        "open": [{"id": r["#"], "text": inline(r["Decision"]), "owner": inline(r["Owner"])} for r in open_],
        "progress": [{"area": r["Area"], "item": inline(r["Item"]), "status": r["Status"],
                      "evidence": inline(r["Evidence"])} for r in progress],
        "pilot": [{k: inline(v) for k, v in r.items()} for r in (pilot[0] if pilot else [])],
        "commits": commits(a.base, a.remote),
        "head": git("rev-parse", "--short", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain", "--", str(HERE.parent)).strip()),
    }
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*@DATA@*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    Path(a.out).write_text(page, encoding="utf-8")
    print(f"{a.out}: {len(data['progress'])} progress items, {len(data['decided'])} decided, "
          f"{len(data['open'])} open, {len(data['commits'])} commits", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
