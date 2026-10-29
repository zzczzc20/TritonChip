"""The workspace an agent gets, and the gate its changes pass through.

export: the task's base commit, file by file, as a repository of ONE commit --
no history, no remote, no tags -- without agent instruction files, without the
harnesses, and without whatever the task hides. Equal trees give equal commit
hashes (fixed author, date and message).

gate: what the agent changed, judged against the base commit read from the
SOURCE repository. The workspace's own .git is the agent's and is never
trusted. A change under one of the task's design paths goes into the
candidate; a change anywhere else is not the candidate's business and is
recorded and dropped (the evaluator uses its own copy of those files). A
symbolic link or special file under a design path rejects the candidate: it
could point the evaluator's build at something outside the tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

# Never in a workspace, at any depth: what a CLI would read as instructions.
INSTRUCTION_NAMES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", ".claude", ".codex", ".cursor", ".gemini",
                     ".opencode", ".mcp.json", "opencode.json")
# Never in a workspace: the harnesses and the developers' own measurements.
ALWAYS_HIDDEN = ("tools/tritonchip_agent_harness/", "tools/tritonchip_fpga_harness/", "evidence/")

_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "LC_ALL": "C",
        "GIT_AUTHOR_NAME": "tritonchip-agent-harness", "GIT_AUTHOR_EMAIL": "harness@invalid",
        "GIT_COMMITTER_NAME": "tritonchip-agent-harness", "GIT_COMMITTER_EMAIL": "harness@invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"}


class WorkspaceError(RuntimeError):
    pass


def git(*args: str, cwd: Optional[Path] = None, input_bytes: Optional[bytes] = None) -> bytes:
    env = {**os.environ, **_ENV}
    proc = subprocess.run(["git", *args], cwd=cwd, input=input_bytes, capture_output=True, env=env)
    if proc.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args[:3])} failed: {proc.stderr.decode(errors='replace').strip()[:300]}")
    return proc.stdout


def under(path: str, entry: str) -> bool:
    return path.startswith(entry) if entry.endswith("/") else path == entry


def hidden(path: str, hide: Iterable[str]) -> bool:
    parts = path.split("/")
    if any(p in INSTRUCTION_NAMES for p in parts):
        return True
    return any(under(path, e) for e in (*ALWAYS_HIDDEN, *hide))


def base_tree(repo: Path, commit: str) -> dict:
    """path -> (mode, blob id) for every file in the commit."""
    out = git("-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit)
    tree = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, oid = meta.decode().split(" ")
        if kind != "blob":
            raise WorkspaceError(f"{path.decode()}: a {kind} entry is not supported")
        tree[os.fsdecode(path)] = (mode, oid)
    return tree


def visible_tree(repo: Path, commit: str, hide: Iterable[str]) -> dict:
    """The base tree without what the task hides. A hide entry that hides nothing is an error.

    A typo, or a directory written without its trailing slash, would otherwise
    leave the very thing it names in the workspace without a word.
    """
    hide = tuple(hide)
    tree = base_tree(repo, commit)
    for entry in hide:
        if not any(under(p, entry) for p in tree):
            hint = " (a directory needs a trailing slash)" if any(p.startswith(entry + "/") for p in tree) else ""
            raise WorkspaceError(f"hide entry {entry!r} matches nothing in {commit[:12]}{hint}")
    return {p: e for p, e in tree.items() if not hidden(p, hide)}


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def export(repo: Path, commit: str, hide: Iterable[str], dest: Path) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise WorkspaceError(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    tree = visible_tree(repo, commit, hide)
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, env={**os.environ, **_ENV})
    try:
        for path, (mode, oid) in sorted(tree.items()):
            proc.stdin.write(oid.encode() + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "120000":
                os.symlink(os.fsdecode(data), target)
            else:
                target.write_bytes(data)
                os.chmod(target, 0o755 if mode == "100755" else 0o644)
    finally:
        proc.stdin.close()
        proc.wait()
        proc.stdout.close()
    git("init", "-q", "--template=", str(dest))
    git("-C", str(dest), "symbolic-ref", "HEAD", "refs/heads/base")
    git("-C", str(dest), "add", "-A", "-f")
    git("-C", str(dest), "-c", "commit.gpgsign=false", "-c", "core.logAllRefUpdates=false",
        "commit", "-q", "--no-verify", "-m", "base")
    if base_tree(dest, "HEAD") != tree:
        raise WorkspaceError("the exported commit does not contain exactly the visible base tree")
    return {"base_commit": commit, "files": len(tree),
            "workspace_commit": git("-C", str(dest), "rev-parse", "HEAD").decode().strip()}


def _untracked(workspace: Path) -> list:
    """Files git would show as new, using the base tree's ignore rules but not the workspace's .git."""
    with tempfile.TemporaryDirectory() as d:
        git("init", "-q", "--template=", d)
        out = git("-c", "core.excludesFile=" + os.devnull, f"--git-dir={d}/.git", f"--work-tree={workspace}",
                  "ls-files", "--others", "--exclude-standard", "-z")
    return [os.fsdecode(p) for p in out.split(b"\0") if p]


def _entry(path: Path) -> tuple:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return ("120000", blob_id(os.fsencode(os.readlink(path))))
    if not stat.S_ISREG(st.st_mode):
        return ("special", "")
    return ("100755" if st.st_mode & stat.S_IXUSR else "100644", blob_id(path.read_bytes()))


def gate(repo: Path, commit: str, hide: Iterable[str], design: Iterable[str], workspace: Path,
         out: Optional[Path] = None, evaluator_owned: Iterable[str] = ()) -> dict:
    design, owned = tuple(design), tuple(evaluator_owned)
    base = visible_tree(repo, commit, hide)
    changes = []
    for path, entry in sorted(base.items()):
        full = workspace / path
        try:
            now = _entry(full)
        except (FileNotFoundError, NotADirectoryError):
            changes.append({"path": path, "status": "deleted"})
            continue
        if now != entry:
            changes.append({"path": path, "status": "modified", "kind": now[0]})
    for path in _untracked(workspace):
        if path.endswith("/"):
            changes.append({"path": path, "status": "added", "kind": "repository"})
        elif path not in base:
            changes.append({"path": path, "status": "added", "kind": _entry(workspace / path)[0]})

    taken, dropped, rejected = [], [], []
    for c in changes:
        is_design = any(under(c["path"], d) for d in design) and not hidden(c["path"], tuple(hide))
        if any(under(c["path"], o) for o in owned):
            # The evaluator's own: it builds and drives the device with its copy.
            dropped.append({**c, "reason": "evaluator-owned"})
        elif not is_design:
            dropped.append(c)
        elif c.get("kind") in ("120000", "special", "repository"):
            rejected.append({**c, "reason": "not a regular file"})
        else:
            taken.append(c)
    verdict = "REJECT" if rejected else "PASS"
    report = {"verdict": verdict, "base_commit": commit, "taken": taken, "dropped": dropped, "rejected": rejected}
    if out is not None and verdict == "PASS":
        out.mkdir(parents=True, exist_ok=True)
        files = out / "files"
        for c in taken:
            if c["status"] != "deleted":
                (files / c["path"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(workspace / c["path"], files / c["path"])
        (out / "candidate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def apply_candidate(repo: Path, commit: str, candidate: Path, dest: Path, only: Optional[tuple] = None) -> None:
    """The evaluator's tree: the FULL base (nothing hidden) with the candidate's files on top.

    `only` limits the files laid on to those under these prefixes: the device
    the evaluator builds takes the candidate's hardware and nothing else.
    """
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    if not dest.exists() or not any(dest.iterdir()):
        export(repo, commit, (), dest)
    for c in report["taken"]:
        if only is not None and not c["path"].startswith(tuple(only)):
            continue
        target = dest / c["path"]
        if c["status"] == "deleted":
            if target.exists():
                target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate / "files" / c["path"], target)


__all__ = ["ALWAYS_HIDDEN", "INSTRUCTION_NAMES", "WorkspaceError", "apply_candidate", "export", "gate",
           "visible_tree"]
