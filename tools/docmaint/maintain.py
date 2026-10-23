"""One call maintains every stale document, in dependency order, safely.

Safety model:

1. The agent never runs in the repository. Each run exports HEAD (git
   archive) into a private scratch copy with its own throwaway git index, and
   the agent works there.
2. One agent call edits one document. Afterwards the copy's git status must
   show no change except that document; anything else (code, other docs,
   new files, commits) is reverted and the document is rejected.
3. Only that Markdown file is copied back into the repository, after checks:
   regular file, .md, UTF-8, non-empty, size limit, and the repository's copy
   unmodified since HEAD (local edits are never overwritten).
4. The repository itself is fingerprinted before and after each agent call;
   if it changed, the run stops.

Staleness: a document's input fingerprint covers the blob hashes of every
tracked file its ``sources`` match at HEAD, plus the current text of every
document it ``depends_on`` (so upstream edits made earlier in the same run
propagate). The fingerprint after a successful maintenance is recorded in the
state file next to the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .backends import Backend
from .graph import Manifest, expand_sources, topo_order, tracked_files

MAX_DOC_BYTES = 2 * 1024 * 1024
CONTEXT_DIR = ".docmaint"
DIFF_CAP = 400_000


def git(root: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=root, check=check, capture_output=True,
                          text=True).stdout


def state_path(root: Path, m: Manifest) -> Path:
    return root / Path(m.path).with_name("docmap.state.json")


def load_state(root: Path, m: Manifest) -> dict:
    p = state_path(root, m)
    return json.loads(p.read_text()) if p.is_file() else {"docs": {}}


def blob_index(root: Path) -> dict[str, str]:
    out = git(root, "ls-files", "-s", "-z")
    idx = {}
    for rec in out.split("\0"):
        if rec:
            meta, path = rec.split("\t", 1)
            idx[path] = meta.split()[1]
    return idx


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "absent"


def fingerprint(work: Path, m: Manifest, doc: str, files: list[str], blobs: dict[str, str]) -> tuple[str, list[str]]:
    d = m.docs[doc]
    h = hashlib.sha256()
    inputs = []
    state_file = str(Path(m.path).with_name("docmap.state.json"))
    for f in expand_sources(d.sources, files):
        # Maintained documents are ordered only through depends_on; a folder
        # source that happens to contain other documents must not bypass the
        # topological order (it would never converge).
        if f == doc or f in m.docs or f == state_file:
            continue
        h.update(f"S {f} {blobs.get(f, '?')}\n".encode())
        inputs.append(f)
    for up in sorted(d.depends_on):
        h.update(f"D {up} {sha256_file(work / up)}\n".encode())
        inputs.append(up)
    return h.hexdigest(), inputs


@dataclass
class DocResult:
    path: str
    status: str            # fresh | updated | unchanged | rejected | failed | skipped | planned
    reason: str = ""
    seconds: float = 0.0
    log: str = ""
    violations: list[str] = field(default_factory=list)


def repo_snapshot(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").strip() + "\n" + git(root, "status", "--porcelain=v1", "-uall")


# ------------------------------------------------------------ scratch copy
class Scratch:
    """HEAD exported to a private directory with its own git index, so the
    agent cannot reach the repository's history, refs or working tree."""

    def __init__(self, root: Path, base: Path):
        self.root = root
        base.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix="docmaint-", dir=base))
        archive = subprocess.run(["git", "archive", "HEAD"], cwd=root, check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(self.dir)], input=archive, check=True)
        for args in (["init", "-q"], ["add", "-A"],
                     ["-c", "user.name=docmaint", "-c", "user.email=docmaint@localhost",
                      "commit", "-q", "-m", "base", "--no-verify"]):
            git(self.dir, *args)
        (self.dir / ".git/info/exclude").write_text(f"/{CONTEXT_DIR}/\n")
        (self.dir / CONTEXT_DIR).mkdir()
        self.base = git(self.dir, "rev-parse", "HEAD").strip()

    def changes(self) -> list[str]:
        out = git(self.dir, "status", "--porcelain=v1", "-z", "-uall")
        paths = []
        for rec in out.split("\0"):
            if rec:
                paths.append(rec[3:])
        if git(self.dir, "rev-parse", "HEAD").strip() != self.base:
            paths.append("<HEAD moved: the agent committed>")
        return sorted(set(paths))

    def accept(self, doc: str) -> None:
        """Make the accepted document part of the copy's baseline."""
        git(self.dir, "add", "--", doc)
        git(self.dir, "-c", "user.name=docmaint", "-c", "user.email=docmaint@localhost",
            "commit", "-q", "-m", f"maintain {doc}", "--no-verify")
        self.base = git(self.dir, "rev-parse", "HEAD").strip()

    def revert(self) -> None:
        git(self.dir, "reset", "-q", "--hard", self.base)
        git(self.dir, "clean", "-q", "-fdx", "-e", f"/{CONTEXT_DIR}/")

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------ prompt
PROMPT = """You are maintaining exactly one documentation file of this repository.

TARGET DOCUMENT: {doc}

Your task: bring {doc} up to date with the current state of its inputs.

Inputs to verify against (read them; they are the ground truth):
- Upstream documents, already maintained in this run, read first: {ups}
- Sources (files and folders this document describes): {srcs}
- What changed since this document was last maintained: {why}
{diffnote}
{notes}
Rules (mandatory):
1. Edit ONLY {doc}. Do not create, delete, rename or modify any other file,
   including source code, tests, configuration, scripts and other documents.
   Do not run git commands that change state (commit, checkout, reset, stash).
   Any other change is detected, reverted, and your edit is discarded.
2. Write in professional technical English. If parts of the document are in
   another language, rewrite them in professional English, preserving every
   technical fact, number, identifier, table and link.
3. Every statement must be supported by the inputs. Correct what is wrong or
   outdated, remove what no longer exists, add what is missing and important.
   Never invent results, measurements or status.
4. Keep the document's purpose and structure; be concise. Prefer tables and
   short paragraphs; remove duplication with upstream documents by linking
   to them with relative Markdown links instead of copying their content.{limit}
5. Keep relative links valid. Do not touch fenced code blocks that reproduce
   commands or file contents unless the sources show they changed.
6. If the document is already accurate and in professional English, leave
   it unchanged.

When finished, reply with one short paragraph summarising what you changed.
"""


def build_prompt(m: Manifest, doc: str, inputs: list[str], why: str, diff_file: str | None) -> str:
    d = m.docs[doc]
    srcs = ", ".join(d.sources) or "(none)"
    ups = ", ".join(d.depends_on) or "(none)"
    diffnote = (f"- The relevant git diff is in {diff_file} (read it; it may be truncated).\n"
                if diff_file else "")
    notes = f"Maintainer notes for this document: {d.notes}\n" if d.notes else ""
    limit = f"\n   Keep it under {d.max_lines} lines." if d.max_lines else ""
    lang = "" if m.language == "en" else f" (target language code: {m.language})"
    return PROMPT.format(doc=doc, ups=ups, srcs=srcs, why=why, diffnote=diffnote,
                         notes=notes, limit=limit) + lang


# ------------------------------------------------------------------- run
@dataclass
class RunOptions:
    only: list[str] | None = None
    force: bool = False
    dry_run: bool = False
    commit: bool = False
    timeout: int = 1800
    keep_scratch: bool = False
    cache: Path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "docmaint"


def _check_doc(p: Path) -> str | None:
    if p.is_symlink() or not p.is_file():
        return "not a regular file"
    if p.suffix != ".md":
        return "not Markdown"
    data = p.read_bytes()
    if not data.strip():
        return "empty"
    if len(data) > MAX_DOC_BYTES:
        return f"larger than {MAX_DOC_BYTES} bytes"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "not UTF-8"
    return None


def _why(root: Path, rec: dict | None, inputs: list[str], scratch: Scratch | None) -> tuple[str, str | None]:
    if not rec:
        return "never maintained by this tool; review the whole document", None
    old = rec.get("commit")
    paths = [i for i in inputs if (root / i).exists() or i]
    if old and subprocess.run(["git", "cat-file", "-e", f"{old}^{{commit}}"], cwd=root,
                              capture_output=True).returncode == 0:
        stat = git(root, "diff", "--stat", old, "HEAD", "--", *paths).strip()
        if scratch is not None:
            diff = git(root, "diff", old, "HEAD", "--", *paths)
            f = f"{CONTEXT_DIR}/{hashlib.sha1(old.encode()).hexdigest()[:8]}-{len(diff)}.diff"
            (scratch.dir / f).write_text(diff[:DIFF_CAP] + ("\n[... truncated ...]\n" if len(diff) > DIFF_CAP else ""))
            return (f"since commit {old[:12]}:\n{stat or '(only upstream documents changed in this run)'}", f)
        return f"since commit {old[:12]}: {stat.splitlines()[-1] if stat else 'upstream documents changed'}", None
    return "previous commit unknown; review against all inputs", None


def run(root: Path, m: Manifest, backend: Backend | None, opt: RunOptions) -> list[DocResult]:
    root = root.resolve()
    order = topo_order(m)
    targets = set(opt.only) if opt.only else set(order)
    unknown = targets - set(order)
    if unknown:
        raise SystemExit(f"not in the manifest: {', '.join(sorted(unknown))}")
    state = load_state(root, m)
    head = git(root, "rev-parse", "HEAD").strip()
    files = tracked_files(root)
    blobs = blob_index(root)
    results: list[DocResult] = []

    if opt.dry_run:
        for doc in order:
            fp, _ = fingerprint(root, m, doc, files, blobs)
            rec = state["docs"].get(doc)
            stale = opt.force or not rec or rec.get("fingerprint") != fp
            results.append(DocResult(doc, "planned" if doc in targets and stale else "fresh",
                                     "" if stale else "inputs unchanged"))
        return results

    if backend is None or not backend.available():
        raise SystemExit(f"backend {getattr(backend, 'name', None)!r} is not available on this machine")
    run_id = time.strftime("%Y%m%dT%H%M%S")
    logdir = opt.cache / hashlib.sha1(str(root).encode()).hexdigest()[:12] / f"run-{run_id}"
    logdir.mkdir(parents=True, exist_ok=True)
    scratch = Scratch(root, logdir.parent / "scratch")
    changed: list[str] = []
    try:
        for doc in order:
            fp, inputs = fingerprint(scratch.dir, m, doc, files, blobs)
            rec = state["docs"].get(doc)
            if doc not in targets or (not opt.force and rec and rec.get("fingerprint") == fp):
                results.append(DocResult(doc, "fresh", "inputs unchanged"))
                continue
            if git(root, "status", "--porcelain", "--", doc).strip():
                results.append(DocResult(doc, "skipped", "the repository copy has local edits"))
                continue
            why, diff_file = _why(root, rec, inputs, scratch)
            prompt = build_prompt(m, doc, inputs, why, diff_file)
            (logdir / (doc.replace("/", "__") + ".prompt.txt")).write_text(prompt)
            before_repo = repo_snapshot(root)
            before_doc = sha256_file(scratch.dir / doc)
            res = backend.run(prompt, scratch.dir, opt.timeout, logdir / (doc.replace("/", "__") + ".log"))
            if repo_snapshot(root) != before_repo:
                scratch.revert()
                results.append(DocResult(doc, "failed", "the repository changed during the agent call; run stopped",
                                         res.seconds, res.log))
                break
            violations = [c for c in scratch.changes() if c != doc]
            if violations:
                scratch.revert()
                results.append(DocResult(doc, "rejected", "the agent changed files other than the target",
                                         res.seconds, res.log, violations))
                continue
            if res.timed_out or res.returncode != 0:
                scratch.revert()
                results.append(DocResult(doc, "failed", f"backend exit {res.returncode}"
                                         + (" (timeout)" if res.timed_out else ""), res.seconds, res.log))
                continue
            bad = _check_doc(scratch.dir / doc)
            if bad:
                scratch.revert()
                results.append(DocResult(doc, "rejected", f"edited document is {bad}", res.seconds, res.log))
                continue
            after_doc = sha256_file(scratch.dir / doc)
            if after_doc != before_doc:
                if git(root, "status", "--porcelain", "--", doc).strip():
                    scratch.revert()
                    results.append(DocResult(doc, "skipped", "local edits appeared during the run", res.seconds, res.log))
                    continue
                shutil.copyfile(scratch.dir / doc, root / doc)
                scratch.accept(doc)
                changed.append(doc)
                status = "updated"
            else:
                status = "unchanged"
            fp_after, _ = fingerprint(scratch.dir, m, doc, files, blobs)
            state["docs"][doc] = {"fingerprint": fp_after, "commit": head, "doc_sha256": after_doc,
                                  "backend": backend.name, "time": run_id}
            state_path(root, m).write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
            results.append(DocResult(doc, status, "", res.seconds, res.log))
    finally:
        if not opt.keep_scratch:
            scratch.close()
        (logdir / "report.json").write_text(json.dumps([asdict(r) for r in results], indent=1))

    if opt.commit and results:
        paths = changed + [str(state_path(root, m).relative_to(root))]
        git(root, "add", "--", *paths)
        if git(root, "diff", "--cached", "--name-only").strip():
            git(root, "commit", "-q", "-m",
                f"docs: maintain {len(changed)} document(s) with docmaint ({backend.name})", "--", *paths)
    return results


def baseline(root: Path, m: Manifest, only: list[str] | None = None) -> int:
    """Record every listed document as current without running an agent."""
    state = load_state(root, m)
    files, blobs = tracked_files(root), blob_index(root)
    head = git(root, "rev-parse", "HEAD").strip()
    n = 0
    for doc in topo_order(m):
        if only and doc not in only:
            continue
        fp, _ = fingerprint(root, m, doc, files, blobs)
        state["docs"][doc] = {"fingerprint": fp, "commit": head, "doc_sha256": sha256_file(root / doc),
                              "backend": "baseline", "time": time.strftime("%Y%m%dT%H%M%S")}
        n += 1
    state_path(root, m).write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    return n
