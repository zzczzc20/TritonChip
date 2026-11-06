"""The documentation dependency graph.

A manifest (TOML) lists every maintained Markdown document with two kinds of
dependency:

* ``depends_on``: other documents. A document is maintained after everything
  it depends on, so a change made upstream in this run is visible downstream.
* ``sources``: files, folders or glob patterns in the repository. They are
  what the document describes; a change in any of them makes it stale.

Example::

    [settings]
    language = "en"
    backend = "agy"

    [[doc]]
    path = "doc/architecture_overview.md"
    depends_on = ["doc/isa/TNCA-ISA-v0.1.md"]
    sources = ["rtl/core/", "rtl/nuca/*.v"]
    notes = "High-level architecture; keep tables in sync with the RTL."
    max_lines = 600
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_MANIFEST = "doc/docmap.toml"


class GraphError(Exception):
    """The manifest is malformed or its graph is not a DAG."""


@dataclass
class Doc:
    path: str
    depends_on: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: str = ""
    max_lines: int | None = None


@dataclass
class Manifest:
    docs: dict[str, Doc]
    language: str = "en"
    backend: str = "agy"
    path: str = DEFAULT_MANIFEST


def _norm(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or ".." in PurePosixPath(p).parts:
        raise GraphError(f"path must be relative to the repository root and stay inside it: {p!r}")
    return p


def load_manifest(root: Path, rel: str = DEFAULT_MANIFEST) -> Manifest:
    f = root / rel
    if not f.is_file():
        raise GraphError(f"manifest not found: {rel} (create one with `docmaint discover --write`)")
    data = tomllib.loads(f.read_text(encoding="utf-8"))
    settings = data.get("settings", {})
    docs: dict[str, Doc] = {}
    for entry in data.get("doc", []):
        if "path" not in entry:
            raise GraphError(f"doc entry without path: {entry}")
        d = Doc(path=_norm(entry["path"]),
                depends_on=[_norm(x) for x in entry.get("depends_on", [])],
                sources=[_norm(x) for x in entry.get("sources", [])],
                notes=str(entry.get("notes", "")),
                max_lines=entry.get("max_lines"))
        if not d.path.endswith(".md"):
            raise GraphError(f"maintained documents must be Markdown (.md): {d.path}")
        if d.path in docs:
            raise GraphError(f"document listed twice: {d.path}")
        docs[d.path] = d
    m = Manifest(docs=docs, language=settings.get("language", "en"),
                 backend=settings.get("backend", "agy"), path=rel)
    validate(m, root)
    return m


def validate(m: Manifest, root: Path) -> None:
    for d in m.docs.values():
        if not (root / d.path).is_file():
            raise GraphError(f"document does not exist: {d.path}")
        for up in d.depends_on:
            if up not in m.docs:
                raise GraphError(f"{d.path} depends on {up}, which is not in the manifest")
            if up == d.path:
                raise GraphError(f"{d.path} depends on itself")
    topo_order(m)  # raises on cycles


def topo_order(m: Manifest) -> list[str]:
    """Kahn's algorithm; ties broken by path so the order is deterministic."""
    indeg = {p: 0 for p in m.docs}
    down: dict[str, list[str]] = {p: [] for p in m.docs}
    for d in m.docs.values():
        for up in d.depends_on:
            indeg[d.path] += 1
            down[up].append(d.path)
    ready = sorted(p for p, n in indeg.items() if n == 0)
    order: list[str] = []
    while ready:
        p = ready.pop(0)
        order.append(p)
        for q in sorted(down[p]):
            indeg[q] -= 1
            if indeg[q] == 0:
                ready.append(q)
        ready.sort()
    if len(order) != len(m.docs):
        raise GraphError("dependency cycle: " + " -> ".join(_find_cycle(m, set(m.docs) - set(order))))
    return order


def _find_cycle(m: Manifest, nodes: set[str]) -> list[str]:
    start = sorted(nodes)[0]
    seen: list[str] = []
    p = start
    while p not in seen:
        seen.append(p)
        p = next(u for u in m.docs[p].depends_on if u in nodes)
    return seen[seen.index(p):] + [p]


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return sorted(x for x in out.decode().split("\0") if x)


def expand_sources(patterns: list[str], files: list[str]) -> list[str]:
    """Tracked files matched by a doc's sources. A pattern ending in "/" (or
    naming a directory) means everything below it; otherwise fnmatch."""
    hit: set[str] = set()
    for pat in patterns:
        prefix = pat.rstrip("/") + "/"
        for f in files:
            if f == pat or f.startswith(prefix) or fnmatch.fnmatchcase(f, pat):
                hit.add(f)
    return sorted(hit)


# --------------------------------------------------------------- discovery
_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
_CODE = re.compile(r"`([A-Za-z0-9_./-]+)`")


def discover(root: Path, include: list[str] | None = None) -> str:
    """Propose a manifest from the Markdown files themselves: links to other
    Markdown files become depends_on, references to tracked files or folders
    become sources. The result is a starting point to review, not a truth."""
    files = tracked_files(root)
    fileset = set(files)
    dirs = {str(PurePosixPath(f).parent) for f in files}
    mds = [f for f in files if f.endswith(".md")
           and (not include or any(f.startswith(i.rstrip("/") + "/") or f == i for i in include))]
    mdset = set(mds)
    entries: dict[str, tuple[list[str], list[str]]] = {}
    for md in mds:
        text = (root / md).read_text(encoding="utf-8", errors="replace")
        base = PurePosixPath(md).parent
        deps, srcs = set(), set()
        for raw in _LINK.findall(text) + _CODE.findall(text):
            if "://" in raw:
                continue
            for cand in (str(PurePosixPath(base, raw)), raw):
                try:
                    c = _norm(str(PurePosixPath(cand)))
                    c = str(PurePosixPath(*[x for x in PurePosixPath(c).parts]))
                except GraphError:
                    continue
                c = _collapse(c)
                if c in mdset and c != md:
                    deps.add(c); break
                if c in fileset and not c.endswith(".md"):
                    srcs.add(c); break
                if c.rstrip("/") in dirs and c.rstrip("/") not in ("", "."):
                    srcs.add(c.rstrip("/") + "/"); break
        entries[md] = (sorted(deps), sorted(srcs))
    _break_cycles(entries)
    lines = ['# Documentation dependency graph, maintained by tools/docmaint.',
             '# Generated by `docmaint discover`; review depends_on and sources.',
             '', '[settings]', 'language = "en"', 'backend = "agy"', '']
    for md, (deps, srcs) in sorted(entries.items()):
        lines += ['[[doc]]', f'path = "{md}"',
                  'depends_on = [' + ", ".join(f'"{d}"' for d in deps) + ']',
                  'sources = [' + ", ".join(f'"{s}"' for s in srcs) + ']', '']
    return "\n".join(lines)


def _collapse(p: str) -> str:
    parts: list[str] = []
    for x in PurePosixPath(p).parts:
        if x == "..":
            if not parts:
                raise GraphError(p)
            parts.pop()
        elif x != ".":
            parts.append(x)
    return "/".join(parts)


def _break_cycles(entries: dict[str, tuple[list[str], list[str]]]) -> None:
    """Links between documents are often mutual; drop back edges found by DFS
    so the proposal is a DAG (the dropped edges are listed as comments)."""
    state: dict[str, int] = {}

    def visit(n: str) -> None:
        state[n] = 1
        deps = entries[n][0]
        for d in list(deps):
            if state.get(d) == 1:
                deps.remove(d)
            elif d not in state:
                visit(d)
        state[n] = 2

    for n in sorted(entries):
        if n not in state:
            visit(n)
