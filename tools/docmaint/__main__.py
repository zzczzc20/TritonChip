"""docmaint: dependency-ordered, agent-driven maintenance of the repository's
Markdown documentation. Agents may edit Markdown documents only.

    python -m tools.docmaint check                  validate the manifest
    python -m tools.docmaint plan                   show the order and stale documents
    python -m tools.docmaint run [--backend agy]    maintain every stale document
    python -m tools.docmaint baseline               mark all documents current
    python -m tools.docmaint discover [--write]     propose a manifest from links
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import graph, maintain
from .backends import BACKENDS, get_backend


def _root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    out = maintain.git(Path.cwd(), "rev-parse", "--show-toplevel").strip()
    return Path(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docmaint", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="repository root (default: the current git repository)")
    ap.add_argument("--manifest", default=graph.DEFAULT_MANIFEST)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--force", action="store_true")
    p_run = sub.add_parser("run")
    p_run.add_argument("--backend", choices=sorted(BACKENDS))
    p_run.add_argument("--model")
    p_run.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p_run.add_argument("--only", nargs="+", help="maintain only these documents")
    p_run.add_argument("--force", action="store_true", help="maintain even if inputs are unchanged")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--commit", action="store_true", help="commit updated documents and the state file")
    p_run.add_argument("--timeout", type=int, default=1800, help="seconds per document")
    p_run.add_argument("--keep-scratch", action="store_true")
    p_base = sub.add_parser("baseline")
    p_base.add_argument("--only", nargs="+")
    p_disc = sub.add_parser("discover")
    p_disc.add_argument("--include", nargs="+", help="only Markdown files under these paths")
    p_disc.add_argument("--write", action="store_true", help="write the manifest (refuses to overwrite)")
    a = ap.parse_args(argv)
    root = _root(a.root)

    if a.cmd == "discover":
        text = graph.discover(root, a.include)
        if a.write:
            f = root / a.manifest
            if f.exists():
                print(f"refusing to overwrite {a.manifest}", file=sys.stderr)
                return 1
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text + "\n")
            print(f"wrote {a.manifest}")
        else:
            print(text)
        return 0

    try:
        m = graph.load_manifest(root, a.manifest)
    except graph.GraphError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        return 1

    if a.cmd == "check":
        print(f"ok: {len(m.docs)} documents, acyclic")
        return 0
    if a.cmd == "baseline":
        print(f"recorded {maintain.baseline(root, m, a.only)} documents as current")
        return 0
    if a.cmd == "plan":
        res = maintain.run(root, m, None, maintain.RunOptions(dry_run=True, force=a.force))
        for i, r in enumerate(res, 1):
            print(f"{i:4d}  {r.status:8s}  {r.path}")
        print(f"{sum(r.status == 'planned' for r in res)} of {len(res)} documents need maintenance")
        return 0

    backend = None
    if not a.dry_run:
        kw = {k: v for k, v in (("model", a.model), ("effort", a.effort)) if v}
        backend = get_backend(a.backend or m.backend, **kw)
    opt = maintain.RunOptions(only=a.only, force=a.force, dry_run=a.dry_run, commit=a.commit,
                              timeout=a.timeout, keep_scratch=a.keep_scratch)
    res = maintain.run(root, m, backend, opt)
    for r in res:
        extra = f"  [{', '.join(r.violations[:5])}]" if r.violations else ""
        print(f"{r.status:9s} {r.path}  {r.reason}{extra}")
    bad = [r for r in res if r.status in ("failed", "rejected")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
