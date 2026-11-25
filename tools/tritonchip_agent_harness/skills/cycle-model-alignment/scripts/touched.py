#!/usr/bin/env python3
"""Which mechanisms your change moves: the same cases on the base and on your tree, on the CA model, compared.

    python /opt/tnca-skills/cycle-model-alignment/scripts/touched.py CASE_OR_POINT [...] [--base REF]
        [--config CFG] [--base-config CFG] [--profile P] [--base-profile P] [--top N] [--json OUT]

Run it from the repository checkout you work in (or set TNCA_ROOT). A case is a case file (`make()`) or the id
of a development point (its file in /opt/tnca-task/dev_cases; a stand-in has an id of its own there). Each
case runs twice on the CA model (mechanisms.py's measurement, one process each, two at once):

  base     the base commit's tree (default: your checkout's first commit, the base every island starts
           from; `--base REF` for another), exported once to /tmp/tnca-touched/base-<sha>, its package data
           staged, its own model built from its own source, with its configuration (`--base-config`, default
           the same path as `--config`) and `--base-profile`;
  changed  your working tree as it is, its model (the --model-build directory, built incrementally) with
           `--config` and `--profile`.

For each case it prints the model's cycles on both sides, every mechanism either side reaches with its count
on each side and the change (largest first; `new` and `gone` for a mechanism only one side reaches), the
matrix instructions of each side's compiled code, and the `--top` raw counters that moved most.

What it can see is what the model can see. The model knows your machine only through its configuration and
its code: an RTL change the configuration does not carry moves nothing here, and a compiler change shows as
the program it compiles to. A mechanism whose count moved is one your change reaches on that case; one that
stayed is not proof of the opposite on other programs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import align_case as A  # noqa: E402
import mechanisms as M  # noqa: E402

DEV_CASES = Path(os.environ.get("TNCA_DEV_CASES", "/opt/tnca-task/dev_cases"))
CACHE = Path(os.environ.get("TNCA_TOUCHED_DIR", "/tmp/tnca-touched"))


def git(r: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(r), *args], capture_output=True, text=True, check=True).stdout.strip()


def base_tree(r: Path, ref: str | None) -> Path:
    """The base commit's tree, exported once (git archive), with its package data staged."""
    sha = git(r, "rev-parse", f"{ref}^{{commit}}") if ref else git(r, "rev-list", "--max-parents=0", "HEAD").split()[-1]
    dest = CACHE / f"base-{sha[:12]}"
    if (dest / ".complete").exists():
        return dest
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    dest.mkdir(parents=True)
    archive = subprocess.Popen(["git", "-C", str(r), "archive", sha], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    if archive.wait():
        raise SystemExit(f"git archive {sha} failed")
    p = subprocess.run(["make", "-s", "python/tnca/data/.staged"], cwd=dest, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"staging the base's package data failed:\n{(p.stdout + p.stderr)[-1500:]}")
    (dest / ".complete").write_text(sha + "\n")
    return dest


def resolve(name: str) -> Path:
    """A case file, or a development point's file by its id."""
    if Path(name).is_file():
        return Path(name).resolve()
    index = DEV_CASES / "INDEX.json"
    if index.is_file():
        for c in json.loads(index.read_text(encoding="utf-8"))["cases"]:
            if c["point"] == name:
                return (DEV_CASES / c["file"]).resolve()
    raise SystemExit(f"{name}: neither a case file nor a development point (see {index})")


def change(b: float, c: float) -> str:
    if not b and c:
        return "new"
    if b and not c:
        return "gone"
    return f"{(c - b) / b * 100:+.1f}%" if b else "0"


def moved(b: float, c: float) -> float:
    """How far a count moved, for the order: a mechanism that appears or vanishes first, then by log ratio."""
    if bool(b) != bool(c):
        return math.inf
    return abs(math.log(c / b)) if b and c else 0.0


def compare(base: dict, mine: dict, top: int = 15) -> dict:
    names = sorted(set(base["mechanisms"]) | set(mine["mechanisms"]),
                   key=lambda n: -moved(base["mechanisms"].get(n, 0), mine["mechanisms"].get(n, 0)))
    mech = [{"mechanism": n, "base": base["mechanisms"].get(n, 0), "changed": mine["mechanisms"].get(n, 0),
             "change": change(base["mechanisms"].get(n, 0), mine["mechanisms"].get(n, 0))} for n in names]
    keys = set(base.get("counters") or {}) | set(mine.get("counters") or {})
    rows = [(k, (base.get("counters") or {}).get(k, 0), (mine.get("counters") or {}).get(k, 0)) for k in keys]
    rows = sorted((r for r in rows if r[1] != r[2]), key=lambda r: (-moved(r[1], r[2]), -abs(r[2] - r[1])))
    b, c = base.get("device_cycles"), mine.get("device_cycles")
    return {"case": mine["case"], "status": {"base": base["status"], "changed": mine["status"]},
            "cycles": {"base": b, "changed": c, "speedup_on_model": round(b / c, 4) if b and c else None},
            "mechanisms": mech,
            "touched": [m["mechanism"] for m in mech if m["base"] != m["changed"]],
            "static": {"base": base.get("static") or {}, "changed": mine.get("static") or {}},
            "counters_moved": [{"counter": k, "base": x, "changed": y, "change": change(x, y)} for k, x, y in rows[:top]],
            "problems": {"base": base.get("problems") or [], "changed": mine.get("problems") or []}}


def show(d: dict) -> str:
    cy = d["cycles"]
    out = [f"{Path(d['case']).name}: base {d['status']['base']}, changed {d['status']['changed']}; cycles "
           f"{cy['base']} -> {cy['changed']}" + (f" ({cy['speedup_on_model']}x on the model)" if cy["speedup_on_model"] else "")]
    out.append(f"  {'mechanism':34} {'base':>12} {'changed':>12}  change")
    for m in d["mechanisms"]:
        out.append(f"  {m['mechanism']:34} {m['base']:>12} {m['changed']:>12}  {m['change']}")
    if d["static"]["base"] != d["static"]["changed"]:
        out.append(f"  matrix instructions (compiled code): base {json.dumps(d['static']['base'])} -> "
                   f"changed {json.dumps(d['static']['changed'])}")
    elif d["static"]["base"]:
        out.append(f"  matrix instructions (compiled code), both: {json.dumps(d['static']['base'])}")
    if d["counters_moved"]:
        out.append("  counters that moved most:")
        out += [f"    {r['counter']:44} {r['base']:>12} -> {r['changed']:>12}  {r['change']}" for r in d["counters_moved"]]
    for side, probs in d["problems"].items():
        out += [f"  ! {side}: {p}" for p in probs]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="+", help="case files or development point ids")
    M.common_args(ap)
    ap.add_argument("--base", help="the base commit (default: your checkout's first commit)")
    ap.add_argument("--base-config", help="the base's model configuration, in its tree (default: --config)")
    ap.add_argument("--base-profile", help="the profile the base compiles for (default: --profile)")
    ap.add_argument("--top", type=int, default=15, help="raw counters to list per case")
    ap.add_argument("--json", type=Path, help="write one JSON record per case here")
    a = ap.parse_args(argv)
    r = A.root()
    cases = [resolve(c) for c in a.cases]
    btree = base_tree(r, a.base)
    print(f"base: {btree}", flush=True)
    sides = {
        "base": (btree, *M.model_for(btree, str(btree / ".model-build"), a.base_config or a.config),
                 a.base_profile or a.profile),
        "changed": (r, *M.model_for(r, a.model_build, a.config), a.profile),
    }
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for case in cases:
            futs = {side: pool.submit(M.measure, tree, case, binary=binary, config=cfg, profile=prof,
                                      active_cores=a.active_cores, timeout=a.timeout)
                    for side, (tree, binary, cfg, prof) in sides.items()}
            d = compare(futs["base"].result(), futs["changed"].result(), a.top)
            print(show(d), flush=True)
            out.append(d)
    if a.json:
        a.json.write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
    return 0 if all(d["status"]["base"] == d["status"]["changed"] == "pass" for d in out) else 1


if __name__ == "__main__":
    sys.exit(main())
