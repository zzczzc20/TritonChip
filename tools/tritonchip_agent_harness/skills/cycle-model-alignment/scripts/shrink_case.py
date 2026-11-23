#!/usr/bin/env python3
"""The cheapest size of a case that still reaches the mechanisms you name, on the CA model.

    python /opt/tnca-skills/cycle-model-alignment/scripts/shrink_case.py CASE.py --dim NAME
        [--reach MECHANISM ...] [--min V] [--step S] [--out NEW.py] [--config CFG] [--profile P]

Run it from the repository checkout you work in (or set TNCA_ROOT). The case is a development case
(`_point_make(seed, NAME=...)`) or a case of your own whose `make()` takes NAME as a keyword with a default;
that default is the size it starts from. `--reach` names mechanisms as mechanisms.py lists them (`--list`
there); without it, every mechanism the case reaches at its own size. The script measures the case at its
own size, then bisects NAME over the multiples of `--step` from `--min` up (default: the largest power of two
up to 64 that divides the starting size), each probe one run of the case on the CA model, for the smallest
size whose run passes and reaches every named mechanism. It assumes a larger size reaches at least what a
smaller one does; every probe is printed, so a mechanism that comes and goes shows. Of the sizes it measured
that reach them, it keeps the one with the fewest model cycles: a smaller size is not always a cheaper
program (a size that is not a power of two can compile to a slower one), and the simulator's time is cycles.

It writes the case at that size to `--out` (default `<case>_<NAME><size>.py` in the current directory): the
original source with `make()` redefined at its end, and DEMAND dropped (the declared demand was the original
size's; tnca_case.py then measures it from the case). The cycles printed are the model's, its prediction of
what the RTL simulator will take; the simulator runs some tens of device cycles per second of wall time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import align_case as A  # noqa: E402
import mechanisms as M  # noqa: E402


def default_step(start: int) -> int:
    step = 1
    while step < 64 and start % (step * 2) == 0:
        step *= 2
    return step


def bisect(sizes: list, ok) -> tuple:
    """The smallest size in the ascending `sizes` for which ok(size) holds, assuming it holds at the last one and
    is monotone; and every probe made, in order."""
    probes = []
    lo, hi = 0, len(sizes) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        good = ok(sizes[mid])
        probes.append((sizes[mid], good))
        if good:
            hi = mid
        else:
            lo = mid + 1
    return sizes[lo], probes


def cheapest(runs: dict, want: set):
    """Of the sizes measured, the one that passes and reaches `want` in the fewest model cycles (the smaller
    size on a tie): a smaller size can compile to a slower program, and the simulator's time is its cycles."""
    good = [(rec.get("device_cycles") or 0, size) for size, rec in runs.items()
            if rec["status"] == "pass" and want <= set(rec["mechanisms"])]
    return min(good)[1]


def write_case(src: Path, out: Path, dim: str, size, start, dev_format: bool, note: list) -> None:
    call = f"_shrunk_from(0, {dim}={size!r})" if dev_format else f"_shrunk_from({dim}={size!r})"
    tail = "\n".join([
        "",
        "",
        f"# --- shrink_case.py: `{dim}` {start} -> {size} " + "-" * 40,
        *[f"# {line}" for line in note],
        "# The demand declared above was the original size's: tnca_case.py measures this one's from the case.",
        "POINT = None",
        "DEMAND = None",
        f"_shrunk_from = {'_point_make' if dev_format else 'make'}",
        "",
        "",
        "def make():",
        f"    return {call}",
        "",
    ])
    out.write_text(src.read_text(encoding="utf-8").rstrip("\n") + "\n" + tail, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", help="a case file")
    ap.add_argument("--dim", required=True, help="the dimension to shrink (a keyword of the case's builder)")
    ap.add_argument("--reach", action="append", default=[], help="a mechanism the smaller case must still reach")
    ap.add_argument("--min", type=int, help="the smallest size to try (default: --step)")
    ap.add_argument("--step", type=int, help="sizes are multiples of this")
    ap.add_argument("--out", type=Path, help="where the smaller case goes")
    M.common_args(ap)
    a = ap.parse_args(argv)
    for name in a.reach:
        if name not in M.RULES:
            ap.error(f"--reach {name!r}: not a mechanism; mechanisms.py --list names them")
    case = Path(a.case).resolve()
    mod = M.case_module(case)
    _, defaults = M.sized(mod, {a.dim: 0})
    start = defaults.get(a.dim)
    if not isinstance(start, int) or start < 2:
        ap.error(f"the case's {a.dim!r} has no integer default to start from ({start!r})")
    step = a.step or default_step(start)
    first_size = -(-max(a.min or step, step) // step) * step       # the first multiple of step at or above --min
    sizes = list(range(first_size, start, step)) + [start]
    r = A.root()
    binary, config = M.model_for(r, a.model_build, a.config)
    runs = {}

    def run(size):
        if size not in runs:
            runs[size] = M.measure(r, case, binary=binary, config=config, profile=a.profile,
                                   active_cores=a.active_cores, dims={a.dim: size}, timeout=a.timeout)
        return runs[size]

    first = run(start)
    print(f"{a.dim}={start}: {first['status']}, {first.get('device_cycles')} cycles, reaches "
          f"{sorted(first['mechanisms'])}", flush=True)
    if first["status"] != "pass":
        print("\n".join(first.get("problems") or []))
        return 2
    want = set(a.reach) or set(first["mechanisms"])
    if not want <= set(first["mechanisms"]):
        print(f"the case does not reach {sorted(want - set(first['mechanisms']))} at its own size")
        return 2

    def ok(size):
        rec = run(size)
        missing = sorted(want - set(rec["mechanisms"]))
        good = rec["status"] == "pass" and not missing
        print(f"  {a.dim}={size}: {rec['status']}, {rec.get('device_cycles')} cycles"
              + (f", missing {missing}" if missing else ", reaches all"), flush=True)
        return good

    smallest, _ = bisect(sizes, ok)
    size = cheapest(runs, want)
    got = run(size)
    print(f"smallest that reaches: {a.dim}={smallest} ({run(smallest).get('device_cycles')} cycles on the model); "
          f"cheapest: {a.dim}={size} ({got.get('device_cycles')} cycles; {start}: {first.get('device_cycles')})")
    out = a.out or Path.cwd() / f"{case.stem}_{a.dim}{size}.py"
    note = [f"Reaches on the CA model ({Path(config).name}): {', '.join(sorted(want))}.",
            f"{got.get('device_cycles')} model cycles here; {first.get('device_cycles')} at {a.dim}={start}."]
    write_case(case, out, a.dim, size, start, hasattr(mod, "_point_make"), note)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
