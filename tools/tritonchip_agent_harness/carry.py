"""A base's kept measurements, carried to a platform commit that cannot change them (decided 2026-11-09).

The evaluator keys the base's simulated cycles (evaluator.baseline_key) and its card cycles
(board.baseline_key) by the whole platform commit, so a new platform commit measures the base again --
an hour of simulation and a card session -- even when it changed only what neither measurement runs.
When the two platform commits differ only in MEASUREMENT_NEUTRAL files, the kept results are copied to
the new commit's keys, each with a provenance file beside it (`<key>.carried.json`). Anything else is
refused: the base is measured again.

    MEASUREMENT_NEUTRAL   python/tnca/backends/board_relay.py: the card relay's log of anomalies (the
                          read race fixed on 2026-11-09). The simulator never imports it, and it counts
                          a finished launch the same before and after. And that fix's own test, which
                          no measurement runs.

The whole tree's difference counts, not only the task's platform paths: the card host runs a checkout of
the platform commit, not just those files.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path

from . import board, evaluator
from .task import Task
from .workspace import git

MEASUREMENT_NEUTRAL = ("python/tnca/backends/board_relay.py", "python/tnca/test_board_relay_observe_cpu.py")


class CarryError(RuntimeError):
    pass


def changed_files(repo: Path, old: str, new: str) -> list:
    return sorted(x for x in git("-C", str(repo), "diff", "--name-only", old, new).decode().splitlines() if x)


def carry(state_dir: Path, repo: Path, task: Task, from_platform: str) -> dict:
    """The base's kept simulation and card results under `from_platform`, copied to the task's platform commit."""
    platform = task.evaluation.get("platform")
    if not platform:
        raise CarryError("the task has no evaluation.platform")
    to = platform["commit"]
    files = changed_files(repo, from_platform, to)
    if not files:
        raise CarryError(f"{from_platform[:12]} and {to[:12]} have the same tree: nothing to carry")
    other = [f for f in files if f not in MEASUREMENT_NEUTRAL]
    if other:
        raise CarryError(f"{from_platform[:12]}..{to[:12]} changes {', '.join(other[:10])}, which a measurement "
                         "may read: measure the base again")
    old = dataclasses.replace(task, evaluation={**task.evaluation, "platform": {**platform, "commit": from_platform}})
    out: dict = {"from_platform": from_platform, "to_platform": to, "changed": files}
    for kind, root, old_key, new_key in (
            ("simulation", state_dir / "baselines", evaluator.baseline_key(old), evaluator.baseline_key(task)),
            ("card", state_dir / "board-baselines", board.baseline_key(old), board.baseline_key(task))):
        src, dst = root / f"{old_key}.json", root / f"{new_key}.json"
        if dst.exists():
            out[kind] = {"status": "already there", "key": new_key}
            continue
        if not src.exists():
            out[kind] = {"status": "nothing kept to carry", "from_key": old_key, "key": new_key}
            continue
        tmp = dst.with_suffix(".json.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        provenance = {"carried_from": src.name, "to": dst.name, "from_platform": from_platform, "to_platform": to,
                      "changed": files, "why": "the platform commits differ only in files no measurement reads "
                      "(carry.MEASUREMENT_NEUTRAL)", "at": time.time()}
        (root / f"{new_key}.carried.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
        out[kind] = {"status": "carried", "from_key": old_key, "key": new_key}
    return out


__all__ = ["CarryError", "MEASUREMENT_NEUTRAL", "carry", "changed_files"]
