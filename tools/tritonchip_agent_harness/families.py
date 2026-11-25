"""The board score: one number per family, then their weighted geometric mean (decided 2026-11-10, B2).

A point's family is its operator (`op.matmul/...` -> matmul) unless the task names a family for it
(`evaluation.families`: R41-F3 puts llama decode and prefill into one end-to-end family of weight 2).
Each family's score is the geometric mean of its points' speedups over the base on the card; a point that
did not pass counts as FAILED_SPEEDUP, a penalty that is never zero. The score is the weighted geometric
mean of the families, so a family counts once however many points it has.
"""

from __future__ import annotations

import math
from typing import Optional

FAILED_SPEEDUP = 0.5


def family_of(point: str, families: Optional[dict] = None) -> str:
    for name, spec in (families or {}).items():
        if any(point.startswith(p) for p in spec.get("prefixes", ())):
            return name
    head = point.split("/", 1)[0]
    return head.split(".", 1)[1] if "." in head else head


def weight_of(family: str, families: Optional[dict] = None) -> float:
    return float(((families or {}).get(family) or {}).get("weight", 1.0))


def scores(rows: list, families: Optional[dict] = None) -> dict:
    """{"score": weighted geometric mean, "families": {family: geometric mean}} from evaluator.score's rows."""
    by: dict = {}
    for r in rows or ():
        s = r.get("speedup") if r.get("failure") is None else None
        by.setdefault(family_of(r["point"], families), []).append(s if isinstance(s, (int, float)) and s > 0
                                                                 else FAILED_SPEEDUP)
    fam = {f: math.exp(sum(math.log(x) for x in xs) / len(xs)) for f, xs in by.items() if xs}
    total = sum(weight_of(f, families) for f in fam)
    score = (math.exp(sum(weight_of(f, families) * math.log(v) for f, v in fam.items()) / total)
             if total else None)
    return {"score": score, "families": {f: round(v, 4) for f, v in sorted(fam.items())}}


__all__ = ["FAILED_SPEEDUP", "family_of", "scores", "weight_of"]
