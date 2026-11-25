"""What an agent expects of a snapshot on the card, set beside what the card measured (decided 2026-11-23).

With a visible development set, a `score` or `flow` request may carry a prediction: a JSON object, point id ->
the speedup over the base the agent expects on the card, for any points it chooses, and optionally "score" for
the board score. A stand-in's id counts for the development point it stands in for. Nothing is charged or
refused for it and it changes no score; when the card measures exactly that snapshot (the same digest), the
reply sets each expectation beside the card's value, and the run's record keeps both. What the agent chose to
predict is part of what a run shows (the fidelity question): which points, how far off, in which direction.
"""

from __future__ import annotations

import math
from typing import Optional

MAX_ENTRIES = 200
DIRECTION_BAND = 0.01      # a change within 1 % either way has no direction


class PredictionError(ValueError):
    pass


def check(value, points: list, standins: dict) -> dict:
    """A request's prediction, checked: {point or "score": float > 0}. Ids must be development points, their
    stand-ins, or "score"; a stand-in is mapped to its point here."""
    if not isinstance(value, dict) or not value:
        raise PredictionError("a prediction is a non-empty JSON object: point id -> expected speedup over the base")
    if len(value) > MAX_ENTRIES:
        raise PredictionError(f"a prediction has at most {MAX_ENTRIES} entries")
    known = set(points)
    out: dict = {}
    for key, x in value.items():
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise PredictionError(f"{key}: an expected speedup is a positive number")
        target = key if key == "score" or key in known else standins.get(key)
        if target is None:
            raise PredictionError(f"{key}: not a development point, a stand-in of one, or \"score\"")
        if target in out:
            raise PredictionError(f"{key}: its point {target} is predicted twice (a stand-in counts for its point)")
        out[target] = float(x)
    return out


def _direction(x: float) -> int:
    return 0 if abs(math.log(x)) < math.log1p(DIRECTION_BAND) else (1 if x > 1 else -1)


def grade(expected: dict, full: Optional[dict]) -> dict:
    """Each expectation beside the card's measurement (evaluator.score's rows; `full["score"]` the board score):
    per point the measured speedup and measured / expected; overall how many, the median of |measured / expected
    - 1|, and how often the direction (faster, slower, within 1 %) agreed."""
    rows = {r.get("point"): r for r in ((full or {}).get("points") or ())}
    out, errors, agree, n_dir = [], [], 0, 0
    for point, e in sorted(expected.items()):
        if point == "score":
            continue
        r = rows.get(point)
        m = r.get("speedup") if r and r.get("failure") is None else None
        entry = {"point": point, "expected": round(e, 4), "measured": round(m, 4) if m else None}
        if r is not None and r.get("failure") is not None:
            entry["failure"] = r["failure"]
        if m:
            entry["measured_over_expected"] = round(m / e, 4)
            errors.append(abs(m / e - 1))
            n_dir += 1
            agree += _direction(m) == _direction(e)
        out.append(entry)
    summary: dict = {"points": len(out), "measured": len(errors)}
    if errors:
        errors.sort()
        mid = len(errors) // 2
        summary["median_abs_error_pct"] = round(100 * (errors[mid] if len(errors) % 2 else
                                                       (errors[mid - 1] + errors[mid]) / 2), 2)
        summary["direction_agreed"] = f"{agree}/{n_dir}"
    if "score" in expected:
        got = (full or {}).get("score")
        summary["score"] = {"expected": round(expected["score"], 4), "measured": round(got, 4) if got else None}
    return {"points": out, "summary": summary}


__all__ = ["MAX_ENTRIES", "PredictionError", "check", "grade"]
