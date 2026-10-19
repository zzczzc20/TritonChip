"""Small persistent token ledger for managed Harness workers.

The ledger controls requests issued by a worker that owns the ledger.  It does
not claim to meter an already-running external Codex conversation; that mode is
advisory and should only use the evidence/work-item limits in the manifest.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any


class BudgetError(RuntimeError):
    pass


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class TokenLedger:
    def __init__(self, path: Path, hard_limit: int) -> None:
        if hard_limit <= 0:
            raise BudgetError("hard_limit must be positive")
        self.path = Path(path).expanduser().resolve()
        self.hard_limit = int(hard_limit)

    def _initial(self) -> dict[str, Any]:
        return {"hard_limit": self.hard_limit, "spent": 0, "reserved": 0, "entries": {}}

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock = lock_path.open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if self.path.is_file() and self.path.stat().st_size > 0:
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                lock.close()
                raise BudgetError(f"cannot read ledger {self.path}: {exc}") from exc
        else:
            value = self._initial()
        if value.get("hard_limit") != self.hard_limit:
            lock.close()
            raise BudgetError("ledger hard_limit does not match requested limit")
        return lock, value

    def reserve(self, request_id: str, amount: int) -> dict[str, Any]:
        if not request_id or amount <= 0:
            raise BudgetError("request_id and positive amount are required")
        lock, value = self._locked()
        try:
            entries = value.setdefault("entries", {})
            existing = entries.get(request_id)
            if existing is not None:
                if existing.get("reserved") != amount:
                    raise BudgetError(f"request {request_id!r} already reserved with a different amount")
                return value
            if int(value["spent"]) + int(value["reserved"]) + amount > self.hard_limit:
                raise BudgetError("token budget exhausted")
            entries[request_id] = {"reserved": amount, "state": "RESERVED"}
            value["reserved"] = int(value["reserved"]) + amount
            _write_atomic(self.path, value)
            return value
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def settle(self, request_id: str, actual: int | None) -> dict[str, Any]:
        lock, value = self._locked()
        try:
            entry = value.setdefault("entries", {}).get(request_id)
            if entry is None:
                raise BudgetError(f"unknown request {request_id!r}")
            if entry.get("state") in ("SETTLED", "UNKNOWN"):
                return value
            reserved = int(entry["reserved"])
            if actual is None:
                entry["state"] = "UNKNOWN"
                entry["actual"] = None
                # Keep the reservation: unknown provider usage cannot be
                # treated as zero or released for another request.
            else:
                if actual < 0:
                    raise BudgetError("actual usage cannot be negative")
                value["reserved"] = int(value["reserved"]) - reserved
                value["spent"] = int(value["spent"]) + actual
                entry.update({"state": "SETTLED", "actual": actual})
            _write_atomic(self.path, value)
            return value
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def status(self) -> dict[str, Any]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return self._initial()
        return json.loads(self.path.read_text(encoding="utf-8"))


__all__ = ["BudgetError", "TokenLedger"]
