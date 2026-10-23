"""Agent backends. A backend runs one prompt inside a working directory and
returns. It is never trusted: the runner checks afterwards which files changed
and keeps only the one Markdown document the agent was asked to edit.

To add a backend, subclass Backend, implement ``command`` (and optionally
``available``), and register it in BACKENDS.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BackendResult:
    returncode: int
    seconds: float
    timed_out: bool
    log: str


class Backend:
    name = "base"
    executable = ""

    def available(self) -> bool:
        return bool(self.executable) and shutil.which(self.executable) is not None

    def command(self, prompt: str, workdir: Path, timeout: int) -> list[str]:
        raise NotImplementedError

    def run(self, prompt: str, workdir: Path, timeout: int, log_path: Path) -> BackendResult:
        cmd = self.command(prompt, workdir, timeout)
        t0 = time.monotonic()
        timed_out = False
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                p = subprocess.run(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout + 60)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc, timed_out = -1, True
        return BackendResult(rc, time.monotonic() - t0, timed_out, str(log_path))


class AgyBackend(Backend):
    """Antigravity CLI in print mode, with edits auto-approved. The workspace is
    the isolated worktree the runner passes as workdir."""
    name = "agy"
    executable = "agy"

    def __init__(self, model: str | None = None, effort: str | None = None):
        self.model, self.effort = model, effort

    def command(self, prompt: str, workdir: Path, timeout: int) -> list[str]:
        cmd = ["agy", "--print", prompt, "--dangerously-skip-permissions",
               "--mode", "accept-edits", "--disable-slash-commands",
               "--print-timeout", f"{timeout}s"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd


class _Planned(Backend):
    """Placeholder for a backend that is designed for but not yet supported."""

    def available(self) -> bool:
        return False

    def command(self, prompt: str, workdir: Path, timeout: int) -> list[str]:
        raise NotImplementedError(
            f"backend {self.name!r} is not implemented yet; implement {type(self).__name__}.command "
            f"(the runner's isolation and file guard apply unchanged)")


class CodexBackend(_Planned):
    # Intended: ["codex", "exec", "-C", str(workdir), "--sandbox", "workspace-write", prompt]
    name = "codex"
    executable = "codex"


class ClaudeBackend(_Planned):
    # Intended: ["claude", "-p", prompt, "--permission-mode", "acceptEdits",
    #            "--allowedTools", "Read,Grep,Glob,Edit"]
    name = "claude"
    executable = "claude"


BACKENDS: dict[str, type[Backend]] = {
    "agy": AgyBackend,
    "codex": CodexBackend,
    "claude": ClaudeBackend,
}


def get_backend(name: str, **kw) -> Backend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(sorted(BACKENDS))}")
    cls = BACKENDS[name]
    return cls(**kw) if kw and cls is AgyBackend else cls()
