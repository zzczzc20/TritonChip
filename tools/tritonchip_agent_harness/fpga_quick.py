"""The FPGA service's quick tier: out-of-context synthesis and timing of the device, on the build worker.

The candidate's hardware (the device tree: the base with only the candidate's
`rtl/` and profile on top) is copied to the build worker named in the site
configuration and synthesised there with the repository's own
`flow/synth/vivado_ooc.sh`, inside a bubblewrap sandbox: no network, a
read-only view of the system and of the Vivado installation, no home
directory, only the job's own directory writable. Candidate RTL is read by
Vivado, and the sandbox is what keeps it from reading anything else.

The FPGA top's parameters are still written by hand (P1 has not reached it),
so the profile the run is scored on is passed to it as NAME=VALUE: the
defaults in rtl/common/tnca_uarch_defaults.vh, then the profile's own values.

Only what the candidate's own design produced comes back: worst slack,
failing endpoints, the frequency the slack implies, the worst paths and the
utilisation. A post-synthesis estimate leaves out most of the wire, so the
frequency is optimistic; the routed design is the slow tier's to judge.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

TOP = "tnca_fpga_top"
DEFAULT_PART = "xcu280-fsvh2892-2L-e"
# Parameters of the top that are not the machine's: the platform's own.
NOT_MACHINE = {"CORE_ID", "NUM_CORES", "HOST_TAG", "WORDS", "ADDR_W", "ADDR_BITS", "PHYS_VREGS", "PHYS_W"}

_DEFINE = re.compile(r"`define\s+TNCA_(?:IF|UARCH)_([A-Z][A-Z0-9_]*)\s+(-?\d+)\s*$", re.M)
_PARAM = re.compile(r"^\s*parameter\s+([A-Z][A-Z0-9_]*)\s*=", re.M)


def effective_params(tree: Path, profile: str) -> dict:
    """The scored profile's parameters as the FPGA top takes them: defaults, then the profile."""
    params = {k: int(v) for k, v in _DEFINE.findall((tree / "rtl/common/tnca_uarch_defaults.vh").read_text())}
    p = json.loads((tree / "spec/r1_profiles.json").read_text())["profiles"].get(profile) or {}
    params.update(p.get("elaboration") or {})
    params.update((p.get("implementation") or {}).get("rtl_parameters") or {})
    top = set(_PARAM.findall((tree / "rtl/fpga" / f"{TOP}.v").read_text()))
    return {k: v for k, v in sorted(params.items())
            if k in top and k not in NOT_MACHINE and isinstance(v, int) and not isinstance(v, bool)}


def parse_summary(text: str) -> dict:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and i + 2 < len(lines):
            v = lines[i + 2].split()
            try:
                return {"wns_ns": float(v[0]), "tns_ns": float(v[1]), "failing_endpoints": int(v[2]),
                        "total_endpoints": int(v[3])}
            except (ValueError, IndexError):
                return {}
    return {}


_UTIL = {"CLB LUTs": "lut", "CLB Registers": "ff", "Block RAM Tile": "bram", "URAM": "uram", "DSPs": "dsp"}


def parse_util(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3:
            name = cells[1].rstrip("*").strip()
            if name in _UTIL and _UTIL[name] not in out:
                try:
                    out[_UTIL[name]] = int(float(cells[2]))
                except ValueError:
                    pass
    return out


def parse_paths(text: str, n: int = 5) -> list:
    paths, cur = [], {}
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "Slack" and len(w) >= 4:
            cur = {"slack_ns": _ns(w[3])}
        elif w[0] == "Source:" and len(w) > 1:
            cur["start"] = w[1]
        elif w[0] == "Destination:" and len(w) > 1:
            cur["end"] = w[1]
        elif line.strip().startswith("Data Path Delay:") and len(w) > 3:
            cur["delay_ns"] = _ns(w[3])
        elif line.strip().startswith("Logic Levels:") and len(w) > 2:
            try:
                cur["levels"] = int(w[2])
            except ValueError:
                pass
            if "start" in cur:
                paths.append(cur)
            cur = {}
            if len(paths) >= n:
                break
    return paths


def _ns(word: str) -> Optional[float]:
    try:
        return float(word.rstrip("ns"))
    except ValueError:
        return None


def threads(worker: dict) -> int:
    """Vivado's thread limit for a job on the worker: the site's `threads`, else 4."""
    return int(worker.get("threads", 4))


def _sandboxed(worker: dict, remote: str, period: float, generics: dict, seconds: int) -> str:
    """The command the build worker runs: the flow's own script, inside bubblewrap, with a hard deadline."""
    binds = ["--ro-bind", "/usr", "/usr", "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
             "--ro-bind-try", "/lib32", "/lib32", "--ro-bind", "/bin", "/bin", "--ro-bind-try", "/sbin", "/sbin",
             "--ro-bind", "/etc", "/etc"]
    for path in worker.get("ro_binds", []):
        binds += ["--ro-bind", path, path]
    # What flow/synth/vivado_ooc.sh does, but into a short directory: the script names its output
    # after every NAME=VALUE pair, and a whole profile's worth of them is longer than a file name may be.
    out = f"{remote}/sim/build/ooc/{TOP}"
    files = "$(cd rtl && ls common/*.v core/*.v gen/*.v nuca/*.v noc/*.v tile/*.v fpga/*.v *.v)"
    inner = (f"source {shlex.quote(worker['vivado_settings'])} >/dev/null 2>&1; mkdir -p {out} && "
             f"OOC_GENERICS={shlex.quote(' '.join(f'{k}={v}' for k, v in generics.items()))} "
             f"vivado -mode batch -nojournal -nolog -source flow/synth/vivado_ooc_timing.tcl -tclargs "
             f"{TOP} {period:g} {shlex.quote(worker.get('part', DEFAULT_PART))} {out} {remote}/rtl {files} "
             f"> {out}/vivado.log 2>&1")
    bwrap = (["timeout", "--kill-after=30", str(seconds), "bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
             + binds + ["--bind", remote, remote, "--tmpfs", "/tmp", "--tmpfs", "/home/sandbox",
                        "--setenv", "HOME", "/home/sandbox", "--proc", "/proc", "--dev", "/dev", "--chdir", remote,
                        "/bin/bash", "-c", inner])
    return (f"cd {shlex.quote(remote)} && sed -i 's/maxThreads 8/maxThreads {threads(worker)}/' "
            f"flow/synth/vivado_ooc_timing.tcl && {' '.join(shlex.quote(a) for a in bwrap)} > run.log 2>&1")


def run(worker: dict, tree: Path, profile: str, period: float, seconds: int, job_dir: Path, job_id: str) -> dict:
    """One quick-tier job, end to end. Returns the full result; the caller decides what the agent sees."""
    job_dir.mkdir(parents=True, exist_ok=True)
    generics = effective_params(tree, profile)
    remote = f"{worker['workdir'].rstrip('/')}/{job_id}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", worker["ssh"]]
    result: dict = {"profile": profile, "period_ns": period, "part": worker.get("part", DEFAULT_PART),
                    "parameters": generics}
    started = time.time()
    try:
        subprocess.run(ssh + [f"mkdir -p {shlex.quote(remote)}"], check=True, capture_output=True, timeout=60)
        subprocess.run(["rsync", "-aR", "--delete", "-e", "ssh -o BatchMode=yes",
                        f"{tree}/./rtl", f"{tree}/./flow/synth", f"{worker['ssh']}:{remote}/"],
                       check=True, capture_output=True, timeout=300)
        proc = subprocess.run(ssh + [_sandboxed(worker, remote, period, generics, seconds)],
                              capture_output=True, text=True, timeout=seconds + 120)
        result["exit"] = proc.returncode
        subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{worker['ssh']}:{remote}/run.log",
                        f"{worker['ssh']}:{remote}/sim/build/ooc", str(job_dir) + "/"],
                       capture_output=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        result["exit"] = None
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        result["seconds"] = round(time.time() - started, 1)
        subprocess.run(ssh + [f"rm -rf {shlex.quote(remote)}"], capture_output=True, timeout=60)
    reports = next((d for d in (job_dir / "ooc").glob("*") if (d / f"{TOP}_summary.rpt").exists()), None) \
        if (job_dir / "ooc").exists() else None
    if reports is None:
        log = (job_dir / "run.log").read_text(errors="replace") if (job_dir / "run.log").exists() else ""
        vlog = next(iter((job_dir / "ooc").glob("*/vivado.log")), None) if (job_dir / "ooc").exists() else None
        errors = [l for l in ((vlog.read_text(errors="replace") if vlog else "") + log).splitlines()
                  if "ERROR" in l or "timed out" in l.lower()]
        tail = [l for l in log.splitlines() if l.strip()][-5:]
        result.update(status="failed", errors=errors[-15:] or tail or ["no report: the job did not finish in time"])
        return result
    summary = parse_summary((reports / f"{TOP}_summary.rpt").read_text(errors="replace"))
    result.update(status="ok" if summary else "failed", **summary,
                  utilization=parse_util((reports / f"{TOP}_util.rpt").read_text(errors="replace")),
                  worst_paths=parse_paths((reports / f"{TOP}_paths.rpt").read_text(errors="replace"))
                  if (reports / f"{TOP}_paths.rpt").exists() else [])
    if "wns_ns" in summary and period - summary["wns_ns"] > 0:
        result["fmax_est_mhz"] = round(1000.0 / (period - summary["wns_ns"]), 1)
    return result


def cancel(worker: dict, job_id: str) -> None:
    """Stop a job's processes on the worker. The bracket keeps pkill from matching its own command line."""
    pattern = f"[{job_id[0]}]{job_id[1:]}"
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", worker["ssh"],
                    f"pkill -f {shlex.quote(pattern)}"], capture_output=True, timeout=60)


__all__ = ["cancel", "effective_params", "parse_paths", "parse_summary", "parse_util", "run", "threads"]
