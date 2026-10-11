"""Harness lifecycle, source identity, leases, and adapter execution."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform as host_platform
import shlex
import socket
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .schema import SchemaError, dump_json, load_json, validate_manifest
from . import __version__ as HARNESS_VERSION
from .adapters.u50 import U50Adapter as VendorU50Adapter, U50Config
from .adapters.u50 import execute_command
from .adapters.u280 import U280FlowAdapter
from .measurement import MeasurementError, archbench_python, board_preflight, image_profile_contract, measure_board


class HarnessError(RuntimeError):
    """A recoverable harness or flow error."""


# Where this package lives in a candidate checkout.  A checkout that differs
# from the implemented commit only below this path builds the same image.
HARNESS_TREE = "tools/tritonchip_fpga_harness"


def repo_root(start: Path) -> Path:
    start = start.resolve()
    out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if out.returncode:
        raise HarnessError(f"{start} is not inside a git repository")
    return Path(out.stdout.strip()).resolve()


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if out.returncode:
        detail = out.stderr.decode("utf-8", errors="replace").strip()
        raise HarnessError(f"git {' '.join(args)} failed: {detail}")
    # `git diff --binary` is intentionally part of the provenance record.  A
    # candidate may contain a generated binary, so decoding with strict UTF-8
    # would make identity collection fail before the harness can record it.
    return out.stdout.decode("utf-8", errors="surrogateescape").strip()


def source_identity(root: Path) -> dict[str, Any]:
    """Return an identity that also detects an intentionally dirty candidate.

    Vendor scratch/output trees are explicitly excluded from the source
    identity.  Their names and locations are still recorded so a run explains
    why a checkout containing an old build is not treated as a different RTL
    candidate.
    """
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "ls-files", "-s")
    diff = _git(root, "diff", "--binary", "HEAD")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    ignored_prefixes = {".ipcache", "u50_build", ".tritonchip-fpga-harness"}
    source_paths: list[str] = []
    ignored_paths: list[str] = []
    inventory: list[dict[str, Any]] = []
    for relative in untracked.splitlines() if untracked else []:
        path = root / relative
        if relative.split("/", 1)[0] in ignored_prefixes or path.suffix == ".log":
            ignored_paths.append(relative)
            continue
        source_paths.append(relative)
        entry: dict[str, Any] = {"path": relative, "exists": path.is_file()}
        if path.is_file():
            try:
                entry.update({"size": path.stat().st_size, "sha256": file_sha256(path)})
            except OSError as exc:
                entry["error"] = str(exc)
        inventory.append(entry)
    payload = (tree + "\n--diff--\n" + diff + "\n--untracked-inventory--\n" + json.dumps(inventory, sort_keys=True)).encode(
        "utf-8", errors="surrogateescape"
    )
    return {
        "commit": commit,
        "tree_digest": hashlib.sha256(tree.encode()).hexdigest(),
        "working_tree_digest": hashlib.sha256(payload).hexdigest(),
        "dirty": bool(diff or source_paths),
        "untracked_files": source_paths,
        "ignored_generated_count": len(ignored_paths),
        "ignored_generated_digest": hashlib.sha256("\n".join(ignored_paths).encode()).hexdigest(),
        "ignored_generated_sample": ignored_paths[:20],
        "untracked_inventory": inventory,
    }


def software_identity(root: Path, implemented_commit: str | None, current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Relate the checkout that drives the card to the commit the image was built from.

    Conformance and ArchBench run the runtime and compiler of ``root``.  That
    is the implemented commit when build and measurement are one ``submit``;
    a later ``measure`` may run from another checkout, and the measurement
    has to say which.  ``relation`` is ``same-commit``, ``same-tree`` (other
    commit, no file differs), ``harness-only-diff`` (every differing path is
    below ``HARNESS_TREE``), ``differs`` or ``unknown`` (the implemented
    commit is not in this repository, so nothing can be compared).
    """
    current = current or source_identity(root)
    changed: list[str] | None
    if not implemented_commit:
        relation, changed = "unknown", None
    elif implemented_commit == current["commit"]:
        relation, changed = "same-commit", []
    else:
        out = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", implemented_commit, "HEAD"],
            capture_output=True, text=True,
        )
        if out.returncode:
            relation, changed = "unknown", None
        else:
            changed = [line for line in out.stdout.splitlines() if line]
            if not changed:
                relation = "same-tree"
            elif all(path.startswith(HARNESS_TREE + "/") for path in changed):
                relation = "harness-only-diff"
            else:
                relation = "differs"
    return {
        "commit": current["commit"],
        "implemented_commit": implemented_commit,
        "relation": relation,
        "dirty": current["dirty"],
        "tree_digest": current["tree_digest"],
        "working_tree_digest": current["working_tree_digest"],
        "changed_count": None if changed is None else len(changed),
        "changed_paths": None if changed is None else changed[:50],
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        return value or None
    except OSError:
        return None


def _process_starttime(pid: int) -> str | None:
    """Return Linux process start ticks, used to avoid PID reuse."""
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii").split()
        return fields[21]
    except (OSError, IndexError):
        return None


def _process_identity(pid: int, pgid: int | None = None) -> dict[str, Any]:
    return {
        "pid": pid,
        "pgid": pgid if pgid is not None else pid,
        "pid_starttime": _process_starttime(pid),
        "boot_id": _boot_id(),
        "recorded_at": time.time(),
    }


def _same_process(identity: dict[str, Any]) -> bool:
    pid = int(identity.get("pid", -1))
    if pid <= 0 or _boot_id() != identity.get("boot_id"):
        return False
    start = _process_starttime(pid)
    return start is not None and start == identity.get("pid_starttime")


def _host_identity() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "uid": getattr(os, "getuid", lambda: None)(),
        "username": os.environ.get("USER") or os.environ.get("USERNAME"),
        "pid": os.getpid(),
        "boot_id": _boot_id(),
        "os": host_platform.platform(),
        "system": host_platform.system(),
        "release": host_platform.release(),
        "machine": host_platform.machine(),
        "python": host_platform.python_version(),
    }


def _probe(argv: Sequence[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Probe a tool without allowing a failed probe to abort a run."""
    result: dict[str, Any] = {"argv": list(argv), "found": False, "returncode": None, "version": None}
    try:
        path = shutil.which(str(argv[0]))
        result["path"] = path
        if path is None:
            return result
        result["found"] = True
        out = subprocess.run(list(argv), capture_output=True, text=True, timeout=10, env=env, check=False)
        result["returncode"] = out.returncode
        result["version"] = (out.stdout or out.stderr).strip()[:2000]
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    return result


def _probe_with_settings(tool: str, settings: Path) -> dict[str, Any]:
    if not settings.is_file():
        return _probe([tool, "--version"])
    result: dict[str, Any] = {"argv": [tool, "--version"], "settings": str(settings), "found": False, "returncode": None, "version": None}
    try:
        command = f"source {shlex.quote(str(settings))} >/dev/null 2>&1 && command -v {shlex.quote(tool)} && {shlex.quote(tool)} --version"
        out = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=20, check=False)
        result["returncode"] = out.returncode
        result["version"] = (out.stdout or out.stderr).strip()[:2000]
        lines = (out.stdout or "").splitlines()
        result["path"] = lines[0] if lines else None
        result["found"] = out.returncode == 0 and bool(lines)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    return result


def _path_fingerprint(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return value
    try:
        stat = path.stat()
        value.update({"is_file": path.is_file(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if path.is_file():
            value["sha256"] = file_sha256(path)
        elif path.is_dir():
            names = sorted(p.name for p in path.iterdir())[:200]
            value["entries"] = names
            value["entries_sha256"] = hashlib.sha256("\n".join(names).encode()).hexdigest()
    except OSError as exc:
        value["error"] = str(exc)
    return value


def environment_snapshot(root: Path, profile: Profile, adapter_name: str) -> dict[str, Any]:
    """Capture the host/tool/platform contract used by a run.

    This is intentionally descriptive: an unavailable vendor tool is a
    preflight error for a real run, while the snapshot remains useful for a
    dry-run and for diagnosing the correct FPGA server.
    """
    data = profile.data
    default_vitis = "/home/x4/Software/Xilinx/Vitis/2022.2" if adapter_name in {"u50", "real"} else ""
    vitis_root = Path(str(data.get("vitis_root", os.environ.get("XILINX_VITIS", default_vitis))))
    platforms_root = Path(str(data.get("platforms_root", os.environ.get("XILINX_PLATFORMS", "/opt/xilinx/platforms"))))
    settings = vitis_root / "settings64.sh" if str(vitis_root) else Path("")
    return {
        "schema_version": 1,
        "harness_version": HARNESS_VERSION,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": _host_identity(),
        "repo": str(root),
        "target": profile.name,
        "adapter": adapter_name,
        "board": data.get("board"),
        "part": data.get("part"),
        "platform": data.get("platform"),
        "kernel_clock_mhz": data.get("kernel_clock_mhz"),
        "device_clock_mhz": data.get("device_clock_mhz"),
        "device_divider": data.get("device_divider", 4),
        "paths": {
            "vitis_root": _path_fingerprint(vitis_root),
            "settings64": _path_fingerprint(settings),
            "platform": _path_fingerprint(platforms_root / str(data.get("platform", ""))),
            "package_script": _path_fingerprint(root / str(data.get("package_script", ""))),
            "link_script": _path_fingerprint(root / str(data.get("link_script", ""))),
            "program_script": _path_fingerprint(root / str(data.get("program_script", ""))),
        },
        "tools": {
            "vivado": _probe(["vivado", "-version"]),
            "v++": _probe_with_settings("v++", settings),
            "xclbinutil": _probe_with_settings("xclbinutil", settings),
            # XRT, not Vitis, ships xbutil, and settings64.sh does not put it
            # on PATH; the programmer calls it by this path too.
            "xbutil": _probe([str(Path(os.environ.get("XILINX_XRT", "/opt/xilinx/xrt")) / "bin" / "xbutil"), "--version"]),
        },
        "environment": {str(k): str(v) for k, v in data.get("environment", {}).items()},
    }


def host_resource_snapshot(path: Path) -> dict[str, Any]:
    """Record cheap host state around a run without requiring psutil."""
    snapshot: dict[str, Any] = {"timestamp": time.time()}
    try:
        usage = shutil.disk_usage(path)
        snapshot.update({"disk_free_bytes": usage.free, "disk_total_bytes": usage.total})
    except OSError:
        snapshot.update({"disk_free_bytes": None, "disk_total_bytes": None})
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                snapshot["mem_available_kb"] = int(line.split()[1])
                break
    except (OSError, ValueError, IndexError):
        snapshot["mem_available_kb"] = None
    snapshot.setdefault("mem_available_kb", None)
    return snapshot


@dataclass(frozen=True)
class Profile:
    name: str
    data: dict[str, Any]

    @property
    def expected_clock(self) -> float:
        return float(self.data["kernel_clock_mhz"])

    @property
    def expected_device_divider(self) -> int:
        return int(self.data.get("device_divider", 4))

    @property
    def expected_adapter(self) -> str:
        return str(self.data.get("adapter", self.data.get("board", "u50")))


def validate_profile(profile: Profile) -> None:
    data = profile.data
    required = ("board", "part", "platform", "kernel_clock_mhz", "device_clock_mhz", "full_slots")
    missing = [key for key in required if key not in data]
    if missing:
        raise SchemaError(f"profile {profile.name} missing: {', '.join(missing)}")
    divider = profile.expected_device_divider
    if divider < 1 or float(data["kernel_clock_mhz"]) / divider != float(data["device_clock_mhz"]):
        raise SchemaError(
            f"profile {profile.name} has inconsistent clocks: "
            f"device_clock_mhz={data['device_clock_mhz']} != kernel_clock_mhz/device_divider"
        )
    if int(data["full_slots"]) < 1:
        raise SchemaError(f"profile {profile.name} has invalid resource slots")
    expected = data.get("expected_artifacts", [])
    if len(expected) != 2 or not all(isinstance(item, str) and item for item in expected):
        raise SchemaError(f"profile {profile.name} must define two expected_artifacts")


def load_profile(package_dir: Path, target: str) -> Profile:
    path = package_dir / "profiles" / f"{target.replace('/', '_')}.json"
    if target == "u50-legacy":
        path = package_dir / "profiles" / "u50.json"
    if not path.is_file():
        raise HarnessError(f"unknown target {target!r}; expected {path}")
    value = load_json(path)
    profile = Profile(target, value)
    validate_profile(profile)
    return profile


@contextlib.contextmanager
def lease(state_dir: Path, mode: str) -> Iterator[None]:
    """Acquire one full or one of two OOC slots with a process-safe lock."""
    import fcntl  # POSIX is the supported FPGA worker host.

    lease_dir = state_dir / "leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    if mode == "full":
        paths = [lease_dir / "full.lock"]
    else:
        paths = [lease_dir / "ooc-0.lock", lease_dir / "ooc-1.lock"]
    handles = []
    try:
        for path in paths:
            fh = path.open("a+")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({"pid": os.getpid(), "mode": mode, "started": time.time()}))
            fh.flush()
            handles.append(fh)
            break
        if not handles:
            raise HarnessError(f"no {mode} slot is available")
        yield
    finally:
        for fh in handles:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


def _run_process(argv: Sequence[str], cwd: Path, log: Path, timeout_s: int | None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as out:
        out.write("$ " + " ".join(argv) + "\n")
        out.flush()
        proc = subprocess.Popen(argv, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = 124
            out.write(f"\nHARNESS_TIMEOUT after {timeout_s}s\n")
        out.write(f"\nHARNESS_RC={rc}\nHARNESS_ELAPSED_S={time.time() - started:.3f}\n")
    return rc


class Adapter:
    name = "base"

    def doctor(self, root: Path, profile: Profile) -> list[dict[str, Any]]:
        raise NotImplementedError

    def run(self, root: Path, run_dir: Path, profile: Profile, mode: str, timeout_s: int | None) -> list[dict[str, Any]]:
        raise NotImplementedError


class MockAdapter(Adapter):
    name = "mock"

    def doctor(self, root: Path, profile: Profile) -> list[dict[str, Any]]:
        return [{"name": "mock-adapter", "ok": True, "detail": "deterministic local simulation"}]

    def run(self, root: Path, run_dir: Path, profile: Profile, mode: str, timeout_s: int | None) -> list[dict[str, Any]]:
        stages = []
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        expected = list(profile.data.get("expected_artifacts", ["tnca_u50.xo", "tnca_u50.xclbin"]))
        # OOC is a synthesis screening endpoint.  It must not fabricate a
        # linked xclbin; only the package/XO boundary is exercised here.  A
        # full run is the only endpoint that produces an xclbin.
        stages_to_run = ("package", "link") if mode == "full" else ("package",)
        for name in stages_to_run:
            log = run_dir / "logs" / f"{name}.log"
            payload = {"adapter": self.name, "stage": name, "target": profile.name, "mode": mode}
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            if name == "package":
                (artifacts / expected[0]).write_bytes(b"TRITONCHIP-MOCK-XO\n")
            else:
                (artifacts / expected[-1]).write_bytes(b"TRITONCHIP-MOCK-XCLBIN\n")
                dump_json(artifacts / "timing_summary.json", {"kernel_clock_mhz": profile.expected_clock, "wns_ns": 0.25})
            stages.append({"name": name, "returncode": 0, "log": str(log.relative_to(run_dir)), "elapsed_s": 0.0})
        return stages


class U50FlowAdapter(Adapter):
    name = "u50"

    def _impl(self, root: Path, profile: Profile) -> VendorU50Adapter:
        expected_artifacts = list(profile.data["expected_artifacts"])
        return VendorU50Adapter(U50Config(
            repo=root,
            vitis_root=Path(os.environ.get("XILINX_VITIS", "/home/x4/Software/Xilinx/Vitis/2022.2")),
            platforms_root=Path(os.environ.get("XILINX_PLATFORMS", "/opt/xilinx/platforms")),
            platform=str(profile.data["platform"]),
            expected_ap_clk_mhz=int(profile.expected_clock),
            expected_device_divider=profile.expected_device_divider,
            require_routed_timing=bool(profile.data.get("require_routed_timing", False)),
            require_drc=bool(profile.data.get("require_drc", False)),
            xo_name=str(expected_artifacts[0]),
            xclbin_name=str(expected_artifacts[1]),
            package_script=root / str(profile.data["package_script"]),
            link_script=root / str(profile.data["link_script"]),
        ))

    def doctor(self, root: Path, profile: Profile) -> list[dict[str, Any]]:
        impl = self._impl(root, profile)
        # Doctor is a readiness check, so missing Vivado/Vitis/platform must
        # be reported as a blocking check.  `submit --dry-run` is the command
        # that deliberately downgrades vendor availability to warnings.
        result = impl.preflight(root.parent / ".tritonchip-fpga-harness-doctor", stage="full", dry_run=False)
        return [{"name": c.name, "ok": c.ok, "detail": c.detail, "severity": c.severity} for c in result.checks]

    def run(self, root: Path, run_dir: Path, profile: Profile, mode: str, timeout_s: int | None) -> list[dict[str, Any]]:
        if mode == "ooc":
            raise HarnessError("the current U50 adapter supports full implementation only; add an OOC adapter explicitly")
        build = run_dir / "artifacts"
        result = self._impl(root, profile).run(build, mode="full", timeout_s=timeout_s)
        stages = []
        for command in result.commands:
            name = "package" if command.argv and command.argv[0] == "vivado" else "link"
            stages.append({"name": name, "returncode": command.returncode, "log": str(Path(command.log_path).relative_to(run_dir)), "timed_out": command.timed_out, "resource": command.resource})
        if result.artifacts is not None:
            dump_json(run_dir / "artifact_validation.json", result.artifacts.as_dict())
        return stages

    def validate_existing(self, root: Path, profile: Profile, build_dir: Path) -> dict[str, Any]:
        """Validate an existing U50 build without invoking a vendor tool."""
        validation = self._impl(root, profile).validate_artifacts(build_dir, require_xclbin=True)
        return validation.as_dict()


def adapter_for(name: str) -> Adapter:
    if name == "mock":
        return MockAdapter()
    if name in ("u50", "real"):
        return U50FlowAdapter()
    if name == "u280":
        return U280FlowAdapter()
    raise HarnessError(f"unknown adapter {name!r}")


def doctor(package_dir: Path, root: Path, target: str, adapter_name: str,
           manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Readiness of this host for ``target``; with a manifest, for its board stage too."""
    if manifest is not None:
        validate_manifest(manifest)
        if manifest["target"] != target:
            raise SchemaError(f"manifest target {manifest['target']!r} is not {target!r}")
    profile = load_profile(package_dir, target)
    if adapter_name in ("u50", "u280", "real") and profile.expected_adapter != ("u50" if adapter_name == "real" else adapter_name):
        raise SchemaError(f"target {profile.name} requires adapter {profile.expected_adapter}, got {adapter_name}")
    adapter = adapter_for(adapter_name)
    checks = adapter.doctor(root, profile)
    if manifest is not None:
        checks = checks + board_preflight(root=root, profile=profile, manifest=manifest)
    return {
        "schema_version": 1,
        "target": target,
        "adapter": adapter.name,
        "checks": checks,
        "environment": environment_snapshot(root, profile, adapter.name),
        "ok": all(c["ok"] for c in checks),
    }


def validate_existing(package_dir: Path, root: Path, target: str, build_dir: Path, *, historical: bool = False) -> dict[str, Any]:
    profile = load_profile(package_dir, target)
    adapter = adapter_for(str(profile.data.get("adapter", "u50")))
    if not isinstance(adapter, (U50FlowAdapter, U280FlowAdapter)):
        raise HarnessError("vendor adapter is unavailable")
    source = source_identity(root)
    resolved_build = Path(build_dir).expanduser().resolve()
    lineage = resolved_build / "run.json"
    if not lineage.is_file():
        lineage = resolved_build.parent / "run.json"
    if not lineage.is_file() and not historical:
        return {
            "target": target,
            "source": source,
            "build_dir": str(resolved_build),
            "scope": "unattributed",
            "status": "UNATTRIBUTED",
            "error": "build has no harness run.json; pass --historical to inspect legacy artifacts",
        }
    validation = adapter.validate_existing(root, profile, resolved_build)
    return {
        "target": target,
        "source": source,
        "build_dir": str(resolved_build),
        "scope": "historical" if historical and not lineage.is_file() else "harness-run",
        "status": "HISTORICAL" if historical and not lineage.is_file() and validation.get("ok") else ("PASS" if validation.get("ok") else "FAIL"),
        "artifact_validation": validation,
    }


def plan(package_dir: Path, root: Path, manifest: dict[str, Any], adapter_name: str) -> dict[str, Any]:
    validate_manifest(manifest)
    profile = load_profile(package_dir, manifest["target"])
    if adapter_name in ("u50", "u280", "real") and profile.expected_adapter != ("u50" if adapter_name == "real" else adapter_name):
        raise SchemaError(
            f"target {profile.name} requires adapter {profile.expected_adapter}, got {adapter_name}"
        )
    if manifest["mode"] == "ooc" and adapter_name in ("u50", "u280", "real"):
        config = manifest.get("ooc")
        if not isinstance(config, dict):
            raise SchemaError("mode=ooc requires an explicit ooc configuration")
        command = config.get("command")
        artifacts = config.get("artifacts")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise SchemaError("ooc.command must be a non-empty argv list; shell strings are not accepted")
        if not isinstance(artifacts, list) or not artifacts or not all(isinstance(item, str) and item for item in artifacts):
            raise SchemaError("ooc.artifacts must list at least one output relative to the run directory")
        for artifact in artifacts:
            artifact_path = Path(artifact)
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                raise SchemaError(f"ooc artifact must be relative to the run directory: {artifact}")
        result_file = Path(str(config.get("result_file", "ooc_result.json")))
        if result_file.is_absolute() or ".." in result_file.parts:
            raise SchemaError("ooc.result_file must be relative to the run directory")
    identity = source_identity(root)
    requested_commit = manifest.get("source_commit")
    if requested_commit and requested_commit not in ("HEAD", identity["commit"]):
        raise SchemaError(f"source_commit {requested_commit!r} does not match current HEAD {identity['commit']}")
    expected = manifest.get("clock_mhz", profile.expected_clock)
    if float(expected) != profile.expected_clock:
        raise SchemaError(f"clock_mhz {expected} does not match target {profile.expected_clock}")
    return {
        "candidate_id": manifest["candidate_id"],
        "target": profile.name,
        "adapter": adapter_name,
        "mode": manifest["mode"],
        "clock_mhz": expected,
        "source": identity,
        "dirty_warning": identity["dirty"],
        "resource_policy": {"full_slots": profile.data["full_slots"], "ooc": "user-configured"},
        "board_measurement": {
            "requested": bool(manifest.get("board")),
            "backend": (manifest.get("board") or {}).get("backend", f"fpga-{profile.data.get('board', 'unknown')}"),
            "bdf": (manifest.get("board") or {}).get("bdf"),
            "archbench": bool((manifest.get("board") or {}).get("archbench")),
            "archbench_python": archbench_python(manifest["board"])
            if isinstance(manifest.get("board"), dict) and manifest["board"].get("archbench") else None,
            "completion_fields": ["done_cycles", "done_retired"],
            "image_profile": image_profile_contract(root, manifest["board"]["profile"])
            if isinstance(manifest.get("board"), dict) else None,
        },
        "profile_contract": {
            "expected_adapter": profile.expected_adapter,
            "kernel_clock_mhz": profile.data["kernel_clock_mhz"],
            "device_clock_mhz": profile.data["device_clock_mhz"],
            "device_divider": profile.expected_device_divider,
            "expected_artifacts": profile.data["expected_artifacts"],
            "program_script": profile.data.get("program_script"),
        },
    }


def _expected_artifacts(run_dir: Path, profile: Profile, mode: str, manifest: dict[str, Any] | None = None) -> list[Path]:
    if mode == "ooc":
        config = (manifest or {}).get("ooc", {})
        return [run_dir / str(path) for path in config.get("artifacts", [])]
    return [run_dir / "artifacts" / name for name in profile.data["expected_artifacts"]]


def _configured_ooc(root: Path, run_dir: Path, manifest: dict[str, Any], timeout_s: int | None) -> list[dict[str, Any]]:
    """Run a user-selected OOC command under a standard Harness contract.

    The command is argv-only (no shell) and receives a few explicit path
    substitutions. The command must write ``ooc_result.json`` (or the
    configured result_file) with status PASS/FAIL and may report timing and
    utilization details. This keeps module/top/constraint mapping in the
    user's OOC script instead of pretending it is common to every RTL module.
    """
    config = manifest["ooc"]
    run_dir = run_dir.resolve()
    substitutions = {
        "{repo}": str(root.resolve()),
        "{run_dir}": str(run_dir),
        "{top}": str(config.get("top", "")),
        "{clock_mhz}": str(config.get("clock_mhz", manifest.get("clock_mhz", ""))),
    }
    argv: list[str] = []
    for token in config["command"]:
        value = token
        for key, replacement in substitutions.items():
            value = value.replace(key, replacement)
        argv.append(value)
    log = run_dir / "logs" / "ooc.log"
    result = execute_command(argv, cwd=root, env={"TNCA_HARNESS_MODE": "ooc", "TNCA_HARNESS_RUN_DIR": str(run_dir)}, log_path=log, timeout_s=timeout_s)
    stages = [{
        "name": "ooc",
        "returncode": result.returncode,
        "log": str(log.relative_to(run_dir)),
        "timed_out": result.timed_out,
        "resource": result.resource,
        "argv": list(result.argv),
    }]
    result_file = run_dir / str(config.get("result_file", "ooc_result.json"))
    checks: list[dict[str, Any]] = []
    payload: dict[str, Any] | None = None
    if result_file.is_file():
        try:
            payload = load_json(result_file)
        except SchemaError as exc:
            checks.append({"name": "ooc_result_json", "ok": False, "detail": str(exc)})
    else:
        checks.append({"name": "ooc_result_json", "ok": False, "detail": str(result_file)})
    if payload is not None:
        checks.append({"name": "ooc_command", "ok": result.returncode == 0 and not result.timed_out, "detail": str(result.returncode)})
        checks.append({"name": "ooc_result_status", "ok": payload.get("status") == "PASS", "detail": str(payload.get("status"))})
    expected = _expected_artifacts(run_dir, load_profile(Path(__file__).resolve().parent, manifest["target"]), "ooc", manifest)
    for artifact in expected:
        checks.append({"name": f"ooc_artifact:{artifact.relative_to(run_dir)}", "ok": artifact.is_file() and artifact.stat().st_size > 0, "detail": str(artifact)})
    validation = {
        "ok": bool(checks) and all(check["ok"] for check in checks),
        "mode": "ooc",
        "checks": checks,
        "result_file": str(result_file.relative_to(run_dir)) if result_file.is_file() else None,
        "result": payload,
        "artifacts": {str(path.relative_to(run_dir)): file_sha256(path) for path in expected if path.is_file()},
    }
    dump_json(run_dir / "artifact_validation.json", validation)
    return stages


def _implementation_verdict(run_dir: Path, profile: Profile, manifest: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge the package/link (or OOC) stages from what they left behind."""
    missing = [str(p.relative_to(run_dir)) for p in _expected_artifacts(run_dir, profile, manifest["mode"], manifest) if not p.is_file()]
    stage_ok = all(s.get("returncode") == 0 for s in stages)
    validation = run_dir / "artifact_validation.json"
    validation_value = load_json(validation) if validation.is_file() else None
    validation_ok = validation_value is None or bool(validation_value.get("ok", False))
    failed_checks = {check.get("name", "") for check in (validation_value or {}).get("checks", []) if not check.get("ok", False)}
    if stage_ok and not missing and validation_ok:
        status = "PASS"
    elif manifest["mode"] == "ooc" and not validation_ok:
        status = "OOC_FAIL"
    elif failed_checks & {"timing_clean", "drc_clean"}:
        # A routed report exists and says the design did not close.  v++ exits
        # non-zero for that too, so this has to be asked before "the link
        # command failed" -- otherwise a timing failure is filed as a build
        # failure and the report that explains it is never looked at.
        status = "TIMING_FAIL"
    elif not stage_ok or missing:
        status = "BUILD_FAIL"
    elif any(name.startswith("timing_") or name in {"routed_timing_report", "drc_report"} for name in failed_checks):
        status = "TIMING_FAIL"
    else:
        status = "BUILD_FAIL"
    return {"status": status, "missing": missing, "validation": validation_value}


def submit(package_dir: Path, root: Path, manifest: dict[str, Any], state_dir: Path, adapter_name: str, allow_dirty: bool = False, timeout_s: int | None = None, run_id: str | None = None, skip_board: bool = False) -> dict[str, Any]:
    validate_manifest(manifest)
    source = source_identity(root)
    requested_commit = manifest.get("source_commit")
    if requested_commit and requested_commit not in ("HEAD", source["commit"]):
        raise HarnessError(f"source_commit {requested_commit!r} does not match current HEAD {source['commit']}")
    if source["dirty"] and not allow_dirty:
        raise HarnessError("working tree is dirty; commit the candidate or pass --allow-dirty")
    p = plan(package_dir, root, manifest, adapter_name)
    profile = load_profile(package_dir, manifest["target"])
    adapter = adapter_for(adapter_name)
    run_id = run_id or (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    run_dir = state_dir / "runs" / run_id
    if run_dir.exists():
        raise HarnessError(f"run already exists: {run_id}")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    resource_before = host_resource_snapshot(state_dir)
    environment = environment_snapshot(root, profile, adapter.name)
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "PENDING",
        "created_at": started,
        "adapter": adapter.name,
        "manifest": manifest,
        "source": source,
        "profile": profile.data,
        "profile_hash": hashlib.sha256(json.dumps(profile.data, sort_keys=True).encode()).hexdigest(),
        "run_spec": p,
        "environment": environment,
        "run_dir": str(run_dir),
        "controller": {**_process_identity(os.getpid()), "kind": "harness-controller"},
        "resource_before": resource_before,
    }
    dump_json(run_dir / "run.json", run)
    dump_json(run_dir / "manifest.json", manifest)
    dump_json(run_dir / "environment.json", environment)
    try:
        if source["dirty"] and allow_dirty:
            (run_dir / "source.diff").write_text(_git(root, "diff", "--binary", "HEAD"), encoding="utf-8")
            snapshot_dir = run_dir / "source_snapshot"
            for entry in source.get("untracked_inventory", []):
                relative = Path(str(entry.get("path", "")))
                source_path = root / relative
                if not source_path.is_file() or relative.parts and relative.parts[0] in {".tritonchip-fpga-harness", "u50_build"}:
                    continue
                destination = snapshot_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        run["status"] = "RUNNING"
        dump_json(run_dir / "run.json", run)
        measurement: dict[str, Any] | None = None
        measurement_error: str | None = None
        board_status: str | None = None
        board_requested = manifest["mode"] == "full" and bool(manifest.get("board")) and not skip_board
        board_checks = board_preflight(root=root, profile=profile, manifest=manifest) if board_requested else []
        if any(not check["ok"] for check in board_checks):
            # The board stage could not start on this host.  Say so now: the
            # link in front of it takes hours and would end in the same place.
            result = {"schema_version": 1, "run_id": run_id, "status": "ENV_FAIL", "source_commit": source["commit"], "target": manifest["target"], "mode": manifest["mode"], "stages": [], "missing_artifacts": [], "artifact_validation": None, "measurement": None, "measurement_error": None, "board_preflight": board_checks, "error": "the board stage cannot start on this host: " + "; ".join(f"{c['name']} ({c['detail']})" for c in board_checks if not c["ok"]), "resource_before": resource_before, "resource_after": host_resource_snapshot(state_dir), "elapsed_s": time.time() - started}
            dump_json(run_dir / "result.json", result)
            run.update({"status": "ENV_FAIL", "finished_at": time.time(), "error": result["error"]})
            dump_json(run_dir / "run.json", run)
            return result
        if manifest["mode"] == "ooc" and adapter_name != "mock":
            # OOC concurrency is deliberately user-configured through the
            # submitted experiment; profiles no longer reserve phantom OOC
            # slots. Full implementations still use the host-wide lease.
            stages = _configured_ooc(root, run_dir, manifest, timeout_s)
            verdict = _implementation_verdict(run_dir, profile, manifest, stages)
        else:
            lease_context = lease(state_dir, "full") if manifest["mode"] == "full" else contextlib.nullcontext()
            with lease_context:
                stages = adapter.run(root, run_dir, profile, manifest["mode"], timeout_s)
                verdict = _implementation_verdict(run_dir, profile, manifest, stages)
                # Programming and board execution are a second, explicit gate
                # after link validation.  They run under the lease the build
                # took, not a new one, so a second worker cannot replace the
                # image between the link, conformance and the ArchBench run.
                # Manifests without ``board`` keep the build-only behaviour.
                if board_requested and verdict["status"] == "PASS":
                    try:
                        measurement = measure_board(
                            root=root, run_dir=run_dir, profile=profile,
                            manifest=manifest, timeout_s=timeout_s,
                            software=software_identity(root, source["commit"], source),
                        )
                    except MeasurementError as exc:
                        measurement, measurement_error, board_status = exc.partial, str(exc), exc.status
                    except (OSError, ValueError) as exc:
                        measurement_error, board_status = str(exc), "BOARD_FAIL"
                    if measurement:
                        stages.extend(measurement.get("stages", []))
        final_status = board_status or verdict["status"]
        missing = verdict["missing"]
        validation_value = verdict["validation"]
        result = {"schema_version": 1, "run_id": run_id, "status": final_status, "source_commit": source["commit"], "target": manifest["target"], "mode": manifest["mode"], "stages": stages, "missing_artifacts": missing, "artifact_validation": validation_value, "measurement": measurement, "measurement_error": measurement_error, "board_deferred": bool(skip_board and manifest.get("board")), "artifacts": {str(p.relative_to(run_dir)): file_sha256(p) for p in run_dir.rglob("*") if p.is_file() and p.parent.name == "artifacts"}, "resource_before": resource_before, "resource_after": host_resource_snapshot(state_dir), "elapsed_s": time.time() - started}
        dump_json(run_dir / "result.json", result)
        run.update({"status": result["status"], "finished_at": time.time()})
        dump_json(run_dir / "run.json", run)
        return result
    except Exception as exc:
        result = {"schema_version": 1, "run_id": run_id, "status": "FAIL", "error": str(exc), "source_commit": source["commit"], "target": manifest["target"], "mode": manifest["mode"], "elapsed_s": time.time() - started}
        dump_json(run_dir / "result.json", result)
        run.update({"status": "FAIL", "finished_at": time.time(), "error": str(exc)})
        dump_json(run_dir / "run.json", run)
        raise


def status(state_dir: Path, run_id: str) -> dict[str, Any]:
    path = state_dir / "runs" / run_id / "run.json"
    if not path.is_file():
        raise HarnessError(f"unknown run {run_id}")
    run = load_json(path)
    if run.get("status") == "RUNNING":
        controller = run.get("controller", {})
        if controller and not _same_process(controller):
            run["status"] = "ORPHANED"
            run["recovery_required"] = True
            dump_json(path, run)
    result = state_dir / "runs" / run_id / "result.json"
    if result.is_file():
        run["result"] = load_json(result)
    return run


def measure_existing(
    package_dir: Path,
    root: Path,
    state_dir: Path,
    run_id: str,
    timeout_s: int | None = None,
    *,
    board_manifest: dict[str, Any] | None = None,
    allow_dirty: bool = False,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Run the board stage for an already-linked Harness run.

    This is useful when implementation and board-access windows are managed
    separately.  It reuses the exact XCLBIN recorded by the run; callers
    cannot accidentally measure a different artifact by pointing at a
    convenient build directory.

    The board configuration is the one the run was submitted with.  A run
    submitted without one can be given it here (``board_manifest``): that
    manifest must name the same candidate, target and mode, and may only name
    the commit the run was built from.

    ``root`` supplies the runtime, conformance suite and ArchBench that drive
    the card, and it may have moved on since the build.  It must be clean, and
    it may differ from the implemented commit only below ``HARNESS_TREE``;
    anything else needs the explicit override and is recorded as such.
    """
    run_dir = state_dir / "runs" / run_id
    run_path = run_dir / "run.json"
    result_path = run_dir / "result.json"
    if not run_path.is_file() or not result_path.is_file():
        raise HarnessError(f"run {run_id} has no completed implementation result")
    run = load_json(run_path)
    result = load_json(result_path)
    manifest = run.get("manifest")
    if not isinstance(manifest, dict):
        raise HarnessError(f"run {run_id} recorded no manifest")
    implemented_commit = (run.get("source") or {}).get("commit") or result.get("source_commit")
    board_config_source = "submit"
    if board_manifest is not None:
        validate_manifest(board_manifest)
        if not isinstance(board_manifest.get("board"), dict):
            raise HarnessError("the manifest given to measure has no board block")
        for key in ("candidate_id", "target", "mode"):
            if board_manifest.get(key) != manifest.get(key):
                raise HarnessError(
                    f"the manifest given to measure names {key} {board_manifest.get(key)!r}; "
                    f"run {run_id} was built as {manifest.get(key)!r}"
                )
        wanted = board_manifest.get("source_commit")
        if wanted and wanted != implemented_commit:
            raise HarnessError(
                f"the manifest given to measure names source_commit {wanted!r}; "
                f"run {run_id} was built from {implemented_commit!r}"
            )
        manifest = dict(manifest, board=board_manifest["board"])
        board_config_source = "measure"
    if not manifest.get("board"):
        raise HarnessError(
            f"run {run_id} has no board measurement configuration; "
            "pass measure --manifest with a board block"
        )
    if result.get("mode") != "full" or result.get("status") not in {"PASS", "BOARD_FAIL", "ARCHBENCH_FAIL"}:
        raise HarnessError(f"run {run_id} is not a successful full implementation")
    profile = load_profile(package_dir, str(manifest["target"]))
    current_profile_hash = hashlib.sha256(
        json.dumps(profile.data, sort_keys=True).encode()
    ).hexdigest()
    recorded_profile_hash = run.get("profile_hash")
    if recorded_profile_hash and recorded_profile_hash != current_profile_hash:
        raise HarnessError(
            f"run {run_id} profile changed after implementation; "
            "rebuild the implementation before measuring it"
        )
    software = software_identity(root, implemented_commit)
    if software["dirty"] and not allow_dirty:
        raise HarnessError(
            "working tree is dirty; commit the checkout that drives the card or pass --allow-dirty"
        )
    if software["relation"] in {"differs", "unknown"} and not allow_source_mismatch:
        raise HarnessError(
            f"this checkout ({software['commit']}) is not the commit run {run_id} was built from "
            f"({implemented_commit}) and differs outside {HARNESS_TREE} "
            f"(relation: {software['relation']}); check out the implemented commit "
            "or pass --allow-source-mismatch"
        )
    blocked = [c for c in board_preflight(root=root, profile=profile, manifest=manifest) if not c["ok"]]
    if blocked:
        # Nothing ran, so nothing is written against the run.
        raise HarnessError(
            "the board stage cannot start on this host: "
            + "; ".join(f"{c['name']} ({c['detail']})" for c in blocked)
        )
    if board_config_source == "measure":
        dump_json(run_dir / "board_manifest.json", board_manifest)
    measurement: dict[str, Any] | None = None
    error: str | None = None
    final_status = "PASS"
    with lease(state_dir, "full"):
        try:
            measurement = measure_board(root=root, run_dir=run_dir, profile=profile,
                                         manifest=manifest, timeout_s=timeout_s,
                                         software=software)
        except MeasurementError as exc:
            measurement, error, final_status = exc.partial, str(exc), exc.status
        except (OSError, ValueError) as exc:
            error, final_status = str(exc), "BOARD_FAIL"
    # The implementation stages keep their place; the board stages of an
    # earlier attempt are replaced by this one's.
    board_stage_names = {"program_check", "program", "conformance", "archbench"}
    stages = [s for s in result.get("stages", []) if s.get("name") not in board_stage_names]
    stages.extend((measurement or {}).get("stages", []))
    result.update({
        "stages": stages,
        "measurement": measurement,
        "measurement_error": error,
        "status": final_status,
        "board_deferred": False,
        "board_config_source": board_config_source,
    })
    dump_json(result_path, result)
    run.update({"status": final_status, "finished_at": time.time()})
    if error:
        run["error"] = error
    else:
        run.pop("error", None)
    dump_json(run_path, run)
    return result


def reconcile(state_dir: Path, run_id: str, *, terminate: bool = False) -> dict[str, Any]:
    """Reconcile a run after controller loss and optionally kill its vendor group."""
    run_dir = state_dir / "runs" / run_id
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise HarnessError(f"unknown run {run_id}")
    run = load_json(run_path)
    # execute_command leaves the record two levels above its log: the run
    # directory for a build stage, logs/ for a board stage.
    active = None
    for active_path in (run_dir / "active_process.json", run_dir / "logs" / "active_process.json"):
        if active_path.is_file():
            active = load_json(active_path)
            break
    alive = bool(active and _same_process(active))
    terminated = False
    if terminate and alive:
        try:
            os.killpg(int(active["pgid"]), signal.SIGTERM)
            terminated = True
        except (OSError, KeyError, ValueError):
            terminated = False
    if run.get("status") == "RUNNING" and not alive:
        run.update({"status": "ORPHANED", "recovery_required": True, "reconciled_at": time.time()})
        dump_json(run_path, run)
    return {"run_id": run_id, "status": run.get("status"), "active_process": active, "process_alive": alive, "terminated": terminated}


def export_evidence(state_dir: Path, run_id: str, output: Path) -> Path:
    run_dir = state_dir / "runs" / run_id
    if not (run_dir / "result.json").is_file():
        raise HarnessError(f"run {run_id} has no result")
    output = output.resolve()
    if output == run_dir or run_dir in output.parents:
        raise HarnessError("evidence output must be outside the run directory")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / run_id
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(run_dir, destination, ignore=shutil.ignore_patterns("active_process.json"))
    files: dict[str, str] = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "evidence-manifest.json":
            files[str(path.relative_to(destination))] = file_sha256(path)
    dump_json(destination / "evidence-manifest.json", {"schema_version": 1, "run_id": run_id, "files": files})
    (destination / "README.txt").write_text(
        "This directory is a self-contained FPGA harness evidence packet. "
        "The hashes in evidence-manifest.json cover logs, reports, source snapshot, "
        "XO/XCLBIN and run metadata.\n", encoding="utf-8"
    )
    return destination
