"""Manifest/result validation and deterministic JSON helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class SchemaError(ValueError):
    """A candidate manifest or result failed the harness contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must contain a JSON object")
    return value


def dump_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError("manifest must be a JSON object")
    required = ("candidate_id", "target", "mode")
    missing = [k for k in required if not value.get(k)]
    if missing:
        raise SchemaError("manifest missing: " + ", ".join(missing))
    if value["mode"] not in ("full", "ooc"):
        raise SchemaError("mode must be 'full' or 'ooc'")
    for key in ("candidate_id", "target", "mode"):
        if not isinstance(value[key], str):
            raise SchemaError(f"{key} must be a string")
    if "clock_mhz" in value and (not isinstance(value["clock_mhz"], (int, float)) or value["clock_mhz"] <= 0):
        raise SchemaError("clock_mhz must be a positive number")
    if "source_commit" in value and not isinstance(value["source_commit"], str):
        raise SchemaError("source_commit must be a string")
    if "schema_version" in value and value["schema_version"] != 1:
        raise SchemaError("unsupported manifest schema_version; expected 1")
    if "hypothesis" in value and not isinstance(value["hypothesis"], str):
        raise SchemaError("hypothesis must be a string")
    if "image" in value:
        image = value["image"]
        # A device profile to build instead of the RTL defaults (image.py).
        if not isinstance(image, dict) or set(image) != {"profile"} or not isinstance(image["profile"], str) \
                or not image["profile"]:
            raise SchemaError("image must be {\"profile\": \"<a profile of spec/r1_profiles.json>\"}")
        if value["mode"] != "full":
            raise SchemaError("image is for a full build")
        if isinstance(value.get("board"), dict) and value["board"].get("profile") not in (None, image["profile"]):
            raise SchemaError("board.profile must be the image's profile")
    if "board" in value:
        board = value["board"]
        if not isinstance(board, dict):
            raise SchemaError("board must be an object")
        unknown_board = sorted(set(board) - {"backend", "profile", "bdf", "program", "program_script", "conformance", "archbench"})
        if unknown_board:
            raise SchemaError("board has unsupported fields: " + ", ".join(unknown_board))
        for key in ("backend", "profile", "bdf", "program_script"):
            if key in board and not isinstance(board[key], str):
                raise SchemaError(f"board.{key} must be a string")
        if not board.get("profile"):
            raise SchemaError("board.profile is required: name the device profile of the image")
        for key in ("program", "conformance"):
            if key in board and not isinstance(board[key], (bool, dict)):
                raise SchemaError(f"board.{key} must be a boolean or object")
        if "program" in board and isinstance(board["program"], dict):
            program = board["program"]
            unknown = sorted(set(program) - {"command", "check_command"})
            if unknown:
                raise SchemaError("board.program has unsupported fields: " + ", ".join(unknown))
            for key in ("command", "check_command"):
                if key in program and (not isinstance(program[key], list) or not program[key] or
                                       not all(isinstance(item, str) and item for item in program[key])):
                    raise SchemaError(f"board.program.{key} must be a non-empty argv list")
        if "conformance" in board and isinstance(board["conformance"], dict):
            conf = board["conformance"]
            unknown = sorted(set(conf) - {"command", "result_file"})
            if unknown:
                raise SchemaError("board.conformance has unsupported fields: " + ", ".join(unknown))
            if "command" in conf and (not isinstance(conf["command"], list) or not conf["command"] or
                                       not all(isinstance(item, str) and item for item in conf["command"])):
                raise SchemaError("board.conformance.command must be a non-empty argv list")
            if "result_file" in conf and not isinstance(conf["result_file"], str):
                raise SchemaError("board.conformance.result_file must be a string")
        if "archbench" in board:
            bench = board["archbench"]
            if not isinstance(bench, dict):
                raise SchemaError("board.archbench must be an object")
            unknown = sorted(set(bench) - {"command", "result_file", "tiers", "points", "jobs", "counters", "timeout_s", "python"})
            if unknown:
                raise SchemaError("board.archbench has unsupported fields: " + ", ".join(unknown))
            if "command" in bench and (not isinstance(bench["command"], list) or not bench["command"] or
                                        not all(isinstance(item, str) and item for item in bench["command"])):
                raise SchemaError("board.archbench.command must be a non-empty argv list")
            if "result_file" in bench and not isinstance(bench["result_file"], str):
                raise SchemaError("board.archbench.result_file must be a string")
            if "python" in bench and (not isinstance(bench["python"], str) or not bench["python"]):
                raise SchemaError("board.archbench.python must be an interpreter path")
            if "tiers" in bench and (not isinstance(bench["tiers"], list) or not bench["tiers"] or
                                      not all(isinstance(item, str) and item for item in bench["tiers"])):
                raise SchemaError("board.archbench.tiers must be a non-empty string list")
            if "points" in bench and (not isinstance(bench["points"], list) or not bench["points"] or
                                       not all(isinstance(item, str) and item for item in bench["points"])):
                raise SchemaError("board.archbench.points must be a non-empty list of point ids")
            if "points" in bench and "tiers" in bench:
                raise SchemaError("board.archbench takes tiers or points, not both")
            if "jobs" in bench and (not isinstance(bench["jobs"], int) or bench["jobs"] < 1):
                raise SchemaError("board.archbench.jobs must be a positive integer")
            if "counters" in bench and not isinstance(bench["counters"], bool):
                raise SchemaError("board.archbench.counters must be boolean")
            if "timeout_s" in bench and (not isinstance(bench["timeout_s"], (int, float)) or bench["timeout_s"] <= 0):
                raise SchemaError("board.archbench.timeout_s must be positive")
    if "ooc" in value:
        if not isinstance(value["ooc"], dict):
            raise SchemaError("ooc must be an object")
        command = value["ooc"].get("command")
        artifacts = value["ooc"].get("artifacts")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise SchemaError("ooc.command must be a non-empty argv list")
        if not isinstance(artifacts, list) or not artifacts or not all(isinstance(item, str) and item for item in artifacts):
            raise SchemaError("ooc.artifacts must be a non-empty string list")
        for key in ("top", "script", "clock_mhz", "result_file"):
            if key in value["ooc"] and not isinstance(value["ooc"][key], (str, int, float)):
                raise SchemaError(f"ooc.{key} has an invalid type")
        unknown_ooc = sorted(set(value["ooc"]) - {"top", "script", "clock_mhz", "result_file", "command", "artifacts"})
        if unknown_ooc:
            raise SchemaError("ooc has unsupported fields: " + ", ".join(unknown_ooc))
    # Keep the checked-in JSON schema and the executable contract aligned for
    # the fields that affect orchestration.  This deliberately avoids a
    # third-party jsonschema dependency on FPGA hosts.
    allowed = {"schema_version", "candidate_id", "target", "mode", "clock_mhz", "source_commit", "hypothesis", "ooc", "board", "image"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SchemaError("manifest has unsupported fields: " + ", ".join(unknown))
    return value


WORKER_OPERATIONS = ("doctor", "submit", "status", "export", "measure")


def validate_worker_request(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the JSON envelope exchanged by the SSH controller/worker."""
    if not isinstance(value, dict):
        raise SchemaError("worker request must be a JSON object")
    if value.get("schema_version", 1) != 1:
        raise SchemaError("unsupported worker schema_version; expected 1")
    required = ("operation", "board", "adapter", "manifest")
    missing = [key for key in required if key not in value]
    if missing:
        raise SchemaError("worker request missing: " + ", ".join(missing))
    if value["operation"] not in WORKER_OPERATIONS:
        raise SchemaError("worker operation must be one of " + ", ".join(WORKER_OPERATIONS))
    if value["board"] not in ("u50", "u280"):
        raise SchemaError("worker board must be 'u50' or 'u280'")
    # `mock` plays the build without Vivado; it exists to exercise a split
    # pipeline end to end, and its runs say so (run.json adapter).
    if value["adapter"] not in ("u50", "u280", "mock"):
        raise SchemaError("worker adapter must be 'u50', 'u280' or 'mock'")
    if value["adapter"] != "mock" and value["adapter"] != value["board"]:
        raise SchemaError("worker adapter must match board")
    validate_manifest(value["manifest"])
    for key in ("allow_dirty", "skip_board", "detach", "light", "allow_source_mismatch"):
        if key in value and not isinstance(value[key], bool):
            raise SchemaError(f"worker {key} must be boolean")
    if "timeout" in value and (not isinstance(value["timeout"], int) or value["timeout"] <= 0):
        raise SchemaError("worker timeout must be a positive integer")
    if "state_dir" in value and (not isinstance(value["state_dir"], str) or not value["state_dir"]):
        raise SchemaError("worker state_dir must be a non-empty string")
    # A run id names a directory under runs/: a plain name, never a path.
    if "run_id" in value and (not isinstance(value["run_id"], str)
                              or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value["run_id"])):
        raise SchemaError(f"worker run_id {value.get('run_id')!r} is not a plain name")
    if "scope" in value and value["scope"] not in ("build", "board", "source"):
        raise SchemaError("worker scope must be 'build', 'board' or 'source'")
    if "step" in value and value["step"] not in ("submit", "measure"):
        raise SchemaError("worker step must be 'submit' or 'measure'")
    if value["operation"] in ("status", "export", "measure") and "run_id" not in value:
        raise SchemaError(f"worker operation {value['operation']} needs run_id")
    unknown = sorted(set(value) - {
        "schema_version", "operation", "board", "adapter", "manifest",
        "state_dir", "allow_dirty", "skip_board", "timeout", "run_id",
        "scope", "detach", "light", "step", "allow_source_mismatch",
    })
    if unknown:
        raise SchemaError("worker request has unsupported fields: " + ", ".join(unknown))
    return value
