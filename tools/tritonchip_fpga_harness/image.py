"""A device profile as the image's parameters.

The flow builds whatever the wrapper's parameter defaults elaborate to unless
TNCA_IMAGE_CONFIG names a Tcl file of parameters (flow/fpga/package_xo.tcl).
This module writes that file for a profile of spec/r1_profiles.json, the way
simulation reads the profile: the values in rtl/common/tnca_uarch_defaults.vh
first, then the profile's `elaboration` and `implementation.rtl_parameters`.
Every switch the wrapper exposes is set explicitly, so a switch the profile
does not name is built at simulation's default and not at the wrapper's.

A parameter the profile names but the wrapper cannot carry is refused, never
dropped: the image would otherwise be another machine than the one simulated.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

WRAPPER = "rtl/board/tnca_u50_xo.v"
DEFAULTS = "rtl/common/tnca_uarch_defaults.vh"
PROFILES = "spec/r1_profiles.json"
# The wrapper's own parameters, which are the platform's and not the machine's.
PLATFORM = frozenset({"S_AXI_ADDR_W", "ADDR_BITS", "WORDS", "ADDR_W", "DEV_DIV"})
# Values the FPGA top derives from others: allowed in a profile when they agree.
DERIVED = frozenset({"PHYS_VREGS"})

_WRAPPER_PARAM = re.compile(r"^\s*parameter\s+([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+)\s*,?", re.M)
_DEFINE = re.compile(r"^\s*`define\s+TNCA_(?:IF|UARCH)_([A-Z][A-Z0-9_]*)\s+(-?\d+)\s*$", re.M)


class ImageError(ValueError):
    """The profile cannot be built as an image by this checkout's flow."""


def wrapper_parameters(root: Path) -> dict[str, int]:
    text = (root / WRAPPER).read_text(encoding="utf-8")
    header = text[text.index("module tnca_u50_xo"):text.index(") (")] if "module tnca_u50_xo" in text else ""
    return {name: int(value) for name, value in _WRAPPER_PARAM.findall(header) if name not in PLATFORM}


def image_parameters(root: Path, profile: str) -> dict[str, int]:
    """The wrapper parameters that build ``profile``, every machine switch set explicitly."""
    profiles = json.loads((root / PROFILES).read_text(encoding="utf-8"))["profiles"]
    if profile not in profiles:
        raise ImageError(f"no profile {profile!r} in {PROFILES}")
    spec = profiles[profile]
    if int(spec.get("num_cores", 1)) != 1:
        raise ImageError(f"profile {profile!r} has {spec.get('num_cores')} cores; the image is single-core")
    stated: dict[str, int] = dict(spec.get("elaboration") or {})
    stated.update((spec.get("implementation") or {}).get("rtl_parameters") or {})
    exposed = wrapper_parameters(root)
    values = {name: int(value) for name, value in _DEFINE.findall((root / DEFAULTS).read_text(encoding="utf-8"))
              if name in exposed}
    values.update({name: int(value) for name, value in stated.items() if name in exposed})
    missing = sorted(name for name in exposed if name not in values)
    if missing:
        raise ImageError(f"no value for {', '.join(missing)}: neither {DEFAULTS} nor the profile sets it")
    if "PHYS_VREGS" in stated and stated["PHYS_VREGS"] != values["NUM_RCTX"] * values["VREGS_PER_RCTX"]:
        raise ImageError(f"profile {profile!r} sets PHYS_VREGS {stated['PHYS_VREGS']}, the image derives "
                         f"{values['NUM_RCTX'] * values['VREGS_PER_RCTX']}")
    cannot = sorted(name for name in stated if name not in exposed and name not in DERIVED)
    if cannot:
        raise ImageError(f"profile {profile!r} sets {', '.join(cannot)}, which the image cannot carry "
                         f"(not a parameter of {WRAPPER})")
    return dict(sorted(values.items()))


def write_image_config(root: Path, profile: str, out_dir: Path) -> dict[str, Any]:
    """Write image_config.tcl (for TNCA_IMAGE_CONFIG) and image_config.json (the record)."""
    params = image_parameters(root, profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcl = out_dir / "image_config.tcl"
    tcl.write_text("set TNCA_IMAGE_PARAMS [dict create " + " ".join(f"{k} {v}" for k, v in params.items()) + "]\n",
                   encoding="utf-8")
    record = {
        "schema_version": 1,
        "profile": profile,
        "parameters": params,
        "profiles_sha256": hashlib.sha256((root / PROFILES).read_bytes()).hexdigest(),
        "defaults_sha256": hashlib.sha256((root / DEFAULTS).read_bytes()).hexdigest(),
        "tcl_sha256": hashlib.sha256(tcl.read_bytes()).hexdigest(),
    }
    (out_dir / "image_config.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def applied(package_log: str, params: dict[str, int]) -> list[str]:
    """The parameters the packager did not report setting to their value (`TNCA_IMAGE_PARAM name=value`)."""
    seen = dict(re.findall(r"TNCA_IMAGE_PARAM ([A-Z][A-Z0-9_]*)=(-?\d+)", package_log))
    return [name for name, value in params.items() if seen.get(name) != str(value)]


__all__ = ["ImageError", "applied", "image_parameters", "wrapper_parameters", "write_image_config"]
