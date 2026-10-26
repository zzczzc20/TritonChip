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


def _checkout_devices(root: Path):
    """The checkout's own python/tnca/devices.py: the image is the one the
    checkout's generator describes, not one an installed package would."""
    import importlib.util
    import sys
    pkg = root / "python" / "tnca"
    sources = [pkg / f for f in ("devices.py", "profiles.py", "errors.py")]
    if not all(f.is_file() for f in sources):
        raise ImageError(f"{root}: this checkout has no python/tnca/devices.py (no device descriptions)")
    # Keyed by the code as well as the place: a checkout changed in place is
    # another generator, and must not be answered by the one loaded before.
    digest = hashlib.sha256(str(root.resolve()).encode() + b"".join(f.read_bytes() for f in sources)).hexdigest()
    name = f"_tnca_devices_{digest[:16]}"
    if f"{name}.devices" in sys.modules:
        return sys.modules[f"{name}.devices"]
    # devices.py imports the checkout's profiles module; load both as a
    # private package so an installed tnca is never consulted.
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(pkg)]
    sys.modules[name] = package
    sub = importlib.util.spec_from_file_location(f"{name}.devices", pkg / "devices.py")
    module = importlib.util.module_from_spec(sub)
    sys.modules[f"{name}.devices"] = module
    try:
        sub.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- the checkout's code, reported, not trusted
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
        raise ImageError(f"{root}: python/tnca/devices.py does not load: {exc}") from exc
    return module


def device_description(root: Path, device: str) -> dict[str, Any]:
    """A device description of the checkout (spec/devices), validated by its generator."""
    devices = _checkout_devices(root)
    folder = (root / "spec" / "devices").resolve()
    path = root / "spec" / "devices" / f"{device}.json"
    if path.resolve().parent != folder:
        raise ImageError(f"device {device!r} is not a description in spec/devices")
    try:
        # The generator reads profiles through tnca.profiles, which prefers a
        # staged copy (python/tnca/data, git-ignored, `make r2-stage-data`) to
        # the checkout's spec. A stale one would build another machine than the
        # commit's under the commit's name, and no dirty check would see it.
        staged = devices.profiles.spec()
        own = json.loads((root / PROFILES).read_text(encoding="utf-8"))
    except (OSError, ValueError, devices.profiles.ProfileMismatch) as exc:
        raise ImageError(f"{root}: the profiles cannot be read: {exc}") from exc
    if staged != own:
        raise ImageError(f"{root}: python/tnca/data/{PROFILES} is not {PROFILES}; restage it "
                         "(make r2-stage-data) or remove it")
    try:
        return devices.load(path)
    except devices.DeviceError as exc:
        raise ImageError(f"device {device!r}: {exc}") from exc


def device_parameters(root: Path, device: str) -> dict[str, int]:
    """The image parameters a device description generates."""
    return dict(sorted(_checkout_devices(root).image_params(device_description(root, device)).items()))


def write_device_config(root: Path, device: str, out_dir: Path) -> dict[str, Any]:
    """image_config.tcl and image_config.json for a device description."""
    desc = device_description(root, device)
    devices = _checkout_devices(root)
    params = dict(sorted(devices.image_params(desc).items()))
    out_dir.mkdir(parents=True, exist_ok=True)
    tcl = out_dir / "image_config.tcl"
    tcl.write_text(devices.image_tcl(desc), encoding="utf-8")
    source = root / "spec" / "devices" / f"{device}.json"
    record = {
        "schema_version": 1,
        "device": device,
        "profile": desc["per_core_profile"],
        "platform": desc["platform"],
        "parameters": params,
        "fingerprint": devices.fingerprint(desc),
        "device_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "profiles_sha256": hashlib.sha256((root / PROFILES).read_bytes()).hexdigest(),
        "tcl_sha256": hashlib.sha256(tcl.read_bytes()).hexdigest(),
    }
    (out_dir / "image_config.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def applied(package_log: str, params: dict[str, int]) -> list[str]:
    """The parameters the packager did not report setting to their value (`TNCA_IMAGE_PARAM name=value`)."""
    seen = dict(re.findall(r"TNCA_IMAGE_PARAM ([A-Z][A-Z0-9_]*)=(-?\d+)", package_log))
    return [name for name, value in params.items() if seen.get(name) != str(value)]


__all__ = ["ImageError", "applied", "device_description", "device_parameters", "image_parameters",
           "wrapper_parameters", "write_device_config", "write_image_config"]
