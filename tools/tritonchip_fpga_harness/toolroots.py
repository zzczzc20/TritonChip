"""Where a host keeps the vendor tools a target is built with.

A target describes a card (`targets/*.json`); where Vitis is installed is a
property of the host that runs the flow. The target's ``vitis_root`` and
``platforms_root`` are its defaults -- the card host's paths -- and a host that
keeps them elsewhere says so with ``FPGA_HARNESS_VITIS_ROOT`` and
``FPGA_HARNESS_PLATFORMS_ROOT``. The build host sets these through its entry in
a hosts file (see ``hosts.py``); a run records which roots it used and where
they came from.

A target may also list the Vitis versions its card accepts images from
(``vitis_versions``). The version is read from the root's last path component,
which is how AMD lays out an installation (``.../Vitis/2022.2``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

VITIS_ENV = "FPGA_HARNESS_VITIS_ROOT"
PLATFORMS_ENV = "FPGA_HARNESS_PLATFORMS_ROOT"
_VERSION = re.compile(r"^\d{4}\.\d+$")


def tool_roots(data: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The Vitis and platform roots a run on this host uses for a target."""
    env = os.environ if environ is None else environ
    vitis_env = env.get(VITIS_ENV)
    platforms_env = env.get(PLATFORMS_ENV)
    vitis = vitis_env or data.get("vitis_root")
    platforms = platforms_env or data.get("platforms_root") or "/opt/xilinx/platforms"
    return {
        "vitis_root": str(vitis) if vitis else None,
        "platforms_root": str(platforms),
        "vitis_source": "host" if vitis_env else ("target" if data.get("vitis_root") else None),
        "platforms_source": "host" if platforms_env else ("target" if data.get("platforms_root") else "default"),
        "vitis_version": vitis_version(vitis) if vitis else None,
    }


def vitis_version(root: str | Path) -> str | None:
    """``2022.2`` for ``/opt/.../Vitis/2022.2``; ``None`` when the path does not say."""
    name = Path(str(root)).name
    return name if _VERSION.fullmatch(name) else None


__all__ = ["PLATFORMS_ENV", "VITIS_ENV", "tool_roots", "vitis_version"]
