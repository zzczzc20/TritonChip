"""A backend a platform branch could have written.

R36-HAND-01. It is registered from outside the package, through the
published extension point, and it builds its device from the HARDWARE
EXPORT rather than from the repository: `tnca_shell.v` and everything
beside it in the package's `rtl/`, plus the reference binding in its
`sim/`.

That is the substitution a real board makes, with the simulator standing
in for the board: the same public top, a different thing wrapped around
it. If this file needed anything from the main tree, the export would be
incomplete -- which is the fact the rehearsal is for.
"""
import os
import shutil
import subprocess
import sys

from tnca import backend
from tnca.errors import TncaError
from tnca.transport import Transport


class ExportBackend(Transport):
    """The file-mailbox protocol, over a device built from the export.

    Subclassing the reference transport rather than reimplementing the
    protocol is the point: a platform branch inherits the host side and
    supplies the device. What it overrides is where the device comes
    from, which is the only thing that is actually different.
    """

    name = 'export-sim'
    state = backend.REFERENCE_AVAILABLE

    def capabilities(self):
        out = super().capabilities()
        out.update({'medium': 'a shared directory',
                    'reaches': 'a device built from the hardware export '
                               'package, not from the repository',
                    'built_from': self.package})
        return out


def build_device(package, out=None):
    """Elaborate the export's own reference binding.

    Nothing of ours on the command line: the file list comes from the
    package's manifest, the include path is the package's own, and the
    working directory is the package.
    """
    import json
    manifest = os.path.join(package, 'MANIFEST.json')
    if not os.path.exists(manifest):
        raise TncaError(f"{package} is not a hardware export package: "
                        f"no MANIFEST.json",
                        code='ARGUMENT_INVALID')
    with open(manifest) as fh:
        m = json.load(fh)
    iverilog = shutil.which('iverilog')
    if iverilog is None:
        raise TncaError('iverilog is not on PATH', code='TOOL_MISSING')
    out = out or os.path.join(package, 'device.vvp')
    srcs = [e['file'] for e in m['rtl']]
    srcs += [e['file'] for e in m['sim'] if e['file'].endswith('.v')]
    argv = [iverilog, '-g2005', '-I', 'rtl', '-I', 'sim',
            '-s', m['simulation_top'],
            # The banked register file, which is the profile the rest of
            # the rehearsal compiles for. A platform picks its
            # parameters here, which is what a parameter is for.
            '-P', f"{m['simulation_top']}.USE_BANKED_VRF=1",
            '-o', out] + srcs
    r = subprocess.run(argv, cwd=package, capture_output=True, text=True,
                       timeout=1800)
    if r.returncode != 0 or not os.path.exists(out):
        raise TncaError(
            f"the export would not elaborate: "
            f"{(r.stderr or r.stdout).strip()[-300:]}",
            code='DEVICE_LOST')
    return out


def install(package, name='export-sim'):
    """Register this backend. The whole extension point, in one call."""
    device = build_device(package)

    def factory(**kw):
        kw.pop('bench', None)
        b = ExportBackend(device, **kw)
        b.package = package
        return b

    backend.register(name, factory,
                     state=backend.REFERENCE_AVAILABLE,
                     what='a device built from the hardware export '
                          'package, standing in for a board')
    return name, device
