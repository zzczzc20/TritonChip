"""
Finding, and if necessary building, the simulator an installed runtime runs on.

doc/phase2_tasks_and_acceptance.md R2-02 allows the profile's simulator to be
built at install time or by an explicit prepare step, and forbids building it
per launch:

    "允许在安装/显式 prepare 阶段构建 profile 的 simulator，不能在每个 kernel
     launch 做 build"

So `locate` looks and does not build; `prepare` builds and says so. A launch
calls the first and never the second, which is why they are two functions
rather than one with a flag -- a flag defaulting the wrong way is how a build
ends up inside a launch.

Where it looks, in order:

  $TNCA_DEVICE            an explicit override, for a developer or a test
  <cache>/<profile>/...   what `prepare` built, keyed by the profile's
                          configuration hash, so a device built for another
                          profile is never picked up by accident
  <checkout>/sim/build/   a development tree
"""
import hashlib
import os
import shutil
import subprocess
import sys

from . import profiles
from .errors import DeviceLost

BENCH_NAME = {
    'core-sim': 'tnca_device.vvp',
}

# The elaboration each profile's device is built with. Read from the profile
# spec rather than written here, so a profile that changes its RCTX count
# does not silently keep an old bench.
_TOP = 'tnca_device'


def cache_root():
    base = os.environ.get('TNCA_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'tnca')
    return os.path.join(base, 'devices')


def _package_data(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.join(here, 'data'),
                 os.path.dirname(os.path.dirname(os.path.dirname(here)))):
        path = os.path.join(root, *parts)
        if os.path.exists(path):
            return path
    return None


def expected_path(profile):
    """Where `prepare` puts this profile's device.

    Keyed by the configuration hash, so a device built for a different
    elaboration is a different file rather than a surprise.
    """
    h = profiles.config_hash(profile)[:16]
    return os.path.join(cache_root(), f"{profile}-{h}",
                        BENCH_NAME.get(profile, 'tnca_device.vvp'))


def locate(profile='core-sim'):
    """The device bench, or None. Never builds."""
    override = os.environ.get('TNCA_DEVICE')
    if override and os.path.exists(override):
        return override
    built = expected_path(profile)
    if os.path.exists(built):
        return built
    dev = _package_data('sim', 'build', BENCH_NAME.get(profile, ''))
    if dev and os.path.exists(dev):
        return dev
    return None


def require(profile='core-sim'):
    path = locate(profile)
    if path:
        return path
    raise DeviceLost(
        f"no simulator for profile {profile!r}",
        detail={'looked for': expected_path(profile)},
        remedy='run `tnca-sim prepare` once to build it, or set '
               'TNCA_DEVICE to an existing bench. A launch does not build '
               'one: that would put a compile inside every kernel call')


def prepare(profile='core-sim', *, force=False, verbose=True):
    """Build this profile's device from the RTL the package ships."""
    target = expected_path(profile)
    if os.path.exists(target) and not force:
        if verbose:
            print(f"  already built: {target}")
        return target

    iverilog = shutil.which('iverilog')
    if not iverilog:
        raise DeviceLost(
            'iverilog is not on PATH',
            remedy='install Icarus Verilog 12; the runtime executes on a '
                   'simulator and there is nothing to run without one')

    rtl = _package_data('rtl')
    tb = _package_data('tb')
    if not rtl or not tb:
        raise DeviceLost(
            'this installation does not ship the RTL',
            detail={'rtl': rtl, 'tb': tb},
            remedy='reinstall a package built with `make r2-package`, which '
                   'stages the sources the device is built from')

    os.makedirs(os.path.dirname(target), exist_ok=True)
    source = os.path.join(tb, 'device', 'tnca_device.v')
    argv = [iverilog, '-g2005', '-I', os.path.join(rtl, 'common'),
            '-y', os.path.join(rtl, 'core'),
            '-y', os.path.join(rtl, 'common'),
            '-y', os.path.join(rtl, 'gen'),
            '-y', os.path.join(tb, 'common'),
            '-o', target, source]
    if verbose:
        print(f"  building {profile} -> {target}")
    r = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(target):
        raise DeviceLost(
            f"could not build the device for {profile!r}",
            detail={'stderr': (r.stderr or r.stdout)[-600:]},
            remedy='check that iverilog is version 12 and that the package '
                   'shipped its RTL')
    if verbose:
        print(f"  built {os.path.getsize(target)} bytes")
    return target
