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
import warnings
import subprocess
import tempfile
import sys
import time

from . import profiles
from .errors import DeviceLost

BENCH_NAME = {
    'core-sim': 'tnca_device.vvp',
    'core-vrf-sync-v1': 'tnca_device_vrf.vvp',
    # The same two, on a back-pressuring and reordering memory. C04 and
    # C05 are specified on one, and a memory mode is a condition of the
    # run rather than a different machine -- the ISA, the resources and
    # the fingerprint are those of the profile it is derived from.
    'core-sim/slow': 'tnca_device_slow.vvp',
    'core-vrf-sync-v1/slow': 'tnca_device_vrf_slow.vvp',
    'core-vrf-sync-v2': 'tnca_device_vrf2.vvp',
    'core-vrf-sync-v2/slow': 'tnca_device_vrf2_slow.vvp',
    'core-vrf-sync-v3': 'tnca_device_vrf3.vvp',
    'core-vrf-sync-v3/slow': 'tnca_device_vrf3_slow.vvp',
    'core-vrf-sync-v4': 'tnca_device_vrf4.vvp',
    'core-vrf-sync-v4/slow': 'tnca_device_vrf4_slow.vvp',
    'core-vrf-sync-v5': 'tnca_device_vrf5.vvp',
    'core-vrf-sync-v5/slow': 'tnca_device_vrf5_slow.vvp',
    # R40-04: external memory. Every address from the heap up goes through
    # rtl/fpga/tnca_axi_bridge.v to an AXI memory, as on the U280's HBM image;
    # the architecture is v5's. Its timing is the AXI model's (+axi_* plusargs,
    # TNCA_DEVICE_PLUSARGS).
    'core-vrf-sync-v5/hbm': 'tnca_device_vrf5_hbm.vvp',
}

# Parameters a profile's device is elaborated with. Both profiles come from
# the same bench source, so what distinguishes them is here and nowhere
# else -- and a difference between them cannot be a difference in the
# testbench.
_SLOW = ['-P', 'tnca_device.MEM_LATENCY=7',
         '-P', 'tnca_device.RANDOM_STALL=1',
         '-P', 'tnca_device.MEM_REORDER=1']
# Which microarchitecture a profile's device is is no longer said here: the
# profile itself is compiled into it, as one generated header
# (profiles.uarch_header). What is left is the platform's own business, the
# memory model a variant runs on: '/slow' above, and '/hbm' below.
#
# R40-04: the AXI window of the /hbm memory mode, in 64-bit words (32 MiB).
HBM_SIM_WORDS = 1 << 22
HBM_LOCAL_LIMIT = 0x20000
_HBM = ['-P', 'tnca_device.DATA_AXI=1', '-P', f'tnca_device.AXI_WORDS={HBM_SIM_WORDS}',
        '-P', f'tnca_device.LOCAL_LIMIT={HBM_LOCAL_LIMIT}']


# R39 ablations of the hardware switches on the device simulation:
# TNCA_RTL_PARAMS="RCTX_SWITCH=0,LSU_INFLIGHT=2" elaborates the profile's
# device with these on top of its own parameters. Only timing switches --
# nothing that changes the ISA, the registers or the resources, so the
# profile's artifacts and fingerprint still describe the machine. The
# overrides are part of the cache key, and archbench records them with
# `--set` like any other knob.
RTL_SWITCHES = ('VRF_BYPASS', 'FE_NOBUBBLE', 'VRF_PREFETCH', 'RCTX_SWITCH', 'LSU_INFLIGHT',
                'FPU_OVERLAP', 'MTX_ABUF',
                # R40-04's AXI mechanisms, for an /hbm memory mode
                # (tb/device/tnca_device.v): burst lines, buffered lines,
                # lines ahead, writes in flight
                'AXI_RAM_CACHE', 'AXI_LOCAL_PIPE', 'AXI_LINE', 'AXI_LB_N', 'AXI_PF', 'AXI_WAIT_N', 'AXI_OUT_N')
# Of those, the ones that belong to the /hbm memory mode's bench rather than to
# the device: they are parameters of tb/device/tnca_device.v (its AXI bridge),
# so they go there by -P like the memory model's own, not into the header.
# Derived, not listed again: an AXI switch added above and missed here would go
# into the header and set nothing.
BENCH_SWITCHES = tuple(k for k in RTL_SWITCHES if k.startswith('AXI_'))


def rtl_overrides():
    """[(name, value)] from TNCA_RTL_PARAMS, sorted; refuses anything else."""
    raw = os.environ.get('TNCA_RTL_PARAMS', '').strip()
    out = {}
    for kv in filter(None, (x.strip() for x in raw.split(','))):
        k, _, v = kv.partition('=')
        if k not in RTL_SWITCHES or not v.strip().isdigit():
            raise ValueError(f"TNCA_RTL_PARAMS: {kv!r} is not one of "
                             f"{', '.join(RTL_SWITCHES)}=<int>")
        out[k] = int(v)
    return sorted(out.items())


def bench_params(profile):
    """The -P arguments this profile's device is elaborated with: the memory
    model only -- a variant's, and any TNCA_RTL_PARAMS ablation of the /hbm
    bench's AXI bridge (BENCH_SWITCHES). The profile, and every other
    ablation, is the header."""
    params = list(_SLOW) if profile.endswith('/slow') else []
    if profile.endswith('/hbm'):
        params += _HBM
    for k, v in rtl_overrides():
        if k in BENCH_SWITCHES:
            params += ['-P', f'tnca_device.{k}={v}']
    return params


def uarch_header(profile):
    """The generated header this profile's device is compiled with."""
    return profiles.uarch_header(base_profile(profile),
                                 [(k, v) for k, v in rtl_overrides() if k not in BENCH_SWITCHES])


def memory_limit(profile):
    """The end of the device memory a Session's heap may use, in bytes: the
    512 KiB the on-chip devices have, or for an /hbm memory mode the AXI
    window above the on-chip region."""
    if profile.endswith('/hbm'):
        return HBM_LOCAL_LIMIT + HBM_SIM_WORDS * 8
    return 0x80000


def base_profile(name):
    """The architectural profile behind a memory-mode variant.

    'core-sim/slow' is core-sim on a slower memory, not another machine:
    same ISA, same resources, same fingerprint. Keeping the suffix out of
    everything that asks about the ARCHITECTURE is what stops a memory
    mode being mistaken for an implementation -- which is the confusion
    R3V2-BASE-02 refuses in the inventory.
    """
    return name.split('/', 1)[0]

# The elaboration each profile's device is built with. Read from the profile
# spec rather than written here, so a profile that changes its RCTX count
# does not silently keep an old bench.
_TOP = 'tnca_device'


def cache_root():
    base = os.environ.get('TNCA_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'tnca')
    return os.path.join(base, 'devices')


def _package_data(*parts):
    """Shared with every other module through `_data`.

    This had its own copy of the root calculation, with the same off-by-one:
    three dirnames from a DIRECTORY rather than from `__file__`, which lands
    on `/`. A checkout then could not find the bench it had just built, and
    the error said "no simulator" about a file that was right there.
    """
    from . import _data
    path = _data.path(*parts)
    return path if os.path.exists(path) else None


def expected_path(profile):
    """Where `prepare` puts this profile's device.

    Keyed by the configuration hash, so a device built for a different
    elaboration is a different file rather than a surprise.
    """
    h = profiles.config_hash(base_profile(profile))[:16]
    # The header too: the config hash covers the elaboration and the
    # capabilities, not the implementation's switches, and a device built
    # with different switches must be a different file.
    src = hashlib.sha256((source_fingerprint(profile)
                          + uarch_header(profile)).encode()).hexdigest()[:12]
    ov = rtl_overrides()
    tag = ('-' + '-'.join(f'{k.lower()}{v}' for k, v in ov)) if ov else ''
    return os.path.join(cache_root(),
                        f"{profile.replace('/', '-')}-{h}-{src}{tag}",
                        BENCH_NAME.get(profile, 'tnca_device.vvp'))


# The directories `prepare` compiles from. A bench older than any file in
# them was built from different RTL than the one on disk.
# `rtl` itself, not only its subdirectories. tnca_shell.v -- the public
# top the bench instantiates -- lives directly in rtl/, and leaving the
# directory out meant rewriting the device's entire command sequencer
# changed nothing about the cache key: the runtime kept serving the
# device it had built from the previous one. The walk below is recursive,
# so naming rtl/ covers its subdirectories too; they are kept for the
# case where only part of the tree was staged.
_SOURCE_DIRS = (('rtl',), ('rtl', 'core'), ('rtl', 'common'),
                ('rtl', 'gen'), ('tb', 'common'), ('tb', 'device'))


_FINGERPRINT = {}


def source_fingerprint(profile='core-sim'):
    """A hash of every source the device is compiled from.

    This is in the cache key, which is what makes a cached bench incapable
    of being stale: a different source tree is a different key and a
    different file, not the same file with a question about its age.

    The first version of this compared MTIMES instead, and refused a cache
    entry older than the sources. That is right within one checkout and
    wrong across several, because ~/.cache is shared and a fresh `git
    archive` stamps every file with the export time -- so a clean checkout
    declared a perfectly good cache stale and refused to run at all. A
    hash does not care when a file was written.
    """
    if profile in _FINGERPRINT:
        return _FINGERPRINT[profile]
    h = hashlib.sha256()
    for parts in _SOURCE_DIRS:
        root = _package_data(*parts)
        if not root or not os.path.isdir(root):
            continue
        for base, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                if not f.endswith(('.v', '.vh', '.svh')):
                    continue
                path = os.path.join(base, f)
                h.update(os.path.relpath(path, root).encode())
                try:
                    with open(path, 'rb') as fh:
                        h.update(fh.read())
                except OSError:
                    h.update(b'<unreadable>')
    digest = h.hexdigest()
    _FINGERPRINT[profile] = digest
    return digest


def newest_source(profile='core-sim'):
    """(path, mtime) of the most recently changed source, or (None, 0).

    Sources are not always present -- an installed wheel that did not stage
    them has none -- and then nothing can be said about staleness, which is
    different from saying the build is fresh.
    """
    newest, when = None, 0.0
    for parts in _SOURCE_DIRS:
        root = _package_data(*parts)
        if not root or not os.path.isdir(root):
            continue
        for base, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(('.v', '.vh', '.svh')):
                    continue
                path = os.path.join(base, f)
                try:
                    m = os.path.getmtime(path)
                except OSError:
                    continue
                if m > when:
                    newest, when = path, m
    return newest, when


def stale(path, profile='core-sim'):
    """The source that is newer than this build, or None.

    Worth its own function because the alternative is what actually
    happened: an edited bench, a build from before the edit still sitting
    in sim/build, and a test reporting on RTL that was no longer there.
    Silence here is indistinguishable from a passing test.
    """
    try:
        built = os.path.getmtime(path)
    except OSError:
        return None
    src, when = newest_source(profile)
    return src if src is not None and when > built else None


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


def require(profile='core-sim', *, allow_stale=False):
    path = locate(profile)
    if path:
        # A bench in the cache is named after the sources it was built
        # from, so it cannot be stale. Anything else -- TNCA_DEVICE, or a
        # build sitting in sim/build -- can be, and that is where the
        # question is worth asking: a build older than the RTL reports on
        # code that is no longer on disk and reads like a passing test.
        in_cache = os.path.abspath(path).startswith(
            os.path.abspath(cache_root()))
        src = None if (allow_stale or in_cache) else stale(path, profile)
        if src is None:
            return path
        raise DeviceLost(
            f"the device for profile {profile!r} is older than the RTL it "
            f"was built from",
            detail={'device': path, 'newer source': src},
            remedy='run `tnca-sim prepare --force` to rebuild it. Running '
                   'it anyway would report on RTL that is no longer on '
                   'disk, which reads exactly like a passing test')
    raise DeviceLost(
        f"no simulator for profile {profile!r}",
        detail={'looked for': expected_path(profile)},
        remedy='run `tnca-sim prepare` once to build it, or set '
               'TNCA_DEVICE to an existing bench. A launch does not build '
               'one: that would put a compile inside every kernel call')


KEEP_BUILDS = 3

#: How recently a build must have been touched to be left alone,
#: whatever its age rank. See prune().
RECENT_SECONDS = 900


def prune(profile='core-sim', keep=KEEP_BUILDS, verbose=False):
    """Drop all but the newest `keep` builds of this profile.

    The cache key holds a hash of the sources, which is what makes a
    cached bench incapable of being stale -- and also what makes a new
    directory appear every time the RTL changes. At 29 MB each that is a
    gigabyte after a few dozen commits, so old ones go. Only this
    profile's own directories are considered, and the newest are kept.
    """
    root = cache_root()
    if not os.path.isdir(root):
        return []
    mine = []
    for name in os.listdir(root):
        # An EXACT profile match, not a prefix one. A directory is named
        # "<profile>-<config hash>-<source fingerprint>", and both
        # hashes are hex, so `core-sim-` as a prefix also matched
        # `core-sim-slow-...` -- preparing one profile could evict
        # another profile's builds. Splitting from the right is what
        # tells the two apart.
        parts = name.rsplit('-', 2)
        if len(parts) != 3 or parts[0] != profile:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            mine.append((os.path.getmtime(path), path))
        except OSError:
            continue
    mine.sort(reverse=True)
    removed = []
    # Never remove a build that something might be running right now.
    #
    # Two installations sharing a home directory -- the checkout and a
    # wheel in a virtual environment, which is exactly what the handoff
    # rehearsal sets up -- prepare the same profile and prune each
    # other. One of them then exec'd a path the other had just deleted:
    # "tnca_device_vrf.vvp: No such file or directory", from a device
    # that had been there a moment earlier.
    #
    # An age floor is the cheap fix: a build younger than this is
    # somebody's, and disk is not so short that a few minutes of
    # patience costs anything.
    now = time.time()
    for _when, path in mine[keep:]:
        if now - _when < RECENT_SECONDS:
            continue
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError:
            pass
    if verbose and removed:
        print(f"  pruned {len(removed)} older build(s) of {profile}")
    return removed


_WARNED_STALE = []


def warn_if_staged_copy_is_stale():
    """Say so when the staged data no longer matches the working tree.

    `tnca/data/` is what _data.path reads whenever it exists, and it
    exists as soon as anything has packaged once. Nothing re-makes it
    when a source changes, so editing rtl/ or tb/ in a checkout reaches
    the Makefile's benches and NOT this one: the device gets rebuilt
    from a snapshot, and every result afterwards describes code that is
    no longer in the tree.

    The Makefile has a stamp with prerequisites now, which fixes it for
    anything run through make. This covers the rest -- a script run
    directly, an interactive session -- because the failure is silent
    and expensive: it cost this project an afternoon of measuring the
    wrong device and believing the numbers.

    Warns once per process, and only when both copies exist.
    """
    from . import _data
    roots = _data.roots()
    if len(roots) < 2 or _WARNED_STALE:
        return None
    staged, tree = roots[0], roots[1]
    if not os.path.isdir(staged):
        return None
    differ = []
    for parts in _SOURCE_DIRS:
        a = os.path.join(staged, *parts)
        b = os.path.join(tree, *parts)
        if not (os.path.isdir(a) and os.path.isdir(b)):
            continue
        if _dir_digest(a) != _dir_digest(b):
            differ.append('/'.join(parts))
    if not differ:
        return None
    _WARNED_STALE.append(True)
    msg = (f"the staged copy under {staged} differs from the working "
           f"tree in {differ}. The runtime reads the staged copy, so it "
           f"is about to build a device from sources that are not the "
           f"ones you edited. Run `make r2-stage-data`.")
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return msg


def _dir_digest(root):
    h = hashlib.sha256()
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith(('.v', '.vh', '.svh')):
                continue
            p = os.path.join(base, f)
            h.update(os.path.relpath(p, root).encode())
            try:
                with open(p, 'rb') as fh:
                    h.update(fh.read())
            except OSError:
                h.update(b'<unreadable>')
    return h.hexdigest()


def prepare(profile='core-sim', *, force=False, verbose=True):
    """Build this profile's device from the RTL the package ships."""
    warn_if_staged_copy_is_stale()
    target = expected_path(profile)
    if os.path.exists(target) and not force:
        src = stale(target, profile)
        if src is None:
            if verbose:
                print(f"  already built: {target}")
            return target
        if verbose:
            print(f"  rebuilding: {os.path.basename(src)} is newer than "
                  f"the build")

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

    # NOTE: the directory is created after prune(), not here. See below.
    source = os.path.join(tb, 'device', 'tnca_device.v')
    # The public top is NAMED, not found by library search. `rtl/` itself
    # is deliberately not a -y directory: it holds the tops, and a bench
    # that picks up a top by accident is how two benches end up
    # disagreeing about which one they are testing.
    shell = os.path.join(rtl, 'tnca_shell.v')
    argv = ([iverilog, '-g2005', '-I', os.path.join(rtl, 'common'),
             '-y', os.path.join(rtl, 'core'),
             '-y', os.path.join(rtl, 'common'),
             '-y', os.path.join(rtl, 'gen'),
             '-y', os.path.join(tb, 'common'),
             # rtl/fpga for the AXI bridge an /hbm memory mode puts in front
             # of its memory; a module there is only elaborated if named.
             '-y', os.path.join(rtl, 'fpga')]
            + bench_params(profile)
            + ['-o', target, source, shell])
    if verbose:
        print(f"  building {profile} -> {target}")
    # Prune FIRST, then create the directory, then build into it.
    #
    # This used to create the output directory, then prune, then run
    # iverilog. Pruning between creating a directory and writing into it
    # is a race with itself: the gate saw
    # "core-vrf-sync-v1-.../tnca_device_vrf.vvp: No such file or
    # directory" followed by "Code generator failure: -1" from a build
    # whose target directory had just been made. Ordering it this way
    # removes the whole class rather than the one case.
    prune(profile, verbose=verbose)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    header = os.path.join(os.path.dirname(target), 'tnca_uarch.vh')
    with open(header, 'w') as f:
        f.write(uarch_header(profile))
    argv.insert(argv.index(source), header)
    # Publish only a complete executable. Parallel preparations must never
    # truncate the cached file while another Session is starting vvp.
    fd, temporary = tempfile.mkstemp(prefix='.building-', suffix='.vvp',
                                      dir=os.path.dirname(target))
    os.close(fd)
    argv[argv.index('-o') + 1] = temporary
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not os.path.getsize(temporary):
            raise DeviceLost(
                f"could not build the device for {profile!r}",
                detail={'stderr': (r.stderr or r.stdout)[-600:]},
                remedy='check that iverilog is version 12 and that the package '
                       'shipped its RTL')
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if verbose:
        print(f"  built {os.path.getsize(target)} bytes")
    return target
