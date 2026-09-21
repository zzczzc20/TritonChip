#!/usr/bin/env python3
"""
The public build entry point -- phase3_6_tasks_and_acceptance.md
R36-BUILD-01.

One command, four things it can produce, and an honest answer about the
one it cannot:

    package     the installable wheel
    sim-assets  the simulation run assets: the staged data and the
                device the runtime builds from it
    hardware    the generic hardware export (tools/r36/export.py)
    platform    external platform extensions -- FPGA or ASIC flows

The standard has two halves and the second is the interesting one:
*missing optional FPGA/ASIC tools must not block the mainline package
and simulation build, and must not falsely report a successful platform
build*. Those pull in opposite directions. The way out is that "not
built" is a first-class outcome with a reason attached, and that asking
for a platform build you cannot have is an error only when you say you
require it:

    tools/r36/build.py --what all                 # platform: not built
    tools/r36/build.py --what platform --require  # exits non-zero

A build report is written either way, so what was and was not produced
is a record rather than something to be reconstructed from a log.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

WHAT = ('package', 'sim-assets', 'hardware', 'platform', 'all')

# What an FPGA or ASIC flow would need before it could claim anything.
# Named rather than probed generically, because "some tool was missing"
# is not a useful thing to tell a maintainer.
PLATFORM_TOOLS = {
    'fpga': [('vivado', 'Xilinx Vivado'), ('quartus', 'Intel Quartus'),
             ('nextpnr-ecp5', 'nextpnr for ECP5')],
    'asic': [('openroad', 'OpenROAD'), ('innovus', 'Cadence Innovus'),
             ('icc2_shell', 'Synopsys IC Compiler II')],
}


def _tool_versions():
    out = {'python': sys.version.split()[0]}
    for name, argv in (('iverilog', ['iverilog', '-V']),
                       ('yosys', ['yosys', '-V'])):
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=60)
            out[name] = (r.stdout or r.stderr).splitlines()[0].strip()
        except Exception as e:                             # noqa: BLE001
            out[name] = f"unavailable: {e}"
    return out


def _head():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:                                      # noqa: BLE001
        return ''


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=kw.pop('timeout', 3600), **kw)


def _sha_tree(root):
    """A stable digest of a directory: names and contents, sorted."""
    h = hashlib.sha256()
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(base, f)
            h.update(os.path.relpath(p, root).encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()


def _pip_python():
    """An interpreter that can build a wheel, or None.

    The Makefile's default $(PYTHON) is whatever `python3` is, and on a
    MacPorts install that is an interpreter with no pip. A build that
    failed with "No module named pip" three lines into a make recipe is
    a worse answer than one that says which interpreter it needs.
    """
    for cand in (os.environ.get('TRITON_PYTHON'), sys.executable,
                 shutil.which('python3')):
        if not cand or not os.path.exists(cand):
            continue
        r = subprocess.run([cand, '-c', 'import pip'],
                           capture_output=True, timeout=60)
        if r.returncode == 0:
            return cand
    return None


def build_package(rep, log):
    t0 = time.time()
    py = _pip_python()
    if py is None:
        rep['package'] = {
            'built': False, 'artifacts': [],
            'why': 'no interpreter on this machine has pip, so no wheel '
                   'can be built here. Set TRITON_PYTHON to one that '
                   'does.',
            'seconds': 0.0}
        log.append("  package     NOT BUILT  no interpreter with pip")
        log.append(f"              {rep['package']['why'][:88]}")
        return False
    r = _run(['make', 'r2-package', f'PYTHON={py}'])
    ok = r.returncode == 0
    dist = os.path.join(ROOT, 'sim', 'build', 'dist')
    wheels = sorted(f for f in os.listdir(dist)
                    if f.endswith('.whl')) if os.path.isdir(dist) else []
    rep['package'] = {
        'built': ok and bool(wheels),
        'artifacts': [{'file': f"sim/build/dist/{w}",
                       'sha256_16': hashlib.sha256(
                           open(os.path.join(dist, w), 'rb').read()
                       ).hexdigest()[:16],
                       'bytes': os.path.getsize(os.path.join(dist, w))}
                      for w in wheels],
        'seconds': round(time.time() - t0, 1),
        'interpreter': py,
    }
    if not ok:
        rep['package']['why'] = (r.stderr or r.stdout).strip()[-400:]
    log.append(f"  package     {'built' if rep['package']['built'] else 'FAILED'}"
               f"  {len(wheels)} wheel(s)")
    return rep['package']['built']


def build_sim_assets(rep, log):
    t0 = time.time()
    r = _run(['make', 'r2-stage-data'])
    ok = r.returncode == 0
    data = os.path.join(ROOT, 'python', 'tnca', 'data')
    n = sum(len(fs) for _b, _d, fs in os.walk(data)) if os.path.isdir(data) \
        else 0
    # And the device the runtime builds from them, which is the asset the
    # simulation actually runs.
    dev_ok, dev_path = False, None
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [os.path.join(ROOT, 'python'), env.get('PYTHONPATH', '')])
    r2 = _run([sys.executable, '-c',
               "from tnca import device;"
               "device.prepare('core-sim', verbose=False);"
               "print(device.expected_path('core-sim'))"], env=env)
    if r2.returncode == 0:
        dev_path = r2.stdout.strip().splitlines()[-1]
        dev_ok = os.path.exists(dev_path)
    rep['sim-assets'] = {
        'built': ok and n > 0 and dev_ok,
        'staged_files': n,
        'staged_digest': _sha_tree(data)[:16] if os.path.isdir(data) else '',
        'device': dev_path,
        'seconds': round(time.time() - t0, 1),
    }
    if not dev_ok:
        rep['sim-assets']['why'] = (r2.stderr or r2.stdout).strip()[-400:]
    log.append(f"  sim-assets  "
               f"{'built' if rep['sim-assets']['built'] else 'FAILED'}"
               f"  {n} staged file(s), device "
               f"{'present' if dev_ok else 'MISSING'}")
    return rep['sim-assets']['built']


def build_hardware(rep, log, out):
    t0 = time.time()
    r = _run([sys.executable, os.path.join('tools', 'r36', 'export.py'),
              '--out', out])
    ok = r.returncode == 0 and os.path.isdir(out)
    manifest = {}
    if ok:
        with open(os.path.join(out, 'MANIFEST.json')) as fh:
            manifest = json.load(fh)
    rep['hardware'] = {
        'built': ok,
        'path': os.path.relpath(out, ROOT),
        'digest': _sha_tree(out)[:16] if ok else '',
        'rtl_files': len(manifest.get('rtl', [])),
        'sim_files': len(manifest.get('sim', [])),
        'claim_level': manifest.get('claim_level'),
        'seconds': round(time.time() - t0, 1),
    }
    if not ok:
        rep['hardware']['why'] = (r.stderr or r.stdout).strip()[-400:]
    log.append(f"  hardware    {'built' if ok else 'FAILED'}"
               f"  {rep['hardware']['rtl_files']} synthesizable file(s), "
               f"claim {rep['hardware']['claim_level']}")
    return ok


def build_platform(rep, log, require):
    """The one that is allowed to say no.

    Nothing here ever reports success. There is no platform flow in this
    repository and section 1 puts device selection, vendor IP and board
    constraints in a separate branch, so the only honest outcomes are
    "not built, no tool" and "not built, nothing to run".
    """
    found = {}
    for family, tools in PLATFORM_TOOLS.items():
        found[family] = [name for name, _label in tools
                         if shutil.which(name)]
    have = {k: v for k, v in found.items() if v}
    rep['platform'] = {
        'built': False,
        'why': ('no platform flow exists in this repository: section 1 of '
                'the phase contract puts device selection, vendor IP, '
                'placement, routing, bitstreams and board bring-up in a '
                'separate branch'),
        'tools_searched': {k: [n for n, _l in v]
                           for k, v in PLATFORM_TOOLS.items()},
        'tools_found': found,
        'claim_level': None,
    }
    log.append(f"  platform    NOT BUILT  "
               f"{'tools present: ' + str(have) if have else 'no vendor tool on PATH'}")
    log.append(f"              {rep['platform']['why'][:88]}")
    if require:
        log.append("              --require was given, so this is a "
                   "failure rather than a note")
    return not require


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--what', nargs='*', default=['all'], choices=WHAT)
    ap.add_argument('--require', action='store_true',
                    help='a target that cannot be built is an error')
    ap.add_argument('--out', default=os.path.join(ROOT, 'sim', 'build',
                                                  'tnca-export'))
    ap.add_argument('--json')
    args = ap.parse_args()

    want = set(args.what)
    if 'all' in want:
        want = {'package', 'sim-assets', 'hardware', 'platform'}

    # The tools this build used, so the verifier can check that the
    # deliverables and the checkers came from one machine rather than
    # from two runs stapled together.
    rep = {'targets': sorted(want), 'require': args.require,
           'commit': _head(), 'tools': _tool_versions(),
           'platform': platform.platform(), 'machine': platform.machine(),
           'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    log = []
    print("TNCA build")
    ok = True
    if 'package' in want:
        ok &= build_package(rep, log)
    if 'sim-assets' in want:
        ok &= build_sim_assets(rep, log)
    if 'hardware' in want:
        ok &= build_hardware(rep, log, args.out)
    if 'platform' in want:
        ok &= build_platform(rep, log, args.require)
    for line in log:
        print(line)
    rep['ok'] = bool(ok)
    print(f"  -> {'ok' if ok else 'FAILED'}")
    if args.json:
        R.write_json(args.json, rep)
        print(f"  report {args.json}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
