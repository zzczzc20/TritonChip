#!/usr/bin/env python3
"""
PKG-01 smoke: a clean, non-editable wheel reads only the data it shipped.

PKG-01 (tools/r2/package_tests.py) installs the wheel into fresh
environments and runs compile -> run -> download with the repository out of
the way. That path never imported tnca_cc.upstream, so it never noticed that
upstream/version.py found the toolchain lock by counting four directories up
from itself -- the repository root in a checkout, `<prefix>/lib` in an
install, where it looked for lib/spec/r1_toolchain.lock.json (PR #3 review).

This is the narrow, fast check for that class of bug:

  1. a fresh venv, the wheel installed non-editable, no dependencies;
  2. a Python started with -I (no cwd, no PYTHONPATH, no user site) in an
     empty directory, so the checkout is unreachable;
  3. an audit hook records every file the package opens while it imports
     its modules and calls each function that reads packaged data;
  4. every file opened must exist and lie inside the environment, and the
     values read must equal the checkout's.

    python3 tools/r2/wheel_smoke.py [--wheel sim/build/dist/tnca-*.whl]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

PROBE = r'''
import json, os, sys
opened = []
def hook(event, args):
    if event == 'open' and isinstance(args[0], str):
        opened.append(os.path.abspath(args[0]))
sys.addaudithook(hook)

out = {}
# Every packaged reader of checkout-relative data, called once.
from tnca_cc.upstream import version as V
out['pinned'] = V.pinned_version()
out['describe'] = sorted(V.describe())
from tnca_cc import bankcost
out['banks'] = bankcost.hardware()['banks']
from tnca_cc import isa, target
out['target'] = target.TargetInfo.load('tnca-sim-v0.1').name
import tnca
from tnca import _data, profiles
out['profiles'] = len(profiles.names())
with open(_data.path('spec', 'r1_toolchain.lock.json')) as fh:
    out['lock_via_tnca'] = json.load(fh)['triton']['version']
out['tnca'] = tnca.__file__
out['opened'] = sorted(set(p for p in opened if p.endswith('.json')))
print('PROBE ' + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--wheel', help='the wheel (default: the newest in sim/build/dist)')
    args = ap.parse_args()
    wheel = args.wheel or max(glob.glob(os.path.join(ROOT, 'sim', 'build', 'dist', 'tnca-*.whl')),
                              key=os.path.getmtime, default=None)
    if not wheel:
        sys.exit('no wheel: run `make r2-package` first')
    expect = json.load(open(os.path.join(ROOT, 'spec', 'r1_toolchain.lock.json')))['triton']['version']

    base = tempfile.mkdtemp(prefix='pkg01-smoke-')
    try:
        venv = os.path.join(base, 'env')
        subprocess.run([sys.executable, '-m', 'venv', venv], check=True)
        py = os.path.join(venv, 'bin', 'python')
        r = subprocess.run([py, '-m', 'pip', 'install', '-q', '--no-deps', '--no-cache-dir', wheel],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr)
            sys.exit('FAIL: the wheel would not install')
        work = os.path.join(base, 'empty')
        os.mkdir(work)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'TNCA_'))}
        r = subprocess.run([py, '-I', '-c', PROBE], cwd=work, env=env,
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith('PROBE ')), None)
        if r.returncode or line is None:
            print(r.stdout[-2000:] + r.stderr[-3000:])
            sys.exit('FAIL: the installed package could not read its own data')
        got = json.loads(line[6:])
        bad = []
        if got['pinned'] != expect:
            bad.append(f"tnca_cc.upstream.version read {got['pinned']}, the lock says {expect}")
        if got['lock_via_tnca'] != expect:
            bad.append(f"tnca._data read {got['lock_via_tnca']}, the lock says {expect}")
        if not got['tnca'].startswith(venv):
            bad.append(f"tnca imported from {got['tnca']}, not the environment")
        for p in got['opened']:
            if not p.startswith(venv):
                bad.append(f'opened outside the environment: {p}')
            elif not os.path.exists(p):
                bad.append(f'opened a file that does not exist: {p}')
        for b in bad:
            print('  FAIL', b)
        print(f"  PKG-01 smoke: {len(got['opened'])} data file(s) read, all inside the "
              f"environment; triton pinned {got['pinned']}; target {got['target']}; "
              f"{got['banks']} VRF banks" if not bad else '')
        print(f"wheel smoke: {'PASS' if not bad else 'FAIL'} ({os.path.basename(wheel)})")
        return 1 if bad else 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
