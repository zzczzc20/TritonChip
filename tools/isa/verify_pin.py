#!/usr/bin/env python3
"""
verify_pin.py -- confirm the pinned toolchain still describes upstream.

A version pin that was right once and is never rechecked is a comment. This
asks GitHub whether the recorded Triton tag still resolves to the recorded
commit, and whether that commit still names the recorded LLVM revision.

Needs network; skips rather than fails when offline, so it does not break an
air-gapped build.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PIN = os.path.join(ROOT, 'compiler', 'pin', 'toolchain.json')
TIMEOUT = 25


class Offline(Exception):
    pass


def get(url):
    """Fetch via curl.

    Not urllib: the system Python on macOS ships without a CA bundle, so
    urllib fails with CERTIFICATE_VERIFY_FAILED on a host where the network
    is perfectly fine. That would read as "upstream changed" when it means
    "this Python cannot do TLS".
    """
    if not shutil.which('curl'):
        raise Offline('curl not installed')
    r = subprocess.run(['curl', '-sS', '--fail', '--max-time', str(TIMEOUT), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Offline(r.stderr.strip() or f'curl exit {r.returncode}')
    return r.stdout


def main():
    pin = json.load(open(PIN))
    tri, llvm = pin['triton'], pin['llvm']

    try:
        tags = json.loads(get('https://api.github.com/repos/triton-lang/triton/tags?per_page=100'))
    except (Offline, OSError) as e:
        print(f"SKIP: cannot reach github ({e}); pin left unverified")
        return 0

    bad = []

    found = {t['name']: t['commit']['sha'] for t in tags}
    if tri['tag'] not in found:
        bad.append(f"triton tag {tri['tag']} no longer exists upstream")
    elif found[tri['tag']] != tri['commit']:
        bad.append(f"triton {tri['tag']} now resolves to {found[tri['tag']]}, "
                   f"pin says {tri['commit']}")
    else:
        print(f"  triton {tri['tag']} = {tri['commit'][:12]}  ok")

    try:
        info = json.loads(get('https://raw.githubusercontent.com/triton-lang/'
                              f"triton/{tri['commit']}/cmake/llvm-info.json"))
    except (Offline, OSError) as e:
        print(f"SKIP: cannot read llvm-info.json ({e})")
        info = None

    if info is not None:
        if info.get('llvm_hash') != llvm['commit']:
            bad.append(f"triton {tri['tag']} pins LLVM {info.get('llvm_hash')}, "
                       f"our pin says {llvm['commit']}")
        else:
            print(f"  llvm {llvm['commit'][:12]}  ok")
        for plat, want in llvm['prebuilt_sha256'].items():
            got = info.get('sha256sum', {}).get(plat)
            if got != want:
                bad.append(f"llvm prebuilt {plat} sha256 changed: {got}")

    for msg in bad:
        print(f"  MISMATCH: {msg}")
    if bad:
        print("FAIL: compiler/pin/toolchain.json no longer matches upstream")
        return 1
    print("PASS: pinned toolchain matches upstream")
    return 0


if __name__ == '__main__':
    sys.exit(main())
