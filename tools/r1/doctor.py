#!/usr/bin/env python3
"""
`make r1-doctor` -- what this machine can and cannot certify.

doc/phase1_tasks_and_acceptance.md R1-09 requires acceptance on **Linux
x86_64**, in two independent environments, from a locked toolchain. This
reports where it is running and refuses to imply more than that: on macOS it
succeeds as a development check and states plainly that it is not an R1
acceptance environment.

That refusal is the feature. The contract says macOS may not stand in for the
Linux dual-environment acceptance, and a doctor that printed a green tick here
would be the first step to claiming otherwise.
"""
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
LOCK = os.path.join(ROOT, 'spec', 'r1_toolchain.lock.json')

TOOLS = [
    ('python3', [sys.executable, '--version']),
    ('iverilog', ['iverilog', '-V']),
    ('vvp', ['vvp', '-V']),
    ('yosys', ['yosys', '-V']),
    ('git', ['git', '--version']),
    ('make', ['make', '--version']),
    ('g++', ['g++', '--version']),
    ('cmake', ['cmake', '--version']),
]


def probe(name, argv):
    exe = shutil.which(argv[0]) or (argv[0] if os.path.exists(argv[0]) else None)
    if not exe:
        return dict(tool=name, found=False, version=None, path=None)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        ver = (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception as e:
        ver = f"probe failed: {e}"
    return dict(tool=name, found=True, version=ver, path=exe)


def version_number(text):
    import re
    m = re.search(r'(\d+)\.(\d+)', text or '')
    return tuple(int(x) for x in m.groups()) if m else None


def main():
    host = dict(system=platform.system(), machine=platform.machine(),
                release=platform.release(), python=platform.python_version())
    tools = [probe(n, a) for n, a in TOOLS]

    with open(LOCK) as fh:
        lock = json.load(fh)
    plat = lock['acceptance_platform']

    # Required means "the lock requires it", not "this file happens to probe
    # it". yosys is probed because its version is worth recording and absent
    # because the gate does not synthesise anything.
    missing = [t['tool'] for t in tools if not t['found']
               and lock['tools'].get(t['tool'], {}).get('source') != 'optional']

    is_accept_env = (host['system'] == 'Linux' and
                     host['machine'] in ('x86_64', 'amd64'))

    # Compare against the lock rather than against expectations in the code.
    lock_notes = []
    for name, want in lock['tools'].items():
        t = next((x for x in tools if x['tool'] == name), None)
        if not t or not t['found']:
            if want.get('source') != 'optional':
                lock_notes.append(f"{name}: not found (lock requires it)")
            continue
        got = version_number(t['version'])
        need = version_number(want.get('version') or want.get('min') or '')
        if got and need and got < need:
            lock_notes.append(f"{name}: {t['version']} is below the locked "
                              f"{want.get('version') or want.get('min')}")
    for item in lock.get('not_yet_locked', []):
        lock_notes.append(f"NOT LOCKED: {item}")

    print(f"  host: {host['system']} {host['machine']} (python {host['python']})")
    for t in tools:
        state = t['version'] if t['found'] else 'NOT FOUND'
        print(f"    {t['tool']:<10s} {state}")

    # GNU Make 3.81 ignores .SHELLFLAGS, which is why no recipe may rely on it.
    mk = next((t for t in tools if t['tool'] == 'make'), None)
    if mk and mk['found'] and 'GNU Make 3.8' in (mk['version'] or ''):
        print("    note: this make predates .SHELLFLAGS; recipes must not "
              "depend on it (see the Makefile header)")

    print()
    optional_missing = [t['tool'] for t in tools if not t['found']
                        and t['tool'] not in missing]
    if optional_missing:
        print(f"    optional, absent: {', '.join(optional_missing)}")
    if missing:
        print(f"  r1-doctor              FAIL   missing: {', '.join(missing)}")
        return 1
    if is_accept_env:
        # Size is part of R1-09's requirement, so a machine below it is
        # reported as such rather than silently accepted.
        size = []
        try:
            cpus = os.cpu_count() or 0
            if cpus < plat['reference_runner']['vcpu']:
                size.append(f"{cpus} vCPU < {plat['reference_runner']['vcpu']}")
            import shutil as _sh
            free_gib = _sh.disk_usage(ROOT).free // (1 << 30)
            if free_gib < plat['reference_runner']['disk_gib']:
                size.append(f"{free_gib} GiB free < "
                            f"{plat['reference_runner']['disk_gib']}")
        except Exception:
            pass
        print("  r1-doctor              PASS   platform matches the lock")
        if size:
            print(f"    below the reference runner: {'; '.join(size)}")
        print("    R1-09 still requires TWO independent environments; this "
              "checks one.")
    else:
        print("  r1-doctor              PASS   development tools present")
        print(f"    NOT an R1 acceptance environment: R1-09 requires Linux "
              f"x86_64, this is {host['system']} {host['machine']}.")
        print("    macOS is a supported development platform and may not "
              "stand in for the Linux dual-environment acceptance.")

    if lock_notes:
        print("  lock:")
        for n in lock_notes:
            print('    ' + n)

    out = dict(host=host, tools=tools,
               is_r1_acceptance_environment=is_accept_env,
               lock_notes=lock_notes,
               toolchain_lock=os.path.relpath(LOCK, ROOT))
    path = os.path.join(ROOT, 'sim', 'r1', 'doctor.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
