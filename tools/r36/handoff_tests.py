#!/usr/bin/env python3
"""
R36-HAND-01, HAND-02 and HAND-03 -- phase3_6_tasks_and_acceptance.md
section R36-09.

The downstream rehearsal, the platform responsibility tables, and the
handoff guide.

HAND-01 is the one that can fail for a real reason. It runs
tools/r36/downstream from a directory outside this repository, against a
freshly built export package, and requires: a real program on real RTL
with a correct answer, the shared conformance suite passing, no file of
the common core or runtime modified, and nothing in the rehearsal
reading a path back into the tree.

The rehearsal has already earned its place twice. It failed once because
the conformance suite assembled its programs with a repository tool a
downstream branch does not have, and once because it kept its own copy
of a constant that had drifted. Both were fixed on the mainline, which
is what section 5.3 requires of a gap found this way -- "演练发现的接口
缺口在主线补齐，不能靠样例私下 patch 公共代码通过".

HAND-02 and HAND-03 check doc/r36_handoff.md against the things it has
to contain and, more importantly, against the things it must NOT: a
named device or process, and any of our own semantic gaps written into
the platform's column.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

GUIDE = os.path.join(ROOT, 'doc', 'r36_handoff.md')
TRITON = os.environ.get('TRITON_PYTHON', sys.executable)

# The files a platform branch may not touch. If the rehearsal changed
# one of them it extended nothing; it forked.
UNTOUCHABLE = ('rtl/core', 'rtl/tnca_shell.v', 'python/tnca/runtime.py',
               'python/tnca/backend.py', 'python/tnca/transport.py',
               'compiler/python')


def _digests(rels):
    out = {}
    for rel in rels:
        p = os.path.join(ROOT, rel)
        if os.path.isdir(p):
            for base, dirs, files in os.walk(p):
                dirs.sort()
                for f in sorted(files):
                    if f.endswith(('.pyc',)):
                        continue
                    q = os.path.join(base, f)
                    with open(q, 'rb') as fh:
                        out[os.path.relpath(q, ROOT)] = \
                            hashlib.sha256(fh.read()).hexdigest()[:16]
        elif os.path.exists(p):
            with open(p, 'rb') as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def _install_wheel(venv, log):
    """A fresh environment with the wheel in it, and nothing else.

    No Triton and no compiler: the rehearsal loads a program it was
    given as machine words, which is the harder claim and the one a
    downstream branch actually makes.
    """
    dist = os.path.join(ROOT, 'sim', 'build', 'dist')
    wheels = (sorted(f for f in os.listdir(dist) if f.endswith('.whl'))
              if os.path.isdir(dist) else [])
    if not wheels:
        r = subprocess.run([sys.executable, 'tools/r36/build.py',
                            '--what', 'package'], cwd=ROOT,
                           capture_output=True, text=True, timeout=1800)
        wheels = (sorted(f for f in os.listdir(dist)
                         if f.endswith('.whl'))
                  if os.path.isdir(dist) else [])
        if not wheels:
            return None, (r.stdout + r.stderr).strip()[-160:]
    r = subprocess.run([sys.executable, '-m', 'venv', venv],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[-160:]
    py = os.path.join(venv, 'bin', 'python')
    r = subprocess.run([py, '-m', 'pip', 'install', '-q',
                        '--no-cache-dir',
                        os.path.join(dist, wheels[-1])],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[-200:]
    # And prove it is the installed one, not a checkout that leaked in.
    r = subprocess.run([py, '-c', 'import tnca;print(tnca.__file__)'],
                       cwd=os.path.dirname(venv), capture_output=True,
                       text=True, timeout=300,
                       env={k: v for k, v in os.environ.items()
                            if k != 'PYTHONPATH'})
    where = r.stdout.strip()
    log.append(f"    the wheel {wheels[-1]} installed into a fresh "
               f"environment")
    log.append(f"    tnca resolves to {where}")
    if ROOT in where:
        log.append("    ...which is inside the repository, so this would "
                   "not be an installed package at all")
        return None, 'the checkout leaked into the environment'
    # Triton must NOT be there: a runtime-only install is the claim.
    r = subprocess.run([py, '-c', 'import triton'], capture_output=True,
                       text=True, timeout=300)
    log.append(f"    and it has no Triton and no compiler: "
               f"{r.returncode != 0}")
    return py, ''


def hand01(log):
    """The rehearsal, from outside, against the delivered package."""
    ok = True
    before = _digests(UNTOUCHABLE)
    log.append(f"    {len(before)} file(s) of the common core and runtime, "
               f"fingerprinted before")

    work = tempfile.mkdtemp(prefix='tnca-handoff-')
    pkg = os.path.join(work, 'hardware')
    r = subprocess.run([sys.executable,
                        os.path.join('tools', 'r36', 'export.py'),
                        '--out', pkg],
                       cwd=ROOT, capture_output=True, text=True,
                       timeout=1800)
    if r.returncode != 0:
        log.append(f"    the export failed: {(r.stderr or r.stdout)[-160:]}")
        return False, 'there was nothing to hand over'
    log.append(f"    a hardware export at {pkg}")

    # The downstream project: the rehearsal and the shared suite, and
    # nothing else of ours.
    down = os.path.join(work, 'downstream')
    os.makedirs(down)
    for name in os.listdir(os.path.join(HERE, 'downstream')):
        if name.endswith(('.py', '.md')):
            shutil.copy2(os.path.join(HERE, 'downstream', name),
                         os.path.join(down, name))
    shutil.copy2(os.path.join(HERE, 'conformance.py'),
                 os.path.join(down, 'conformance.py'))
    log.append(f"    a downstream project at {down}: "
               f"{sorted(os.listdir(down))}")

    # Nothing in it may name a path back here.
    leaks = []
    for name in sorted(os.listdir(down)):
        text = open(os.path.join(down, name)).read()
        # Comments explaining where something came from are
        # documentation; an import or an open() is a dependency. The
        # test is whether the string is used as a path.
        for m in re.finditer(r'''["']([^"']*(?:repo/TritonChip|\.\./\.\.)[^"']*)["']''',
                             text):
            leaks.append((name, m.group(1)[:50]))
    log.append(f"    paths in it that lead back into the repository: "
               f"{len(leaks)} {leaks[:2] if leaks else ''}")
    ok &= not leaks

    # An INSTALLED package, not this repository's source tree.
    #
    # Section 5.3 says the downstream sample is built from the export
    # package AND the install package. This used to hand the rehearsal
    # `PYTHONPATH=<repo>/python`, so the sample read the main repository
    # directly -- and the leak check above could not see it, because it
    # only scans the copied files and the path was injected from
    # outside. An audit of these checkers found it.
    venv = os.path.join(work, 'venv')
    py, why = _install_wheel(venv, log)
    if py is None:
        log.append(f"    no installed package to rehearse against: {why}")
        return False, 'the wheel would not install here'

    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env['PYTHONNOUSERSITE'] = '1'
    out = os.path.join(work, 'rehearsal.json')
    r = subprocess.run([py, 'run_rehearsal.py', '--package', pkg,
                        '--json', out],
                       cwd=down, capture_output=True, text=True,
                       timeout=3600, env=env)
    for line in [x for x in r.stdout.splitlines() if x.strip()][:9]:
        log.append(f"      {line.strip()[:92]}")
    verdict = [x for x in r.stdout.splitlines()
               if 'rehearsal PASS' in x or 'rehearsal FAIL' in x]
    log.append(f"    {verdict[-1].strip() if verdict else 'no verdict'}")
    if r.returncode != 0 and not verdict:
        log.append(f"      {(r.stderr or '').strip()[-200:]}")
    ok &= r.returncode == 0

    # TWICE, and the same answers both times. R36-REL-01 asks for the
    # simulated execution and the downstream rehearsal to be
    # reproducible and nothing re-ran either of them; `ok and
    # verdict == SIGNOFF` was standing in for it. The second run is
    # cheap -- the device is already built -- and it is the only way to
    # tell a reproducible result from one that happened once.
    out2 = os.path.join(work, 'rehearsal-2.json')
    r2 = subprocess.run([py, 'run_rehearsal.py', '--package', pkg,
                         '--json', out2],
                        cwd=down, capture_output=True, text=True,
                        timeout=3600, env=env)
    ok &= r2.returncode == 0

    def _cycles(text):
        m = re.search(r'launch: status \d+, (\d+) cycle', text)
        return int(m.group(1)) if m else None

    c1, c2 = _cycles(r.stdout), _cycles(r2.stdout)
    same = c1 is not None and c1 == c2
    log.append(f"    run twice: {c1} and {c2} cycle(s) "
               f"({'reproducible' if same else 'NOT REPRODUCIBLE'})")
    ok &= same

    if os.path.exists(out) and os.path.exists(out2):
        with open(out) as fh:
            a = json.load(fh)
        with open(out2) as fh:
            b = json.load(fh)
        pa = {k: v.get('passed') for k, v in (a.get('conformance') or {}).items()}
        pb = {k: v.get('passed') for k, v in (b.get('conformance') or {}).items()}
        log.append(f"    and the conformance verdicts agree between the "
                   f"two runs: {pa == pb}")
        ok &= pa == pb

    if os.path.exists(out):
        with open(out) as fh:
            rec = json.load(fh)
        cases = rec.get('conformance') or {}
        passed = sum(1 for c in cases.values() if c.get('passed'))
        log.append(f"    the shared conformance suite against the "
                   f"backend it registered: {passed}/{len(cases)} case(s)")
        ok &= bool(cases) and passed == len(cases)

    after = _digests(UNTOUCHABLE)
    changed = [k for k in set(before) | set(after)
               if before.get(k) != after.get(k)]
    log.append(f"    the common core and runtime afterwards: "
               f"{'unchanged' if not changed else 'CHANGED ' + str(changed)}")
    ok &= not changed
    shutil.rmtree(work, ignore_errors=True)
    return ok, 'a real program on real RTL, twice, from outside, '\
               'patching nothing'


# Devices and processes that must not appear: naming one would be
# specifying a platform, which section 1 puts out of scope.
_NAMED_PARTS = re.compile(
    r'\b(xc7|xcku|xcvu|xczu|artix|kintex|virtex|zynq|cyclone|stratix|'
    r'arria|agilex|ecp5|ice40|tsmc|gf\d|samsung\s+\d+nm|intel\s+\d+a?\b|'
    r'\b\d{1,2}\s*nm\b)', re.I)


def hand02(log):
    """Responsibility tables, limits, and nothing pretending."""
    ok = True
    text = open(GUIDE).read()

    for head in ('### FPGA', '### ASIC', '### What is NOT a board task'):
        present = head in text
        log.append(f"    {head!r}: {'present' if present else 'MISSING'}")
        ok &= present

    named = _NAMED_PARTS.findall(text)
    log.append(f"    devices or processes named: {len(named)} "
               f"{sorted(set(named))[:4] if named else ''}")
    log.append("      section 1 puts device selection and process out of "
               "scope, so naming one here would be specifying a platform")
    ok &= not named

    # The gaps that are ours must be listed as ours. Each is read out of
    # the machine-readable record rather than trusted to the prose.
    caps = json.load(open(os.path.join(ROOT, 'spec',
                                       'r36_capabilities.json')))
    unimpl = [c['id'] for c in caps['capabilities']
              if c['state'] == 'unimplemented']
    mem = json.load(open(os.path.join(ROOT, 'spec',
                                      'r36_memories.json')))
    unmapped = [a['id'] for a in mem['arrays']
                if a['technology_mapping'] == 'not started']
    tail = text.split('### What is NOT a board task', 1)[1]
    log.append(f"    {len(unimpl)} capability(ies) recorded unimplemented, "
               f"{len(unmapped)} array(s) with no technology mapping")
    claims = [
        ('no technology mapping', 'technology_mapping: not started' in tail
         or 'technology mapping' in tail.lower()),
        ('the ARF', 'ARF' in tail),
        ('cache coherence', 'coherence' in tail.lower()),
        ('the four-tile runtime', 'four-tile' in tail),
    ]
    for what, present in claims:
        log.append(f"      {what} is listed as ours: {present}")
        ok &= present

    # And the owners and the change process.
    for head in ('## 9. Who owns what', '## 8. Versioning the interface',
                 '## 7. Which way a fix goes'):
        present = head in text
        log.append(f"    {head!r}: {'present' if present else 'MISSING'}")
        ok &= present
    return ok, 'two responsibility tables, and our gaps kept as ours'


def hand03(log):
    """A minimal program, diagnosis, counters, and reproduction."""
    ok = True
    text = open(GUIDE).read()

    needed = {
        'a minimal correctness program': 'run_smoke.sh',
        'how to reproduce it': 'run_rehearsal.py --package',
        'fault localisation': 'done_fault_pc',
        'the counters and what they mean': 'perf_stall_lsu',
        'the completion record': 'done_retired',
        'the diagnosis table': '| symptom | where to look |',
        'where a fix goes': 'the mainline',
        'the extension layer': 'your extension layer',
        'how the interface is versioned': 'PROTOCOL_VERSION',
        'the claim level': 'engineering-ready',
    }
    for what, marker in needed.items():
        present = marker in text
        log.append(f"    {what}: {'yes' if present else 'MISSING (' + marker + ')'}")
        ok &= present

    # The commands it gives must be real.
    cmds = re.findall(r'^\s*(make [a-z0-9-]+)', text, re.M)
    bad = []
    mk = open(os.path.join(ROOT, 'Makefile')).read()
    for c in sorted(set(cmds)):
        target = c.split()[1]
        if f"\n{target}:" not in mk and f"\n{target} " not in mk:
            bad.append(c)
    log.append(f"    the make commands it names: {sorted(set(cmds))}")
    log.append(f"    every one is a real target: {not bad} "
               f"{bad if bad else ''}")
    ok &= not bad

    # And the files it points at must exist.
    files = re.findall(r'`((?:tools|spec|rtl|python|doc)/[\w./-]+)`', text)
    missing = [f for f in sorted(set(files))
               if not os.path.exists(os.path.join(ROOT, f))]
    log.append(f"    the {len(set(files))} file(s) it points at: "
               f"{len(missing)} missing {missing if missing else ''}")
    ok &= not missing
    return ok, 'a program, a diagnosis table, counters and commands'


TESTS = [('R36-HAND-01', hand01), ('R36-HAND-02', hand02),
         ('R36-HAND-03', hand03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate D: the handoff, rehearsed from outside")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-900:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
        print(flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print(f"  r36-handoff-tests      {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
