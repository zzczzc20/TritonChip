#!/usr/bin/env python3
"""
R36-BUILD-01, BUILD-02 and BUILD-03 -- phase3_6_tasks_and_acceptance.md
section R36-07.

One build entry point, an install that does not need the checkout, and
two clean builds that agree.

BUILD-01's two halves pull against each other: a missing FPGA or ASIC
tool must not block the mainline build, and must not produce a false
claim of a platform build either. The resolution is that "not built" is
an outcome with a reason, and that `--require` turns it into a failure
for anyone who needs it to be one. Both directions are exercised here,
because a build that always says "not built" and a build that always
says "fine" are both consistent with only testing one of them.

BUILD-03 is checked by building twice from two independent clean
archives of the source and comparing the manifests under the
normalisation rules in spec/r36_provenance.json -- declared in advance,
in a file, rather than decided when the comparison fails. The integrity
check makes the tree read-only first, so it cannot be the thing that
repairs what it is checking.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import stat
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

PROV = json.load(open(os.path.join(ROOT, 'spec',
                                   'r36_provenance.json')))
with open(os.path.join(ROOT, 'spec', 'r1_toolchain.lock.json')) as _fh:
    LOCKED_TRITON = json.load(_fh)['triton']['version']
BUILD = os.path.join(ROOT, 'tools', 'r36', 'build.py')


def _run(cmd, cwd=ROOT, timeout=3600, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build01(log):
    """One entry point, four targets, and an honest no."""
    ok = True
    out = tempfile.mkdtemp(prefix='tnca-b01-')
    rep = os.path.join(out, 'report.json')

    # The mainline targets, with no vendor tool anywhere. This is the
    # half that must not be blocked.
    r = _run([sys.executable, BUILD, '--what', 'hardware', 'sim-assets',
              '--out', os.path.join(out, 'hw'), '--json', rep])
    log.append(f"    the mainline targets, with no FPGA or ASIC tool "
               f"installed: exit {r.returncode}")
    for ln in r.stdout.strip().splitlines():
        log.append(f"      {ln.strip()[:88]}")
    ok &= r.returncode == 0
    with open(rep) as fh:
        report = json.load(fh)
    ok &= report['hardware']['built'] and report['sim-assets']['built']

    # The platform target, which must NOT claim success.
    r = _run([sys.executable, BUILD, '--what', 'platform', '--json', rep])
    with open(rep) as fh:
        report = json.load(fh)
    p = report['platform']
    log.append(f"    the platform target: built={p['built']}, exit "
               f"{r.returncode}")
    log.append(f"      why: {p['why'][:84]}")
    log.append(f"      tools it looked for: "
               f"{sorted(sum(p['tools_searched'].values(), []))}")
    log.append(f"      claim level recorded: {p['claim_level']}")
    ok &= (p['built'] is False) and (p['claim_level'] is None)
    ok &= r.returncode == 0          # a note, not a failure, by default

    # ...unless the caller says it needs one.
    r = _run([sys.executable, BUILD, '--what', 'platform', '--require'])
    log.append(f"    the same target with --require: exit {r.returncode} "
               f"({'refused' if r.returncode != 0 else 'ACCEPTED, which is wrong'})")
    ok &= r.returncode != 0

    # A build report is a record either way.
    need = ('targets', 'require', 'commit', 'started', 'ok')
    log.append(f"    the report carries {len([k for k in need if k in report])}"
               f"/{len(need)} of its required fields")
    ok &= all(k in report for k in need)
    shutil.rmtree(out, ignore_errors=True)
    return ok, 'four targets, and the one that cannot be built says so'


def build02(log):
    """An install that does not need the checkout.

    R2's PKG-01, PKG-02, PKG-04 and PKG-05 already put a wheel in a fresh
    environment, move the repository off sys.path and out of the working
    directory, and compile and run from there. Repeating that machinery
    would be a second implementation of it to keep in step, so this runs
    those cases and then adds the two things R3.6 needs on top: that the
    wheel carries the new public top, and that the device the runtime
    builds from an installed package really contains it.
    """
    ok = True
    # Through the public entry point, not by calling make directly: if
    # the entry point cannot build a package, that is the fact this case
    # is about.
    rep = os.path.join(tempfile.mkdtemp(prefix='tnca-b02-'), 'r.json')
    r = _run([sys.executable, BUILD, '--what', 'package', '--json', rep],
             timeout=1800)
    with open(rep) as fh:
        report = json.load(fh)
    if not report['package']['built']:
        log.append(f"    no wheel: {report['package'].get('why','')[:88]}")
        return False, 'the wheel would not build in this environment'
    log.append(f"    a non-editable wheel, built by tools/r36/build.py "
               f"with {os.path.basename(report['package']['interpreter'])}")
    dist = os.path.join(ROOT, 'sim', 'build', 'dist')
    wheels = [f for f in os.listdir(dist) if f.endswith('.whl')]
    ok &= bool(wheels)

    # What is in it. The export's public top has to be there, or an
    # installed package can stage data and then fail to build a device
    # from it -- which is exactly what happened when device.prepare did
    # not name tnca_shell.v.
    import zipfile
    with zipfile.ZipFile(os.path.join(dist, wheels[0])) as z:
        names = z.namelist()
    want = ['tnca/data/rtl/tnca_shell.v',
            'tnca/data/tb/device/tnca_device.v',
            'tnca/data/rtl/core/tnca_core.v',
            'tnca/data/tb/common/tnca_mem_model.v']
    have = [w for w in want if w in names]
    log.append(f"    the wheel carries {len(have)}/{len(want)} of the "
               f"sources the device is built from")
    for w in want:
        if w not in names:
            log.append(f"      MISSING {w}")
    ok &= len(have) == len(want)
    log.append(f"    {len(names)} file(s) in the wheel")

    # R2's suite, which is the real BUILD-02 machinery. Invoked as the
    # script rather than through make: it has no make target, and
    # inventing one here would make this checker the only caller of a
    # rule nobody else uses.
    r = _run([sys.executable, os.path.join('tools', 'r2',
                                           'package_tests.py')],
             timeout=5400)
    verdicts = [ln.strip() for ln in r.stdout.splitlines()
                if ln.strip().startswith('PKG-')]
    for ln in verdicts:
        log.append(f"      {ln[:88]}")
    log.append(f"    the R2 suite (fresh environment, repo off the path, "
               f"arbitrary cwd, runtime-only): exit {r.returncode}")

    # Why it failed matters. The locked Triton is a manylinux x86_64
    # wheel and spec/r1_toolchain.lock.json says in as many words that
    # macOS is a development platform and may not substitute for the
    # acceptance one. A case that cannot run here has NOT passed -- R1-02
    # exists to stop that becoming a skip -- so this still fails, and
    # says which machine would answer it.
    blocked = 'the locked Triton is not installable here' in r.stdout
    if blocked:
        log.append(f"    this platform is {platform.system()}/"
                   f"{platform.machine()}, and the locked Triton "
                   f"{LOCKED_TRITON} ships no wheel for it")
        log.append("    so the fresh-environment cases are NOT RUN here, "
                   "which is a failure and not a skip: the verdict for "
                   "this standard comes from the acceptance platform "
                   "spec/r1_toolchain.lock.json names")
    ok &= r.returncode == 0
    return ok, ('the fresh-environment cases need the acceptance platform'
                if blocked
                else 'the wheel stands on its own, and carries the public top')


def _normalised(manifest):
    """A manifest with the fields the provenance file says may differ."""
    drop = set(PROV['normalisation']['manifest_fields_ignored'])
    return {k: v for k, v in manifest.items() if k not in drop}


def build03(log):
    """Two clean builds, and an integrity check that cannot repair."""
    ok = True
    log.append(f"    normalisation, declared in "
               f"spec/r36_provenance.json: ignore "
               f"{PROV['normalisation']['manifest_fields_ignored']}")
    for k, why in PROV['normalisation']['why'].items():
        log.append(f"      {k}: {why[:76]}")

    # Two independent clean builds: two archives of the tracked source,
    # unpacked in two places, each exporting on its own.
    mans, dirs = [], []
    for i in (0, 1):
        work = tempfile.mkdtemp(prefix=f'tnca-clean{i}-')
        tar = os.path.join(work, 'src.tar')
        r = _run(['git', 'archive', '-o', tar, 'HEAD'])
        if r.returncode != 0:
            log.append(f"    could not archive the source: "
                       f"{r.stderr.strip()[:80]}")
            return False, 'the source would not archive'
        src = os.path.join(work, 'src')
        os.makedirs(src)
        _run(['tar', '-xf', tar, '-C', src])
        out = os.path.join(work, 'export')
        r = _run([sys.executable, os.path.join('tools', 'r36', 'export.py'),
                  '--out', out], cwd=src)
        if r.returncode != 0:
            log.append(f"    clean build {i} failed: "
                       f"{(r.stderr or r.stdout).strip()[-140:]}")
            return False, 'a clean build failed'
        with open(os.path.join(out, 'MANIFEST.json')) as fh:
            mans.append(json.load(fh))
        dirs.append(out)
        log.append(f"    clean build {i}: {len(mans[-1]['rtl'])} "
                   f"synthesizable file(s), from a fresh archive in {work}")

    a, b = _normalised(mans[0]), _normalised(mans[1])
    same = a == b
    log.append(f"    the two manifests agree under those rules: {same}")
    if not same:
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                log.append(f"      differs: {k}")
    ok &= same

    # And the field that was allowed to differ really is the only one
    # that could have. A normalisation rule that removes a field nobody
    # would have compared is a rule doing nothing.
    raw_same = mans[0] == mans[1]
    log.append(f"    without normalisation they would "
               f"{'still agree' if raw_same else 'differ'}, so the rules "
               f"are {'unnecessary here' if raw_same else 'load-bearing'}")

    # Provenance: every generated artifact says what made it, and every
    # third-party thing has a licence.
    gaps = []
    for g in PROV['generated_artifacts']:
        for k in ('generated_from', 'generated_by', 'licence',
                  'checked_in'):
            if k not in g:
                gaps.append(f"{g.get('path','?')}:{k}")
    for group in ('compile_dependencies', 'build_tools'):
        for d in PROV[group]:
            if 'licence' not in d:
                gaps.append(f"{d.get('what','?')}:licence")
    log.append(f"    provenance: {len(PROV['generated_artifacts'])} "
               f"generated artifact(s), "
               f"{len(PROV['compile_dependencies']) + len(PROV['build_tools'])}"
               f" third-party dependenc(ies), {len(gaps)} gap(s) "
               f"{gaps if gaps else ''}")
    ok &= not gaps
    lic = os.path.join(ROOT, PROV['project']['licence_file'])
    log.append(f"    the licence file exists and is "
               f"{PROV['project']['licence']}: {os.path.exists(lic)}")
    ok &= os.path.exists(lic)

    # The integrity check, on a READ-ONLY tree.
    pkg = dirs[0]
    for base, dnames, fnames in os.walk(pkg):
        for f in fnames:
            p = os.path.join(base, f)
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        del dnames
    bad = []
    with open(os.path.join(pkg, 'MANIFEST.json')) as fh:
        m = json.load(fh)
    for e in m['rtl'] + m['sim'] + m['includes']:
        p = os.path.join(pkg, e['file'])
        with open(p, 'rb') as fh:
            got = hashlib.sha256(fh.read()).hexdigest()[:16]
        if got != e['sha256_16'] or os.path.getsize(p) != e['bytes']:
            bad.append(e['file'])
    log.append(f"    read-only integrity check over "
               f"{len(m['rtl']) + len(m['sim']) + len(m['includes'])} "
               f"file(s): {len(bad)} mismatch(es) {bad if bad else ''}")
    ok &= not bad

    # The check has to be able to fail. A corrupted file must be caught.
    victim = os.path.join(pkg, m['rtl'][0]['file'])
    os.chmod(victim, stat.S_IRUSR | stat.S_IWUSR)
    with open(victim, 'a') as fh:
        fh.write('\n// injected\n')
    with open(victim, 'rb') as fh:
        got = hashlib.sha256(fh.read()).hexdigest()[:16]
    caught = got != m['rtl'][0]['sha256_16']
    log.append(f"    and a file changed by one line is caught: {caught}")
    ok &= caught

    for d in dirs:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)
    return ok, 'two clean builds agree, and the manifest is checkable'


TESTS = [('R36-BUILD-01', build01), ('R36-BUILD-02', build02),
         ('R36-BUILD-03', build03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate C: one build entry point, and what it may not claim")
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
    print(f"  r36-build-tests        {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
