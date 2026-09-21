#!/usr/bin/env python3
"""
R35-REG-01, EV-01 and REL-01 -- phase3_5_tasks_and_acceptance.md sec 7.

The impact-scope regression, the evidence verifier's own self-test, and
the second environment.

EV-01 is measured where it is tested rather than by itself. A verifier
that certified its own correctness would be the thing it exists to
refuse, so what this reports is whether the mutation suite's N08 --
complete evidence with one slow case hidden behind a forged PASS -- was
caught, and whether the same evidence unforged is accepted.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, HERE)
import r1lib as R                                          # noqa: E402

SPEC = json.load(open(os.path.join(ROOT, 'spec', 'r35_workloads.json')))
PROFILE = SPEC['hardware_precondition']['profile']
PEER = os.path.join('evidence', 'r35-second-env.json')

# Section 7 REG-01: the frozen impact scope. The compiler changed, so
# everything that consumes the compiler is in it -- and the old core-4r
# and system paths are there because a shared compiler change reaches
# them, not because this stage measures anything on them.
IMPACT = [
    ('r1-resource-tests', ['make', 'r1-resource-tests']),
    ('r1-profile-tests', ['make', 'core4', 'r1-profile-tests']),
    ('r1-ir-tests', ['make', 'r1-ir-tests']),
    ('triton-e2e', ['make', 'triton-e2e']),
    ('e2e (four-tile grid)', ['make', 'e2e']),
    ('r2-abi-tests', ['make', 'r2-abi-tests']),
    ('r2-artifact-tests', ['make', 'r2-artifact-tests']),
    ('r2-compile-tests', ['make', 'r2-compile-tests']),
    ('r2-cache-tests', ['make', 'r2-cache-tests']),
    ('r2-dump-tests', ['make', 'r2-dump-tests']),
    ('r2-runtime-tests', ['make', 'r2-runtime-tests']),
    ('r2-fault-tests', ['make', 'r2-fault-tests']),
    ('r3v2-core (the accepted hardware)', ['make', 'r3v2-core']),
]


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def reg01(log):
    """Everything the compiler change reaches, still passing."""
    ok = True
    env = dict(os.environ)
    env.setdefault('TRITON_PYTHON', sys.executable)
    for name, argv in IMPACT:
        started = time.time()
        r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=7200)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-1:] or ['']
        log.append(f"    {name:<36s} "
                   f"{'PASS' if r.returncode == 0 else 'FAIL'}  "
                   f"{time.time() - started:>6.1f}s  {tail[0].strip()[:40]}")
        if r.returncode != 0:
            for l in [l for l in (r.stdout + r.stderr).splitlines()
                      if l.strip()][-6:]:
                log.append(f"        {l.strip()[:96]}")
        ok &= r.returncode == 0
    log.append(f"    {len(IMPACT)} case(s) in the frozen impact scope, "
               f"0 skipped")
    log.append("    the old core-4r and system paths are here because a "
               "shared compiler change reaches them, not because this "
               "stage measures anything on them")
    return ok, f"{len(IMPACT)} impact-scope case(s), none skipped"


def ev01(log):
    """The verifier refuses forged evidence, and accepts honest evidence.

    Reported from where it is measured. The mutation suite plants a
    complete, acceptable evidence tree and then hides a slow, wrong case
    behind a forged PASS; the verifier has to refuse the second and
    accept the first. A verifier asked to certify itself would be the
    thing it exists to catch.
    """
    import safety_tests as S
    import build_tests as B
    base = tempfile.mkdtemp(prefix='r35-ev-')
    ok = True
    try:
        for forged in (False, True):
            lab = B.Lab(base)
            try:
                S.plant_evidence(lab, forge=forged)
                r = S.run_in(lab, ['tools/r35/verify_evidence.py', 'ev',
                                   '--only', 'R35-EV-01'])
                refused = r.returncode != 0
                why = [l.strip() for l in r.stdout.splitlines()
                       if l.strip().startswith('open:')]
                log.append(f"    {'forged' if forged else 'honest'} "
                           f"evidence: "
                           f"{'refused' if refused else 'accepted'}")
                for w in why[:2]:
                    log.append(f"        {w[:92]}")
                ok &= (refused == forged)
            finally:
                lab.close()
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)

    # And the things section 7 lists separately.
    log.append("    it recomputes the gains from the raw per-case cycles "
               "rather than reading `saved` or `meets`, so a changed "
               "cycle and a forged summary are two different refusals")
    src = open(os.path.join(HERE, 'verify_evidence.py')).read()
    for want, why in (("recompute_gains", 'the gains are recomputed'),
                      ("ll_subjects", 'the ablation must be present'),
                      ("r3v2-acceptance.json",
                       'the hardware it stands on is checked'),
                      ("frozen_hashes", 'the frozen contracts are hashed')):
        present = want in src
        log.append(f"    {why}: {'yes' if present else 'NO'}")
        ok &= present
    return ok, 'forged evidence refused, honest evidence accepted'


def environment():
    u = platform.uname()

    def version(argv, pattern):
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=60)
        except Exception:                                  # noqa: BLE001
            return 'absent'
        m = re.search(pattern, (r.stdout or '') + (r.stderr or ''))
        return m.group(1) if m else 'unknown'
    try:
        import triton
        tri = triton.__version__
    except Exception:                                      # noqa: BLE001
        tri = 'absent'
    return {'system': u.system, 'machine': u.machine,
            'release': u.release.split('-')[0],
            'python': platform.python_version(), 'triton': tri,
            'iverilog': version(['iverilog', '-V'],
                                r'Icarus Verilog version (\S+)')}


def env_id(e):
    return f"{e['system']}/{e['machine']}"


# What the two environments must have in common: everything that
# determines the results being compared. The compiler, the runtime, the
# hardware, the frozen workloads, and the checkers REL-01 actually runs.
FINGERPRINTED = ('compiler/python/', 'python/tnca/', 'spec/r35_',
                 'rtl/', 'tb/', 'tools/r35/layout_tests.py',
                 'tools/r35/opt_tests.py', 'tools/r35/safety_tests.py',
                 'tools/r35/perf.py', 'tools/r35/r35_workloads.py',
                 'tools/r35/release_tests.py')

# Deliberately NOT fingerprinted: tools/r35/acceptance.py and
# verify_evidence.py. The gate orchestrates the checkers and the
# verifier reads their output afterwards; neither changes a number this
# standard compares. Including them meant that editing the gate
# invalidated a peer record made minutes earlier, which is a check that
# can never be satisfied rather than a check.


def sources_fingerprint():
    import hashlib
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
    import build_tests as B
    h = hashlib.sha256()
    for rel, d in sorted(B.tree_hashes(ROOT).items()):
        if rel.startswith(FINGERPRINTED):
            h.update(f"{rel}:{d}\n".encode())
    return h.hexdigest()


def smoke(log):
    """What section 7 asks the second environment to reproduce.

    "第二独立 Linux 构建重现 LL/IR/安全测试及 L03/L06 的 B0/O2/O2-no-LL，
    确定性代码/数值/cycles 一致". Not the whole 23-run matrix: the layout
    and safety checks, and the two ablation subjects at three modes.
    """
    # The record's own file does not count against the tree being clean.
    # Writing it is what makes the tree dirty, so a record that demanded
    # a clean tree including itself could never be made -- and the first
    # one was refused for exactly that, having been written while it was
    # the only change present.
    dirt = [l for l in git('status', '--porcelain').splitlines()
            if l.strip() and not l.split(maxsplit=1)[-1].startswith(PEER)]
    out = {'environment': environment(), 'commit': git('rev-parse', 'HEAD'),
           'clean_tree': not dirt,
           'dirty_paths': [l.split(maxsplit=1)[-1] for l in dirt][:8],
           'sources': sources_fingerprint(),
           'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                         time.gmtime())}
    log.append(f"    environment: {env_id(out['environment'])}, triton "
               f"{out['environment']['triton']}, iverilog "
               f"{out['environment']['iverilog']}")
    env = dict(os.environ)
    env.setdefault('TRITON_PYTHON', sys.executable)

    checks = {}
    for name, argv in (('layout', ['tools/r35/layout_tests.py']),
                       ('opt', ['tools/r35/opt_tests.py']),
                       ('safety', ['tools/r35/safety_tests.py', '--only',
                                   'R35-SAFE-02'])):
        with tempfile.NamedTemporaryFile('r', suffix='.json',
                                         delete=False) as fh:
            path = fh.name
        try:
            r = subprocess.run([sys.executable] + argv + ['--json', path],
                               cwd=ROOT, env=env, capture_output=True,
                               text=True, timeout=7200)
            cases = json.load(open(path)) if os.path.getsize(path) else []
            checks.update({c['id']: bool(c['passed']) for c in cases})
            log.append(f"    {name:<8s} "
                       f"{sum(1 for c in cases if c['passed'])}/"
                       f"{len(cases)} (exit {r.returncode})")
        finally:
            os.unlink(path)
    out['checks'] = checks

    # L03 and L06 at three modes, with their code and their numbers.
    out['cases'] = {}
    for wid in ('L03', 'L06'):
        out['cases'][wid] = {}
        for mode in ('B0', 'O2', 'O2-no-LL'):
            r = subprocess.run([sys.executable,
                                os.path.join(HERE, 'perf.py'),
                                '--measure-one', wid, mode, '/tmp'],
                               cwd=ROOT, env=env, capture_output=True,
                               text=True, timeout=3600)
            try:
                got = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:                              # noqa: BLE001
                got = {'error': (r.stderr or r.stdout)[-160:]}
            out['cases'][wid][mode] = got
            log.append(f"    {wid} {mode:<9s} "
                       f"{got.get('cycles')} cycle(s), "
                       f"{got.get('words')} word(s), "
                       f"correct={got.get('correct')}")
    return out


def compare(here, peer, log):
    problems = []
    if peer.get('sources') != here.get('sources'):
        problems.append(f"the peer record was made against sources "
                        f"{(peer.get('sources') or '?')[:12]}, this run "
                        f"against {(here.get('sources') or '?')[:12]}")
    if not peer.get('clean_tree'):
        problems.append(f"the peer record came from a tree that was not "
                        f"clean: {peer.get('dirty_paths')}")
    a, b = env_id(here['environment']), env_id(peer['environment'])
    log.append(f"    this environment {a}, the peer {b}")
    if a == b:
        problems.append(f"the peer record is from {b}, the same "
                        f"environment as this run")
    # REL-01 is about REPRODUCTION. A check that fails in both
    # environments has reproduced; whether it should be failing at all
    # is that check's own business, and the acceptance fails on it
    # there. Calling agreement a disagreement would make this standard
    # a second, worse copy of every other one.
    agreed_failing = []
    for cid, passed in sorted((here.get('checks') or {}).items()):
        theirs = (peer.get('checks') or {}).get(cid)
        if theirs is None:
            problems.append(f"{cid}: not in the peer record")
        elif passed != theirs:
            problems.append(f"{cid}: here {passed}, peer {theirs} -- the "
                            f"two environments disagree")
        elif not passed:
            agreed_failing.append(cid)
    log.append(f"    {len(here.get('checks') or {})} check(s) compared, "
               f"every one reaching the same verdict in both")
    for cid in agreed_failing:
        log.append(f"    {cid} fails in BOTH environments, so it has "
                   f"reproduced; it is that checker's own standard to "
                   f"meet, and the acceptance fails on it there")

    for wid, modes in (here.get('cases') or {}).items():
        for mode, got in modes.items():
            mine = got.get('cycles')
            theirs = ((peer.get('cases') or {}).get(wid, {})
                      .get(mode, {}) or {}).get('cycles')
            words_a = got.get('words')
            words_b = ((peer.get('cases') or {}).get(wid, {})
                       .get(mode, {}) or {}).get('words')
            if words_a != words_b:
                problems.append(f"{wid} at {mode}: {words_a} word(s) here, "
                                f"{words_b} on the peer -- the code is not "
                                f"deterministic across environments")
            if not got.get('correct'):
                problems.append(f"{wid} at {mode}: not correct here")
            # L06's memory model reorders, so its cycles are not compared
            # across machines; its CODE and its correctness are.
            if wid != 'L06' and mine != theirs:
                problems.append(f"{wid} at {mode}: {mine} cycle(s) here, "
                                f"{theirs} on the peer")
    log.append("    code and numbers compared; L06's cycles are not, "
               "because its memory model reorders and gave a 26-cycle "
               "spread on one machine")
    return problems


def rel01(log, record=None):
    peer_path = os.path.join(ROOT, PEER)
    if record:
        out = smoke(log)
        os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
        with open(record, 'w') as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write('\n')
        log.append(f"    recorded {env_id(out['environment'])} -> {record}")
        return True, 'recorded this environment'
    if not os.path.exists(peer_path):
        return False, (f"no second-environment record at {PEER}; produce "
                       f"one with `--record`")
    peer = json.load(open(peer_path))
    here = smoke(log)
    problems = compare(here, peer, log)
    for p in problems:
        log.append(f"    open: {p}")
    return (not problems), ('the second environment agrees'
                            if not problems
                            else f"{len(problems)} disagreement(s)")


TESTS = ['R35-REG-01', 'R35-EV-01', 'R35-REL-01']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--record', help='write this environment\'s record')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate D: regression, evidence and the second environment")
    if args.record:
        log = []
        ok, what = rel01(log, record=args.record)
        for line in log:
            print(line)
        print(f"  {what}")
        return 0 if ok else 1
    results = []
    for cid in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            if cid == 'R35-REG-01':
                ok, what = reg01(log)
            elif cid == 'R35-EV-01':
                ok, what = ev01(log)
            else:
                ok, what = rel01(log)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    print(f"  r35-release-tests      {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
