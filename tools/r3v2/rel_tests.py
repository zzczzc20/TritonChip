#!/usr/bin/env python3
"""
R3V2-REL-01 -- doc/phase3_tasks_and_acceptance.md G7.

The main environment runs everything; a second environment runs a smoke,
and the two must agree on the numbers.

The inventory is explicit that the second environment's runs are budgeted
separately and "do not combine with the main environment's to make up
coverage" -- a case run once in each place has been run once, not twice.
So this compares; it does not count.

What it refuses:

    a peer record produced at a different commit
    a peer record from a tree that was not clean
    a peer record from the SAME environment -- a second environment that
      is this machine is not a second environment
    any numerical or deterministic-count disagreement

Wall time is recorded and never compared: the two machines are not the
same speed and pretending otherwise would make the check fail for the one
reason it should not.

    tools/r3v2/rel_tests.py --record evidence/r3v2-second-env.json
    tools/r3v2/rel_tests.py --json out.json        # compares against it
"""
import argparse
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
import r1lib as R                                         # noqa: E402

DEFAULT_PEER = os.path.join('evidence', 'r3v2-second-env.json')
INVENTORY = json.load(open(os.path.join(ROOT, 'spec',
                                        'r3v2_inventory.json')))
SMOKE = INVENTORY['second_environment']

# The five component cases the smoke names, and the checker that owns each.
COMPONENT_OWNER = {
    'RF-01': 'rf_tests', 'RF-02': 'rf_tests', 'RF-03': 'rf_tests',
    'CORE-02': 'core_tests', 'FLOW-01': 'flow_tests',
}


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def tool_version(argv, pattern):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 'absent'
    m = re.search(pattern, (r.stdout or '') + (r.stderr or ''))
    return m.group(1) if m else 'unknown'


# What the two environments must have in common. The record is committed,
# which moves HEAD -- so comparing on the commit would mean the record can
# never match the run that reads it. This hashes the things a result could
# actually depend on, so committing the record leaves it valid and touching
# the design does not.
FINGERPRINTED = ('rtl/', 'tb/', 'tools/r3v2/', 'python/tnca/',
                 'compiler/python/', 'spec/r3v2_contract.json',
                 'spec/r3v2_inventory.json', 'spec/r1_profiles.json')


def sources_fingerprint():
    import hashlib
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
    import build_tests as B
    h = hashlib.sha256()
    for rel, d in sorted(B.tree_hashes(ROOT).items()):
        if rel.startswith(FINGERPRINTED):
            h.update(f"{rel}:{d}\n".encode())
    return h.hexdigest()


def environment():
    u = platform.uname()
    return {
        'system': u.system,
        'release': u.release.split('-')[0],
        'machine': u.machine,
        'python': platform.python_version(),
        'iverilog': tool_version(['iverilog', '-V'],
                                 r'Icarus Verilog version (\S+)'),
        'yosys': tool_version(['yosys', '-V'], r'Yosys (\S+)'),
    }


def env_id(e):
    """What makes two environments different for this purpose."""
    return f"{e['system']}/{e['machine']}"


# ---------------------------------------------------------------------------
# the smoke itself

def run_components(log):
    """The five component cases, in this environment."""
    out = {}
    by_checker = {}
    for case, checker in COMPONENT_OWNER.items():
        by_checker.setdefault(checker, []).append(f'R3V2-{case}')
    for checker, ids in sorted(by_checker.items()):
        with tempfile.NamedTemporaryFile('r', suffix='.json',
                                         delete=False) as fh:
            path = fh.name
        try:
            started = time.time()
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, f'{checker}.py'),
                 '--only'] + ids + ['--json', path],
                cwd=ROOT, capture_output=True, text=True, timeout=7200)
            took = time.time() - started
            cases = json.load(open(path)) if os.path.getsize(path) else []
            for c in cases:
                out[c['id']] = bool(c['passed'])
            log.append(f"    {checker:<12s} "
                       f"{sum(1 for c in cases if c['passed'])}/"
                       f"{len(cases)} passed in {took:.0f}s "
                       f"(exit {r.returncode})")
            if not cases:
                log.append(f"        {(r.stdout + r.stderr)[-300:]}")
        finally:
            os.unlink(path)
    return out


def run_core(log):
    """C02, C03 and C05 on the candidate, for their numbers."""
    import e2e_tests as E
    out = {}
    if not E.have_triton():
        log.append('    no locked Triton here; the core smoke cannot run')
        return None
    with tempfile.TemporaryDirectory() as tmp:
        for want in SMOKE['core']:
            cid = want['id']
            case = next(c for c in INVENTORY['core'] if c['id'] == cid)
            sub = []
            started = time.time()
            if case['workload'] == 'gemm':
                ok, values = E.run_gemm(case, 'candidate', sub, tmp)
            else:
                art = E.compile_case(case, tmp,
                                     E.profile_for(case, 'candidate'))
                ok, values = E.run_case(case, 'candidate', art, sub)
            took = time.time() - started
            out[cid] = {'ok': bool(ok),
                        'values': [float(v) for v in values],
                        'wall_seconds': round(took, 1)}
            log.append(f"    {cid:<5s} {'ok' if ok else 'FAILED':<7s} "
                       f"{len(values)} value(s) in {took:.0f}s")
    return out


def run_runtime(log):
    """RT-A/B/C: the same artifact, three alphas."""
    import e2e_tests as E
    import workloads as W
    from tnca import device
    from tnca.runtime import Session
    if not E.have_triton():
        log.append('    no locked Triton here; the runtime smoke cannot run')
        return None
    out = {}
    case = next(c for c in INVENTORY['core'] if c['id'] == 'C02')
    n = case['sizes']['n']
    slots = 16 * case['sizes']['grid']
    xs = W.inputs(slots, seed=case['seed'])
    ys = W.inputs(slots, seed=case['seed'] + 1)
    with tempfile.TemporaryDirectory() as tmp:
        art = E.compile_case(case, tmp, E.CAND)
        device.prepare(E.CAND, verbose=False)
        with Session(profile=E.CAND, deadline_cycles=50000) as s:
            mod = s.load(art.path)
            x = s.allocate(slots * 4, name='x')
            y = s.allocate(slots * 4, name='y')
            s.upload(x, W.f32(xs))
            s.upload(y, W.f32(ys))
            for rt in SMOKE['runtime']:
                spec = next(c for c in INVENTORY['runtime']
                            if c['id'] == rt)
                alpha = spec['alpha']
                o = s.allocate(slots * 4, name=f'o{rt}')
                s.upload(o, struct.pack('<I', W.SENTINEL) * slots)
                ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                    'n': n, 'alpha': alpha},
                              grid=(case['sizes']['grid'], 1, 1))
                s.wait(ev)
                got = W.unf32(s.download(o, slots * 4))[:n]
                want = [xs[i] * alpha + ys[i] for i in range(n)]
                out[rt] = {'ok': bool(ev.ok and got == want),
                           'alpha': alpha,
                           'values': [float(v) for v in got]}
                log.append(f"    {rt:<5s} alpha={alpha:<6} "
                           f"{'ok' if out[rt]['ok'] else 'FAILED'}")
    return out


def smoke(log):
    started = time.time()
    rec = {
        'contract_id': 'r3-v2-core-vrf',
        'commit': git('rev-parse', 'HEAD'),
        'clean_tree': git('status', '--porcelain') == '',
        'sources': sources_fingerprint(),
        'environment': environment(),
        'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    log.append(f"    environment: {env_id(rec['environment'])}, "
               f"python {rec['environment']['python']}, "
               f"iverilog {rec['environment']['iverilog']}")
    rec['components'] = run_components(log)
    rec['core'] = run_core(log)
    rec['runtime'] = run_runtime(log)
    rec['wall_seconds'] = round(time.time() - started, 1)
    return rec


# ---------------------------------------------------------------------------
# the comparison

def compare(here, peer, log):
    problems = []

    if peer.get('sources') != here.get('sources'):
        problems.append(f"the peer record was made against sources "
                        f"{(peer.get('sources') or '?')[:12]}, this run "
                        f"against {(here.get('sources') or '?')[:12]} -- "
                        f"the design or the checkers changed since")
    log.append(f"    sources {here.get('sources', '?')[:12]}, peer "
               f"recorded at commit {(peer.get('commit') or '?')[:12]}")
    if not peer.get('clean_tree'):
        problems.append('the peer record came from a tree that was not '
                        'clean')
    a, b = env_id(here['environment']), env_id(peer['environment'])
    log.append(f"    this environment {a}, the peer {b}")
    if a == b:
        problems.append(f"the peer record is from {b}, the same "
                        f"environment as this run -- that is not a second "
                        f"environment")

    for cid in sorted(COMPONENT_OWNER):
        k = f'R3V2-{cid}'
        h, p = here['components'].get(k), (peer.get('components') or
                                           {}).get(k)
        if h is None or p is None:
            problems.append(f"{k}: {'this run' if h is None else 'the peer'}"
                            f" has no result")
        elif not (h and p):
            problems.append(f"{k}: here {h}, peer {p}")
    log.append(f"    {len(COMPONENT_OWNER)} component case(s) compared")

    for section in ('core', 'runtime'):
        mine, theirs = here.get(section), peer.get(section)
        if mine is None or theirs is None:
            problems.append(f"the {section} smoke did not run "
                            f"{'here' if mine is None else 'on the peer'}")
            continue
        for cid in sorted(mine):
            if cid not in theirs:
                problems.append(f"{cid}: not in the peer record")
                continue
            if not (mine[cid]['ok'] and theirs[cid]['ok']):
                problems.append(f"{cid}: here ok={mine[cid]['ok']}, peer "
                                f"ok={theirs[cid]['ok']}")
            if mine[cid]['values'] != theirs[cid]['values']:
                bad = next((i for i, (u, v) in
                            enumerate(zip(mine[cid]['values'],
                                          theirs[cid]['values']))
                            if u != v), 0)
                problems.append(
                    f"{cid}: element {bad} is "
                    f"{mine[cid]['values'][bad]} here and "
                    f"{theirs[cid]['values'][bad]} on the peer")
        log.append(f"    {len(mine)} {section} case(s) compared element by "
                   f"element")

    log.append(f"    wall time recorded and not compared: "
               f"{here.get('wall_seconds', 0):.0f}s here, "
               f"{peer.get('wall_seconds', 0):.0f}s on the peer")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', help='write this environment\'s record '
                                     'and stop')
    ap.add_argument('--peer', default=os.environ.get('R3V2_PEER',
                                                     DEFAULT_PEER))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    if args.record:
        print("R3 v2 second-environment smoke")
        log = []
        rec = smoke(log)
        for line in log:
            print(line)
        os.makedirs(os.path.dirname(os.path.abspath(args.record)),
                    exist_ok=True)
        with open(args.record, 'w') as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
            fh.write('\n')
        print(f"\n  recorded {env_id(rec['environment'])} at "
              f"{rec['commit'][:12]} -> {args.record}")
        return 0

    if args.only and 'R3V2-REL-01' not in args.only:
        print(f"  the --only filter {args.only} matched nothing")
        return 2

    print("R3 v2 G7: the release, in two environments")
    log = []
    peer_path = args.peer if os.path.isabs(args.peer) \
        else os.path.join(ROOT, args.peer)
    if not os.path.exists(peer_path):
        ok, what = False, (f"no second-environment record at "
                           f"{args.peer}; produce one with "
                           f"`rel_tests.py --record`")
        problems = [what]
    else:
        peer = json.load(open(peer_path))
        here = smoke(log)
        problems = compare(here, peer, log)
        ok = not problems
        what = (f"the smoke agrees across "
                f"{env_id(here['environment'])} and "
                f"{env_id(peer['environment'])}") if ok else \
            f"{len(problems)} disagreement(s)"

    print(f"  R3V2-REL-01  {'PASS' if ok else 'FAIL':<5s} {what}")
    for line in log:
        print(line)
    for p in problems:
        print(f"    open: {p}")
    results = [dict(id='R3V2-REL-01', passed=bool(ok), detail=what)]
    if args.json:
        R.write_json(args.json, results)
    print()
    print(f"  r3v2-release-tests     {'PASS' if ok else 'FAIL':<6s} "
          f"1 check(s)")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
