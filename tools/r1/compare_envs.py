#!/usr/bin/env python3
"""
Compare two acceptance environments -- R1-09's second half.

    "两个环境全部 R1 测试通过；ISA/profile/XLS 生成物、目标 binary 的 hash
     相同。中间 host 可执行文件若含路径/时间戳可不要求 byte-identical，但规范
     化去除字段必须预先列入受控清单，不能去除语义配置、数值或 cycle 计数。"

Two environments both reporting ACCEPTED is not the claim. The claim is that
they produced the SAME machine, and that is a comparison neither environment
can make about itself. So this takes the archived evidence from both and
compares what the contract names: the generated ISA files, the profile
configuration hashes, and the built artefacts.

Where a difference is allowed it is allowed by a rule written down in advance
-- spec/r1_normalization.json -- and never by one invented here to make a
comparison pass. The normalisation list says what may be removed and, more
importantly, what may not: parameter values, timescales, scope names, numeric
literals and cycle counts.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import r1lib as R  # noqa: E402

NORMALIZATION = os.path.join(ROOT, 'spec', 'r1_normalization.json')


def find_runs(root):
    """Every evidence directory under `root`, whatever it was unpacked as."""
    runs = []
    for base, dirs, files in os.walk(root):
        if 'acceptance.json' in files and 'run.json' in files:
            runs.append(base)
            dirs[:] = []
    return sorted(runs)


def load(run, name):
    p = os.path.join(run, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# R1-09 asks whether two environments agree, and R1-12 signs off once
# everything else has. Neither question can be answered by one run, so the
# gate leaves them NOT PASS and says so; this is where they are settled.
DEFERRED = ('R1-09', 'R1-12')


def combined_verdict(a, b, problems):
    """The verdict over BOTH environments.

    A single gate run reports NOT ACCEPTED while R1-09 stands open, because
    one environment cannot say whether two agree. That is not a defect in the
    run and it is not a pass either -- it is a question a single run is not
    entitled to answer. This answers it, over two runs that have already
    agreed on everything else.
    """
    out = dict(verdict='NOT ACCEPTED', resolved={}, problems=list(problems))
    accs = [load(a, 'acceptance.json'), load(b, 'acceptance.json')]
    if not all(accs):
        out['problems'].append('one of the runs has no acceptance.json')
        return out

    for wp in DEFERRED:
        out['resolved'][wp] = 'PASS'

    for tag, acc in zip('AB', accs):
        wps = acc.get('work_packages') or {}
        if not wps:
            out['problems'].append(
                f"environment {tag} recorded no work-package roll-up; it was "
                "produced by a gate that predates it")
            return out
        unmet = sorted(k for k, v in wps.items()
                       if v.get('verdict') != 'PASS' and k not in DEFERRED)
        if unmet:
            out['problems'].append(
                f"environment {tag} has work package(s) not passing that "
                f"this comparison cannot resolve: {', '.join(unmet)}")
        # A deferred package still has to be deferred for the RIGHT reason:
        # if R1-09 failed on this environment for something other than the
        # A/B question -- a missing tool, a platform that is not Linux --
        # resolving it here would be resolving a different problem.
        ev = (wps.get('R1-09') or {}).get('evidence', '')
        if 'R1-09 does not accept' in ev:
            out['problems'].append(
                f"environment {tag} is not an R1 acceptance platform, so "
                "there is nothing here for the A/B comparison to confirm")

    if not out['problems']:
        out['verdict'] = 'ACCEPTED'
    return out


def compare(a, b):
    problems = []
    notes = []

    ra, rb = load(a, 'run.json'), load(b, 'run.json')
    ca = (ra or {}).get('git', {}).get('commit')
    cb = (rb or {}).get('git', {}).get('commit')
    notes.append(f"A commit {ca}  B commit {cb}")
    if not ca or ca != cb:
        problems.append(
            f"the two environments built different commits: {ca} vs {cb}")

    # The verdicts. Both must be ACCEPTED, and the counts must agree: two
    # runs can both be "ACCEPTED" over different numbers of cases only if
    # one of them was run with a narrower inventory.
    aa, ab = load(a, 'acceptance.json'), load(b, 'acceptance.json')
    for tag, acc in (('A', aa), ('B', ab)):
        if not acc:
            problems.append(f"environment {tag} has no acceptance.json")
        elif acc.get('verdict') != 'ACCEPTED':
            problems.append(f"environment {tag}: {acc.get('verdict')}")
    if aa and ab:
        notes.append(f"A {aa.get('passed')}/{aa.get('required')} pass, "
                     f"B {ab.get('passed')}/{ab.get('required')}")
        if aa.get('required') != ab.get('required'):
            problems.append(
                f"different inventories: {aa.get('required')} required cases "
                f"in A, {ab.get('required')} in B")

    # Every case must have reached the same verdict in both. A pair of runs
    # that pass different subsets is two partial results, not two agreeing
    # ones.
    sa = (load(a, 'results.json') or {}).get('statuses', {})
    sb = (load(b, 'results.json') or {}).get('statuses', {})
    differ = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    if differ:
        problems.append(f"{len(differ)} case(s) reached different verdicts: "
                        + ', '.join(f"{k} A={sa.get(k)} B={sb.get(k)}"
                                    for k in differ[:5]))

    # The configuration. profiles/manifests.json carries the hash of what
    # decides what each profile IS -- elaboration parameters and capabilities.
    ma, mb = (load(a, 'profiles/manifests.json') or {},
              load(b, 'profiles/manifests.json') or {})
    if set(ma) != set(mb):
        problems.append(f"different profiles: {sorted(set(ma) ^ set(mb))}")
    for name in sorted(set(ma) & set(mb)):
        ha, hb = ma[name].get('config_hash'), mb[name].get('config_hash')
        if ha != hb:
            problems.append(f"profile {name}: config hash {ha} vs {hb}")
        else:
            notes.append(f"profile {name}: {str(ha)[:16]} in both")

    # The generated ISA and target files, byte for byte. These are checked
    # into the tree and regenerated by the build, so a difference here means
    # the two environments generated different sources from the same inputs.
    for sub in ('profiles',):
        fa = {f for f in os.listdir(os.path.join(a, sub))} \
            if os.path.isdir(os.path.join(a, sub)) else set()
        fb = {f for f in os.listdir(os.path.join(b, sub))} \
            if os.path.isdir(os.path.join(b, sub)) else set()
        for f in sorted(fa & fb):
            if not f.endswith('.json') or f == 'manifests.json':
                continue
            with open(os.path.join(a, sub, f), 'rb') as fh:
                da = fh.read()
            with open(os.path.join(b, sub, f), 'rb') as fh:
                db = fh.read()
            if da != db:
                problems.append(f"{sub}/{f} differs between the environments")

    # The built artefacts. An iverilog `.vvp` embeds symbol names derived from
    # load addresses, so it is NOT required to be byte-identical -- and the
    # fields that may be normalised away are listed in advance, in
    # spec/r1_normalization.json, which also lists what may not be.
    ba = (load(a, 'build/artifacts.json') or {})
    bb = (load(b, 'build/artifacts.json') or {})
    common = sorted(set(ba) & set(bb))
    if not common:
        problems.append('neither environment archived any built artefact')
        return problems, notes

    rules = {}
    if os.path.exists(NORMALIZATION):
        with open(NORMALIZATION) as fh:
            rules = json.load(fh)
    declared = rules.get('artifacts', {}).get('*.vvp', {}).get('removed', [])
    never = rules.get('never_removed', [])

    identical = [f for f in common if ba[f]['sha256'] == bb[f]['sha256']]
    notes.append(f"{len(identical)}/{len(common)} built artefact(s) are "
                 f"byte-identical")

    # The question R1-09 asks is whether the two environments produced the
    # same MACHINE, and an iverilog .vvp is never byte-identical across two
    # builds: Icarus names its scope and parameter symbols after the address
    # of the object that created them. The contract allows normalising such
    # fields provided every one of them is listed in advance, which is what
    # spec/r1_normalization.json is; the archives carry the normalised digest
    # so this comparison can use it without the binaries.
    missing_norm = [f for f in common
                    if not ba[f].get('normalized') or not bb[f].get('normalized')]
    if missing_norm:
        problems.append(
            f"{len(missing_norm)} artefact(s) have no normalised digest in "
            f"the archive, e.g. {', '.join(missing_norm[:3])}: the raw hash "
            "alone cannot answer whether two environments built the same "
            "thing")
        return problems, notes

    bad_norm = [f for f in common
                if str(ba[f]['normalized']).startswith('ERROR')
                or str(bb[f]['normalized']).startswith('ERROR')]
    if bad_norm:
        problems.append(
            f"normalisation failed for {', '.join(bad_norm[:3])}; a digest "
            "that could not be computed is not a digest that matched")
        return problems, notes

    agree = [f for f in common
             if ba[f]['normalized'] == bb[f]['normalized']]
    differ = [f for f in common if f not in agree]
    notes.append(
        f"{len(agree)}/{len(common)} agree after the "
        f"{len(declared)} declared normalisation rule(s); never removed: "
        f"{', '.join(never[:3])}")
    if differ:
        problems.append(
            f"{len(differ)} artefact(s) differ even normalised: "
            + ', '.join(differ[:5])
            + ". The declared rules remove addresses and installation paths "
              "and nothing else, so a difference here is a difference in "
              "what was built")
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='directory holding both environments\' evidence')
    ap.add_argument('--json')
    args = ap.parse_args()

    runs = find_runs(args.root)
    print("R1-09 cross-environment comparison")
    for r in runs:
        print(f"  found: {r}")
    if len(runs) != 2:
        print(f"  r1-compare-envs        FAIL   need exactly two evidence "
              f"directories, found {len(runs)}")
        if args.json:
            R.write_json(args.json, dict(ok=False, runs=runs))
        return 1

    problems, notes = compare(runs[0], runs[1])
    for n in notes:
        print('    ' + n)
    final = combined_verdict(runs[0], runs[1], problems)
    if args.json:
        R.write_json(args.json, dict(ok=not final['problems'],
                                     verdict=final['verdict'],
                                     resolved=final['resolved'],
                                     problems=final['problems'],
                                     notes=notes, runs=runs))
    if final['problems']:
        print(f"  r1-compare-envs        FAIL   {len(final['problems'])} "
              f"problem(s)")
        for p in final['problems'][:10]:
            print('    ' + p)
        return 1
    print("  r1-compare-envs        PASS   the two environments agree")
    print(f"    R1-09 and R1-12 resolved here: {final['resolved']}")
    print(f"    FINAL VERDICT: {final['verdict']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
