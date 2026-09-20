#!/usr/bin/env python3
"""
EV-01 .. EV-06 -- doc/phase1_tasks_and_acceptance.md R1-10.

The contract names six ways an evidence directory can be wrong and requires
`r1-verify-evidence` to return non-zero for each:

    缺文件、改一个输出字节、混入另一 commit/profile、删除一个测试、
    伪造 discovered count、缺失结束状态

A verifier nobody has lied to is a verifier nobody has tested, so each one is
injected here into a copy of a real evidence directory -- never the original --
and the verifier must produce a problem that NAMES it. "Returned non-zero" is
not enough on its own: a check that fails for an unrelated reason would pass
this test while missing the tampering entirely, so every case asserts on the
text of the problem and on that problem being absent from the untampered copy.

Four of the six also reseal `checksums.sha256` afterwards. Anyone who edits an
evidence directory and leaves the manifest stale is caught by arithmetic; the
interesting adversary recomputes it, and the set relations and re-derived
configuration are what has to catch that one.
"""
import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import r1lib as R  # noqa: E402

VERIFY = os.path.join(HERE, 'verify_evidence.py')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 16), b''):
            h.update(c)
    return h.hexdigest()


def reseal(run_dir):
    """Recompute checksums.sha256, as a careful forger would."""
    lines = []
    for base, _d, files in os.walk(run_dir):
        for f in sorted(files):
            p = os.path.join(base, f)
            if os.path.basename(p) == 'checksums.sha256':
                continue
            lines.append(f"{sha256(p)}  {os.path.relpath(p, run_dir)}")
    with open(os.path.join(run_dir, 'checksums.sha256'), 'w') as fh:
        fh.write('\n'.join(sorted(lines)) + '\n')


def verify(run_dir):
    r = subprocess.run([sys.executable, VERIFY, run_dir],
                       capture_output=True, text=True, timeout=R.CAPS['meta'])
    return r.returncode, r.stdout + r.stderr


def edit_json(path, fn):
    with open(path) as fh:
        data = json.load(fh)
    fn(data)
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)


# --------------------------------------------------------------------------
# Each injection returns the substring the verifier's complaint must contain.
# --------------------------------------------------------------------------

def ev01(d):
    """A missing file."""
    os.remove(os.path.join(d, 'results.json'))
    return 'missing evidence file'


def ev02(d):
    """One byte of one output changed."""
    target = None
    for base, _dd, files in os.walk(os.path.join(d, 'logs')):
        for f in files:
            if f.endswith('.stdout') and os.path.getsize(
                    os.path.join(base, f)) > 16:
                target = os.path.join(base, f)
                break
        if target:
            break
    if not target:
        raise AssertionError('no log with content to corrupt')
    with open(target, 'r+b') as fh:
        fh.seek(8)
        b = fh.read(1)
        fh.seek(8)
        fh.write(bytes([b[0] ^ 0x20]))
    # NOT resealed: a single flipped byte is exactly what the manifest is for.
    return 'content changed since the run'


def ev03(d):
    """A profile manifest from another configuration mixed in."""
    man = os.path.join(d, 'profiles', 'manifests.json')

    def bend(data):
        for name in data:
            data[name]['config_hash'] = 'f' * 64
    edit_json(man, bend)
    reseal(d)
    return 'does not follow from the archived spec'


def ev04(d):
    """A test deleted from the results."""
    res = os.path.join(d, 'results.json')
    inv = os.path.join(d, 'inventory.json')
    with open(res) as fh:
        data = json.load(fh)
    victim = sorted(data['statuses'])[0]
    del data['statuses'][victim]
    with open(res, 'w') as fh:
        json.dump(data, fh, indent=2)
    # A forger who removes the case removes it from both files.
    edit_json(inv, lambda i: i.__setitem__(
        'discovered', [x for x in i.get('discovered', []) if x != victim]))
    reseal(d)
    return victim


def ev05(d):
    """A faked discovered count."""
    inv = os.path.join(d, 'inventory.json')
    edit_json(inv, lambda i: i.__setitem__(
        'discovered', sorted(set(i.get('discovered', [])) |
                             {'BASE-CORE-invented-fast'})))
    reseal(d)
    return 'no recorded status'


def ev06(d):
    """No end state."""
    acc = os.path.join(d, 'acceptance.json')
    edit_json(acc, lambda a: a.pop('seconds', None))
    reseal(d)
    return 'no end state'


INJECTIONS = [
    ('EV-01', 'a missing evidence file', ev01),
    ('EV-02', 'one changed byte of one output', ev02),
    ('EV-03', 'a profile mixed in from another configuration', ev03),
    ('EV-04', 'a deleted test, removed from both files', ev04),
    ('EV-05', 'a faked discovered count', ev05),
    ('EV-06', 'a run with no end state', ev06),
]


def synthesise(into):
    """A real, small evidence directory to tamper with.

    Not the newest run on disk: this suite runs INSIDE the acceptance gate,
    where the newest directory is the one still being written, and depending
    on whichever run happened to be there last would make the result depend
    on the order somebody ran things in. So it makes its own, with one cheap
    entry -- a real gate run, writing every archive file the contract lists,
    just with little in it.
    """
    r = subprocess.run([sys.executable, os.path.join(HERE, 'gate.py'),
                        '--only', 'gate-selftest', '--evidence', into],
                       cwd=ROOT, capture_output=True, text=True,
                       timeout=R.CAPS['build'])
    # A gate run over one entry is NOT ACCEPTED -- almost everything is NOT
    # RUN -- and that is fine: what is being tested is the verifier, and a
    # directory it can read is all that is needed. What must not happen is
    # the run failing to produce one.
    if not os.path.exists(os.path.join(into, 'acceptance.json')):
        raise AssertionError(
            'the synthetic gate run produced no acceptance.json: '
            + (r.stderr or r.stdout)[-400:])
    return into


def newest_evidence():
    base = os.path.join(ROOT, 'sim', 'r1')
    if not os.path.isdir(base):
        return None
    runs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and os.path.exists(os.path.join(base, d, 'acceptance.json'))]
    return max(runs, key=os.path.getmtime) if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', help='the run directory to copy from')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R1-10 evidence tampering: six lies the verifier must catch")
    results = []
    tmp = tempfile.mkdtemp(prefix='r1-ev-')
    try:
        src = args.evidence
        if not src:
            try:
                src = synthesise(os.path.join(tmp, 'source'))
            except Exception as e:
                print(f"  could not make an evidence directory: {e}")
                if args.json:
                    R.write_json(args.json,
                                 [dict(id=c, passed=False, detail=str(e))
                                  for c, _w, _f in INJECTIONS])
                return 1
        print(f"  source: {src}")

        clean = os.path.join(tmp, 'clean')
        shutil.copytree(src, clean)
        base_rc, base_out = verify(clean)

        for cid, what, fn in INJECTIONS:
            if args.only and cid not in args.only:
                continue
            d = os.path.join(tmp, cid)
            shutil.copytree(src, d)
            try:
                needle = fn(d)
                rc, out = verify(d)
                named = needle in out
                # The same complaint must NOT already be there without the
                # tampering, or the case is passing on someone else's failure.
                pre_existing = needle in base_out
                ok = rc != 0 and named and not pre_existing
                detail = (f"exit {rc}; names the tampering: {named}; "
                          f"present before it: {pre_existing}")
                if not ok:
                    detail += ' | ' + ' / '.join(
                        l.strip() for l in out.splitlines()[:4])
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            results.append(dict(id=cid, passed=bool(ok), detail=detail))
            print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}")
            print(f"           {detail}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-evidence-tests      FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-evidence-tests      PASS   {len(results)} injection(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
