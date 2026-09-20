#!/usr/bin/env python3
"""
R2M-01 .. R2M-08 -- doc/phase2_tasks_and_acceptance.md R2-13.

"为证明新 gate 有效，以下受控 mutation 各自必须被指定测试抓住，未变异正例通过."

Each entry breaks exactly one thing in an isolated copy of the tree and
requires the *named* test to fail because of it. Two conditions keep a pass
honest, and both are checked, not assumed:

  - the same command must PASS on an unmutated copy, so a mutation cannot be
    credited for a test that was already broken;
  - the mutated run must fail inside the named case, so an import error or a
    build failure elsewhere does not count as detection. The contract says
    this outright: "检测必须对应故障根因，不能靠无关的 import/build 失败算检出."

The candidate tree is hashed before and after; a mutation that escaped into
it would make every later result meaningless.

R2M-03, R2M-04 and R2M-07 are NOT RUN. Their detecting tests do not exist
yet -- MEM-01/04 need the four-tile device, MLIR-01/05 are R2-04 -- and a
mutation whose detector is missing cannot be said to have been caught. The
honest report is that the gate is absent, not that the break slipped past.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import build_tests as B    # noqa: E402
import r1lib as R          # noqa: E402


def patch(lab, rel, old, new, *, count=1):
    p = os.path.join(lab.dir, rel)
    text = open(p).read()
    if text.count(old) != count:
        raise AssertionError(
            f"{rel}: expected {count} occurrence(s) of the mutation site, "
            f"found {text.count(old)} -- the mutation did not apply")
    open(p, 'w').write(text.replace(old, new, count))


def run_in(lab, argv, timeout=3600):
    env = dict(os.environ)
    # The lab's own package, not the one this harness imported.
    env['PYTHONPATH'] = os.path.join(lab.dir, 'python')
    env.pop('TNCA_DEVICE', None)
    return subprocess.run(argv, cwd=lab.dir, capture_output=True, text=True,
                          timeout=timeout, env=env)


def suite(name, *cases):
    return [sys.executable, os.path.join('tools', 'r2', name)] + \
           (['--only'] + list(cases) if cases else [])


# --------------------------------------------------------------------------

def m01(lab):
    """Freeze the argument block at the first launch's values.

    This is the failure the whole dynamic-argument path exists to prevent:
    an artifact that works once and then silently reuses the pointers and
    the N it saw the first time.
    """
    patch(lab, 'python/tnca/runtime.py',
          "            block = abi.encode_block(module.params, resolved)",
          "            block = getattr(self, '_r2m01', None)  # R2M-01\n"
          "            if block is None:\n"
          "                block = abi.encode_block(module.params, resolved)\n"
          "                self._r2m01 = block")
    return suite('abi_tests.py', 'ABI-02'), 'ABI-02'


def m02(lab):
    """Report NPID as the tile count rather than the logical grid."""
    patch(lab, 'python/tnca/runtime.py',
          "                                        pid=(x, y, z), "
          "npid=(gx, gy, gz))",
          "                                        pid=(x, y, z), "
          "npid=(4, 1, 1))  # R2M-02")
    return suite('grid_tests.py', 'GRID-01'), 'GRID-01'


def m05(lab):
    """Stop checking that the manifest is the one that was built."""
    patch(lab, 'python/tnca/manifest.py',
          """    got = manifest_hash(manifest)
    if got != claimed:""",
          """    got = manifest_hash(manifest)
    if False:  # R2M-05""")
    return (suite('artifact_tests.py', 'ART-02-integrity'),
            'ART-02-integrity')


def m05b(lab):
    """Stop checking the artifact was built for this profile."""
    patch(lab, 'python/tnca/runtime.py',
          "        if man['target']['profile'] != self.profile:",
          "        if False:  # R2M-05b")
    return suite('artifact_tests.py', 'ART-03'), 'ART-03'


def m06(lab):
    """Believe a response from an older sequence."""
    patch(lab, 'python/tnca/transport.py',
          "                if got.seq == seq:",
          "                if got.seq <= seq:  # R2M-06")
    return suite('fault_tests.py', 'ERR-05'), 'ERR-05'


def m08(lab):
    """Restart the device between launches, quietly."""
    patch(lab, 'python/tnca/runtime.py',
          "            self.dev.upload(task['arg_addr'], task['block'])",
          "            self.dev.stop(); self.dev.start()  # R2M-08\n"
          "            self.dev.upload(task['arg_addr'], task['block'])")
    return suite('runtime_tests.py', 'RT-02'), 'RT-02'


MUTATIONS = [
    ('R2M-01', 'dynamic pointers and N frozen at the first launch', m01),
    ('R2M-02', 'NPID reported as the tile count, not the grid', m02),
    ('R2M-05', 'the artifact integrity check removed', m05),
    ('R2M-05b', 'the profile capability check removed', m05b),
    ('R2M-06', 'an old epoch allowed to complete a new event', m06),
    ('R2M-08', 'the simulator restarted between successful launches', m08),
]

NOT_YET = {
    'R2M-03': ('success reported before the last store retires',
               'the detector is MEM-04, which needs a delay-injection hook '
               'in the memory model (R2-10 RTL half). A mutation with no '
               'detector has not been caught -- it has not been tested'),
    'R2M-04': ('download bypassing dirty-cache coherence',
               'the detector is MEM-01, which needs the four-tile profile: '
               'the single-core memory model is a flat array with no cache '
               'to be dirty'),
    'R2M-07': ('the MLIR pass dumping without compiling',
               'the detectors are MLIR-01/05, which are R2-04 and not '
               'started; there is no main MLIR path to break yet'),
}


def _case_ids():
    """Every case ID this suite knows, for a filter that matched nothing.

    Read off the module's own tables rather than written out again: a second
    list of case IDs is a second thing to forget to update.
    """
    ids = set()
    for name, value in list(globals().items()):
        if name.startswith('_') or not name.isupper():
            continue
        if isinstance(value, dict):
            ids |= {k for k in value if isinstance(k, str) and '-' in k}
        elif isinstance(value, (list, tuple)):
            for item in value:
                if (isinstance(item, (list, tuple)) and item
                        and isinstance(item[0], str) and '-' in item[0]):
                    ids.add(item[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R2-13 mutation: each break must be caught by its named test")
    before = B.tree_hashes(ROOT)
    base = tempfile.mkdtemp(prefix='r2-mut-')
    results = []
    try:
        for cid, what, fn in MUTATIONS:
            if args.only and cid not in args.only:
                continue
            lab = B.Lab(base)
            try:
                argv, names = fn(lab)
                clean = B.Lab(base)
                base_r = run_in(clean, argv)
                clean.close()
                mut_r = run_in(lab, argv)
                caught = base_r.returncode == 0 and mut_r.returncode != 0
                detail = (f"baseline exit={base_r.returncode}, mutated "
                          f"exit={mut_r.returncode} ({names})")
                if caught:
                    # The named case must be the one that failed. Anything
                    # else means the tree stopped working, not that the
                    # checker noticed.
                    line = [l for l in mut_r.stdout.splitlines()
                            if names in l and 'PASS' not in l]
                    if not line:
                        caught = False
                        detail += ' -- it failed, but not in the named case'
                    else:
                        detail += f" -- {line[0].strip()[:70]}"
                elif base_r.returncode != 0:
                    detail += ' -- the baseline does not pass; nothing can '\
                              'be concluded'
            except Exception as e:
                caught, detail = False, f"{type(e).__name__}: {e}"
            finally:
                lab.close()
            results.append(dict(id=cid, passed=bool(caught), detail=detail))
            print(f"  {cid}  {'PASS' if caught else 'FAIL':<5s} {what}",
                  flush=True)
            print(f"           {detail}", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    for cid, (what, why) in sorted(NOT_YET.items()):
        if args.only and cid not in args.only:
            continue
        results.append(dict(id=cid, passed=False, detail=f"NOT RUN: {why}"))
        print(f"  {cid}  NOT RUN  {what}")
        print(f"           {why}")

    after = B.tree_hashes(ROOT)
    leaked = sorted(k for k in before if before[k] != after.get(k))
    if leaked:
        print(f"  CANDIDATE TREE MODIFIED: {leaked}")
        results.append(dict(id='R2M-ISOLATION', passed=False,
                            detail=f"leaked: {leaked}"))
    else:
        print("  candidate source hash unchanged by the mutations")

    if args.json:
        R.write_json(args.json, results)
    # A filter that matched nothing must not report PASS. `--only ART-02`
    # selected zero cases and exited 0, which is how a mutation that removed
    # the manifest integrity check looked like a caught one. An empty run is
    # not a clean run.
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are:")
        for cid in sorted(_case_ids()):
            print(f"      {cid}")
        return 2

    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-mutation-tests      FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-mutation-tests      PASS   {len(results)} mutation(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
