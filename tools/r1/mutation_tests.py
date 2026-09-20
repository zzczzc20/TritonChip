#!/usr/bin/env python3
"""
MUT-01 .. MUT-07 -- doc/phase1_tasks_and_acceptance.md R1-11.

A checker that has never been seen to fail is a checker nobody has tested.
Each mutation below breaks exactly one thing in an isolated copy of the tree
and requires the named test to fail *because of it*; the same test must pass on
the unmutated baseline, and the candidate tree must be byte-identical
afterwards.

"The test count went up" is explicitly not evidence for this work package.
This is.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tests as B   # noqa: E402  (Lab, tree_hashes, sh)
import r1lib as R         # noqa: E402

ROOT = R.ROOT


def patch(lab, rel, old, new, *, count=1):
    """Replace exactly `count` occurrences, or fail loudly.

    A mutation that silently did not apply would make the test pass for the
    wrong reason -- the checker looks like it failed to notice, when in fact
    nothing was broken.
    """
    p = os.path.join(lab.dir, rel)
    with open(p) as fh:
        text = fh.read()
    if text.count(old) != count:
        raise AssertionError(
            f"{rel}: expected {count} occurrence(s) of the mutation site, "
            f"found {text.count(old)} -- the mutation did not apply")
    with open(p, 'w') as fh:
        fh.write(text.replace(old, new, count))


def run_in(lab, argv, timeout=1800):
    return subprocess.run(argv, cwd=lab.dir, capture_output=True, text=True,
                          timeout=timeout)


# --------------------------------------------------------------------------
# Each entry: (id, what it breaks, the mutation, the command, the case IDs
# whose failure proves the checker works)
# --------------------------------------------------------------------------

def mut01(lab):
    """Remove the ownership check on emitted writes."""
    patch(lab, 'compiler/python/tnca_cc/codegen.py',
          """                self.own.check_write(kind, int(dst[1:]),
                                     by=f"{mnem} {dst}")""",
          """                pass  # MUT-01: ownership check removed""")
    return ([sys.executable, 'tools/r1/resource_tests.py', '--only', 'OWN-01'],
            'OWN-01')


def mut02(lab):
    """Size the MMA footprint from the target's default K again."""
    patch(lab, 'compiler/python/tnca_cc/codegen.py',
          "        fm, fn, _k, bits = mma_shape_of(op.attrs, m=tm, n=tn)",
          "        fm, fn, _k = self.target.mma_shape; bits = 16  # MUT-02")
    return ([sys.executable, 'tools/r1/resource_tests.py',
             '--only', 'FP-02', 'FP-04'], 'FP-02/FP-04')


def mut03(lab):
    """Drop generated RTL and includes from the build dependencies."""
    patch(lab, 'Makefile',
          "RTL_DEPS   := $(RTL_ALL) $(RTL_VH)",
          "RTL_DEPS   := $(RTL_CORE) $(RTL_COMMON)  # MUT-03")
    # The fingerprint would still catch it, so remove that edge too: the
    # point is to show BUILD-01 detecting a missing dependency, not to show
    # two mechanisms covering for each other.
    patch(lab, 'Makefile',
          "FP_FILES_tb_core       := $(RTL_DEPS) $(TB)/core/tb_tnca_core.v $(TB)/common/tnca_mem_model.v",
          "FP_FILES_tb_core       := $(TB)/core/tb_tnca_core.v  # MUT-03")
    return ([sys.executable, 'tools/r1/build_tests.py', '--only', 'BUILD-01'],
            'BUILD-01')


def mut04(lab):
    """Claim the four-tile system has an SFU."""
    patch(lab, 'spec/r1_profiles.json',
          '"HAS_SFU": 0,\n        "HAS_ATOMICS": 1,\n        "HAS_SYS_ATOMICS": 0\n      },\n      "bench": "sim/build/tb_top.vvp"',
          '"HAS_SFU": 1,\n        "HAS_ATOMICS": 1,\n        "HAS_SYS_ATOMICS": 0\n      },\n      "bench": "sim/build/tb_top.vvp"')
    run_in(lab, [sys.executable, 'tools/isa/gen_target.py'])
    return ([sys.executable, 'tools/r1/profile_tests.py',
             '--only', 'CFG-01', 'CFG-03'], 'CFG-01/CFG-03')


def mut05(lab):
    """Put the exit-code-swallowing pipeline back.

    The site moved once already: these recipes used to invoke the test files
    directly, and now go through run_suite.py so each test gets its own ID.
    The mutation kept patching the old text, found nothing, and `patch()`
    refused -- which is the framework working. A mutation that silently fails
    to apply makes the checker look like it failed to notice, when in fact
    nothing was broken.
    """
    patch(lab, 'Makefile',
          "	       triton-e2e BASE-TRITON-CORE; \\",
          "	       triton-e2e BASE-TRITON-CORE 2>&1 | tail -4; \\")
    patch(lab, 'Makefile',
          "	       e2e BASE-TRITON-SYSTEM; \\",
          "	       e2e BASE-TRITON-SYSTEM 2>&1 | tail -4; \\")
    return ([sys.executable, 'tools/r1/selftest.py'], 'RUN-01')


# (id, what it protects, mutation, files to carry in from the candidate)
#
# `carry` exists for one reason: MUT-04 claims a capability the *built*
# hardware does not have. Rebuilding the benches from the mutated spec would
# make hardware and spec agree again and hide the very thing the check is for.
def cmake_var(name, default=None):
    """One variable out of the candidate's configured build directory."""
    cache = os.path.join(ROOT, 'compiler', 'build', 'CMakeCache.txt')
    if os.path.exists(cache):
        with open(cache) as fh:
            for line in fh:
                key, _, value = line.partition('=')
                if key.split(':')[0] == name:
                    return value.strip()
    return default


def rebuild_tnca_opt(lab):
    """Configure and build tnca-opt from the LAB's sources."""
    build = os.path.join(lab.dir, 'compiler', 'build')
    shutil.rmtree(build, ignore_errors=True)
    mlir = cmake_var('MLIR_DIR')
    llvm = cmake_var('LLVM_DIR')
    if not mlir or not llvm:
        raise AssertionError('no configured compiler/build to take MLIR_DIR '
                             'and LLVM_DIR from')
    c = run_in(lab, ['cmake', '-G', 'Ninja',
                     '-S', os.path.join(lab.dir, 'compiler'), '-B', build,
                     f'-DMLIR_DIR={mlir}', f'-DLLVM_DIR={llvm}',
                     '-DCMAKE_BUILD_TYPE=' + (cmake_var('CMAKE_BUILD_TYPE')
                                              or 'Release')],
                timeout=1800)
    if c.returncode != 0:
        raise AssertionError('cmake configure in the lab failed: '
                             + (c.stderr or c.stdout)[-400:])
    b = run_in(lab, ['ninja', '-C', build, 'tnca-opt'], timeout=3000)
    if b.returncode != 0:
        raise AssertionError('rebuilding tnca-opt in the lab failed: '
                             + (b.stderr or b.stdout)[-400:])


def mut06(lab):
    """Walk past an unsupported structure instead of refusing it."""
    patch(lab, 'compiler/lib/Dialect/TNCA/Transforms/VerifyAsyncOwnership.cpp',
          "      if (checkSupportBoundary(r, region)) {",
          "      if (false && checkSupportBoundary(r, region)) {  // MUT-06")
    # The rebuild is `prepare`, below: it has to run on the unmutated lab
    # too, or the baseline fails for want of a binary and the comparison
    # says nothing.
    return ([sys.executable, 'tools/r1/ir_tests.py',
             '--only', 'IR-02', 'IR-05'], 'IR-02/IR-05')


def mut07(lab):
    """Free a window when ANY holder completes, not the last one."""
    patch(lab, 'compiler/python/tnca_cc/footprints.py',
          "    def release(self, cmd_id):\n"
          "        self._live = [e for e in self._live if e[0] != cmd_id]",
          "    def release(self, cmd_id):  # MUT-07\n"
          "        freed = [e[1] for e in self._live if e[0] == cmd_id]\n"
          "        self._live = [e for e in self._live\n"
          "                      if e[0] != cmd_id\n"
          "                      and not any(e[1].overlaps(r) for r in freed)]")
    return ([sys.executable, 'tools/r1/resource_tests.py', '--only', 'OWN-07'],
            'OWN-07')


# (id, what it protects, mutation, files to carry in, preparation)
#
# `prepare` runs on the mutated lab AFTER the patch and on the clean lab as
# well, because whatever the experiment needs -- here a rebuilt `tnca-opt` --
# the baseline needs too. Without that the baseline failed for want of a
# binary and the comparison proved nothing.
MUTATIONS = [
    ('MUT-01', 'ownership check on emitted writes', mut01, (), None),
    ('MUT-02', 'MMA footprint from the actual K and dtype', mut02, (), None),
    ('MUT-03', 'generated RTL and includes as build dependencies', mut03, (),
     None),
    ('MUT-04', 'the system profile has no SFU', mut04, ('sim/build',), None),
    ('MUT-05', 'pipelines that swallow an exit code', mut05, (), None),
    ('MUT-06', 'an unsupported structure is refused, not walked past', mut06,
     (), rebuild_tnca_opt),
    ('MUT-07', 'a window survives until the last consumer releases', mut07, (),
     None),
]

NOT_YET = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R1-11 mutation: each break must be caught by its named test")
    before = B.tree_hashes(ROOT)
    base = tempfile.mkdtemp(prefix='r1-mut-')
    results = []
    try:
        for cid, what, fn, carry, prepare in MUTATIONS:
            if args.only and cid not in args.only:
                continue
            lab = B.Lab(base, carry=carry)
            try:
                argv, names = fn(lab)
                if prepare:
                    prepare(lab)
                # Baseline: the same command on the unmutated tree must pass.
                clean = B.Lab(base, carry=carry)
                if prepare:
                    prepare(clean)
                base_r = run_in(clean, argv)
                clean.close()
                mut_r = run_in(lab, argv)
                caught = (base_r.returncode == 0 and mut_r.returncode != 0)
                detail = (f"baseline exit={base_r.returncode}, "
                          f"mutated exit={mut_r.returncode} ({names})")
                if caught:
                    # and the failure must name the case, not be incidental
                    named = any(n.split('/')[0] in mut_r.stdout
                                for n in [names])
                    if not named:
                        caught = False
                        detail += ' -- failed, but not in the named case'
            except Exception as e:
                caught, detail = False, f"{type(e).__name__}: {e}"
            finally:
                lab.close()
            results.append(dict(id=cid, passed=bool(caught), detail=detail))
            print(f"  {cid}  {'PASS' if caught else 'FAIL':<5s} {what}")
            print(f"           {detail}")
    finally:
        shutil.rmtree(base, ignore_errors=True)

    for cid, (what, why) in NOT_YET.items():
        if args.only and cid not in args.only:
            continue
        results.append(dict(id=cid, passed=False, detail=f"NOT RUN: {why}"))
        print(f"  {cid}  NOT RUN  {what}")
        print(f"           {why}")

    after = B.tree_hashes(ROOT)
    leaked = sorted(k for k in before if before[k] != after.get(k))
    if leaked:
        print(f"  CANDIDATE TREE MODIFIED: {leaked}")
        results.append(dict(id='MUT-ISOLATION', passed=False,
                            detail=f"leaked: {leaked}"))
    else:
        print("  candidate source hash unchanged by the mutations")

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-mutation-tests      FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-mutation-tests      PASS   {len(results)} mutation(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
