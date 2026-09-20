#!/usr/bin/env python3
"""
BUILD-01 .. BUILD-07 -- doc/phase1_tasks_and_acceptance.md R1-03.

Does the build graph actually invalidate what it should? The audit found
`tb_core.vvp` with neither `rtl/gen` nor any `.vh` among its prerequisites, so
editing either left a stale simulator in place and the next run "verified" RTL
that no longer existed. A prerequisite list is a claim, and this is the
experiment that tests it.

Every experiment runs in an isolated copy of the tree (contract: "BUILD-01 至
BUILD-04 在隔离 checkout 执行，不污染候选树"), and the candidate's own files
are hashed before and after to prove nothing leaked back.

`make -n` is not accepted as final evidence either, so each experiment builds
for real and compares the artefact.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1lib as R  # noqa: E402

ROOT = R.ROOT
TARGET = 'sim/build/tb_core.vvp'


def sh(cmd, cwd, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def digest(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 16), b''):
            h.update(c)
    return h.hexdigest()


SKIP_DIRS = {'.git', 'sim', 'build', '__pycache__', '.venv'}


def source_rels(root, use_git=True):
    """The files that make up the candidate tree.

    A lab copy has no .git, so `git ls-files` there returns nothing -- which
    is how BUILD-04 first reported a clean tree while the file it had just
    corrupted sat on disk, and how a lab built inside a lab came out empty.
    Falls back to walking when git cannot answer.
    """
    if use_git and os.path.isdir(os.path.join(root, '.git')):
        r = subprocess.run(['git', 'ls-files', '--cached', '--others',
                            '--exclude-standard'],
                           cwd=root, capture_output=True, text=True)
        return r.stdout.split()
    rels = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            rels.append(os.path.relpath(os.path.join(base, f), root))
    return rels


def tree_hashes(root, use_git=True):
    """Hash the source tree, to prove an experiment leaked nothing into it."""
    rels = source_rels(root, use_git)
    out = {}
    for rel in rels:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            out[rel] = digest(p)
    return out


class Lab:
    """An isolated copy of the candidate tree."""

    def __init__(self, base, carry=()):
        self.dir = tempfile.mkdtemp(prefix='r1-build-', dir=base)
        # Tracked *and* untracked-but-not-ignored: the candidate under test is
        # the working tree, and copying only what git has committed would test
        # a different tree from the one the gate runs on.
        for rel in source_rels(ROOT):
            src, dst = os.path.join(ROOT, rel), os.path.join(self.dir, rel)
            if not os.path.exists(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        # `carry` brings ignored build products along -- for an experiment
        # about what a *stale* artefact does, rebuilding it would destroy the
        # very condition under test.
        for rel in carry:
            src, dst = os.path.join(ROOT, rel), os.path.join(self.dir, rel)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    def build(self, target=TARGET, extra=()):
        return sh(['make', target] + list(extra), self.dir)

    def touch_content(self, rel, text):
        """Change a file's CONTENT, not just its timestamp.

        A timestamp-only touch tests make's clock handling; the contract asks
        whether a real edit reaches the artefact, which is a different claim.
        """
        p = os.path.join(self.dir, rel)
        with open(p, 'a') as fh:
            fh.write(text)
        # Push the mtime clear of the artefact built moments ago. Filesystem
        # timestamp granularity is not the property under test -- whether the
        # dependency edge exists is -- and a same-second write would make a
        # present edge look absent.
        future = time.time() + 2
        os.utime(p, (future, future))

    def artefact(self, target=TARGET):
        return digest(os.path.join(self.dir, target))

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def build01(lab, log):
    """Editing core RTL, a .vh, generated RTL or the memory model rebuilds."""
    lab.build()
    base = lab.artefact()
    if base is None:
        return False, 'the baseline build produced no artefact'
    results = {}
    for rel, text in (
            ('rtl/core/tnca_decoder.v', '\n// BUILD-01 core rtl\n'),
            ('rtl/common/tnca_defs.vh', '\n`define R1_BUILD01_PROBE 1\n'),
            ('rtl/gen/tnca_cvt_comb_xls.v', '\n// BUILD-01 generated rtl\n'),
            ('tb/common/tnca_mem_model.v', '\n// BUILD-01 memory model\n')):
        before = lab.artefact()
        lab.touch_content(rel, text)
        r = lab.build()
        after = lab.artefact()
        rebuilt = (r.returncode == 0 and after is not None and after != before)
        results[rel] = rebuilt
        log.append(f"    {rel}: {'rebuilt' if rebuilt else 'NOT REBUILT'}")
    return all(results.values()), json.dumps(results)


def build02(lab, log):
    """A different profile must not reuse another profile's artefact."""
    lab.build('sim/build/tb_core.vvp')
    lab.build('sim/build/tb_core4.vvp')
    a = digest(os.path.join(lab.dir, 'sim/build/tb_core.vvp'))
    b = digest(os.path.join(lab.dir, 'sim/build/tb_core4.vvp'))
    distinct = a is not None and b is not None and a != b

    # And the fingerprints must differ, which is the part a file list misses.
    fa = open(os.path.join(lab.dir, 'sim/build/tb_core.fp')).read().strip()
    fb = open(os.path.join(lab.dir, 'sim/build/tb_core4.fp')).read().strip()
    log.append(f"    tb_core fp={fa[:12]} tb_core4 fp={fb[:12]}")
    return distinct and fa != fb, f"artefacts distinct={distinct}"


def build03(lab, log):
    """Editing a generator or the oracle it imports regenerates its output."""
    vec = 'sim/build/fma_vectors.txt'
    lab.build(vec)
    before = digest(os.path.join(lab.dir, vec))
    # fp32.py is the exact oracle gen_fma_vectors.py imports.
    lab.touch_content('tools/model/fp32.py', '\n# BUILD-03 probe\n')
    after = digest(os.path.join(lab.dir, vec))
    # The content may legitimately be identical -- a comment changes nothing --
    # so what is checked is that make re-ran the generator at all.
    r = lab.build(vec)
    reran = 'gen_fma_vectors.py' in r.stdout
    log.append(f"    regenerated={before != after} generator-re-ran={reran}")
    return reran, f"before={str(before)[:12]} after={str(after)[:12]}"


def build04(lab, log):
    """A stale generated file must fail check-generated, and be left alone."""
    before = tree_hashes(lab.dir, use_git=False)
    lab.touch_content('isa/tnca-isa-v0.1.json', '\n')
    r = sh(['make', 'r1-check-generated'], lab.dir)
    after = tree_hashes(lab.dir, use_git=False)
    # Only the file we deliberately corrupted may differ: the check must not
    # have repaired anything.
    changed = sorted(k for k in before
                     if before[k] != after.get(k)) if after else []
    ok = r.returncode != 0 and changed == ['isa/tnca-isa-v0.1.json']
    log.append(f"    exit={r.returncode} changed={changed}")
    return ok, f"check exit={r.returncode}"


def build05(lab, log):
    """Clean and incremental builds of the same input agree.

    Compared after normalisation, not byte for byte. Icarus names its symbols
    after the address of the object that created them, so ASLR makes two
    builds of identical input differ in every symbol name and nothing else.
    spec/r1_normalization.json lists exactly what is removed, in advance,
    because a normalisation invented after a comparison fails is not one.
    """
    def norm(path):
        r = sh([sys.executable, 'tools/r1/fingerprint.py', 'normalized', path],
               lab.dir)
        return r.stdout.strip() if r.returncode == 0 else None

    lab.build()
    incremental = norm(TARGET)
    raw_inc = lab.artefact()
    sh(['rm', '-rf', 'sim/build'], lab.dir)
    lab.build()
    clean = norm(TARGET)
    raw_clean = lab.artefact()
    log.append(f"    normalized: incremental={str(incremental)[:12]} "
               f"clean={str(clean)[:12]}")
    log.append(f"    raw bytes differ (expected, ASLR symbol names): "
               f"{raw_inc != raw_clean}")
    return incremental == clean and clean is not None, 'normalized digests'


def build06(lab, log):
    """No input change: nothing is recompiled.

    Checked by building for real, not with `make -n`. The fingerprint stamp is
    regenerated every time by design -- that is how it notices a parameter
    change -- so a dry run always *shows* the stamp command and then assumes
    its dependants are stale. Only a real build shows whether the stamp's
    content changed, which is the thing that matters. The contract says `make
    -n` is not acceptable as final evidence, and this is why.
    """
    lab.build()
    r = lab.build()
    recompiled = 'iverilog' in r.stdout
    log.append(f"    second build invoked iverilog: {recompiled}")
    return (r.returncode == 0 and not recompiled), r.stdout.strip()[:200]


def build07(lab, log):
    """An artefact from another profile is rejected BEFORE it is used.

    make cannot catch this and is not expected to: the planted file is newer
    than every prerequisite, so it is up to date by the only question make
    knows how to ask. The provenance record is what notices, and it has to be
    checked before a simulation runs rather than at build time.
    """
    lab.build('sim/build/tb_core4.vvp')
    lab.build('sim/build/tb_core.vvp')
    clean = sh(['make', 'r1-check-artifacts'], lab.dir)

    shutil.copy2(os.path.join(lab.dir, 'sim/build/tb_core4.vvp'),
                 os.path.join(lab.dir, 'sim/build/tb_core.vvp'))
    planted = sh(['make', 'r1-check-artifacts'], lab.dir)

    log.append(f"    before planting: exit={clean.returncode}")
    log.append(f"    after planting:  exit={planted.returncode} "
               f"{planted.stdout.strip()[:120]}")
    return (clean.returncode == 0 and planted.returncode != 0), \
        'provenance check before execution'


TESTS = [('BUILD-01', build01), ('BUILD-02', build02), ('BUILD-03', build03),
         ('BUILD-04', build04), ('BUILD-05', build05), ('BUILD-06', build06),
         ('BUILD-07', build07)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json', help='write results here')
    args = ap.parse_args()

    before = tree_hashes(ROOT)
    base = tempfile.mkdtemp(prefix='r1-build-lab-')
    results = []
    print("R1-03 build invalidation: seven experiments in isolated checkouts")
    try:
        for cid, fn in TESTS:
            if args.only and cid not in args.only:
                continue
            log = []
            lab = Lab(base)
            t0 = time.time()
            try:
                ok, detail = fn(lab, log)
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            finally:
                lab.close()
            results.append(dict(id=cid, passed=bool(ok), detail=detail,
                                seconds=round(time.time() - t0, 1)))
            print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {detail}")
            for line in log:
                print(line)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    after = tree_hashes(ROOT)
    leaked = sorted(k for k in before if before[k] != after.get(k))
    if leaked:
        print(f"  CANDIDATE TREE MODIFIED: {leaked}")
        results.append(dict(id='BUILD-ISOLATION', passed=False,
                            detail=f"leaked into the candidate: {leaked}"))
    else:
        print("  candidate tree unchanged by the experiments")

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-build-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-build-tests         PASS   {len(results)} experiment(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
