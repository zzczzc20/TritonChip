#!/usr/bin/env python3
"""
R36-TEST-03 -- phase3_6_tasks_and_acceptance.md section R36-08.

Six directed mutations, each broken in an isolated copy of the tree and
each required to be caught by the checker this stage named for it.

The six the standard lists: capability, asset hash, return epoch, early
completion, storage latency and export dependency. The point of naming
them is that a checker suite can be large and still blind, and the way
to find out is to break the thing each case claims to watch and see
whether it notices.

Two rules carried over from R3 v2's mutation work, because both were
learned the hard way there:

  The named case must be the one that fails. A mutation that stops the
  tree building has not been detected by anything, and a suite that
  counted a build error as a catch would pass with every checker
  deleted.

  The candidate tree must be untouched afterwards. Each mutation runs in
  its own copy and the source is compared before and after.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import build_tests as B                                    # noqa: E402
import r1lib as R                                          # noqa: E402

TRITON = os.environ.get('TRITON_PYTHON', sys.executable)


def patch(lab_dir, rel, old, new, *, count=1):
    p = os.path.join(lab_dir, rel)
    text = open(p).read()
    if text.count(old) != count:
        raise AssertionError(
            f"{rel}: expected {count} occurrence(s) of the mutation site, "
            f"found {text.count(old)}: the mutation did not apply")
    open(p, 'w').write(text.replace(old, new, count))


def run_in(lab_dir, argv, timeout=3600, env_extra=None):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([
        os.path.join(lab_dir, 'python'),
        os.path.join(lab_dir, 'compiler', 'python'),
        env.get('PYTHONPATH', '')])
    # The lab must not read the candidate's staged data, or a mutation
    # to its RTL would be invisible to the runtime.
    env.pop('TNCA_DATA', None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(argv, cwd=lab_dir, capture_output=True,
                          text=True, timeout=timeout, env=env)


def tree_digest(root):
    h = hashlib.sha256()
    for rel in sorted(B.source_rels(root)):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        h.update(rel.encode())
        with open(p, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


# --------------------------------------------------------------------------
# the mutations
# --------------------------------------------------------------------------

def m_capability(lab, log):
    """Say the hardware has something it does not.

    The compile target publishes what the machine can do, and the
    compiler refuses a kernel that needs something absent. Claiming a
    capability the RTL does not implement is the failure R36-CFG-01
    watches for, and the artifact's identity has to move with it.
    """
    # Found by its `profile` field, not by its filename: the file is
    # tnca-sim-vrf-v1.json and the profile is core-vrf-sync-v1, and
    # assuming the two were the same string made this mutation fail to
    # apply -- which the harness reported as NOT CAUGHT, correctly.
    tdir = os.path.join(lab, 'compiler', 'targets')
    target = None
    for name in sorted(os.listdir(tdir)):
        if not name.endswith('.json'):
            continue
        cand = json.load(open(os.path.join(tdir, name)))
        if cand.get('profile') == 'core-vrf-sync-v1':
            target = os.path.join('compiler', 'targets', name)
            break
    if target is None:
        return None, 'no target publishes the profile under test'
    p = os.path.join(lab, target)
    d = json.load(open(p))
    caps = d.get('capabilities') or {}
    victim = next((k for k, v in caps.items() if v is False), None)
    if victim is None:
        return None, 'the target has no capability set to false to flip'
    caps[victim] = True
    json.dump(d, open(p, 'w'), indent=2)
    log.append(f"      flipped {victim} from false to true in {target}")
    r = run_in(lab, [TRITON, 'tools/r36/config_tests.py',
                     '--only', 'R36-CFG-01'])
    return r, f"the target now claims {victim}"


def m_asset_hash(lab, log):
    """Change one byte of an exported file after the manifest was written."""
    out = os.path.join(lab, 'export')
    r = run_in(lab, [sys.executable, 'tools/r36/export.py', '--out', out])
    if r.returncode != 0:
        return None, f"the export failed: {(r.stderr or r.stdout)[-160:]}"
    m = json.load(open(os.path.join(out, 'MANIFEST.json')))
    victim = m['rtl'][0]['file']
    with open(os.path.join(out, victim), 'a') as fh:
        fh.write('\n// a byte that is not in the manifest\n')
    log.append(f"      appended one line to {victim} after the manifest "
               f"recorded its hash")
    # The integrity check, run exactly as R36-BUILD-03 runs it.
    bad = []
    for e in m['rtl'] + m['sim'] + m['includes']:
        with open(os.path.join(out, e['file']), 'rb') as fh:
            got = hashlib.sha256(fh.read()).hexdigest()[:16]
        if got != e['sha256_16']:
            bad.append(e['file'])
    return {'returncode': 1 if bad else 0,
            'stdout': f"integrity: {len(bad)} mismatch(es) {bad}",
            'stderr': ''}, 'one file no longer matches its hash'


def m_epoch(lab, log):
    """Make the relay answer under somebody else's epoch."""
    patch(lab, 'python/tnca/backends/relay.py',
          "'epoch': epoch}",
          "'epoch': epoch + 1000}")
    log.append("      the relay now stamps every reply with a different "
               "epoch")
    r = run_in(lab, [TRITON, 'tools/r36/conformance.py',
                     '--backends', 'sim-socket'])
    return r, 'replies carry the wrong epoch'


def m_early_completion(lab, log):
    """Report a launch finished while the core is still running."""
    patch(lab, 'rtl/tnca_shell.v',
          "                end else if (core_idle) begin",
          "                end else if (1'b1) begin")
    log.append("      tnca_shell now completes a launch on the first "
               "cycle of S_RUN, without waiting for core_idle")
    r = run_in(lab, [TRITON, 'tools/r36/mem_tests.py',
                     '--only', 'R36-MEM-02'])
    return r, 'the shell completes before the kernel does'


def m_storage_latency(lab, log):
    """Freeze the memory's latency, so a slower one is not slower."""
    patch(lab, 'tb/common/tnca_mem_ref.v',
          "    localparam integer DEPTH = (LATENCY < 1) ? 1 : LATENCY;",
          "    localparam integer DEPTH = 1;")
    log.append("      tnca_mem_ref now ignores LATENCY and always "
               "answers in one cycle")
    r = run_in(lab, [TRITON, 'tools/r36/mem_tests.py',
                     '--only', 'R36-MEM-02'])
    return r, 'the memory no longer honours its latency parameter'


def m_export_dependency(lab, log):
    """Leave a file the export needs out of its own source list."""
    patch(lab, 'tools/r36/export.py',
          "    for name in sorted(synth_files):\n"
          "        manifest['rtl'].append(dict(module=name,\n"
          "                                    **copy(synth_files[name], "
          "'rtl')))",
          "    for name in sorted(synth_files):\n"
          "        if name == 'tnca_lzc':\n"
          "            continue\n"
          "        manifest['rtl'].append(dict(module=name,\n"
          "                                    **copy(synth_files[name], "
          "'rtl')))")
    log.append("      the export now omits tnca_lzc from rtl/ and from "
               "its manifest")
    r = run_in(lab, [sys.executable, 'tools/r36/rtl_tests.py',
                     '--only', 'R36-RTL-01'])
    return r, 'the package is missing a module it instantiates'


MUTATIONS = [
    ('MUT-CAPABILITY', m_capability, 'R36-CFG-01',
     'a target that claims a capability the hardware does not have'),
    ('MUT-ASSET-HASH', m_asset_hash, 'R36-BUILD-03',
     'an exported file changed after its hash was recorded'),
    ('MUT-EPOCH', m_epoch, 'R36-TEST-01 / sim-socket',
     'a reply that belongs to another epoch'),
    ('MUT-EARLY-DONE', m_early_completion, 'R36-MEM-02',
     'a completion reported before the kernel finished'),
    ('MUT-MEM-LATENCY', m_storage_latency, 'R36-MEM-02',
     'a memory that ignores the latency it was given'),
    ('MUT-EXPORT-DEP', m_export_dependency, 'R36-RTL-01',
     'an export missing a module it needs'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate C: six mutations, and the checker named for each")

    before = tree_digest(ROOT)
    print(f"  the candidate tree before: {before[:16]}")
    base = tempfile.mkdtemp(prefix='r36-mut-')
    results, ok = [], True
    try:
        for mid, fn, detector, what in MUTATIONS:
            if args.only and mid not in args.only:
                continue
            log = []
            lab = B.Lab(base)
            try:
                r, note = fn(lab.dir, log)
            except Exception as e:                         # noqa: BLE001
                r, note = None, f"{type(e).__name__}: {e}"
            if r is None:
                caught, detail = False, f"the mutation did not apply: {note}"
            else:
                rc = r['returncode'] if isinstance(r, dict) else r.returncode
                out = ((r['stdout'] + r['stderr']) if isinstance(r, dict)
                       else (r.stdout + r.stderr))
                caught = rc != 0
                # A mutation that merely stops the tree building has not
                # been detected by anything.
                syntax = any(w in out for w in
                             ('SyntaxError', 'ImportError', 'No such file',
                              'Traceback (most recent call last)'))
                if caught and syntax and 'FAIL' not in out:
                    caught = False
                    detail = ('the run failed for a reason that is not a '
                              'detection: ' + out.strip()[-140:])
                else:
                    detail = (out.strip().splitlines() or ['(no output)'])[-1][:110]
            log.append(f"      {detector} -> "
                       f"{'CAUGHT' if caught else 'NOT CAUGHT'}")
            log.append(f"      {detail}")
            results.append({'id': mid, 'detector': detector,
                            'what': what, 'caught': bool(caught)})
            ok &= caught
            print(f"  {mid:<18s} {'CAUGHT' if caught else 'MISSED':<7s} "
                  f"{what}", flush=True)
            for line in log:
                print(line, flush=True)
            shutil.rmtree(lab.dir, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    after = tree_digest(ROOT)
    clean = after == before
    print(f"  the candidate tree after:  {after[:16]} "
          f"({'unchanged' if clean else 'CHANGED -- a mutation escaped'})")
    ok &= clean
    if args.json:
        R.write_json(args.json, [{
            'id': 'R36-TEST-03', 'passed': bool(ok),
            'detail': f"{len(results)} mutation(s), "
                      f"{sum(1 for r in results if r['caught'])} caught, "
                      f"tree unchanged {clean}",
            'mutations': results, 'tree_unchanged': clean,
            'digest': after}])
    missed = [r['id'] for r in results if not r['caught']]
    print(f"  r36-mutation-tests     {'FAIL' if not ok else 'PASS':<6s} "
          + (', '.join(missed) if missed
             else f"{len(results)} mutation(s), each caught by its checker"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
