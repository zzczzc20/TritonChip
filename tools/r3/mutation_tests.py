#!/usr/bin/env python3
"""
R3M-01 .. R3M-06 -- doc/phase3_tasks_and_acceptance.md R3-11.

"以下mutation全部必须检出，注入在隔离副本 ... 变异失败必须由指定语义触发，
 不能用无关build失败算成功."

Each entry breaks one thing in an isolated copy and requires the named
check to fail because of it. Two conditions keep a pass honest, and both
are checked: the same command passes on an unmutated copy, and the mutated
run fails inside the named case rather than anywhere convenient.

The contract also says to catch these in the component test rather than by
re-running a whole workload matrix ("不复制整个GEMM/system workload矩阵"),
so every command below is a single suite with `--only`.

R3M-02 and R3M-03 are NOT RUN: their detectors are FLOW-03 and UNIT/NUM,
which do not exist yet. A mutation whose detector is missing has not been
caught.
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
    env['PYTHONPATH'] = os.path.join(lab.dir, 'python')
    return subprocess.run(argv, cwd=lab.dir, capture_output=True, text=True,
                          timeout=timeout, env=env)


COUNT_PROGRAMS = ['t04_roles', 't11_matrix', 't12_tensor', 't16_vecadd',
                  't20_sfu', 't21_cvt', 't22_atomic', 't25_mmasf',
                  't28_count']


def build_core(lab):
    """Build the core bench and its programs inside the lab.

    Carrying a prebuilt bench in would be faster and wrong: an RTL
    mutation has to be compiled to be tested, and a carried artefact is
    the unmutated design wearing the mutated tree's name.
    """
    targets = ['sim/build/tb_core.vvp'] + \
              [f"sim/build/{p}.hex" for p in COUNT_PROGRAMS]
    r = subprocess.run(['make'] + targets, cwd=lab.dir,
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise AssertionError(
            f"the lab could not build the core bench: "
            f"{(r.stderr or r.stdout)[-300:]}")


def suite(name, *cases):
    return [sys.executable, os.path.join('tools', 'r3', name)] + \
           (['--only'] + list(cases) if cases else [])


# --------------------------------------------------------------------------

def m01a(lab):
    """Return the read one cycle early at latency 2."""
    patch(lab, 'rtl/core/tnca_rf_bank.v',
          "    assign rvalid = (READ_LATENCY == 2) ? s2_valid : s1_valid;\n"
          "    assign rdata  = (READ_LATENCY == 2) ? s2_data  : s1_merged;",
          "    assign rvalid = s1_valid;   // R3M-01a\n"
          "    assign rdata  = s1_merged;")
    return suite('rf_tests.py', 'RF-02'), 'RF-02'


def m01b(lab):
    """Route each bank's response to the wrong requester."""
    patch(lab, 'rtl/core/tnca_vrf_banked.v',
          "                    rsp_data_c[dst*WIDTH +: WIDTH]  =\n"
          "                        bank_rdata[rb*WIDTH +: WIDTH];",
          "                    rsp_data_c[dst*WIDTH +: WIDTH]  =  // R3M-01b\n"
          "                        bank_rdata[((rb + 1) % NUM_BANKS)*WIDTH "
          "+: WIDTH];")
    return suite('rf_tests.py', 'RF-02'), 'RF-02'


def m04a(lab):
    """Let the stall attribution overlap again.

    On x_wb rather than x_lsu. `hazard` and `stall_lsu` happen not to
    overlap on any of the programs in the sweep -- on t16_vecadd they are
    323 and 41 and sum to the total exactly -- so removing x_lsu's
    exclusivity changes no number and would be a mutation nobody could
    catch for a reason that has nothing to do with the checker.
    `wb_steal` is the term that really does overlap: 323 + 41 + 48 = 412
    against a total of 364.
    """
    patch(lab, 'rtl/core/tnca_core.v',
          "    wire x_wb    = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf\n"
          "                   && !x_fence && !x_arr && !x_rsv && wb_steal;",
          "    wire x_wb    = wb_steal;   // R3M-04a")
    return suite('count_tests.py', 'COUNT-02'), 'COUNT-02'


def m04b(lab):
    """Drop one port's grants from the completion count."""
    patch(lab, 'rtl/core/tnca_operand_collector.v',
          "                        grants_c      = grants_c + 1;",
          "                        if (j != 0)   // R3M-04b\n"
          "                            grants_c  = grants_c + 1;")
    return suite('count_tests.py', 'COUNT-01'), 'COUNT-01'


def m05(lab):
    """Give the model a register file with no port limit.

    Not by changing the DEFAULT -- MODEL-02 constructs its Machine
    explicitly, so the default is never read and the mutation would be
    invisible while looking applied. And not by disabling one of the two
    conflict tests, because the other still fires and the model still
    stalls. Making no read take a port at all is the one edit that
    actually means "unlimited".
    """
    patch(lab, 'tools/r3/model.py',
          "            banks = [r % m.rf_banks for r in op.reads]",
          "            banks = []   # R3M-05: no read takes a port")
    return suite('model_tests.py', 'MODEL-02'), 'MODEL-02'


def m06(lab):
    """Sneak a bigger K into the performance list."""
    import json
    p = os.path.join(lab.dir, 'spec', 'r3_workloads.json')
    spec = json.load(open(p))
    cap = spec['caps']['core_gemm']['k']
    bumped = None
    for w in spec['workloads']:
        params = w.get('params') or {}
        if 'k' in params and w.get('profile') == 'core-sim':
            params['k'] = cap * 4
            bumped = w.get('id', '?')
            break
    if bumped is None:
        raise AssertionError('no core GEMM workload with a k to bump')
    json.dump(spec, open(p, 'w'), indent=2)
    return ([sys.executable, os.path.join('tools', 'r3', 'plan.py')],
            'PLAN-02')


MUTATIONS = [
    ('R3M-01a', 'the synchronous read returns a cycle early', m01a, None),
    ('R3M-01b', "each bank's data goes to the wrong requester", m01b, None),
    ('R3M-04a', 'the stall attribution overlaps again', m04a, build_core),
    ('R3M-04b', 'one port is left out of the grant count', m04b, build_core),
    ('R3M-05', 'the model gets a register file with no port limit', m05,
     None),
    ('R3M-06', 'a workload quietly grows its K past the cap', m06, None),
]

NOT_YET = {
    'R3M-02': ('a return or writeback reservation removed',
               'the detectors are FLOW-03 and FLOW-04\'s return half, and '
               'FLOW-03 needs the writeback units of R3-05. FLOW-04 covers '
               'the collector\'s own conservation, which is not the same '
               'claim as the return path\'s'),
    'R3M-03': ('the ARF accumulation reordered, or a masked write dropped',
               'the detectors are UNIT-02/05 and NUM-01, which are R3-05 '
               'and R3-08 and not started; the ARF also still has the '
               'behavioural storage that R3-03 replaced for the VRF'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R3-11 mutation: each break must be caught by its named check")
    before = B.tree_hashes(ROOT)
    base = tempfile.mkdtemp(prefix='r3-mut-')
    results = []
    try:
        for cid, what, fn, prepare in MUTATIONS:
            if args.only and cid not in args.only:
                continue
            lab = B.Lab(base)
            try:
                argv, names = fn(lab)
                if prepare:
                    prepare(lab)
                clean = B.Lab(base)
                if prepare:
                    prepare(clean)
                base_r = run_in(clean, argv)
                clean.close()
                mut_r = run_in(lab, argv)
                caught = base_r.returncode == 0 and mut_r.returncode != 0
                detail = (f"baseline exit={base_r.returncode}, mutated "
                          f"exit={mut_r.returncode} ({names})")
                if caught:
                    line = [l for l in mut_r.stdout.splitlines()
                            if names in l and 'PASS' not in l]
                    if not line:
                        caught = False
                        detail += ' -- it failed, but not in the named case'
                    else:
                        detail += f" -- {line[0].strip()[:66]}"
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
            print(f"            {detail}", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    for cid, (what, why) in sorted(NOT_YET.items()):
        if args.only and cid not in args.only:
            continue
        results.append(dict(id=cid, passed=False, detail=f"NOT RUN: {why}"))
        print(f"  {cid}  NOT RUN  {what}")
        print(f"            {why}")

    after = B.tree_hashes(ROOT)
    leaked = sorted(k for k in before if before[k] != after.get(k))
    if leaked:
        print(f"  CANDIDATE TREE MODIFIED: {leaked}")
        results.append(dict(id='R3M-ISOLATION', passed=False,
                            detail=f"leaked: {leaked}"))
    else:
        print("  candidate source hash unchanged by the mutations")

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3-mutation-tests      FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3-mutation-tests      PASS   {len(results)} mutation(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
