#!/usr/bin/env python3
"""
R3V2-MUT-01 -- doc/phase3_tasks_and_acceptance.md G7.

Seven mutations, each broken in an isolated copy and each required to be
caught by the checker the contract names for it. The clean baseline must
pass, the mutated run must fail INSIDE the named case, and the candidate
tree must be byte-identical afterwards.

"无关语法/编译失败不算检出" -- a mutation that merely stops the tree
building has not been detected by anything. So every entry checks that the
named case is the one that failed, and a build error is a failure of the
mutation rather than a success of the checker.
"""
import argparse
import copy
import json
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


def run_in(lab, argv, timeout=7200):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(lab.dir, 'python')
    return subprocess.run(argv, cwd=lab.dir, capture_output=True,
                          text=True, timeout=timeout, env=env)


def P(name, case):
    """One (invocation, named case) probe from the contract's table."""
    return suite(f'{name}.py', case), case


def suite(name, *cases):
    return [sys.executable, os.path.join('tools', 'r3v2', name)] + \
           (['--only'] + list(cases) if cases else [])


def build_core(lab, mutated=False):
    """The candidate core, built inside the lab.

    An RTL mutation has to be compiled to be tested; carrying a prebuilt
    bench in would be the unmutated design wearing the mutated tree's name.

    Every program is assembled, not a chosen few: FLOW-03 reaches for
    t13_share, t17_descfault and t26_stfault, and a checker that cannot
    find its program fails for a reason that has nothing to do with the
    mutation -- which reads as "the baseline does not pass" and concludes
    nothing.
    """
    import glob
    for target in ('sim/build/tb_core.vvp', 'sim/build/tb_core_vrf.vvp'):
        r = subprocess.run(['make', target], cwd=lab.dir,
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise AssertionError(f"the lab could not build {target}: "
                                 f"{(r.stderr or r.stdout)[-300:]}")
    progs = sorted(os.path.basename(s)[:-2] for s in
                   glob.glob(os.path.join(lab.dir, 'tb', 'core', 'progs',
                                          '*.s')))
    r = subprocess.run(['make'] + [f'sim/build/{p}.hex' for p in progs],
                       cwd=lab.dir, capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        raise AssertionError(f"the lab could not assemble the programs: "
                             f"{(r.stderr or r.stdout)[-300:]}")


# ---------------------------------------------------------------------------

def m01a(lab):
    """The synchronous read returns a cycle early.

    Not by changing READ_LATENCY -- the design is instantiated with 1
    everywhere, so touching the 2-deep arm mutates dead code and proves
    nothing. And not by moving `rvalid`, which deadlocks the handshake:
    a mutation that hangs the bench is caught by the clock, not by the
    checker. This leaves the handshake alone and reads the array at the
    address being presented NOW instead of the one accepted last cycle,
    which is what an off-by-one read port actually delivers.
    """
    patch(lab, 'rtl/core/tnca_rf_bank.v',
          "    assign rdata     = (READ_LATENCY == 2) ? s2_data  : s1_data;",
          "    assign rdata     = (READ_LATENCY == 2) ? s2_data\n"
          "                                           : mem[raddr];"
          "   // M01a")
    return ([P('rf_tests', 'R3V2-RF-02'),
             P('core_tests', 'R3V2-CORE-02')], build_core)


def m01b(lab):
    """A bank's data goes back to the wrong requester."""
    patch(lab, 'rtl/core/tnca_vrf_banked.v',
          "                    rsp_data_c[dst*WIDTH +: WIDTH]  =\n"
          "                        bank_rdata[rb*WIDTH +: WIDTH];",
          "                    rsp_data_c[dst*WIDTH +: WIDTH]  =  // M01b\n"
          "                        bank_rdata[((rb + 1) % NUM_BANKS)"
          "*WIDTH +: WIDTH];")
    return ([P('rf_tests', 'R3V2-RF-02'),
             P('core_tests', 'R3V2-CORE-02')], build_core)


def m02(lab):
    """A refused write is dropped instead of held."""
    patch(lab, 'rtl/core/tnca_vrf_banked.v',
          "            if (we[wi2] && !w_bank_busy[w_bank_of[wi2]]) begin",
          "            if (we[wi2]) begin   // M02: the bank is never busy")
    return [P('flow_tests', 'R3V2-FLOW-01')], build_core


def m03(lab):
    """After a flush, an old tag writes into a new entry."""
    patch(lab, 'rtl/core/tnca_operand_collector.v',
          "                for (j = 0; j < NUM_PORTS; j = j + 1)\n"
          "                    if (port_live[j])\n"
          "                        port_orphan[j] <= 1'b1;",
          "                // M03: the flush forgets what is in flight\n"
          "                for (j = 0; j < NUM_PORTS; j = j + 1)\n"
          "                    if (1'b0)\n"
          "                        port_orphan[j] <= 1'b1;")
    return [P('flow_tests', 'R3V2-FLOW-03')], build_core


def m04a(lab):
    """A masked write drops a lane."""
    patch(lab, 'rtl/core/tnca_rf_bank.v',
          "            for (li = 0; li < LANES; li = li + 1)\n"
          "                if (wmask[li])\n"
          "                    mem[waddr][li*ELEN +: ELEN] <= "
          "wdata[li*ELEN +: ELEN];",
          "            for (li = 0; li < LANES; li = li + 1)\n"
          "                if (wmask[li] && li != 3)   // M04a\n"
          "                    mem[waddr][li*ELEN +: ELEN] <= "
          "wdata[li*ELEN +: ELEN];")
    return [P('rf_tests', 'R3V2-RF-02')], None


def m04b(lab):
    """A source is freed before its operand has been consumed.

    Dropping the `ready[i]` guard is not this mutation: nothing in the
    stimulus asserts `consume` for an entry that is not ready, so the
    guard is redundant and removing it changes no behaviour at all. An
    equivalent mutant tests the checker's patience, not its reach.

    This frees the slot the moment its FIRST source lands, while the rest
    are still outstanding -- the slot is reusable with reads in flight,
    which is the thing the reuse and conservation checks exist for.
    """
    patch(lab, 'rtl/core/tnca_operand_collector.v',
          "            for (i = 0; i < ENTRIES; i = i + 1)\n"
          "                if (consume[i] && ready[i])\n"
          "                    e_busy[i] <= 1'b0;",
          "            for (i = 0; i < ENTRIES; i = i + 1)\n"
          "                if ((consume[i] && ready[i])   // M04b\n"
          "                    || (e_busy[i] && touch_have[i]\n"
          "                        && (have_c[i] != {SRCS{1'b0}})))\n"
          "                    e_busy[i] <= 1'b0;")
    return [P('flow_tests', 'R3V2-FLOW-03')], build_core


def m05a(lab):
    """A stall is counted twice."""
    patch(lab, 'rtl/core/tnca_core.v',
          "            if (|running && x_vrf)   perf_x_vrf    <= "
          "perf_x_vrf    + 64'd1;",
          "            if (|running && x_vrf)   perf_x_vrf    <= "
          "perf_x_vrf    + 64'd2;   // M05a")
    return [P('count_tests', 'R3V2-COUNT-01')], build_core


def m05b(lab):
    """One accepted request is missed."""
    patch(lab, 'rtl/core/tnca_core.v',
          "                vrf_n_accept <= vrf_n_accept +\n"
          "                    {29'd0, vrf_req_accept[0]} + "
          "{29'd0, vrf_req_accept[1]} +\n"
          "                    {29'd0, vrf_req_accept[2]};",
          "                vrf_n_accept <= vrf_n_accept +   // M05b\n"
          "                    {29'd0, vrf_req_accept[1]} +\n"
          "                    {29'd0, vrf_req_accept[2]};")
    return [P('count_tests', 'R3V2-COUNT-01')], build_core


def m06(lab):
    """The model allows unlimited reads from one bank."""
    patch(lab, 'tools/r3v2/vrf_model.py',
          "            b, _r = self.bank_row(p)\n"
          "            if b in taken or not free[b] or port in busy:\n"
          "                continue",
          "            b, _r = self.bank_row(p)   # M06\n"
          "            if not free[b] or port in busy:\n"
          "                continue")
    return [P('model_tests', 'R3V2-MODEL-01')], None


def m07a(lab):
    """A core case loses its candidate."""
    p = os.path.join(lab.dir, 'spec', 'r3v2_inventory.json')
    d = json.load(open(p))
    d['core'][0]['implementations'] = ['reference']
    json.dump(d, open(p, 'w'), indent=2)
    return [P('plan', 'R3V2-BASE-01'),
            P('plan', 'R3V2-BASE-02')], None


def m07b(lab):
    """A workload quietly grows its K."""
    p = os.path.join(lab.dir, 'spec', 'r3v2_inventory.json')
    d = json.load(open(p))
    for c in d['core']:
        if c['workload'] == 'gemm':
            c['sizes']['k'] = d['size_caps']['gemm']['k'] * 4
            break
    json.dump(d, open(p, 'w'), indent=2)
    return [P('plan', 'R3V2-BASE-01'),
            P('plan', 'R3V2-BASE-02')], None


def plant_evidence(lab, corrupt=None):
    """A complete evidence tree the verifier accepts.

    Synthesised from the same frozen inventory the verifier reads, so the
    baseline is genuinely acceptable and a mutation of it is genuinely a
    regression -- not two runs refused for the same missing-file reason.
    """
    sys.path.insert(0, os.path.join(lab.dir, 'tools', 'r3v2'))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        've_lab', os.path.join(lab.dir, 'tools', 'r3v2',
                               'verify_evidence.py'))
    ve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ve)

    ev = os.path.join(lab.dir, 'ev')
    os.makedirs(os.path.join(ev, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(ev, 'contracts'), exist_ok=True)
    import hashlib as H
    frozen = {}
    for name in ('r3v2_contract.json', 'r3v2_inventory.json'):
        src = os.path.join(lab.dir, 'spec', name)
        shutil.copy2(src, os.path.join(ev, 'contracts', name))
        frozen[name] = H.sha256(open(src, 'rb').read()).hexdigest()

    checks = []
    raw = {}
    for checker, ids in sorted(ve.REQUIRED.items()):
        raw[checker] = [{'id': cid, 'passed': True, 'detail': 'ok'}
                        for cid in ids]
        checks.append({'checker': checker, 'exit': 0, 'elapsed_s': 10,
                       'cases': copy.deepcopy(raw[checker])})
    summary = {
        'contract_id': ve.CONTRACT_ID, 'commit': 'a' * 40,
        'clean_tree': True, 'frozen_hashes': frozen,
        'checks': checks, 'verdict': ve.SIGNOFF,
    }
    if corrupt:
        corrupt(raw, summary)
    for checker, cases in raw.items():
        json.dump(cases, open(os.path.join(ev, 'runs', f'{checker}.json'),
                              'w'), indent=2)
    json.dump(summary, open(os.path.join(ev, 'acceptance.json'), 'w'),
              indent=2)


VERIFIER_ARGV = [sys.executable, os.path.join('tools', 'r3v2',
                                              'verify_evidence.py'), 'ev']


def m07c(lab):
    """A result is edited to say it passed.

    The raw result says R3V2-RF-02 failed; the summary quotes it as
    passing. This is exactly the shape of a hand-edited sign-off.
    """
    def corrupt(raw, summary):
        for c in raw['rf']:
            if c['id'] == 'R3V2-RF-02':
                c['passed'] = False
                c['detail'] = 'lane 3 read back stale'

    def setup(lab, mutated=False):
        plant_evidence(lab, corrupt if mutated else None)
    return [(VERIFIER_ARGV, 'the summary says True')], setup


def m07d(lab):
    """A required case has no result at all."""
    def corrupt(raw, summary):
        raw['core'] = [c for c in raw['core']
                       if c['id'] != 'R3V2-CORE-02']
        for e in summary['checks']:
            if e['checker'] == 'core':
                e['cases'] = [c for c in e['cases']
                              if c['id'] != 'R3V2-CORE-02']

    def setup(lab, mutated=False):
        plant_evidence(lab, corrupt if mutated else None)
    return [(VERIFIER_ARGV,
             'R3V2-CORE-02: required by the contract')], setup


MUTATIONS = [
    ('M01a', 'the read returns a cycle early', m01a),
    ('M01b', "a bank's data goes to the wrong requester", m01b),
    ('M02',  'a refused write is dropped instead of held', m02),
    ('M03',  'after a flush, an old tag writes a new entry', m03),
    ('M04a', 'a masked write drops a lane', m04a),
    ('M04b', 'a source is freed before it is consumed', m04b),
    ('M05a', 'a stall is counted twice', m05a),
    ('M05b', 'one accepted request is missed', m05b),
    ('M06',  'the model allows unlimited reads from one bank', m06),
    ('M07a', 'a core case loses its candidate', m07a),
    ('M07b', 'a workload quietly grows its K', m07b),
    ('M07c', 'a result is edited to say it passed', m07c),
    ('M07d', 'a required case has no result at all', m07d),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R3 v2 G7: every mutation caught by its named checker")
    before = B.tree_hashes(ROOT)
    base = tempfile.mkdtemp(prefix='r3v2-mut-')
    caught_all = True
    detail = []
    # The clean baseline for a given checker invocation is the same for
    # every mutation that names it; run it once.
    baseline = {}
    try:
        for mid, what, fn in MUTATIONS:
            if args.only and mid not in args.only:
                continue
            lab = B.Lab(base)
            fired, note = [], []
            caught = False
            try:
                probes, setup = fn(lab)
                if setup:
                    setup(lab, True)
                for argv, marker in probes:
                    key = (tuple(argv), getattr(setup, '__qualname__', ''))
                    if key not in baseline:
                        clean = B.Lab(base)
                        try:
                            if setup:
                                setup(clean, False)
                            r = run_in(clean, argv)
                            baseline[key] = (r.returncode, r.stdout)
                        finally:
                            clean.close()
                    base_rc = baseline[key][0]
                    if base_rc != 0:
                        note.append(f"{marker}: the baseline does not "
                                    f"pass, so nothing can be concluded")
                        continue
                    mut_r = run_in(lab, argv)
                    if mut_r.returncode == 0:
                        note.append(f"{marker}: not detected")
                        continue
                    named = [l for l in mut_r.stdout.splitlines()
                             if marker in l and 'PASS' not in l]
                    if named:
                        fired.append(marker)
                    else:
                        note.append(f"{marker}: the run failed, but not "
                                    f"in the named case")
                # 至少一个合同指定的检出器必须抓到它。
                caught = bool(fired)
            except Exception as e:
                note.append(f"{type(e).__name__}: {e}")
            finally:
                lab.close()
            detail_note = ('caught by ' + ', '.join(fired) if fired
                           else '; '.join(note) or 'no probe ran')
            if fired and note:
                detail_note += ' (' + '; '.join(note) + ')'
            caught_all &= caught
            detail.append({'id': mid, 'caught': bool(caught),
                           'what': what, 'detectors': fired,
                           'note': detail_note})
            print(f"  {mid:<5s} {'caught' if caught else 'MISSED':<7s} "
                  f"{what}", flush=True)
            print(f"        {detail_note}", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    after = B.tree_hashes(ROOT)
    leaked = sorted(k for k in before if before[k] != after.get(k))
    if leaked:
        print(f"  CANDIDATE TREE MODIFIED: {leaked}")
        caught_all = False
    else:
        print("  candidate source hash unchanged by the mutations")

    results = [{'id': 'R3V2-MUT-01', 'passed': bool(caught_all),
                'detail': f"{sum(d['caught'] for d in detail)}/"
                          f"{len(detail)} mutation(s) caught by their "
                          f"named checker",
                'mutations': detail}]
    # R3V2-EV-01 is not a separate run: it is what M07c and M07d are. The
    # verifier refusing forged and incomplete evidence IS the standard, so
    # it is reported from the mutations that test it and from nothing else.
    ev = {d['id']: d['caught'] for d in detail if d['id'] in ('M07c', 'M07d')}
    if len(ev) == 2:
        results.append({
            'id': 'R3V2-EV-01', 'passed': all(ev.values()),
            'detail': 'the verifier refuses a forged result and a missing '
                      'one, and accepts the same evidence without them'})
    elif ev:
        print(f"  R3V2-EV-01 not reported: it needs both M07c and M07d, "
              f"and only {sorted(ev)} ran")
    if args.json:
        R.write_json(args.json, results)
    print()
    if not caught_all:
        print("  r3v2-mutation-tests    FAIL   R3V2-MUT-01")
        return 1
    print(f"  r3v2-mutation-tests    PASS   {len(detail)} mutation(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
