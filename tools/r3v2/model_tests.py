#!/usr/bin/env python3
"""
R3V2-MODEL-01 -- doc/phase3_tasks_and_acceptance.md G5.

Six directed patterns where tools/r3v2/vrf_model.py must predict the
register file's behaviour exactly.

The order matters and is the point. The model runs first, its predictions
are written down and hashed, and only then is the RTL run. "不得读取测量
结果回填模型" is easy to violate without noticing -- a model tuned until
it agrees is a record of the measurement, not a prediction of it -- so the
hash is computed before the simulator is started and checked afterwards.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402
from vrf_model import CandidateVRF, LANES                # noqa: E402

BENCH = os.path.join(ROOT, 'tb', 'core', 'tb_r3v2_vrf.v')
NUM_PORTS = 4
READY_ALL = (1 << NUM_PORTS) - 1


# Each graph: a name, and a list of cycles. A cycle is
#   (write or None, [(port, idx)], ready_mask)
def graphs():
    full = (1 << LANES) - 1

    def w(idx, word):
        return (0, idx, full, word)

    return {
        'serial-one-bank': [
            # v0, v4, v8 are all bank 0: three reads, three turns.
            (w(0, 0x1000), [], READY_ALL),
            (w(4, 0x2000), [], READY_ALL),
            (w(8, 0x3000), [], READY_ALL),
            (None, [(0, 0), (1, 4), (2, 8)], READY_ALL),
            (None, [(0, 0), (1, 4), (2, 8)], READY_ALL),
            (None, [(0, 0), (1, 4), (2, 8)], READY_ALL),
            (None, [], READY_ALL),
        ],
        'parallel-three-banks': [
            # v1, v2, v3: three banks, one cycle.
            (w(1, 0x1000), [], READY_ALL),
            (w(2, 0x2000), [], READY_ALL),
            (w(3, 0x3000), [], READY_ALL),
            (None, [(0, 1), (1, 2), (2, 3)], READY_ALL),
            (None, [], READY_ALL),
        ],
        'four-on-one-bank': [
            # The fairness pattern: every requester on bank 0.
            (w(0, 0x1000), [], READY_ALL),
            (w(4, 0x2000), [], READY_ALL),
            (w(8, 0x3000), [], READY_ALL),
            (w(12, 0x4000), [], READY_ALL),
        ] + [(None, [(0, 0), (1, 4), (2, 8), (3, 12)], READY_ALL)
             for _ in range(8)] + [(None, [], READY_ALL)],
        'held-response': [
            # The consumer stops taking; the file holds and refuses more.
            (w(5, 0x5000), [], READY_ALL),
            (None, [(0, 5)], 0),
            (None, [(0, 5)], 0),
            (None, [(0, 5)], 0),
            (None, [], READY_ALL),
            (None, [], READY_ALL),
        ],
        'write-first-bypass': [
            # A read of the address being written, in the same cycle.
            (w(6, 0x6000), [(0, 6)], READY_ALL),
            (None, [(0, 6)], READY_ALL),
            (None, [], READY_ALL),
        ],
        'zero-then-written': [
            # Unwritten reads zero; the same register reads back after.
            (None, [(0, 20)], READY_ALL),
            (w(20, 0x7000), [], READY_ALL),
            (None, [(0, 20)], READY_ALL),
            (None, [], READY_ALL),
        ],
    }


def predict(cycles):
    """What the model says, before anything is simulated."""
    m = CandidateVRF()
    out = []
    for n, (write, reads, ready) in enumerate(cycles):
        rd = {p: bool(ready >> p & 1) for p in range(NUM_PORTS)}
        acc, vis = m.cycle(write,
                           [(p, 0, idx) for p, idx in reads], rd)
        out.append({'cycle': n,
                    'accepted': sorted(acc),
                    'responses': sorted((p, v[0]) for p, v in vis)})
    return out


def stimulus(cycles):
    lines = []
    for write, reads, ready in cycles:
        we = 1 if write else 0
        _r, widx, wmask, word = write or (0, 0, 0, 0)
        idxs = [0] * NUM_PORTS
        rv = 0
        for p, idx in reads:
            idxs[p] = idx
            rv |= 1 << p
        lines.append(f"{we} {widx} {wmask:04x} {word} {rv} " +
                     ' '.join(str(i) for i in idxs) + f" {ready}")
    return '\n'.join(lines) + '\n'


def measure(vvp, cycles, tmp, name):
    path = os.path.join(tmp, f'{name}.txt')
    open(path, 'w').write(stimulus(cycles))
    r = subprocess.run([vvp, '+case=5', f'+stim={path}'],
                       capture_output=True, text=True, timeout=900)
    acc, rsp = {}, {}
    for m in re.finditer(r'ACC (\d+) ([01]+)', r.stdout):
        bits = m.group(2)[::-1]
        acc[int(m.group(1))] = sorted(i for i, b in enumerate(bits)
                                      if b == '1')
    for m in re.finditer(r'RSP (\d+) (\d+) ([0-9a-fA-FxX]+)', r.stdout):
        lane0 = m.group(3)[-8:]
        val = None if any(c in 'xX' for c in lane0) else int(lane0, 16)
        rsp.setdefault(int(m.group(1)), []).append((int(m.group(2)), val))
    out = []
    for n in range(len(cycles)):
        out.append({'cycle': n,
                    'accepted': acc.get(n, []),
                    'responses': sorted(rsp.get(n, []))})
    return out


def model01(log, ctx):
    """Six graphs, predicted first, then measured."""
    ok = True
    gs = graphs()

    # Predict everything and seal it before the simulator runs.
    predictions = {name: predict(cycles) for name, cycles in gs.items()}
    blob = json.dumps(predictions, sort_keys=True).encode()
    sealed = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(ctx['tmp'], 'prediction.json'), 'wb') as fh:
        fh.write(blob)
    log.append(f"    {len(gs)} graph(s) predicted and sealed as "
               f"{sealed[:16]} before the simulator was started")

    for name, cycles in gs.items():
        got = measure(ctx['vvp'], cycles, ctx['tmp'], name)
        want = predictions[name]
        diffs = [(w['cycle'], w, g) for w, g in zip(want, got)
                 if w['accepted'] != g['accepted']
                 or w['responses'] != g['responses']]
        served = sum(len(c['accepted']) for c in got)
        log.append(f"    {name:<22s} {len(cycles):>2d} cycle(s), "
                   f"{served:>2d} grant(s), "
                   f"{len(diffs)} disagreement(s)")
        for cyc, w, g in diffs[:3]:
            log.append(f"        cycle {cyc}: model "
                       f"acc={w['accepted']} rsp={w['responses']}; "
                       f"rtl acc={g['accepted']} rsp={g['responses']}")
        ok &= not diffs

    # The prediction must not have moved.
    after = hashlib.sha256(
        json.dumps({n: predict(c) for n, c in gs.items()},
                   sort_keys=True).encode()).hexdigest()
    log.append(f"    the prediction recomputed after measuring is still "
               f"{after[:16]}: {after == sealed}")
    ok &= after == sealed

    # And it has to be able to disagree.
    bad = CandidateVRF(num_banks=1)
    control = []
    for n, (write, reads, ready) in enumerate(gs['parallel-three-banks']):
        rd = {p: True for p in range(NUM_PORTS)}
        a, _v = bad.cycle(write, [(p, 0, i) for p, i in reads], rd)
        control.append(sorted(a))
    real = [c['accepted'] for c in predictions['parallel-three-banks']]
    log.append(f"    control: the same graph against a one-bank model -> "
               f"{'disagrees' if control != real else 'AGREES (the comparison is not looking)'}")
    ok &= control != real
    return ok, 'six graphs, sealed first, exact on every cycle'


TESTS = [('R3V2-MODEL-01', model01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G5: the micro-model against the register file")
    tmp = tempfile.mkdtemp(prefix='r3v2-model-')
    vvp = os.path.join(tmp, 'vrf.vvp')
    b = subprocess.run(
        ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'rtl', 'core'),
         '-y', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'tb', 'common'), '-o', vvp, BENCH],
        capture_output=True, text=True, timeout=1800, cwd=ROOT)
    if b.returncode != 0:
        print(f"  the bench did not build: {b.stderr[-400:]}")
        return 1
    ctx = {'vvp': vvp, 'tmp': tmp}
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log, ctx)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-900:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3v2-model-tests       FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-model-tests       PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
