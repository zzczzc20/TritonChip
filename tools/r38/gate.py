#!/usr/bin/env python3
"""S1-09: promotion gates, the DevOps log and last-known-good.

    tools/r38/gate.py l0 [--rtl DIR]     schema/encoding/metric self-checks
    tools/r38/gate.py l1 [--rtl DIR]     the smoke suite on the candidate
    tools/r38/gate.py l2 [--rtl DIR]     l0 + profiler + baseline check + smoke
    tools/r38/gate.py selftest           inject a numerically wrong candidate

Every run appends one line to sim/build/r38/devops_log.jsonl: the candidate
hash, the level, the verdict, the feedback latency (start of the gate to its
verdict), and on failure the tail of the evidence and where it was kept.
sim/build/r38/last_known_good.json moves ONLY when l2 passes; a failed
candidate never becomes the baseline the next change is compared with
(roadmap section 3.4).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness as H                                         # noqa: E402
import manifest                                             # noqa: E402

OUT = ROOT / 'sim' / 'build' / 'r38'
LOG = OUT / 'devops_log.jsonl'
LKG = OUT / 'last_known_good.json'
TRITON_PY = str(Path.home() / 'src' / 'tritonenv' / 'bin' / 'python')
SLO = {'l0': 300, 'l1': 900}


def steps(level, rtl, out):
    py = TRITON_PY if Path(TRITON_PY).exists() else sys.executable
    rtl_args = ['--rtl', str(rtl)] if rtl else []
    l0 = [[sys.executable, str(HERE / 'gen_counters.py'), '--check'],
          [sys.executable, str(HERE / 'evidence_tests.py')],
          [sys.executable, str(HERE / 'archbench_tests.py'), '--allow-pending']]
    l1 = [[py, str(HERE / 'run.py'), '--suite', 'smoke', '--fail-fast', '--out', str(out / 'smoke')] + rtl_args]
    if level == 'l0':
        return l0
    if level == 'l1':
        return l1
    return l0 + [[py, str(HERE / 'profiler_tests.py')], [py, str(HERE / 'baseline.py'), '--check']] + l1


def gate(level, rtl=None, out=None):
    out = Path(out or OUT / 'gate' / level)
    out.mkdir(parents=True, exist_ok=True)
    cand = manifest.candidate(hardware_root=rtl)
    started = time.monotonic()
    entry = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), level=level,
                 candidate_hash=cand['candidate_hash'], components=cand['components'],
                 rtl=str(rtl) if rtl else 'working tree')
    failed = None
    for i, cmd in enumerate(steps(level, rtl, out)):
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f'step{i}.log').write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            failed = dict(step=' '.join(Path(c).name if '/' in c else c for c in cmd[1:3]),
                          evidence=str((out / f'step{i}.log').relative_to(ROOT)),
                          tail=(r.stdout + r.stderr).strip().splitlines()[-6:])
            break
    latency = round(time.monotonic() - started, 1)
    entry.update(verdict='fail' if failed else 'pass', feedback_latency_s=latency, failure=failed,
                 slo_s=SLO.get(level), within_slo=(latency <= SLO[level]) if level in SLO else None)
    if not failed and level == 'l2':
        LKG.write_text(json.dumps(dict(candidate=cand, promoted=entry['time']), indent=1) + '\n')
        entry['promoted'] = True
    with LOG.open('a') as fh:
        fh.write(json.dumps(entry) + '\n')
    print(f'{level}: {entry["verdict"].upper()} in {latency}s, candidate {cand["candidate_hash"][:12]}'
          + (f' -- {failed["step"]}: {failed["tail"][-1] if failed["tail"] else ""}' if failed else ''))
    return entry


def selftest():
    """A numerically wrong candidate fails L1, keeps its evidence, and does
    not move last-known-good."""
    before = LKG.read_text() if LKG.exists() else None
    lab = OUT / 'gate' / 'selftest-bad-fp-one'
    H.rtl_copy(lab, [('rtl/core/tnca_vector_fpu.v', "FP_ONE   = 32'h3f800000", "FP_ONE   = 32'h3f810000")])
    entry = gate('l1', rtl=lab, out=lab / 'out')
    problems = []
    if entry['verdict'] != 'fail':
        problems.append('a candidate whose vector add is wrong passed L1')
    if entry['failure'] and not (ROOT / entry['failure']['evidence']).exists():
        problems.append('the failure evidence was not kept')
    after = LKG.read_text() if LKG.exists() else None
    if after != before:
        problems.append('last-known-good moved')
    last = json.loads(LOG.read_text().splitlines()[-1])
    if last['candidate_hash'] != entry['candidate_hash'] or last['verdict'] != 'fail':
        problems.append('the failure is not the last devops log entry')
    for p in problems:
        print('FAIL', p)
    print(f'gate selftest: {"PASS" if not problems else "FAIL"}')
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('level', choices=['l0', 'l1', 'l2', 'selftest'])
    ap.add_argument('--rtl', type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.level == 'selftest':
        return selftest()
    return 0 if gate(args.level, rtl=args.rtl)['verdict'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
