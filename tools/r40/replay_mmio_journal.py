#!/usr/bin/env python3
"""Replay captured simulated-time mailbox batches and compare every response.

This is a deterministic functional reproducer, not a performance measurement.
A diagnostic harness override is recorded and still checked against the same
input times, response bytes and batch completion times.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/r1'))
import r1lib as R


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--journal',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    choices=ap.add_mutually_exclusive_group()
    choices.add_argument('--harness',type=Path,help='relocated copy with the captured SHA256')
    choices.add_argument('--diagnostic-harness',type=Path)
    ap.add_argument('--runtime',help='matching vvp runtime; vvp from PATH by default')
    ap.add_argument('--capture-log',type=Path,help='relocated capture stdout/stderr')
    ap.add_argument('--cap-seconds',type=float,default=1800)
    args=ap.parse_args()
    journal=args.journal.resolve()
    metadata=json.loads((journal/'metadata.json').read_text())
    if metadata.get('schema')!='tnca-mmio-timed-journal/1':
        ap.error('unsupported journal metadata')
    harness=(args.diagnostic_harness or args.harness or Path(metadata['harness'])).resolve()
    digest=hashlib.sha256(harness.read_bytes()).hexdigest()
    if args.diagnostic_harness is None and digest!=metadata['harness_sha256']:
        ap.error('captured harness changed; preserve it or explicitly select a diagnostic harness')
    if args.cap_seconds<=0:
        ap.error('positive replay execution cap required')
    terminal=[]
    with (journal/'index.txt').open() as stream:
        for line in stream:
            if line.strip():terminal=line.split()
    if len(terminal)!=3 or terminal[0]!='T' or terminal[2] not in ('0','1'):
        ap.error('capture has no terminal marker; partial/live captures cannot certify complete replay')
    runtime=args.runtime or shutil.which('vvp')
    if runtime is None:
        ap.error('matching vvp runtime required on PATH or through --runtime')
    capture_log=args.capture_log or journal/'capture.log'
    if not capture_log.is_file():
        capture_log=Path(metadata.get('simulator_log',''))
    if not capture_log.is_file():
        ap.error('capture log missing; retain it or provide --capture-log')
    capture_failures=[]
    with capture_log.open(errors='replace') as stream:
        for line in stream:
            if 'FAIL' in line or line.startswith('FATAL'):
                capture_failures.append(line.rstrip())
                if len(capture_failures)==16:break
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'unused.cmd').write_text('0\n')
    command=[runtime,str(harness),
             '+cmd='+str(out/'unused.cmd'),'+rsp='+str(out/'actual.rsp'),
             '+replay='+str(journal),'+assert_fatal']
    result=R.run_step(command,kind='build',case_dir=str(out),name='replay',
                      cwd=str(ROOT),cap_seconds=args.cap_seconds)
    stdout=R.read_text(result['stdout'])
    marker_problems=R.check_sim_markers(result)
    passed=terminal[2]=='1' and not capture_failures and result['status']==R.PASS and not marker_problems and 'MMIO_JOURNAL_REPLAY_PASS ' in stdout
    report=dict(status='replay-pass' if passed else 'replay-failed',
                scope='DUT I/O and stop-history replay; excludes host/reference semantics and performance',
                gate_status='not-measured',capture=metadata,
                replay_harness_sha256=digest,diagnostic_override=args.diagnostic_harness is not None,
                capture_terminal=terminal,capture_failures=capture_failures,
                marker_problems=marker_problems,result=dict(result))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(stdout[-2200:],end='')
    print('MMIO journal replay '+('PASS' if passed else 'FAIL'),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
