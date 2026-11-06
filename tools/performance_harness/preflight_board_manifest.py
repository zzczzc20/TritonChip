"""Validate a single board step up to, but without, its first device probe.

Uses the installed runner and real image/clock/map files. The probe deliberately
raises a sentinel, and process execution is forbidden. No board result is made.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve();spec=json.loads(args.manifest.read_text())
    if len(spec['steps'])!=1 or spec['steps'][0]['kind']!='fpga':
        raise ValueError('preflight covers exactly one registered board step')
    if args.out.exists():raise ValueError('preserve the previous preflight; use a fresh output path')
    path=root/'tools/performance_harness/experiment.py'
    module_spec=importlib.util.spec_from_file_location('frozen_experiment_preflight',path)
    runner=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(runner)
    class ProbeBoundaryReached(RuntimeError):pass
    observed=[]
    def stop_before_probe(board):
        observed.append(board)
        raise ProbeBoundaryReached('device access intentionally excluded from preflight')
    with patch.object(runner.subprocess,'Popen',side_effect=AssertionError('preflight must never execute a command')) as processes:
        try:runner.run(args.manifest,root,args.out,python=Path(sys.executable),board_probe=stop_before_probe)
        except ProbeBoundaryReached:pass
        else:raise AssertionError('runner did not reach the board-probe boundary')
        assert len(observed)==1 and processes.call_count==0
    report=dict(status='registration-preflight-pass-no-device-access',
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        installed_runner_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        software_tree=runner.software_tree(root),board=observed[0],
        board_probe_calls=0,commands_executed=processes.call_count,
        scope='real source/image/clock/map file and command validation only; not device qualification or measurement')
    (args.out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],commands_executed=report['commands_executed']),indent=2))


if __name__=='__main__':main()
