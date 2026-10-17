#!/usr/bin/env python3
"""Watch one existing implementation; extract reports once, never retry a build."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--runner-pid', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-wait', type=float, default=172800)
    ap.add_argument('--poll-seconds', type=float, default=30)
    args = ap.parse_args()
    build, out = args.build.resolve(), args.out.resolve()
    if not build.is_dir() or args.max_wait <= 0 or args.poll_seconds <= 0:
        ap.error('existing build and positive observation limits required')
    out.mkdir(parents=True, exist_ok=False)
    record = dict(scope='report extraction only; no synthesis/place/route/load/retry',
                  gate_status='not-measured', build=str(build), runner_pid=args.runner_pid,
                  image_sha256=None, steps=[], watcher_sha256=sha(Path(__file__)))
    def save(status):
        record.update(status=status, observed_utc=datetime.now(timezone.utc).isoformat())
        temporary = out/'state.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(out/'state.json')
    def report(name, script, arguments, cap=2700):
        command = ['vivado', '-mode', 'batch', '-log', str(out/(name+'.vivado.log')),
                   '-journal', str(out/(name+'.vivado.jou')), '-source', str(script),
                   '-tclargs', *map(str, arguments)]
        step = dict(name=name, command=command, script_sha256=sha(script))
        with (out/(name+'.log')).open('x') as stream:
            proc = subprocess.Popen(command, cwd=str(out), stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            step['pid'] = proc.pid
            try:
                step['exit_code'] = proc.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                # The execution cap applies only to this owned report process.
                # An observation limit never terminates the implementation.
                step['report_execution_cap_seconds'] = cap
                try:
                    if os.getpgid(proc.pid) == proc.pid:
                        os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        if os.getpgid(proc.pid) == proc.pid:
                            os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                step['exit_code'] = proc.returncode
        record['steps'].append(step)
        save('extracting-reports')
        return step['exit_code'] == 0 and 'report_execution_cap_seconds' not in step
    save('watching-existing-build')
    terminal = build/'build/attempt.exit'
    deadline = time.monotonic()+args.max_wait
    while True:
        try:
            terminal_value = terminal.read_text().strip()
        except FileNotFoundError:
            terminal_value = ''
        if terminal_value:
            try:
                record['implementation_exit_code'] = int(terminal_value)
            except ValueError:
                save('invalid-terminal-marker-needs-inspection')
                return 2
            break
        if time.monotonic() >= deadline:
            save('observation-timeout-build-verdict-unknown')
            return 2
        try:
            command = (Path('/proc')/str(args.runner_pid)/'cmdline').read_bytes()
        except FileNotFoundError:
            # Children may still be alive. Preserve uncertainty; do not retry.
            save('runner-missing-without-terminal-marker-needs-inspection')
            return 2
        if str(build).encode() not in command:
            save('runner-pid-no-longer-matches-needs-inspection')
            return 2
        time.sleep(min(args.poll_seconds, max(0.01, deadline-time.monotonic())))
    runs = build/'xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1'
    tools = ROOT/'flow/fpga'
    if record['implementation_exit_code'] != 0:
        candidates = [runs/'level0_wrapper_routed_error.dcp', runs/'level0_wrapper_routed.dcp']
        checkpoint = next((p for p in candidates if p.is_file()), None)
        if checkpoint:
            record.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
            report('failed_checkpoint', tools/'diagnose_r40_checkpoint.tcl', [checkpoint, out/'failed_checkpoint_reports'])
        save('implementation-failed-evidence-retained-no-retry')
        return 1
    checkpoint, image = runs/'level0_wrapper_routed.dcp', build/'xo/tnca_u280.xclbin'
    if not checkpoint.is_file() or not image.is_file():
        save('successful-marker-missing-checkpoint-or-image-needs-inspection')
        return 2
    record.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  image=str(image), image_sha256=sha(image))
    save('extracting-reports')
    audit_ok = report('postroute', tools/'audit_r40_image.tcl', [build/'xo'])
    placed_ok = report('actual_cores', tools/'audit_r40_placed_cores.tcl', [checkpoint, out/'actual_cores.rpt'])
    audit = build/'xo/r40_postroute_audit'
    expected = [audit/(name+'.rpt') for name in ('route_status', 'cdc', 'clock_interaction',
                                               'utilization_slr', 'timing', 'bus_skew', 'checks')]
    expected.append(out/'actual_cores.rpt')
    record['reports'] = {str(p):sha(p) for p in expected if p.is_file() and p.stat().st_size}
    complete = audit_ok and placed_ok and len(record['reports']) == len(expected)
    save('reports-ready-review-required' if complete else 'report-extraction-incomplete-needs-inspection')
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
