#!/usr/bin/env python3
"""Kill actual owners, not process groups: native model/pass must both stop."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run import ROOT, sha
from validate_chip import assemble


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_for(callback, message):
    until = time.monotonic()+10
    while time.monotonic() < until:
        value = callback()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError(message)


def children(pid):
    try:
        return [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    binary, base, out = args.binary.resolve(), args.config.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    values = dict(line.split('#')[0].strip().split('=', 1) for line in base.read_text().splitlines()
                  if '=' in line.split('#')[0])
    values.update(memory_bytes=str(1 << 24), trace_events='0', dram_path='1')
    config = out/'model.cfg'
    config.write_text(''.join(f'{k}={v}\n' for k,v in values.items()))
    image = out/'infinite.hex'
    assemble('.li s1, #1\nspin:\nbr.nz s1, spin\n', image)
    commands = out/'infinite.commands'
    commands.write_text(f'code {image}\nlaunch 1 0 0 1 1 1\n')
    rows = []
    env = dict(os.environ, TNCA_DRAM_PATH='slow', TNCA_TLM_MONITOR='off')
    for kind in ('pass-owner-death', 'backend-owner-death', 'runner-owner-death'):
        directory = out/kind
        directory.mkdir()
        owned = []
        if kind == 'pass-owner-death':
            model_out = directory/'model';model_out.mkdir()
            argv = [str(binary), str(config), str(commands), str(model_out)]
        else:
            script = directory/'owner.py'
            if kind == 'backend-owner-death':
                code = (
                    'import tlm_backend,time\n'
                    f'b=tlm_backend.make_backend_class()(binary={str(binary)!r},config={str(config)!r},out={str(directory)!r})\n'
                    'b.start()\n'
                    f'open({str(directory/"model.pid")!r},"w").write(str(b.proc.pid))\n'
                    f'data=b"".join(int(w,16).to_bytes(8,"little") for w in open({str(image)!r}).read().split())\n'
                    'b.upload(0,data)\n'
                    'b.wfile.write(b"G 0 0 1 1 1 1 1 0 1 0\\n")\n'
                    'time.sleep(60)\n')
            else:
                code = ('from run import run\n'
                        f'run({str(binary)!r},{str(config)!r},{str(commands)!r},{str(directory/"model")!r},timeout=60,mode="chip",monitor="off")\n')
            script.write_text(code)
            argv = [sys.executable, str(script)]
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'python'), str(Path(__file__).resolve().parent),
                                           str(ROOT/'compiler/python'), str(ROOT/'tools/asm')])
        with (directory/'stdout.log').open('w') as stdout, (directory/'stderr.log').open('w') as stderr:
            owner = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                owned.append(owner.pid)
                if kind == 'pass-owner-death':
                    model = owner.pid
                else:
                    def native_child():
                        for pid in children(owner.pid):
                            try:
                                if Path(f'/proc/{pid}/exe').resolve() == binary:
                                    return pid
                            except (FileNotFoundError, ProcessLookupError):
                                pass
                        return None
                    model = wait_for(native_child, 'owner did not start native model')
                    owned.append(model)
                child = wait_for(lambda: next(iter(children(model)), None), 'native slow path did not fork a live pass')
                owned.append(child)
                assert alive(owner.pid) and alive(model) and alive(child), 'test did not observe live processes'
                # Deliberately leave other process groups alone. This tests
                # kernel-enforced parent ownership, including the fork pass.
                owner.kill();owner.wait(timeout=5)
                wait_for(lambda: not alive(model) and not alive(child), 'native model or slow pass survived its owner')
                rows.append(dict(case=kind, owner_pid=owner.pid, model_pid=model, pass_pid=child,
                                 owner_signal='SIGKILL', status='pass'))
            finally:
                for pid in reversed(owned):
                    if alive(pid):
                        os.kill(pid, signal.SIGKILL)
                owner.wait(timeout=5)
    report = dict(schema='tnca-slow-process-lifetime/1', status='pass', rows=rows,
                  binary_sha256=sha(binary), config_sha256=sha(config), image_sha256=sha(image))
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
