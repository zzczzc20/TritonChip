#!/usr/bin/env python3
"""Wait for a complete run-2 plan, then start its guarded board queue once."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import auto


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--poll', type=float, default=20)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('poll must be 1..60 seconds')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'watch.lock').open('w') as lock, (a.out/'watch.jsonl').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            try:
                plan = json.loads(a.plan.read_text())
                auto.validate_plan(plan, a.image)
                program = Path(plan['root'])/'flow/fpga/program.sh'
                if not program.is_file():
                    raise ValueError('registered program.sh missing')
                reason = None
            except (OSError, ValueError, KeyError, TypeError) as error:
                reason = f'{type(error).__name__}: {error}'
            if reason != previous or reason is None:
                log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                          event='waiting' if reason else 'starting', reason=reason)) + '\n')
                previous = reason
            if reason is None:
                command = [sys.executable, str(Path(auto.__file__).resolve()), '--plan', str(a.plan),
                           '--image', str(a.image), '--program', str(program), '--out', str(a.out/'queue')]
                with (a.out/'queue-process.log').open('a') as output:
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                          event='queue-exit', returncode=result.returncode)) + '\n')
                return result.returncode
            time.sleep(a.poll)


if __name__ == '__main__':
    raise SystemExit(main())
