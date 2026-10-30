#!/usr/bin/env python3
"""Start the corrected-image queue once its reviews and approval exist."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import candidate_queue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','work','image','approval','out','python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--xclbinutil', default='/opt/xilinx/xrt/bin/xclbinutil')
    p.add_argument('--poll', type=float, default=30)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('poll must be 1..60 seconds')
    root,work = a.root.resolve(),a.work.resolve()
    with (work/'candidate_waiter.lock').open('a') as lock, (work/'candidate_waiter.jsonl').open('a',buffering=1) as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            waiver=work/'clock_waiver.json'
            required=[a.image,a.approval,work/'clock_review.json',work/'cdc_review.json',
                      work/'source_review.json',work/'xclbin_hbm_map.json',work/'reports/cdc.rpt',
                      root/'spec/r41/run3/stages/identity.json']
            missing=[str(x) for x in required if not x.is_file()]
            reason='waiting for '+', '.join(missing) if missing else None
            if not reason:
                try:
                    candidate_queue.preflight(root,work,a.image,a.approval,waiver if waiver.is_file() else None,a.xclbinutil)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    reason=f'{type(error).__name__}: {error}'
            if reason!=previous or reason is None:
                log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           event='waiting' if reason else 'starting',reason=reason))+'\n')
                previous=reason
            if reason is None:
                command=[str(a.python),str(root/'tools/r41/run2_board/candidate_queue.py'),
                    '--root',str(root),'--work',str(work),'--image',str(a.image),
                    '--approval',str(a.approval),'--out',str(a.out),'--python',str(a.python),
                    '--xclbinutil',a.xclbinutil]
                if waiver.is_file():command += ['--waiver',str(waiver)]
                with (work/'candidate_queue_process.log').open('a') as output:
                    result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
                log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           event='queue-exit',returncode=result.returncode))+'\n')
                return result.returncode
            time.sleep(a.poll)


if __name__ == '__main__':
    raise SystemExit(main())
