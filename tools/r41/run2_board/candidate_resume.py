#!/usr/bin/env python3
"""Resume a stopped candidate queue without reprogramming or discarding evidence."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess

import auto
import candidate_queue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'image', 'approval', 'waiver', 'out', 'python'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--xclbinutil', default='xclbinutil')
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    a = p.parse_args()
    root, work, image, out = (v.resolve() for v in (a.root, a.work, a.image, a.out))
    state = candidate_queue.preflight(root, work, image, a.approval, a.waiver,
                                      a.xclbinutil, resume=True)
    if out.exists():
        raise ValueError('resume output already exists')
    out.mkdir(parents=True)
    with (work/'candidate_queue.lock').open('a') as lock, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kw):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kw))+'\n')
        def check_board():
            row = auto.probe(a.sysfs)
            if not auto.idle(row) or row['uuid'] != state['image_uuid'] or auto.clock_mhz(a.sysfs) != 100:
                raise RuntimeError('card busy or candidate image/clock changed')
            return row
        for item in state['registrations']:
            record('board-check', **check_board())
            reg = Path(item['path'])
            if auto.sha(reg) != item['sha256']:
                raise RuntimeError('candidate registration changed')
            target = out/'results'/item['id']
            cmd = [str(a.python), str(root/'tools/r41/run2_board/stage.py'), '--root', str(root),
                   '--out', str(target), '--python', str(a.python), '--registration', str(reg)]
            record('stage-start', stage=item['id'])
            with (out/(item['id']+'.log')).open('x') as f:
                result = subprocess.run(cmd, cwd=root, env=auto.board_environment(), stdout=f,
                                        stderr=subprocess.STDOUT, timeout=259200)
            record('stage-end', stage=item['id'], returncode=result.returncode)
            if result.returncode or json.loads((target/'stage.json').read_text()).get('status') != 'pass':
                raise RuntimeError('candidate stage failed: ' + item['id'])
            check_board()
        record('completed')


if __name__ == '__main__':
    main()
