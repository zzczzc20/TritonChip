"""Package reports, raw compiler logs and assembly, excluding bulky caches."""
from pathlib import Path
import sys
import json
import tarfile
root = Path(sys.argv[1])
output = Path(sys.argv[2])
labels = sys.argv[3:]
with tarfile.open(output, 'w:gz', compresslevel=6) as tar:
    for name in ('harness.sha256', 'int.tree', 'base.tree', 'start-memory.log', 'resource-snapshots.log',
                 'run_audit.py', 'run_gb10.sh', 'attribute_x8.py', 'x8-attribution.json', 'x8-attribution.log'):
        p=root/name
        if p.exists(): tar.add(p, arcname=name)
    for label in labels:
        for suffix in ('.log', '.memory.log', '-compare.json', '-compare.log'):
            p=root/(label+suffix)
            if p.exists(): tar.add(p, arcname=p.name)
        directory=root/label
        if directory.exists():
            for p in sorted(directory.rglob('*')):
                if p.is_file() and (p.name in ('report.json', 'result.json', 'child.log') or p.suffix == '.s'):
                    tar.add(p, arcname=str(p.relative_to(root)))

    if 'x8' in labels:
        for label in ('int-default', 'int-x8'):
            report=json.loads((root/label/'report.json').read_text())
            for point, result in sorted(report['points'].items()):
                if point not in ('op.add/tiny/fp32-n256', 'op.matmul/tiny/fp16-m16-k32-n16', 'op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8'): continue
                for name, kernel in sorted(result['kernels'].items()):
                    asm=root/label/'runs'/point.replace('/', '__')/kernel['asm']
                    for suffix in ('tnir', 'opt.tnir', 'spmd.tnir', 'sched.tnir'):
                        path=asm.with_name(name+'.'+suffix)
                        if path.exists(): tar.add(path, arcname=str(path.relative_to(root)))
