"""Package reports, raw compiler logs and assembly, excluding bulky caches."""
from pathlib import Path
import sys
import tarfile
root = Path(sys.argv[1])
output = Path(sys.argv[2])
labels = sys.argv[3:]
with tarfile.open(output, 'w:gz', compresslevel=6) as tar:
    for name in ('harness.sha256', 'int.tree', 'base.tree', 'start-memory.log', 'resource-snapshots.log',
                 'run_audit.py', 'run_gb10.sh'):
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
