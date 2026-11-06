"""Collect all launch timings while running the original native assertions."""
import json
from pathlib import Path
import runpy
import sys
root = Path(sys.argv.pop(1)).resolve()
script = sys.argv.pop(1)
for p in (root, root/'python', root/'compiler/python', root/'tools/asm'):
    sys.path.insert(0, str(p))
import torch
import triton
from tnca.triton_backend.driver import launch_log
sys.argv[0] = str(root / 'tools/r40' / script)
with launch_log() as launches:
    runpy.run_path(sys.argv[0], run_name='__main__')
print('AUDIT_LAUNCHES ' + json.dumps(launches, sort_keys=True, default=str), flush=True)
