#!/usr/bin/env python3
"""Select the diagnostic tree before Triton, then enable precision casts."""
import runpy
import sys
from pathlib import Path
if len(sys.argv) < 2:
    raise SystemExit('usage: wrapper SCRIPT [ARG...]')
script = Path(sys.argv[1]).resolve()
root = script.parents[2]
paths = [root / 'python', root / 'python/_staged', root / 'compiler/python',
         root / 'tools/asm', root / 'tools/performance_harness/systemc']
sys.path[:0] = [str(p) for p in paths]
import triton.backends.compiler
from torch._inductor import config
sys.argv = sys.argv[1:]
with config.patch(emulate_precision_casts=True):
    runpy.run_path(str(script), run_name='__main__')
