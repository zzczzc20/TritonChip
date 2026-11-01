#!/usr/bin/env python3
"""Enable Inductor precision casts after loading the Triton backend."""
import runpy,sys
import triton.backends.compiler
from torch._inductor import config
if len(sys.argv)<2:raise SystemExit('usage: wrapper SCRIPT [ARG...]')
script=sys.argv[1];sys.argv=sys.argv[1:]
with config.patch(emulate_precision_casts=True):runpy.run_path(script,run_name='__main__')
