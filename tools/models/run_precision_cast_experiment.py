#!/usr/bin/env python3
"""Run a diagnostic script with the non-default Inductor precision-cast setting."""
import runpy
import sys
from torch._inductor import config

if len(sys.argv)<2:
    raise SystemExit('usage: run_precision_cast_experiment.py SCRIPT [ARG ...]')
script=sys.argv[1]
sys.argv=sys.argv[1:]
with config.patch(emulate_precision_casts=True):
    runpy.run_path(script,run_name='__main__')
