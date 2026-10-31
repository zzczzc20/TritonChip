#!/usr/bin/env python3
"""Run an unchanged chat script while periodically preserving Python stacks."""
import faulthandler
import runpy
import sys

if len(sys.argv) < 2:
    raise SystemExit('expected chat script and its original arguments')
script = sys.argv[1]
sys.argv = sys.argv[1:]
faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)
try:
    runpy.run_path(script, run_name='__main__')
finally:
    faulthandler.cancel_dump_traceback_later()
