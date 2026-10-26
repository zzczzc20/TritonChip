#!/bin/bash
set -u
source ~/tc_env/env.sh
cd ~/tritonchip-r40-convergence-8abf527 || exit 2
export PYTHONPATH=python:compiler/python:tools/asm CUDA_VISIBLE_DEVICES=
~/tc_env/torchenv/bin/python -u -c 'import sys; from pathlib import Path; script=Path("/tmp/r40_backend_tests_boundary_fix.py"); nominal=Path.cwd()/"tools/r36/backend_tests.py"; exec(compile(script.read_text(),str(script),"exec"),{"__name__":"__main__","__file__":str(nominal)})' --json /tmp/r40_backend_boundary_validation.json
r40_boundary_rc=$?
printf '%s\n' "$r40_boundary_rc" > /tmp/r40_backend_boundary_validation.exit
exit "$r40_boundary_rc"
