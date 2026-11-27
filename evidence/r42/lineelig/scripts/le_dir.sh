#!/bin/bash
cd ~/r42/lineelig || exit 2; source ~/tc_env/env.sh; export PYTHONPATH=$PWD/tools/asm
nice -n 19 python3 tools/r42/lineelig_tests.py directed
echo DIR_DONE
