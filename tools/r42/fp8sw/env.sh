# source from the tree root on GB10: sets the three-directory PYTHONPATH of R42_COMMON.md
rm -rf python/tnca/data
source ~/tc_env/env.sh
export PYTHONPATH=$PWD:$PWD/python:$PWD/compiler/python:$PWD/tools/asm
export PY=$HOME/tc_env/torchenv/bin/python
