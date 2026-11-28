#!/bin/bash
# run_one.sh TREE BUILD OUTLOG [make args...]   (GB10, nice 19): one `make core-run2` style gate with the given make arguments, log with EXIT line
TREE=$1; BUILD=$2; LOG=$3; shift 3
source ~/tc_env/env.sh
cd $TREE && nice -n 19 make BUILD=$BUILD core-run2 "$@" > $LOG 2>&1; echo "EXIT=$?" >> $LOG
