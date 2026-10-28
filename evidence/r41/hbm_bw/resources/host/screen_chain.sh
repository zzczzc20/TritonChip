#!/bin/bash
# R41 G2: run kernel-synthesis screens one after another on the x86 host
# (one lane; the caller runs at most two lanes). Free screens only.
#   screen_chain.sh WAIT_PID SRC_TGZ COMMIT NAME:IMAGE_TCL [NAME:IMAGE_TCL ...]
# Each screen: ~/r41res/NAME_COMMIT/{src,out,synth.log}, flow/fpga/synth_r40_multicore.tcl.
set -u
wait_pid=$1 tgz=$2 commit=$3; shift 3
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
source ~/x86_link_env.sh >/dev/null 2>&1
for job in "$@"; do
    name=${job%%:*} image=${job#*:}
    d=~/r41res/${name}_${commit}
    [ -e "$d" ] && { echo "skip existing $d"; continue; }
    mkdir -p "$d/src" && tar xzf "$tgz" -C "$d/src" || exit 1
    ( cd "$d" && nice -n 5 vivado -mode batch -nojournal -log synth.log \
        -source src/flow/fpga/synth_r40_multicore.tcl -tclargs src "src/$image" out > /dev/null 2>&1 )
    echo "$(date -u +%FT%TZ) $name exit $?" >> ~/r41res/screen_chain.log
done
