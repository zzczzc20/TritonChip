#!/bin/bash
# R41 G2: run free screens one after another on the x86 host (one lane; the
# caller runs at most two lanes, each Vivado job with <= 4 threads).
#   screen_chain.sh WAIT_PID SRC_TGZ COMMIT JOB [JOB ...]
# JOB is either
#   NAME:IMAGE_TCL                kernel synthesis, flow/fpga/synth_r40_multicore.tcl
#   tile:NAME:SYNTH_DIR:TILE:SLR  tile OOC place and route at 10 ns, flow/fpga/screen_r41_tile.tcl,
#                                 on SYNTH_DIR/out/synth.dcp with SYNTH_DIR/src's flow scripts
# Output: ~/r41res/NAME_COMMIT (kernel) or ~/r41res/NAME (tile).
set -u
wait_pid=$1 tgz=$2 commit=$3; shift 3
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
source ~/x86_link_env.sh >/dev/null 2>&1
for job in "$@"; do
    if [ "${job%%:*}" = tile ]; then
        IFS=: read -r _ name sdir tile slr <<< "$job"
        d=~/r41res/$name
        [ -e "$d" ] && { echo "skip existing $d"; continue; }
        mkdir -p "$d"
        until grep -q '^R40_MULTICORE_SYNTH_DONE=' "$sdir/synth.log" 2>/dev/null; do sleep 30; done
        ( cd "$d" && nice -n 5 vivado -mode batch -nojournal -log screen.log \
            -source "$sdir/src/flow/fpga/screen_r41_tile.tcl" -tclargs "$sdir/out/synth.dcp" "$tile" "$slr" out > /dev/null 2>&1 )
    else
        name=${job%%:*} image=${job#*:}
        d=~/r41res/${name}_${commit}
        [ -e "$d" ] && { echo "skip existing $d"; continue; }
        mkdir -p "$d/src" && tar xzf "$tgz" -C "$d/src" || exit 1
        ( cd "$d" && nice -n 5 vivado -mode batch -nojournal -log synth.log \
            -source src/flow/fpga/synth_r40_multicore.tcl -tclargs src "src/$image" out > /dev/null 2>&1 )
    fi
    echo "$(date -u +%FT%TZ) $job exit $?" >> ~/r41res/screen_chain.log
done
