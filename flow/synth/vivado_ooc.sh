#!/bin/bash
#==============================================================================
# vivado_ooc.sh -- post-synthesis timing of one unit, or the whole device, in
# minutes (flow/synth/vivado_ooc_timing.tcl says what the numbers are).
#
#   flow/synth/vivado_ooc.sh <unit> <period_ns> [NAME=VALUE ...]
#
#   unit  tnca_fpga_top   the board's device (everything under rtl/ but the
#                         board wrapper): what sets the board clock
#         tnca_matrix | tnca_vector_fpu | tnca_tensor | any module under rtl/
#
# Every .v under rtl/{common,core,gen,nuca,noc,tile,fpga} and rtl/*.v is read,
# as flow/fpga/package_xo.tcl reads them, so a unit is timed in the source it
# is built from. NAME=VALUE pairs set the top's parameters (e.g. FP_LAT=3).
# Reports go to sim/build/ooc/<unit>[...]/: *_summary.rpt, *_paths.rpt (one
# worst path per endpoint) and *_util.rpt. Needs Vivado on PATH (source its
# settings64.sh); PART defaults to the U280's.
#
# Measured on 2026-10-08, out of context at 13.333 ns, U280 part:
#   tnca_matrix      14.7 ns (hand-written FMA, acc bypass) -> 8.6 ns
#   tnca_vector_fpu   9.2 ns (hand-written FMA)              -> 6.8 ns
# where "->" is the R38 timing work at unchanged cycles (doc/r38_board_bringup.md
# sec 6). The board's routed worst path ran 1.6x its post-synthesis estimate.
#==============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
UNIT=${1:?unit}
PERIOD=${2:?period in ns}
shift 2
PART=${PART:-xcu280-fsvh2892-2L-e}
TAG=$UNIT${*:+_$(echo "$*" | tr ' =' '__')}
OUT=$ROOT/sim/build/ooc/$TAG
mkdir -p "$OUT"
FILES=$(cd "$ROOT/rtl" && ls common/*.v core/*.v gen/*.v nuca/*.v noc/*.v tile/*.v fpga/*.v *.v)
OOC_GENERICS="$*" vivado -mode batch -nojournal -nolog \
    -source "$ROOT/flow/synth/vivado_ooc_timing.tcl" \
    -tclargs "$UNIT" "$PERIOD" "$PART" "$OUT" "$ROOT/rtl" $FILES > "$OUT/vivado.log" 2>&1
awk '/WNS\(ns\)/{getline; getline; printf "%s: WNS %s ns at %s ns (%s failing of %s endpoints)\n", "'"$TAG"'", $1, "'"$PERIOD"'", $3, $4; exit}' "$OUT"/*_summary.rpt
awk '/^Slack/{s=$4} /Source:/{src=$2} /Destination:/{dst=$2} /Data Path Delay/{d=$4} /Logic Levels/{print "  " s, d, $3 " levels", src, "->", dst}' \
    "$OUT"/*_paths.rpt | head -10
