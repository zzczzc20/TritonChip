#!/bin/bash
#==============================================================================
# final_image_observe.sh -- reports after the last implementation, whatever its
# exit (0, 3 = v++ scaled the clock, other = failure). Read-only on the
# checkpoints; never retries, never loads the card.
#
#   tools/r40/final_image_observe.sh SOURCE_ROOT BUILD_ROOT
#
# Runs flow/fpga/audit_r40_image.tcl on the build, finds the routed checkpoint,
# reads the achieved kernel clock from the routed timing, runs
# flow/fpga/audit_r40_operating_clock.tcl at that clock, and the failing-endpoint
# census when setup fails at 100 MHz. Output under BUILD_ROOT/observe/.
#==============================================================================
set -u
SRC=${1:?source}; BUILD=${2:?build}
OUT=$BUILD/observe
mkdir -p "$OUT"
set +u
source /tools/Xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1
set -u
log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/observe.log"; }
log "attempt exit $(cat "$BUILD/build/attempt.exit" 2>/dev/null)"
vivado -mode batch -nojournal -log "$OUT/image_audit.log" -source "$SRC/flow/fpga/audit_r40_image.tcl" \
    -tclargs "$BUILD" > "$OUT/image_audit.out" 2>&1
log "image audit exit $?"
DCP=$(find "$BUILD" -path '*impl_1*' -name '*_routed.dcp' | head -1)
[ -n "$DCP" ] || { log "no routed checkpoint"; exit 1; }
RPT=$(find "$BUILD" -path '*impl_1*' -name '*timing_summary_routed.rpt' | head -1)
WNS=$(awk '/WNS\(ns\)/{getline; getline; print $1; exit}' "$RPT" 2>/dev/null)
log "routed $DCP WNS(100 MHz report)=$WNS"
MHZ=$(grep -ho 'changed to [0-9.]* MHz' "$BUILD"/xo/link.log 2>/dev/null | tail -1 | grep -o '[0-9.]*' | cut -d. -f1)
MHZ=${MHZ:-100}
vivado -mode batch -nojournal -log "$OUT/operating_clock.log" -source "$SRC/flow/fpga/audit_r40_operating_clock.tcl" \
    -tclargs "$DCP" "$OUT/operating_$MHZ" "$MHZ" placement > "$OUT/operating_clock.out" 2>&1
log "operating clock audit at $MHZ MHz exit $?"
if awk -v w="$WNS" 'BEGIN{exit !(w < 0)}'; then
    vivado -mode batch -nojournal -log "$OUT/census.log" -source "$SRC/tools/r40/census_failing_endpoints.tcl" \
        -tclargs "$DCP" "$OUT/census" 20000 > "$OUT/census.out" 2>&1
    log "census exit $?"
fi
log OBSERVE-DONE
