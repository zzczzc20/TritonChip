#!/bin/bash
#==============================================================================
# final_image_ooc.sh -- Vivado evidence for the final-image candidate (M1-M4),
# run on the FPGA host only while it has >= 30 GB available memory and no
# host-latency experiment is running. Never touches the card; never writes a
# checkpoint outside OUT.
#
#   tools/r40/final_image_ooc.sh units  OLD_ROOT NEW_ROOT OUT
#   tools/r40/final_image_ooc.sh kernel NEW_ROOT IMAGE_TCL OUT
#   tools/r40/final_image_ooc.sh audit  SYNTH_DCP IMAGE_TCL NEW_ROOT OUT
#
# units   post-synthesis timing at 10 ns of every changed module, old source
#         (A3, 54249f7) against the candidate: tnca_vector_fpu (M4),
#         tnca_line_axi / tnca_memory_mode / tnca_multicore_mmio (M3),
#         tnca_idle_host_mux (M1). ~3-10 min and <= ~6 GB each.
# kernel  synth_r40_multicore.tcl on the candidate with the image generics:
#         ~70 min, ~8.5 GB (attempt-3 kernel synth_design took 1 h 08 min).
# audit   tools/r40/audit_cross_slr_cones.tcl on a kernel synthesis DCP with
#         the placement hook (also a hook dry-run). Run on A3's
#         ulp_tnca_u50_1_0.dcp (before) and the candidate's synth.dcp (after).
#         ~10-20 min, ~6-8 GB each.
#==============================================================================
set -euo pipefail
# The host is shared: other users' Vivado jobs hold much of its memory. A
# run starts only when at least OOC_MIN_AVAIL_GB is available (default 30;
# a 3-10 min unit run of <= 6 GB may set 16, leaving 10 GB of headroom).
need_memory() {
    local avail min=${OOC_MIN_AVAIL_GB:-30}
    avail=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$avail" -lt "$min" ]; then
        echo "REFUSED: only ${avail} GB available (< ${min} GB); not starting Vivado" >&2
        exit 3
    fi
}
mode=${1:?mode}; shift
case "$mode" in
units)
    OLD=${1:?old root}; NEW=${2:?new root}; OUT=${3:?out}
    mkdir -p "$OUT"
    # unit:generics, at the image's instance parameters
    UNITS=("tnca_vector_fpu:"
           "tnca_line_axi:SRC_W=2 TAG_W=3 OUT_N=8 DATA_W=256 DYNAMIC_LIMIT=1 ID_W=3"
           "tnca_memory_mode:PORTS=3 N_SRC=3 SRC_W=2 TAG_W=3"
           "tnca_multicore_mmio:N=3 PORTS=3 MEMORY_CONTROLS=1 CACHE_MAINTENANCE=1 AXI_OUT_N=8 REG_ADDR_W=9"
           "tnca_idle_host_mux:TAG_W=5")
    for side in old new; do
        root=$OLD; [ "$side" = new ] && root=$NEW
        files=$(cd "$root/rtl" && ls common/*.v core/*.v gen/*.v nuca/*.v noc/*.v tile/*.v fpga/*.v *.v)
        for entry in "${UNITS[@]}"; do
            unit=${entry%%:*}; generics=${entry#*:}
            need_memory
            dir=$OUT/$side/$unit
            mkdir -p "$dir"
            OOC_SV=1 OOC_GENERICS="$generics" nice -n 10 vivado -mode batch -nojournal -nolog \
                -source "$NEW/flow/synth/vivado_ooc_timing.tcl" \
                -tclargs "$unit" 10 xcu280-fsvh2892-2L-e "$dir" "$root/rtl" $files > "$dir/vivado.log" 2>&1
            printf '%s %s: ' "$side" "$unit"
            awk '/WNS\(ns\)/{getline; getline; printf "WNS %s ns (%s failing of %s)\n", $1, $3, $4; exit}' "$dir"/*_summary.rpt
            awk '/^Slack/{s=$4} /Data Path Delay/{d=$4} /Logic Levels/{print "    " s, d, $3 " levels"; n++} n==1{exit}' "$dir"/*_paths.rpt || true
        done
    done
    ;;
kernel)
    NEW=${1:?new root}; IMAGE=${2:?image.tcl}; OUT=${3:?out}
    need_memory
    nice -n 10 vivado -mode batch -nojournal -log "$OUT.log" \
        -source "$NEW/flow/fpga/synth_r40_multicore.tcl" -tclargs "$NEW" "$IMAGE" "$OUT"
    ;;
audit)
    DCP=${1:?synth dcp}; IMAGE=${2:?image.tcl}; NEW=${3:?new root}; OUT=${4:?out}
    need_memory
    TNCA_IMAGE_CONFIG=$IMAGE nice -n 10 vivado -mode batch -nojournal -log "$OUT.log" \
        -source "$NEW/tools/r40/audit_cross_slr_cones.tcl" \
        -tclargs "$DCP" "$OUT" "$NEW/flow/fpga/place_r40_multicore.tcl"
    cat "$OUT/summary.tsv"
    ;;
*) echo "unknown mode $mode" >&2; exit 2 ;;
esac
