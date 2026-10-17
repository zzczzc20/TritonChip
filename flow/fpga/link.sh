#!/bin/bash
#==============================================================================
# Link tnca_u50.xo into a loadable .xclbin.
#
#   flow/fpga/link.sh <build-dir>
#
# <build-dir> is the directory package_xo.tcl was given. The .xclbin lands
# beside the .xo, with link.log, the v++ logs and the xclbin's own report
# beside it.
#
#------------------------------------------------------------------------------
# THE CLOCK, AND WHY THIS SCRIPT NAMES ONE
#
# The platform states what ap_clk will be, in its own hw.hpfm inside hw.xsa:
#
#   <xd:systemClocks xd:defaultClock="0">
#     <xd:clock xd:id="0" name="ulp_ucs_aclk_kernel_00"
#               xd:frequency="300.000000" xd:period="3.333333"
#               xd:status="scalable"/>
#     <xd:clock xd:id="1" name="ulp_ucs_aclk_kernel_01"
#               xd:frequency="500.000000" xd:period="2.000000"
#               xd:status="scalable"/>
#     <xd:clock xd:id="2" name="_bd_top_blp_s_aclk_freerun_ref_00"
#               xd:frequency="100.000000" xd:status="fixed"/>
#
# Those two frequencies are the clocks' power-on defaults, not the set of values
# the shell can produce. That set is declared by the clocking IP the shell is
# built from -- shell_ucs_subsystem_v3_0, whose parameter file gives
# FREQ_HZ_KERNEL_CLOCK_00 a minimum of 10 and a maximum of 650 MHz. 150 is
# inside that range, and 150 is what --kernel_frequency 0:150 below asks for.
#
# WHY 150
#
# Because the link failed at 300. Of the 173 setup endpoints it reported, 131
# are not reachable by editing RTL: 97 are a zero-logic, 0.722 ns path that a
# -3.884 ns skew between two clock objects the platform never relates fails on
# its own, one has both ends inside the platform, and 33 are a routing outcome
# spending 0.237 ns of its 4.854 in logic. The rest miss by under two
# nanoseconds against a single 3.333 ns period. A longer period reaches every
# one of them at once.
#
# 150 rather than a rounder 100 or 200 because of what it does to the device:
# rtl/board/tnca_u50_xo.v divided ap_clk by four (it divides by DEV_DIV = 2
# now, for 75 MHz: see the divider's header), and 150/4 is 37.5 MHz -- the
# frequency the device ran at in the link that failed, where its own domain
# reported +6.494 ns of slack and nothing failing in it. The device is held
# still deliberately, so every measurement made about it survives the change.
#
# WHY THIS IS NOT WHAT THE SECTION BELOW GUARDS AGAINST
#
# That section guards against v++ moving the clock by itself and reporting
# success anyway. Naming the number here, constraining it in flow/fpga/ooc.xdc,
# counting against it in the divider, and reading it back out of the finished
# .xclbin is the opposite of that: it is the clock the design was measured at,
# written down in four places that have to agree. If the link fails at it the
# next step is a report and a decision, not a smaller number slipped in -- and
# if it comes to that, the number moves in all four places or the .xclbin is not
# the design anybody measured.
#
# WHY 100, AND UNDIVIDED (2026-10-09)
#
# R38's timing work (doc/r38_board_bringup.md sec 6) took the device's worst
# path from a 60-level matrix loop to 6.84 ns post-synthesis, and timing-011
# took the accumulator file out of flip-flops, which cleared the routing
# congestion that held the device's SLR at 97.7 % of its CLBs. Linked at 100
# MHz with DEV_DIV = 1, that RTL routed with the kernel clock at +0.127 ns and
# nothing failing (566 734 endpoints), and on the U280 it passed conformance,
# the profiler and ArchBench tiny with every cycle count equal to the RTL
# simulation, at a measured 100.0 MHz. So the four places now say 100, and the
# divider 1. The /2 image of the same flow (150 MHz, DEV_DIV = 2: 75 MHz) also
# routed and ran; it is the fallback if a change costs this one its slack.
#
#------------------------------------------------------------------------------
# WHY THE RESULTING FREQUENCY IS CHECKED AND NOT ASSUMED
#
# Because v++ can change it without failing.
#
# Vitis 2022.2 ships a rule for exactly this, in scripts/ocl/ocl_rules.cfg:
#
#   AUTO-FREQ-SCALING-04: "One or more timing paths failed timing requirements.
#   The kernel clock %REF has an original frequency equal to %s MHz. The
#   frequency has been automatically changed to %s MHz to enable proper
#   functionality. The clock Id is %s."
#
# and scripts/ocl/ocl_util.tcl is where that path is taken (it reads a flag
# called enable_auto_freq_scale and calls write_new_clk_freq). What it does is
# turn a timing failure into a lowercase success at a lower clock.
#
# That is worse here than a plain failure would be, and not only because the
# number is lower: it would silently rewrite what flow/fpga/ooc.xdc says, what
# the divider comment in rtl/board/tnca_u50_xo.v says, and every measured figure
# this branch is built on -- and it would do all of that in a log whose last
# line says the link succeeded.
#
# There is a switch, and it was looked for.
# `compiler.enableAutoFrequencyScaling` is a real parameter of the v++ compiler:
# it is in the string table of libxv_pic_compiler.so, in a run of three with
# `compiler.skipTimingCheckAndFrequencyScaling` and
# `compiler.errorOnPulseWidthViolation`, which is where it picked up the
# description quoted above. It is not used here. A parameter name recovered
# from a binary is not an interface this repository can point at a line of
# documentation for, and the failure being guarded against is precisely the
# kind where an interface nobody documented gets set and the log still ends by
# saying the build succeeded. The readback below does not depend on any of it.
#
# So the link is checked:
#
#   1. The link log is grepped for the rule IDs. If one that moves or refuses
#      a clock is there, this exits non-zero with the rule's own message.
#   2. The .xclbin is read back with xclbinutil --info and the kernel clock is
#      taken from the entry whose Type is KERNEL. Anything other than the
#      frequency this script names is a failure here.
#
# Both fail closed. A report that cannot be parsed is a failure and not a pass,
# which is the direction an unknown format has to fail in -- and the raw report
# is kept at <build-dir>/xclbin.info so that the parse can be checked by eye.
#
#------------------------------------------------------------------------------
# LINK_EXTRA
#
# More v++ options, logged with the rest. The R40-04 image (package_xo.tcl's
# `hbm` variant) is linked with its master's connectivity here, e.g.
#   LINK_EXTRA="--connectivity.sp tnca_u50_1.hbm:HBM[0:31]"
# so the bank map is a recorded input of the link, not a property of a config
# file nobody kept.
#
# WHAT THIS SCRIPT DOES NOT DO
#
# It does not touch the board. No step here needs root, no step here loads
# anything, and the output is a file. Loading an .xclbin is a separate decision
# and it has not been taken.
#==============================================================================

set -u

OUT=${1:?usage: link.sh <build-dir>    # the directory package_xo.tcl wrote into}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

# Variant metadata is written only after successful XO packaging.
TNCA_KERNEL_VARIANT=none
if [ -f "$OUT/tnca_variant.txt" ]; then
    TNCA_KERNEL_VARIANT=$(cat "$OUT/tnca_variant.txt")
fi
case "$TNCA_KERNEL_VARIANT" in
none|hbm|multicore-hbm) ;;
*) echo "link.sh: unknown packaged variant $TNCA_KERNEL_VARIANT" >&2; exit 2 ;;
esac
if [ "$TNCA_KERNEL_VARIANT" = multicore-hbm ] && [ "${TNCA_BOARD:-u50}" != u280 ]; then
    echo "link.sh: multicore-hbm requires TNCA_BOARD=u280" >&2; exit 2
fi

# The card: TNCA_BOARD=u50 (default) or u280, from flow/fpga/boards.sh.
. "$(dirname "$0")/boards.sh"
VITIS=${XILINX_VITIS:-$BOARD_VITIS}
PLATFORM=$BOARD_PLATFORM
PLATFORMS_DIR=${XILINX_PLATFORMS:-/opt/xilinx/platforms}

# The clock this build asks for, and the one flow/fpga/ooc.xdc constrains and
# rtl/board/tnca_u50_xo.v divides by DEV_DIV (1 since 2026-10-09: the device at
# 100 MHz, measured on the U280; before that 150 MHz divided by 4, 3, then 2).
# The header says why it is not the platform default of 300; the four checks at
# the bottom of this file are what make sure it is this number that reaches the
# .xclbin.
EXPECT_MHZ=100

XO=$OUT/tnca_u50.xo
XCLBIN=$OUT/tnca_${BOARD}.xclbin
LOG=$OUT/link.log
if [ "$TNCA_KERNEL_VARIANT" = multicore-hbm ]; then
    # Keep total HBM capacity fixed; all three aliases need the same BO group.
    # Pass literal paths to v++ without routing them through LINK_EXTRA splitting.
    MULTICORE_CFG=$REPO/fpga/u280_multicore_hbm.cfg
    MULTICORE_PLACE=$REPO/flow/fpga/place_r40_multicore.tcl
    [ -f "$MULTICORE_CFG" ] && [ -f "$MULTICORE_PLACE" ] || {
        echo "link.sh: multicore connectivity/placement inputs missing" >&2; exit 2;
    }
fi
INFO=$OUT/xclbin.info

#------------------------------------------------------------------------------
# Preflight. Each of these is something that would otherwise be found an hour
# into the run, and the last two are the ones that would be found after eight.
#------------------------------------------------------------------------------
[ -f "$XO" ] || { echo "link.sh: no $XO -- run package_xo.tcl first" >&2; exit 2; }
[ -f "$VITIS/settings64.sh" ] || { echo "link.sh: no $VITIS/settings64.sh -- set XILINX_VITIS" >&2; exit 2; }
[ -d "$PLATFORMS_DIR/$PLATFORM" ] || { echo "link.sh: no platform $PLATFORM under $PLATFORMS_DIR" >&2; exit 2; }

source "$VITIS/settings64.sh" >/dev/null 2>&1

command -v v++ >/dev/null || { echo "link.sh: v++ not on PATH after sourcing settings64.sh" >&2; exit 2; }
command -v xclbinutil >/dev/null || { echo "link.sh: xclbinutil not on PATH after sourcing settings64.sh" >&2; exit 2; }

# Vivado's scratch goes under the build directory. / on this machine has been
# down to a few gigabytes, and a link that dies at 90% on ENOSPC is a link that
# gets run again from the beginning.
mkdir -p "$OUT/tmp" "$OUT/vpp_logs"
export TMPDIR=$OUT/tmp TMP=$OUT/tmp TEMP=$OUT/tmp

#------------------------------------------------------------------------------
# LINK_RECHECK=1 re-runs the checks below on a link that has already run,
# without running v++ again: v++'s exit status is taken from the VPP_RC line
# that run left in its log, and the verdict is appended to that log. It exists
# because a link is hours long and the checks are a script; a change to the
# checks should not cost a relink to apply to a build it was made for.
#------------------------------------------------------------------------------
if [ "${LINK_RECHECK:-0}" = 1 ]; then
    [ -f "$LOG" ] || { echo "link.sh: LINK_RECHECK=1 but no $LOG" >&2; exit 2; }
    rc=$(sed -n 's/^VPP_RC=//p' "$LOG" | tail -1)
    [ -n "$rc" ] || { echo "link.sh: LINK_RECHECK=1 but $LOG has no VPP_RC line" >&2; exit 2; }
    {
        echo "=== RECHECK $(date) ==="
        echo "CHECKS_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'not a git tree')"
    } >> "$LOG"
    echo "=== rechecking the link in $OUT (v++ rc=$rc), v++ not run ==="
else
    {
        echo "=== LINK START $(date) ==="
        echo "REPO=$REPO"
        echo "RTL_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'not a git tree')"
        echo "XO=$XO"
        echo "BOARD=$BOARD"
        echo "KERNEL_VARIANT=$TNCA_KERNEL_VARIANT"
        if [ "$TNCA_KERNEL_VARIANT" = multicore-hbm ]; then
            sha256sum "$MULTICORE_CFG" "$MULTICORE_PLACE"
        fi
        echo "BOARD_VPP_OPTS=$BOARD_VPP_OPTS"
        echo "LINK_EXTRA=${LINK_EXTRA:-}"
        echo "PLATFORM=$PLATFORM"
        echo "KERNEL_FREQUENCY_ARG=--kernel_frequency 0:$EXPECT_MHZ"
        echo "EXPECT_MHZ=$EXPECT_MHZ"
        echo "---"
    } > "$LOG"

    #--------------------------------------------------------------------------
    # The link
    #
    # -s keeps the intermediates. Not for debugging: if this fails on timing, the
    # answer to "which path" is in the post-route report, and that report only
    # exists if the routed checkpoint was kept.
    #--------------------------------------------------------------------------
    # VPP_JOBS bounds the parallelism on a shared host (v++ --jobs); unset, v++
    # chooses.
    multicore_args=()
    if [ "$TNCA_KERNEL_VARIANT" = multicore-hbm ]; then
        multicore_args=(--config "$MULTICORE_CFG"
            --vivado.prop "run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=$MULTICORE_PLACE")
    fi
    v++ -l -t hw \
        --platform "$PLATFORM" \
        ${VPP_JOBS:+--jobs $VPP_JOBS} \
        $BOARD_VPP_OPTS \
        "${multicore_args[@]}" \
        ${LINK_EXTRA:-} \
        --kernel_frequency 0:$EXPECT_MHZ \
        --temp_dir "$OUT/tmp" \
        --log_dir "$OUT/vpp_logs" \
        -s \
        -o "$XCLBIN" \
        "$XO" >> "$LOG" 2>&1
    rc=$?

    echo "VPP_RC=$rc" >> "$LOG"
    echo "=== v++ finished rc=$rc at $(date) ==="
fi

#------------------------------------------------------------------------------
# Check 1: did the frequency move?
#
# The rule IDs are quoted verbatim from scripts/ocl/ocl_rules.cfg, so this is
# a search for strings the tool itself defines rather than for a phrasing
# guessed at from its output. Vitis 2023.2 defines eight:
#
#   01, 02, 05   errors: a clock missed timing and could not be rescued
#   03, 04, 06, 07   warnings: "The frequency has been automatically changed"
#   08   advisory: "the auto scaled frequency %s MHz exceeds the original
#        specified frequency. The compiler will select the original specified
#        frequency of %s MHz."
#
# 08 is the one rule that says the clock did not move: the design closed with
# slack to spare and the frequency asked for is kept. The first U280 link that
# met timing (2026-10-07) printed it for both of its clocks, and a check that
# matched every rule ID failed that build. So 08 is the exception and every
# other ID fails -- including any a later release adds, which is the direction
# an unknown rule has to fail in. Check 2 reads the frequency back out of the
# .xclbin either way.
#------------------------------------------------------------------------------
moved=0
if awk '/AUTO-FREQ-SCALING/ && !/AUTO-FREQ-SCALING-08\]/ { f = 1 } END { exit !f }' "$LOG"; then
    moved=1
    echo
    echo "link.sh: v++ reported automatic frequency scaling. The clock moved and"
    echo "         this build is not the design that was measured:"
    echo
    grep -A3 'AUTO-FREQ-SCALING' "$LOG" | sed 's/^/         /'
    echo
elif grep -q 'AUTO-FREQ-SCALING-08\]' "$LOG"; then
    echo "link.sh: AUTO-FREQ-SCALING-08 only: timing closed with slack and the"
    echo "         requested frequencies were kept:"
    grep 'AUTO-FREQ-SCALING-08\]' "$LOG" | sed 's/^/         /'
fi

#------------------------------------------------------------------------------
# Check 2: what frequency actually went into the .xclbin?
#
# Not by looking for the entry whose Type is KERNEL. That was the first
# reading of this file, and it was wrong. On this shell the clock that
# --kernel_frequency 0: names comes out labelled DATA_CLK, and KERNEL_CLK is a
# different clock the design is not wired to. v++ says as much itself:
#
#   INFO: [v++ 60-991] clock name 'ulp_ucs/aclk_kernel_00' (clock ID '0') is
#   being mapped to clock name 'DATA_CLK' in the xclbin
#
#   INFO: [v++ 60-1230] ... System (SYSTEM) clock: hbm_aclk = 450,
#   Kernel (KERNEL) clock: ulp_ucs/aclk_kernel_01 = 500,
#   Kernel (DATA) clock: ulp_ucs/aclk_kernel_00 = 150
#
# A check that matched on "KERNEL" therefore compared 500 against 150 and
# failed a build whose real kernel clock was exactly what was asked for. So
# the name is read out of the tool's own mapping line rather than assumed, and
# the value is read back out of the artifact:
#
#   Scalable Clocks
#   ---------------
#      Name:      DATA_CLK
#      Index:     2
#      Type:      DATA
#      Frequency: 150 MHz
#
# The mapping line is specific to clock ID '0' on purpose: that is the ID this
# script passes, so that is the entry whose value decides this build's clock.
# No mapping line, no such entry, no frequency -- all leave got empty, and
# empty is a failure below. Failing on a missing mapping line is the point:
# the name is not guessed, so a shell that maps it somewhere else stops the
# link rather than being read the wrong way and passed.
#
# WHAT THIS NUMBER IS, AND WHAT IT IS NOT
#
# It is the frequency this build asked the platform for, read back out of the
# artifact that would be loaded -- so it catches a request that did not survive
# the trip into the .xclbin, which is exactly what AUTO-FREQ-SCALING above
# does.
#
# It is not a measurement of the clock. The kernel clock comes out of a clk_wiz
# MMCM spelled aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00 in the link's own
# vivado.log, and Vitis reaches it by re-solving that MMCM and emitting a
# create_generated_clock for the divide and multiply it picked
# (write_user_impl_clock_constraint in ocl_util.tcl). What lands in the
# metadata is the frequency the solver was handed. The shell carries a counter
# of the clock it actually gets -- frequency_counter_aclk_kernel_00, at 0x9000
# in the management aperture -- and reading that is the stronger check. It
# needs the .xclbin loaded, which is a decision this script does not take.
#------------------------------------------------------------------------------
mapped=$(sed -n \
    "s/.*clock name '[^']*' (clock ID '0') is being mapped to clock name '\([^']*\)'.*/\1/p" \
    "$LOG" | tail -1)

got=""
if [ $rc -eq 0 ] && [ -f "$XCLBIN" ]; then
    xclbinutil --info --input "$XCLBIN" > "$INFO" 2>&1
    if [ -n "$mapped" ]; then
        got=$(awk -v want="$mapped" '
            /^[ \t]*Name:/      { n = $2 }
            /^[ \t]*Frequency:/ && n == want { print $2; exit }
        ' "$INFO")
    fi
    echo "clock ID 0 maps to xclbin entry '${mapped:-<no mapping line>}'" >> "$LOG"
    echo "xclbin clock = ${got:-<unreadable>} MHz (expected $EXPECT_MHZ)" >> "$LOG"
    echo "Scalable Clocks block, as reported:" >> "$LOG"
    awk '/^Scalable Clocks/,/^System Clocks/' "$INFO" >> "$LOG"
fi

#------------------------------------------------------------------------------
# The verdict
#------------------------------------------------------------------------------
echo
if [ $rc -ne 0 ]; then
    echo "link.sh: FAILED -- v++ exited $rc. Full log: $LOG"
    echo "         If it failed on timing, the routed report is under"
    echo "         $OUT/tmp. This build already asks for a lowered frequency;"
    echo "         going lower again is a decision to take and write down in"
    echo "         all four places, not a retry -- see WHY 150 at the top."
    echo "LINK_FAILED rc=$rc $(date)" >> "$LOG"
    echo "=== LINK_DONE $(date) ===" >> "$LOG"
    exit $rc
fi

if [ "$moved" -ne 0 ]; then
    echo "link.sh: FAILED -- the link succeeded but the clock frequency was"
    echo "         changed by the tool. See above. Nothing here is loadable as"
    echo "         the design that was measured."
    echo "LINK_FAILED freq-moved $(date)" >> "$LOG"
    echo "=== LINK_DONE $(date) ===" >> "$LOG"
    exit 3
fi

if [ -z "$got" ]; then
    echo "link.sh: FAILED -- the clock this build asked for could not be read"
    echo "         back out of the .xclbin. Either the log carries no 'clock"
    echo "         ID 0 ... is being mapped to ...' line, or the entry it"
    echo "         names has no Frequency after it. The report is at $INFO and"
    echo "         the mapping line, if any, is quoted in $LOG; look at both"
    echo "         before treating this as a pass, because it is not one."
    echo "LINK_FAILED freq-unreadable $(date)" >> "$LOG"
    echo "=== LINK_DONE $(date) ===" >> "$LOG"
    exit 4
fi

if [ "$got" != "$EXPECT_MHZ" ]; then
    echo "link.sh: FAILED -- the kernel clock is ${got} MHz and this build"
    echo "         asked for ${EXPECT_MHZ} MHz. ap_clk is what the divider in"
    echo "         rtl/board/tnca_u50_xo.v counts against, so the device clock"
    echo "         is not what the header says either."
    echo "LINK_FAILED freq=$got $(date)" >> "$LOG"
    echo "=== LINK_DONE $(date) ===" >> "$LOG"
    exit 5
fi

# The divider is read from the wrapper this tree packages, not assumed: the
# message said /4 long after DEV_DIV became 3 and then 2.
div=$(sed -n 's/^[[:space:]]*parameter DEV_DIV[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' \
      "$REPO/rtl/board/tnca_u50_xo.v" | head -1)
dev=$(awk -v f="$got" -v d="${div:-0}" 'BEGIN { if (d > 0) printf "%g", f / d; else printf "?" }')
echo "link.sh: OK -- $XCLBIN"
echo "         kernel clock ${got} MHz, DEV_DIV ${div:-?}, device clock ${dev} MHz"
echo "         bitstream is in the .xclbin; loading it is a separate decision"
echo "         and has not been taken."
echo "LINK_OK $(date)" >> "$LOG"
echo "=== LINK_DONE $(date) ===" >> "$LOG"
exit 0
