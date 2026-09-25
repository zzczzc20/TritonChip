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
# THE CLOCK, AND WHY THIS SCRIPT DOES NOT NAME ONE
#
# The platform states what ap_clk will be, in its own hw.hpfm inside
# hw.xsa:
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
# defaultClock="0" is 300 MHz. That is the number flow/fpga/ooc.xdc constrains
# and the number rtl/board/tnca_u50_xo.v divides by eight to get 37.5 MHz for
# the device. The 500 MHz entry is id 1: it exists, it is not the default, and
# taking it would be a change of design and not a change of build flag.
#
# So --kernel_frequency is not passed. Its own help text says it is there to
# override "a default value from DSA", and a flag that overrides 300 with 300 is
# a flag that is one edit away from overriding 300 with 250. v++ will ask for it
# if the design does not close. The answer to that is not a smaller number.
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
# There is no switch to turn it off that this repository can point at a line of
# documentation for. Rather than reach for one anyway, the link is checked:
#
#   1. The link log is grepped for the rule ID. If it is there, the frequency
#      moved and this exits non-zero with the rule's own message.
#   2. The .xclbin is read back with xclbinutil --info and the kernel clock is
#      taken from the entry whose Type is KERNEL. Anything other than the
#      platform default is a failure here.
#
# Both fail closed. A report that cannot be parsed is a failure and not a pass,
# which is the direction an unknown format has to fail in -- and the raw report
# is kept at <build-dir>/xclbin.info so that the parse can be checked by eye.
#
#------------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES NOT DO
#
# It does not touch the board. No step here needs root, no step here loads
# anything, and the output is a file. Loading an .xclbin is a separate decision
# and it has not been taken.
#==============================================================================

set -u

OUT=${1:?usage: link.sh <build-dir>    # the directory package_xo.tcl wrote into}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

VITIS=${XILINX_VITIS:-/home/x4/Software/Xilinx/Vitis/2022.2}
PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
PLATFORMS_DIR=${XILINX_PLATFORMS:-/opt/xilinx/platforms}

# From hw.hpfm: defaultClock="0", and clock 0 is 300.000000 MHz.
EXPECT_MHZ=300

XO=$OUT/tnca_u50.xo
XCLBIN=$OUT/tnca_u50.xclbin
LOG=$OUT/link.log
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

{
    echo "=== LINK START $(date) ==="
    echo "REPO=$REPO"
    echo "RTL_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'not a git tree')"
    echo "XO=$XO"
    echo "PLATFORM=$PLATFORM"
    echo "KERNEL_FREQUENCY_ARG=none (platform default)"
    echo "EXPECT_MHZ=$EXPECT_MHZ"
    echo "---"
} > "$LOG"

#------------------------------------------------------------------------------
# The link
#
# -s keeps the intermediates. Not for debugging: if this fails on timing, the
# answer to "which path" is in the post-route report, and that report only
# exists if the routed checkpoint was kept.
#------------------------------------------------------------------------------
v++ -l -t hw \
    --platform "$PLATFORM" \
    --temp_dir "$OUT/tmp" \
    --log_dir "$OUT/vpp_logs" \
    -s \
    -o "$XCLBIN" \
    "$XO" >> "$LOG" 2>&1
rc=$?

echo "VPP_RC=$rc" >> "$LOG"
echo "=== v++ finished rc=$rc at $(date) ==="

#------------------------------------------------------------------------------
# Check 1: did the frequency move?
#
# The rule ID is quoted verbatim from scripts/ocl/ocl_rules.cfg, so this is a
# search for a string the tool itself defines rather than for a phrasing
# guessed at from its output.
#------------------------------------------------------------------------------
moved=0
if grep -q 'AUTO-FREQ-SCALING' "$LOG"; then
    moved=1
    echo
    echo "link.sh: v++ reported automatic frequency scaling. The clock moved and"
    echo "         this build is not the design that was measured:"
    echo
    grep -A3 'AUTO-FREQ-SCALING' "$LOG" | sed 's/^/         /'
    echo
fi

#------------------------------------------------------------------------------
# Check 2: what frequency actually went into the .xclbin?
#
# The format, taken from xclbinutil --info on a shipped .xclbin rather than
# assumed:
#
#   Scalable Clocks
#   ---------------
#      Name:      KERNEL_CLK
#      Index:     2
#      Type:      KERNEL
#      Frequency: 500 MHz
#
# Type comes before Frequency inside a block, and exactly one block is the
# kernel's. Anything else -- no such block, no frequency after it -- leaves
# got empty, and empty is a failure below.
#------------------------------------------------------------------------------
got=""
if [ $rc -eq 0 ] && [ -f "$XCLBIN" ]; then
    xclbinutil --info --input "$XCLBIN" > "$INFO" 2>&1
    got=$(awk '
        /^[ \t]*Type:/      { t = $2 }
        /^[ \t]*Frequency:/ && t == "KERNEL" { print $2; exit }
    ' "$INFO")
    echo "xclbin kernel clock = ${got:-<unreadable>} MHz (expected $EXPECT_MHZ)" >> "$LOG"
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
    echo "         $OUT/tmp. Lowering the frequency is not the next step;"
    echo "         flow/fpga/ooc.xdc and rtl/board/tnca_u50_xo.v record why."
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
    echo "link.sh: FAILED -- the kernel clock could not be read out of the"
    echo "         .xclbin. The report is at $INFO; look at it before"
    echo "         treating this as a pass, because it is not one."
    echo "LINK_FAILED freq-unreadable $(date)" >> "$LOG"
    echo "=== LINK_DONE $(date) ===" >> "$LOG"
    exit 4
fi

if [ "$got" != "$EXPECT_MHZ" ]; then
    echo "link.sh: FAILED -- the kernel clock is ${got} MHz and the platform"
    echo "         default is ${EXPECT_MHZ} MHz. ap_clk is what the divider in"
    echo "         rtl/board/tnca_u50_xo.v counts against, so the device clock"
    echo "         is not what the header says either."
    echo "LINK_FAILED freq=$got $(date)" >> "$LOG"
    echo "=== LINK_DONE $(date) ===" >> "$LOG"
    exit 5
fi

echo "link.sh: OK -- $XCLBIN"
echo "         kernel clock ${got} MHz, device clock $((got / 8)) MHz"
echo "         bitstream is in the .xclbin; loading it is a separate decision"
echo "         and has not been taken."
echo "LINK_OK $(date)" >> "$LOG"
echo "=== LINK_DONE $(date) ===" >> "$LOG"
exit 0
