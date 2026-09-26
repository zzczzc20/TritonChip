#!/bin/bash
#==============================================================================
# Put tnca_u50.xclbin onto the U50.
#
#   flow/fpga/program.sh --check <xclbin>   # run every gate, change nothing
#   flow/fpga/program.sh --go    <xclbin>   # program the device
#   flow/fpga/program.sh --status [<xclbin>]# what is loaded, how the last
#                                           # program.sh run went
#
# package_xo.tcl and link.sh each end in a file. This one does not. It is the
# only thing in flow/fpga/ that changes hardware state: it writes a bitstream
# into the device's reconfigurable region, replacing whatever was there.
#
#------------------------------------------------------------------------------
# THE HALF-CONFIGURED STATE
#
# The kernel driver streams the second-stage bitstream into the ICAP. That
# stream is one long write -- no transaction around it, no rollback. If
# whatever is feeding it stops part-way, the region is left holding part of a
# bitstream. Nothing on this side repairs that: the card stays that way until
# `xbutil reset` (whose own help text calls it a "Hot reset") or a reboot.
# This script runs neither, and says so rather than implying it can recover.
#
# So it does not pretend to make the write atomic. It does the two things that
# are actually available:
#
#   1. It refuses to start unless the conditions for a first-try success hold.
#   2. It detaches the loader from the shell that started it.
#
# (2) is not hypothetical. ssh to this host has been reset mid-session, twice
# in one afternoon, with no warning, under the load of other people's Vivado
# runs. `ssh host xbutil program ...` shares that fate and dies at whatever
# byte the stream had reached -- which is exactly the failure this is about.
# `setsid` puts the loader in its own session with all three std fds pointed
# away from the terminal, so nothing this shell does can reach it.
#
#------------------------------------------------------------------------------
# THE GATES
#
# All of them closed: an answer that cannot be read is a failure, not a pass.
# Each is a thing that, if wrong, turns a two-second load into a card that has
# to be reset.
#
#   1. The file parses. xclbinutil reads it end to end. An .xclbin truncated
#      by a full disk or an interrupted copy is caught here, before the device
#      is opened, instead of being streamed until it runs out.
#   2. The interface UUID is the shell's. The .xclbin and the live device each
#      carry one, and they have to be the same string. The driver checks this
#      itself; this script does not rely on where in the driver's sequence
#      that check happens, so it compares the two itself first.
#   3. The platform is the one this was built for.
#   4. Nobody is using the device. kdsstat's contexts, outstanding execs and
#      client pid all have to be empty.
#   5. Nobody else is programming. A second `xbutil program` racing this one
#      is the one way to corrupt a load that would otherwise have been fine.
#   6. Temperature is under the stop thresholds.
#
# Gate 5 reads /proc, so if /proc is mounted with hidepid it sees only this
# user's processes; kdsstat's client pid is the backstop for the rest.
#
# WHAT IT DOES NOT DO
#
# No reset, no reconfigure, no repair, no flash. Not `xbmgmt` in any form, not
# `xbutil reset`, and not `xbutil validate` -- that last one loads the
# platform's own validate xclbins, which would replace whatever is on the card
# just as surely as this script does, only without saying so.
#==============================================================================

set -u

XRT=${XRT_BIN:-/opt/xilinx/xrt/bin}
BDF=${BDF:-0000:01:00.1}
SYS=/sys/bus/pci/devices/$BDF

# The shell this build targets, and the interface UUID that shell and the
# .xclbin both carry. If the live UUID is not this string, the platform was
# swapped underneath the build and nothing downstream of here is valid.
EXPECT_VBNV=xilinx_u50_gen3x16_xdma_5_202210_1
EXPECT_IINTF=16e2362f82d2feab35529da27134b76d

# The stop thresholds. Same numbers as the standing thermal protocol: do not
# add load above these, and stop if a running load crosses them.
TMAX_FPGA=85
TMAX_HBM=80

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

say() { printf '%s\n' "$*"; }
die() { printf 'program.sh: %s\n' "$*" >&2; exit 2; }
field() { sed -n "s/^[[:space:]]*$1[[:space:]]*:[[:space:]]*//p" | head -1; }

FAIL=0
ok()  { printf '  [ ok ] %s\n' "$*"; }
bad() { printf '  [FAIL] %s\n' "$*"; FAIL=1; }

#------------------------------------------------------------------------------
# Gate 1-3: the file.
#------------------------------------------------------------------------------
check_xclbin() {
    say "the .xclbin"
    [ -f "$XCLBIN" ] || { bad "no such file: $XCLBIN"; return; }
    say "  file   $XCLBIN"
    say "  size   $(stat -c%s "$XCLBIN") bytes"
    say "  sha256 $(sha256sum "$XCLBIN" | awk '{print $1}')"

    if ! xclbinutil --info --input "$XCLBIN" > "$WORK/info" 2> "$WORK/err"; then
        bad "xclbinutil cannot read it -- truncated, or not an .xclbin"
        sed 's/^/         /' "$WORK/err" | head -5
        return
    fi
    ok "parses as an .xclbin"

    got_vbnv=$(field 'Platform VBNV' < "$WORK/info" | tr -d '[:space:]')
    if [ "$got_vbnv" = "$EXPECT_VBNV" ]; then
        ok "platform VBNV $got_vbnv"
    else
        bad "platform VBNV is '${got_vbnv:-<unreadable>}', expected $EXPECT_VBNV"
    fi

    got_iintf=$(field 'UUID (IINTF)' < "$WORK/info" | tr -d '[:space:]')
    if [ "$got_iintf" = "$EXPECT_IINTF" ]; then
        ok "interface UUID matches the shell contract"
    else
        bad "interface UUID is '${got_iintf:-<unreadable>}', expected $EXPECT_IINTF"
    fi

    say "  xclbin UUID $(field 'UUID (xclbin)' < "$WORK/info" | tr -d '[:space:]')"
}

#------------------------------------------------------------------------------
# Gate 2 (live side) and 4: the device.
#------------------------------------------------------------------------------
check_device() {
    say
    say "the device"
    [ -d "$SYS" ] || { bad "no $SYS -- device not present"; return; }
    ok "present at $BDF"

    live=$(tr -d '[:space:]' < "$SYS/interface_uuids" 2>/dev/null)
    if [ "$live" = "$EXPECT_IINTF" ]; then
        ok "live shell interface UUID matches"
    else
        bad "live shell interface UUID is '${live:-<unreadable>}', expected $EXPECT_IINTF"
    fi

    ctx=$(field contexts          < "$SYS/kdsstat" 2>/dev/null)
    outs=$(field 'outstanding execs' < "$SYS/kdsstat" 2>/dev/null)
    cpid=$(field 'client pid'     < "$SYS/kdsstat" 2>/dev/null)
    if [ "$ctx" = "0" ] && [ "$outs" = "0" ] && [ -z "$cpid" ]; then
        ok "idle: contexts 0, outstanding 0, no client"
    else
        bad "in use: contexts=${ctx:-<unreadable>} outstanding=${outs:-<unreadable>} client=${cpid:-<unreadable>}"
    fi

    say "  loaded now $(tr -d '[:space:]' < "$SYS/xclbinuuid" 2>/dev/null || echo '<none>')"
}

#------------------------------------------------------------------------------
# Gate 5: nobody else is programming.
#
# Matched on the whole command line, machine-wide. pgrep never matches itself,
# and nothing in this script's own command line contains "xbutil", so it
# cannot find us -- which is the failure mode that a pkill -f style check
# walks into, by matching its own caller and killing it.
#------------------------------------------------------------------------------
check_no_other_programmer() {
    say
    say "other loaders"
    hits=$(pgrep -a -f '(xbutil|xbmgmt).*program' 2>/dev/null || true)
    if [ -z "$hits" ]; then
        ok "no other program/load running"
    else
        bad "another load is running -- a second ICAP stream would corrupt this one:"
        printf '%s\n' "$hits" | sed 's/^/         /'
    fi
}

#------------------------------------------------------------------------------
# Gate 6: temperature.
#------------------------------------------------------------------------------
check_thermal() {
    say
    say "thermal"
    if ! "$XRT/xbutil" examine -d "$BDF" -r thermal --batch > "$WORK/thermal" 2>&1; then
        bad "thermal unreadable"
        return
    fi
    sed -n '/Thermals/,$p' "$WORK/thermal" | sed 's/^/  /'

    fpga=$(awk '/^[[:space:]]*FPGA[[:space:]]+:/     {print $(NF-1); exit}' "$WORK/thermal")
    hbm=$( awk '/^[[:space:]]*FPGA HBM[[:space:]]+:/{print $(NF-1); exit}' "$WORK/thermal")

    if [ -z "$fpga" ] || [ -z "$hbm" ]; then
        bad "could not read FPGA/HBM temperature out of the report"
        return
    fi
    [ "$fpga" -lt "$TMAX_FPGA" ] && ok "FPGA ${fpga} C, under $TMAX_FPGA" \
                                 || bad "FPGA ${fpga} C is at or above $TMAX_FPGA"
    [ "$hbm"  -lt "$TMAX_HBM"  ] && ok "HBM ${hbm} C, under $TMAX_HBM" \
                                 || bad "HBM ${hbm} C is at or above $TMAX_HBM"
}

#------------------------------------------------------------------------------
# The load itself.
#
# The inner runner is written to the build directory rather than typed inline
# because setsid needs something to exec, and because it leaves behind the one
# line that matters -- the loader's exit status -- in the same log as
# everything else. The log is the record; nothing about the outcome depends on
# this shell still being alive to read it.
#------------------------------------------------------------------------------
do_load() {
    OUT=$(cd "$(dirname "$XCLBIN")" && pwd)
    LOG=$OUT/program.log
    RUN=$OUT/program.run

    {
        say "=== PROGRAM START $(date) ==="
        say "BDF=$BDF"
        say "XCLBIN=$XCLBIN"
        say "SHA256=$(sha256sum "$XCLBIN" | awk '{print $1}')"
        say "REPLACING=$(tr -d '[:space:]' < "$SYS/xclbinuuid" 2>/dev/null || echo '<none>')"
        say "THERMAL=$(tr '\n' ' ' < "$WORK/thermal" | tr -s ' ')"
        say "---"
    } > "$LOG"

    cat > "$RUN" <<EOF
#!/bin/bash
# Written by program.sh. Runs detached in its own session; the shell that
# started program.sh cannot reach it, which is the point.
"$XRT/xbutil" program -d "$BDF" -u "$XCLBIN" >> "$LOG" 2>&1
echo "EXIT=\$? $(date)" >> "$LOG"
EOF
    chmod +x "$RUN"

    setsid "$RUN" < /dev/null > /dev/null 2>&1 &
    say
    say "loader detached (setsid, own session, fds off the terminal)"
    say "  log  $LOG"
    say "  it will survive this shell dying, which is not a hypothetical here."

    say "waiting for it to finish (up to 180s)..."
    for _ in $(seq 1 180); do
        grep -q '^EXIT=' "$LOG" 2>/dev/null && break
        sleep 1
    done

    say
    say "--- $LOG ---"
    sed 's/^/  /' "$LOG"
    say "------------"

    if grep -q '^EXIT=0' "$LOG" 2>/dev/null; then
        after=$(tr -d '[:space:]' < "$SYS/xclbinuuid" 2>/dev/null || echo '<none>')
        say
        say "program.sh: OK -- device now reports xclbin $after"
        if [ "$after" != "$(field 'UUID (xclbin)' < "$WORK/info" | tr -d '[:space:]')" ]; then
            say "         but that is not the UUID of the file that was loaded."
            say "         That should not happen, and it means the card is not"
            say "         running what this script just wrote."
            exit 4
        fi
        say "         loading is done; running a kernel on it is a separate step."
        exit 0
    fi

    say
    say "program.sh: the load did NOT report success. Before touching it again:"
    say "  - read $LOG"
    say "  - dmesg | grep -i icap   (the driver logs the download there)"
    say "  - if the region is half-configured the card needs \`xbutil reset -t"
    say "    user\` (a hot reset) or a reboot. This script does not run either,"
    say "    and cannot: one is a hot reset and the other needs root. Stop and"
    say "    get a decision rather than retrying into a worse state."
    exit 3
}

#------------------------------------------------------------------------------
do_status() {
    say "device   $BDF"
    say "loaded   $(tr -d '[:space:]' < "$SYS/xclbinuuid" 2>/dev/null || echo '<none>')"
    say
    say "kdsstat"
    sed 's/^/  /' "$SYS/kdsstat" 2>/dev/null
    say
    say "thermal"
    "$XRT/xbutil" examine -d "$BDF" -r thermal --batch 2>&1 \
        | sed -n '/Thermals/,$p' | sed 's/^/  /'
    if [ -n "${1:-}" ] && [ -f "$1" ]; then
        say
        say "last program.log at $1"
        tail -25 "$1" | sed 's/^/  /'
    fi
}

#------------------------------------------------------------------------------
case "${1:-}" in
    --check|--go)
        MODE=$1
        XCLBIN=${2:-}
        [ -n "$XCLBIN" ] || die "usage: program.sh $MODE <xclbin>"
        command -v xclbinutil >/dev/null \
            || die "xclbinutil not on PATH -- source Vitis settings64.sh first"
        check_xclbin
        check_device
        check_no_other_programmer
        check_thermal
        say
        if [ $FAIL -ne 0 ]; then
            say "program.sh: gates failed. Nothing was loaded."
            exit 1
        fi
        if [ "$MODE" = "--check" ]; then
            say "program.sh: all gates pass. Nothing was loaded (--check)."
            say "         to load:  program.sh --go $XCLBIN"
            exit 0
        fi
        do_load
        ;;
    --status)
        do_status "${2:-}"
        ;;
    *)
        sed -n '3,6p' "$0"
        exit 2
        ;;
esac
