#!/bin/bash
#==============================================================================
# vivado_slot.sh -- run one Vivado job of the R41 Stage 2 timing work on the
# shared FPGA host, holding one of two slots.
#
#   vivado_slot.sh [--min-gb N] LOG -- COMMAND ...
#
# At most two of these jobs run at once: flock on ~/.r41s2_vivado_slot.{0,1}.
# A job starts only while `free -g` shows at least N GB available (default
# 16, which covers a unit OOC of <= ~6 GB; a kernel synthesis takes about 9 GB
# and needs --min-gb 20). Other users' Vivado jobs and the Stage 1 board queue
# share the host. This never touches the U280 card. Exit status is the
# command's; LOG receives its output and start/end lines.
#==============================================================================
set -u
min=16
if [ "${1:-}" = "--min-gb" ]; then min=$2; shift 2; fi
log=${1:?log}; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: vivado_slot.sh [--min-gb N] LOG -- COMMAND ..." >&2; exit 2; }
mkdir -p "$(dirname "$log")"
while true; do
    for s in 0 1; do
        exec {fd}>"$HOME/.r41s2_vivado_slot.$s"
        if flock -n $fd; then
            while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$min" ]; do sleep 60; done
            echo "$(date -u +%FT%TZ) slot $s start (avail $(free -g | awk '/^Mem:/{print $7}') GB): $*" >> "$log"
            set +u; source /tools/Xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1; set -u
            nice -n 10 "$@" >> "$log" 2>&1
            rc=$?
            echo "$(date -u +%FT%TZ) slot $s end exit $rc" >> "$log"
            exec {fd}>&-
            exit $rc
        fi
        exec {fd}>&-
    done
    sleep 30
done
