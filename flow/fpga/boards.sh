# The cards this flow builds for, sourced by link.sh and program.sh and
# passed to package_xo.tcl. TNCA_BOARD selects one; u50 is the default, so
# every command written before this file existed means what it meant.
#
# Each value is read off the card or its platform, not remembered:
#   part, platform   `platforminfo -p <platform>.xpfm` (Board Part, Name)
#   interface UUID   /sys/bus/pci/devices/<bdf>/interface_uuids on the host
#   bdf              `xbutil examine` (the user function, .1)
#
# The device RTL, the kernel and its name (tnca_u50, from
# rtl/board/tnca_u50_xo.v) are the same for every card: the kernel wraps the
# device for an XRT shell, and the U280's gen3x16 shell presents the same
# AXI-Lite control port and kernel clock 0 as the U50's.

case "${TNCA_BOARD:-u50}" in
u50)
    BOARD_PART=xcu50-fsvh2104-2-e
    BOARD_PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
    BOARD_IINTF=16e2362f82d2feab35529da27134b76d
    BOARD_BDF=0000:01:00.1
    BOARD_VITIS=/home/x4/Software/Xilinx/Vitis/2022.2
    BOARD_VPP_OPTS=
    ;;
u280)
    # acf3019.ece.ust.hk, 2026-10-07: shell xilinx_u280_gen3x16_xdma_base_1,
    # platform 202211.1, XRT 2.16 (2023.2), Vitis 2023.2.
    BOARD_PART=xcu280-fsvh2892-2L-e
    BOARD_PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
    BOARD_IINTF=fb2b2c5a19ed63593fea95f51fbc8eb9
    BOARD_BDF=0000:04:00.1
    BOARD_VITIS=/tools/Xilinx/Vitis/2023.2
    # The whole kernel in SLR1. Left to the placer, the first U280 link
    # (2026-10-07) split it over SLR0 and SLR1 and missed 150 MHz by
    # 1.919 ns on 32 endpoints, all one path in the AXI-Lite adapter
    # (w_have -> shadow_lo[*]/CE: one LUT, 8.2 ns of route, an SLR0->SLR1
    # crossing). SLR0 also holds the shell's fixed logic (~108k LUTs);
    # SLR1 was 19 % used. v++ adds the register slices for the control
    # port's crossing from the shell.
    if [ "${TNCA_KERNEL_VARIANT:-none}" = multicore-hbm ]; then
        # Per-tile and link placement is supplied by the board Tcl hook.
        BOARD_VPP_OPTS=
    else
        BOARD_VPP_OPTS="--connectivity.slr tnca_u50_1:SLR1"
    fi
    ;;
*)
    echo "boards.sh: unknown TNCA_BOARD '${TNCA_BOARD}' (u50, u280)" >&2
    exit 2
    ;;
esac
BOARD=${TNCA_BOARD:-u50}
