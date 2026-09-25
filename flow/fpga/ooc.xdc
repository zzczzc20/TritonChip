#==============================================================================
# Out-of-context constraints for the tnca_u50 kernel.
#
# ONE CLOCK, AND ONE LINE.
#
# The RTL Kernel Wizard's own file is
#
#     create_clock -period 4.000 [get_ports ap_clk]
#
# and this is that line with the period the platform actually runs. The U50
# platform's default kernel clock, the one a CU is attached to when nothing
# says otherwise, is 300 MHz -- so 3.333 ns, not the wizard's 4.000, which is
# a placeholder rather than a frequency.
#
# This file governs out-of-context synthesis and nothing else. At link time the
# platform supplies the real period, and the answer this number produces is an
# estimate -- see flow/fpga/README.md for the measured one.
#
# WHY THERE IS NO SECOND create_clock FOR clk_dev
#
# The device clock comes out of BUFGCE_DIV in rtl/board/tnca_u50_xo.v, and
# Vivado propagates the divide ratio through that primitive on its own. A
# create_clock on clk_dev would not be a stricter constraint, it would be a
# second, contradictory claim about a clock that already has a defined source,
# and the CDC it would then be checking against would be checked against a
# period that does not exist.
#
# The crossing itself is handled structurally, not by timing: it is a toggle
# handshake with synchronisers on both sides, and tb/fpga/tb_tnca_axi_lite.v
# sweeps eleven clock ratios over it. No false path or max_delay is declared
# for it here, because a false path on a handshake that is genuinely
# asynchronous is how a metastability bug gets a clean timing report.
#==============================================================================

create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
