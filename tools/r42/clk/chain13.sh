#!/bin/bash
# chain13.sh (GB10): gates of CLKFIX bits 11-13 (RSTTREE, AXI_REQREG, IDLE_SRC) on tree ~/r42/clkb: line_axi matrix at REQ_REG 0/1, hybrid core bench at bit 11,
# three-core stand-in arms r11/r12/r13 (3 points) and all-bits f14 (7 points). Waits for chain12 (the bit 9/10 gates) so only one chain computes at a time.
source ~/tc_env/env.sh
O=~/r42/clk_out
while ! grep -q CHAIN12_DONE $O/g4_rev6.log 2>/dev/null; do sleep 30; done
cd ~/r42/clkb
for rr in 0 1; do REQ_REG=$rr nice -n 19 python3 tools/r40/line_axi_tests.py > $O/line_axi_rr$rr.log 2>&1; echo "line_axi REQ_REG=$rr exit $?" >> $O/chain13.log; done
nice -n 19 python3 tools/r42/clk/core_hybrid_clk.py --fixes 0,2048 --jobs 2 --out $O/hyb_rst > $O/hyb_rst.log 2>&1
for arm in r11 r12 r13; do
  bash ~/r42/clkb/tools/r42/clk/clk_mc.sh clkb tools/r42/clk/images/clk_${arm}_sim.tcl core-vrf-sync-v5-x8-g2c4-dp mc3_${arm} ~/r42/clkb/tools/r42/clk/points_mc3.txt 2 > $O/mc3_${arm}.log 2>&1
done
bash ~/r42/clkb/tools/r42/clk/clk_mc.sh clkb tools/r42/clk/images/clk_f14_sim.tcl core-vrf-sync-v5-x8-g2c4-dp mc7_f14 ~/r42/clkb/tools/r42/clk/points_mc7.txt 2 > $O/mc7_f14.log 2>&1
echo CHAIN13_DONE >> $O/chain13.log
