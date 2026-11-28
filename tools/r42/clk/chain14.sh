#!/bin/bash
# chain14.sh (GB10): the standard RTL gates at CLKFIX 0 on tree ~/r42/clkb (head with bits 11-13): cycle-identical to the base rows of chain6/gate_g2c4_base
source ~/tc_env/env.sh
O=~/r42/clk_out
while ! grep -q CHAIN11_DONE $O/chain11.log 2>/dev/null; do sleep 30; done
cd ~/r42/clkb
nice -n 19 make BUILD=sim/build_g_run2 core-run2 > $O/gate14_run2.log 2>&1
grep -E "^  [a-z0-9_]+ \(.*(PASS|FAIL)" $O/gate14_run2.log > $O/gate14_run2.cycles
nice -n 19 make BUILD=sim/build_g_g2c4 core-g2c4 > $O/gate14_g2c4.log 2>&1
nice -n 19 make BUILD=sim/build_g_mmadp core-mmadp > $O/gate14_mmadp.log 2>&1
nice -n 19 make BUILD=sim/build_g_unit unit > $O/gate14_unit.log 2>&1
echo CHAIN14_DONE >> $O/gate14_unit.log
