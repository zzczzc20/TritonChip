#!/bin/bash
cd ~/r42/lineelig; source ~/tc_env/env.sh; export PYTHONPATH=$PWD/tools/asm
for c in 00 10 11; do
  nice -n 19 iverilog -g2012 -Irtl/common -y rtl/common -y rtl/core -y rtl/gen -y tb/common -s tb_core_hybrid -Ptb_core_hybrid.MEM_LATENCY=6 -Ptb_core_hybrid.RANDOM_STALL=1 -Ptb_core_hybrid.MEM_REORDER=1 -Ptb_core_hybrid.LSU_TAG_W=7 -Ptb_core_hybrid.TENSOR_LINE=${c:0:1} -Ptb_core_hybrid.TENSOR_LINE_HALF=${c:1:1} -Ptb_core_hybrid.HAS_TENSOR_XPOSE=2 -o ~/r42/lineelig_out/ch2_$c.vvp tb/core/tb_core_hybrid.v || exit 1
done
for pa in "t12_tensor:" "t30_xpose:" "t33_xpose8:" "t33_tdesc_load:" "t34_tdesc_store:" "t17_descfault:+expectfault=04" "t35_tdesc_fpad2:+expectfault=04" "t39_freeze_drop:" ; do
  P=${pa%%:*}; A=${pa#*:}
  python3 - <<PY
import sys
sys.path.insert(0,"tools/asm")
from tnca_asm import Assembler,build_image
from pathlib import Path
src=Path("tb/core/progs/$P.s")
Path("$HOME/r42/lineelig_out/$P.hex").write_text("".join(f"{w:016x}\n" for w in build_image(Assembler().assemble(src.read_text()),size_words=65536)))
PY
  for c in 00 10 11; do r=$(nice -n 19 vvp ~/r42/lineelig_out/ch2_$c.vvp +hex=$HOME/r42/lineelig_out/$P.hex +maxcycles=1000000 $A 2>&1 | grep -E "^PASS|FAIL|FATAL|finish" | tr '\n' ' '); echo "$P line=${c:0:1} half=${c:1:1} $r"; done
done
