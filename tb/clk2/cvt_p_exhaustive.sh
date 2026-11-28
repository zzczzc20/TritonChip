#!/bin/bash
# exhaustive tnca_cvt_int_p (REG 0) against tnca_cvt_int: every 32-bit word for fn 0 and 1 (rm 0) and fn 2 and 3 with rm 0..3 (GB10, nice 19)
B=~/r42/clk2_out/cvtp/obj_cvt/Vcvt_int_miter; O=~/r42/clk2_out/cvtp/exh.log
: > $O
for pair in "0 0" "1 0" "2 0" "2 1" "2 2" "2 3" "3 0" "3 1" "3 2" "3 3"; do echo "fn rm $pair" >> $O; nice -n 19 $B $pair >> $O 2>&1; done
echo EXH_DONE >> $O
