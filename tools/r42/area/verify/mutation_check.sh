#!/bin/bash
# Verilator mutation sensitivity of the area stage miters (GB10, from ~/r42/area): one single-point change per mutant
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
cd ~/r42/area
R=rtl; W=${W:-~/r42/area_out/mutvl}; rm -rf $W; mkdir -p $W
SRC="$R/core/tnca_lzc81.v $R/gen/tnca_fma_align_xls.v $R/gen/tnca_fmad_norm_xls.v $R/gen/tnca_fma_round_xls.v"
run() {
  mkdir -p $W/$1; cp $R/core/tnca_fmax_norm.v $R/core/tnca_fmax_add.v $R/core/tnca_fmax_sel.v $R/core/tnca_fmax_round.v $R/core/tnca_fmax_align.v $R/core/tnca_fmax_alignp.v $R/core/tnca_fmax_mul.v $W/$1/
  sed -i "$3" $W/$1/$2
  if cmp -s $W/$1/$2 $R/core/$2; then echo "$1: SED DID NOT APPLY"; return; fi
  (cd $W/$1 && nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module miter_all -Mdir obj -GADD=0 -GSHIFT=${SHIFT:-1} \
     ~/r42/area/tb/area/fast/miter_all.v tnca_fmax_norm.v tnca_fmax_add.v tnca_fmax_sel.v tnca_fmax_round.v tnca_fmax_align.v tnca_fmax_alignp.v tnca_fmax_mul.v $(for f in $SRC; do echo ~/r42/area/$f; done) ~/r42/area/tb/area/fast/fast_miter.cpp -CFLAGS -O2 -j 4 > b.log 2>&1) || { echo "$1: BUILD FAILED"; return; }
  r=$($W/$1/obj/Vmiter_all ${N:-1000000} 5 2>&1 | grep -E "^fast_miter|MISMATCH" | head -2 | tr '\n' ' ')
  echo "$1: $r"
}
run m_sticky_borrow  tnca_fmax_add.v  's/- {48.b0, bor}/- 49'"'"'b0/'
run m_winner         tnca_fmax_sel.v  's/h_neg     ? {dsub_hi, small_lo}/1'"'"'b0     ? {dsub_hi, small_lo}/'
run m_lz8            tnca_fmax_sel.v  's/c81 + 7.d8/c81 + 7'"'"'d7/'
run m_sub_threshold  tnca_fmax_sel.v  's/(ec + 12.sd136)/(ec + 12'"'"'sd135)/'
run m_ressign        tnca_fmax_sel.v  's/small_wins ? s_small : s_big/s_big/'
run m_sticky_round   tnca_fmax_round.v 's/wire        sticky = rest | sticky_al;/wire sticky = rest;/'
run m_tie            tnca_fmax_round.v 's/(sticky | sig\[0\])/(sticky)/'
run m_ones           tnca_fmax_round.v 's/(sig_ones ? 24.h800000 : sig_inc)/sig_inc/'
run m_overflow       tnca_fmax_round.v 's/eb0 >= 12.sd255/eb0 >= 12'"'"'sd256/'
run m_ftz            tnca_fmax_round.v 's/(ftz ? {res_sign, 31.h0}/(1'"'"'b0 ? {res_sign, 31'"'"'h0}/'
run m_mask           tnca_fmax_round.v 's/((90.d1 << sh_mag) - 90.d1)/((90'"'"'d1 << sh_mag) - 90'"'"'d2)/'
run m_leftcut        tnca_fmax_round.v 's/(sh_mag >= 12.d89)/(sh_mag >= 12'"'"'d60)/'
run m_shiftlim       tnca_fmax_round.v 's/(sh_mag >= 12.d90) ? 90.b0/(sh_mag >= 12'"'"'d91) ? 90'"'"'b0/'
run m_eb0_ovf        tnca_fmax_round.v 's/(eb0 >= 12.sd254)/(eb0 >= 12'"'"'sd253)/'
run m_carry          tnca_fmax_round.v 's/wire carry_out  = round_up \& sig_ones;/wire carry_out = round_up;/'
run m_resultzero     tnca_fmax_round.v 's/((sig == 24.d0) \& ~round_up)/((sig == 24'"'"'d0))/'
run a_sticky64      tnca_fmax_alignp.v 's/shamt\[6\] \& (|v0\[63:0\])/shamt[6] \& (|v0[62:0])/'
run a_clamp         tnca_fmax_alignp.v 's/(dabs > 12.sd80) ? 8.d80/(dabs > 12'"'"'sd79) ? 8'"'"'d79/'
run a_shiftc        tnca_fmax_alignp.v 's/c_zero ? 1.b1 : (p_zero ? 1.b0 : ~d_neg)/~d_neg/'
run a_subnormal_c   tnca_fmax_alignp.v 's/{fc, 1.b0}/{1'"'"'b0, fc}/'
run a_pdbl          tnca_fmax_mul.v 's/{1.b0, a_e0} + {1.b0, b_e0}/{1'"'"'b0, a_e0}/'
run a_infinf        tnca_fmax_alignp.v 's/(sp != sc)/(sp == sc)/'
run a_xp            tnca_fmax_alignp.v 's/kab - 12.sd254/kab - 12'"'"'sd253/'
run a_ftz_c         tnca_fmax_alignp.v 's/(c_e0 \& (c_fz | ftz)) | mul_only/(c_e0 \& c_fz) | mul_only/'
run a_ftz_mul       tnca_fmax_mul.v 's/a_e0 \& (a_fz | ftz)/a_e0 \& a_fz/'
run a_infsign       tnca_fmax_alignp.v 's/p_inf ? sp : sc/sp/'
run a_sticky_l1     tnca_fmax_alignp.v 's/l1 = shamt\[1\] \& (|v5\[1:0\])/l1 = shamt[1] \& (|v5[0])/'
