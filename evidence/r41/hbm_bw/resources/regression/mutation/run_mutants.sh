#!/bin/bash
# R41 G2: bench sensitivity. Each mutant breaks one candidate on purpose; the
# lockstep bench must report it. Run from the repository root:
#   bash evidence/r41/hbm_bw/resources/regression/mutation/run_mutants.sh SCRATCH_DIR
set -u
w=${1:?scratch dir}; mkdir -p "$w"
echo "commit $(git rev-parse HEAD) host $(hostname) $(date -u +%FT%TZ)"
slice() { # name sed-expr mode
  sed "$2" rtl/nuca/tnca_nuca_slice.v > "$w/$1.v"
  if cmp -s rtl/nuca/tnca_nuca_slice.v "$w/$1.v"; then echo "MUTANT $1: NOT APPLIED"; return; fi
  iverilog -g2012 -I rtl/common -y rtl/common -P tb_nuca_shared_read.SHARED_MODE=$3 -o "$w/$1.vvp" \
    tb/nuca/tb_nuca_shared_read.v "$w/$1.v" tb/common/tnca_dram_model.v || return
  echo "MUTANT $1 (mode $3): $(vvp "$w/$1.vvp" +seed=1 +cycles=30000 | grep -E '^(PASS|FAIL)' | tail -1)"
}
outlet() { # name sed-expr
  sed "$2" rtl/fpga/tnca_line_axi.v > "$w/$1.v"
  if cmp -s rtl/fpga/tnca_line_axi.v "$w/$1.v"; then echo "MUTANT $1: NOT APPLIED"; return; fi
  iverilog -g2012 -s tb_line_axi -Ptb_line_axi.DATA_W=256 -Ptb_line_axi.OUT_N=8 -Ptb_line_axi.LINE_LUTRAM=1 \
    -o "$w/$1.vvp" "$w/$1.v" tb/fpga/tb_line_axi.v rtl/common/tnca_line_link.v rtl/common/tnca_fifo.v || return
  echo "MUTANT $1: $(vvp "$w/$1.vvp" 2>&1 | grep -E 'FATAL|LINE AXI PASS' | head -1)"
}
slice maint_index_plus1 's/(est == E_MREAD) ? maint_index :/(est == E_MREAD) ? maint_index + 1 :/' 1
slice service_way_xor1 's/^\( *\)service_set\*WAYS + service_way;   \/\/ E_WBRD, E_WAKE/\1(service_set*WAYS + service_way) ^ 1;/' 1
slice mode2_maint_plus1 's/(est == E_MREAD) ? maint_index :/(est == E_MREAD) ? maint_index + 1 :/' 2
outlet rline_wrong_slot 's/assign r_line\[b\*DATA_W+:DATA_W\]=rbuf\[m_rid\];/assign r_line[b*DATA_W+:DATA_W]=rbuf[m_rid^1];/'
outlet wline_fixed_slot 's/assign w_line=wbuf\[issue_slot\];/assign w_line=wbuf[0];/'
