#!/bin/bash
# run_all.sh OUTDIR -- the R42 area verification set on GB10 (nice 19, one process at a time), from the worktree mirror:
#   1. stage miters (tb/area/fast/build.sh + fast_miter): align, norm, round stages and the FMA chain against the generated stages,
#      the round stage also on arbitrary structs
#   2. cycle-level FMA miters (build_cyc.sh) at latency 2 and 3 for the cut/shift/align variants
#   3. MAC lane miters (build_mac.sh), all modes, struct level and result level, BE 5 and 13, cuts 011101, 111101, 111111
#   4. mutation sensitivity (mutation_check.sh)
# Each line of the log is a command, its exit status and its last lines.
set -u
OUT=${1:?outdir}; N=${N:-1000000}
cd ~/r42/area
mkdir -p $OUT; LOG=$OUT/verify.log; : > $LOG
step() { echo "### $*" | tee -a $LOG; }
step "stage miters N=$N x 9 FMA classes + 4 round classes"
for sh in 0 1; do
  SHIFT=$sh bash tb/area/fast/build.sh $OUT/stage_s$sh >> $LOG 2>&1
  nice -n 19 $OUT/stage_s$sh/obj_all/Vmiter_all $N 11 2>&1 | tail -4 >> $LOG
done
step "cycle-level FMA miters N=$N x 9 classes"
for cfg in "1010 1 0 2" "1010 1 1 2" "1010 0 1 2" "1110 1 0 3" "1110 1 1 3" "1110 0 0 3"; do
  set -- $cfg; d=$OUT/cyc_$1_$2_$3
  bash tb/area/fast/build_cyc.sh $d $1 $2 $3 $4 >> $LOG 2>&1
  echo "CUTS=$1 SHIFT=$2 XLS_ALIGN=$3 LAT=$4: $(nice -n 19 $d/obj_cyc/Vfma_cyc_miter $N 13 $4 2>&1 | tail -2 | tr '\n' ' ')" | tee -a $LOG
done
step "MAC lane miters N=$N x 3 classes x 3 modes"
for cfg in "5 011101 5" "13 011101 5" "13 111101 6" "5 111101 6" "13 111111 7"; do
  set -- $cfg; d=$OUT/mac_$1_$2
  bash tb/area/fast/build_mac.sh $d $1 $2 >> $LOG 2>&1
  echo "BE=$1 CUTS=$2: $(nice -n 19 $d/obj_mac/Vmac_miter $N 17 $3 2>&1 | tail -2 | tr '\n' ' ')" | tee -a $LOG
done
step "mutation sensitivity"
N=$N bash tools/r42/area/verify/mutation_check.sh 2>&1 | cut -c1-200 | tee -a $LOG
echo "VERIFY_DONE" | tee -a $LOG
