#!/bin/bash
# finalize.sh : fetch everything, write the evidence tables (rtl_3c_tables.txt, rtl_1c_tables.txt, model_prediction.txt) into the worktree
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
W=/home/ubuntu/r42-wt/img; E=$W/evidence/r42/img
ARMS=A,L,LN,M,MN,LM,LMN,F,LF,LMF,LMFN timeout 100 $S/img/refresh.sh > $E/rtl_3c_tables.txt 2>&1
{ echo "# rtl-3c: every arm against A, tools/r42/img_table.py (generated)"; echo "# names: L, M, LM, F, LF, LMF carry the profile name of their own item set; LN, MN, LMN carry A's profile name (core-vrf-sync-v5-x8-g2c4-dp) and LMFN the F arm's (core-r42-fp32): same image, same software, only the name differs"; cat $E/rtl_3c_tables.txt; } > $E/rtl_3c_tables.tmp && mv $E/rtl_3c_tables.tmp $E/rtl_3c_tables.txt
# rtl-1c records
mkdir -p $E/rtl_1c; rsync -a -e $S/gb_rsh gb10:r42/img_out/c1/ $E/rtl_1c/ --exclude '*_logs' > /dev/null 2>&1
cd $E/rtl_1c
{ echo "# rtl-1c T17 (21 cheap tiny points): origin/R42 tree, merged tree at default parameters, combined profile core-r42-img"; python3 $W/../../../tmp/dummy 2>/dev/null; } > /dev/null
python3 $S/img/cmp1c.py t17_origin.jsonl t17_merged.jsonl t17_lmf.jsonl > $E/rtl_1c_tables.txt 2>&1
for pair in "t17_origin.jsonl t17_m.jsonl" "t17_origin.jsonl t17_f.jsonl"; do [ -f ${pair#* } ] && { echo; python3 $S/img/cmp1c.py $pair >> $E/rtl_1c_tables.txt 2>&1; }; done
[ -f models_lmf_a.jsonl ] && { echo; echo "# the model points of the combined profile against origin/R42 (tiny tier of tiny_origin.jsonl)"; python3 $S/img/cmp1c.py tiny_origin.jsonl models_lmf_a.jsonl | grep -E "conv2d|mlp|resnet|geomean|not pass" >> $E/rtl_1c_tables.txt 2>&1; }
# predictions
cd $W; timeout 100 $S/img/final_predict.sh > $E/model_prediction.txt 2>&1
echo written
