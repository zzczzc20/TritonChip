#!/bin/bash
# fe1b delivery queue: v3 campaign, then (recollect controller waits on it), then the R40-07.policy A/B.
set -u
cd ~/tnca_src_r40_v3x2
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tnca/r40_conditional_6bd07b7_extensions MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/tc_env/torchenv/bin/python
X=~/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin
C=~/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json
L=~/tnca_v3_campaign_x2/queue.log
# the card must be free of other board work
while ps -eo args | grep -E 'archbench.run|v3_campaign.py run|v3_supplement.py run|v3_policy_board.py run' | grep -v grep >/dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) campaign start" >> $L
$PY -u tools/r40/v3_campaign.py run --out ~/tnca_v3_campaign_x2/run --xclbin $X --operating-clock-evidence $C >> $L 2>&1
echo "$(date -u +%FT%TZ) campaign end" >> $L
until python3 -c "import json,sys;sys.exit(0 if json.load(open('/home/zzhouch/tnca_v3_recollect_x2/state.json'))['status'] in ('complete-review-required','failed-preserved') else 1)" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) policy start" >> $L
$PY -u tools/r40/v3_policy_board.py run --out ~/tnca_v3_policy_x2 --xclbin $X --operating-clock-evidence $C >> $L 2>&1
$PY tools/r40/v3_policy_board.py judge --out ~/tnca_v3_policy_x2 >> $L 2>&1
echo "$(date -u +%FT%TZ) QUEUE-X2-DONE" >> $L
