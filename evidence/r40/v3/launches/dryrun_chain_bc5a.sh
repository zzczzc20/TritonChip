#!/bin/bash
# bc5a dry runs while the product fixes are prepared: the experiments that never
# ran on the card. Results are early signals; the delivery identity will change.
set -u
cd ~/tnca_src_r40_v3x
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tnca/r40_conditional_6bd07b7_extensions MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/tc_env/torchenv/bin/python
X=~/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin
C=~/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json
L=~/tnca_v3_dryrun/run.log
$PY -u tools/r40/v3_supplement.py run --out ~/tnca_v3_dryrun/pipeline_roles2 --xclbin $X --operating-clock-evidence $C >> $L 2>&1
$PY tools/r40/v3_supplement.py judge --out ~/tnca_v3_dryrun/pipeline_roles2 >> $L 2>&1
$PY -u tools/r40/v3_campaign.py run --out ~/tnca_v3_campaign/run --xclbin $X --operating-clock-evidence $C --only R40-08.prefill >> $L 2>&1
$PY -u tools/r40/v3_campaign.py run --out ~/tnca_v3_campaign/run --xclbin $X --operating-clock-evidence $C --only R40-05.three-cores R40-05.two-cores R40-09.model R40-01.device-time >> $L 2>&1
echo DRYRUN-DONE >> $L
