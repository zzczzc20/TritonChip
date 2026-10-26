#!/bin/bash
# After the bc5a re-collection releases the card, run the supplementary diagnostics.
set -u
until python3 -c "import json,sys;sys.exit(0 if json.load(open('/home/zzhouch/tnca_v3_recollect/state.json'))['status'] in ('complete-review-required','failed-preserved') else 1)" 2>/dev/null; do sleep 60; done
while pgrep -f 'tnca_src_r40_v3x/tools/r40/(v3_campaign|conditional_delivery|frozen_residency|native_model)' >/dev/null || pgrep -f 'archbench.run' >/dev/null; do sleep 30; done
cd ~/tnca_src_r40_v3x
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tnca/r40_conditional_6bd07b7_extensions MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in ~/tnca_v3_supplement/*/; do
  ~/tc_env/torchenv/bin/python -u tools/r40/v3_supplement.py run --out "$d" \
    --xclbin ~/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin \
    --operating-clock-evidence ~/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json >> ~/tnca_v3_supplement/run.log 2>&1
  ~/tc_env/torchenv/bin/python tools/r40/v3_supplement.py judge --out "$d" >> ~/tnca_v3_supplement/run.log 2>&1
done
echo done >> ~/tnca_v3_supplement/run.log
