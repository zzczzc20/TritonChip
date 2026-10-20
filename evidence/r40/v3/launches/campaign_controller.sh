#!/bin/bash
# Wait for the functional queue to release the card, then run the v3 campaign.
set -u
F=~/tnca_v3_functional/state.json
while true; do
  st=$(python3 -c "import json;print(json.load(open('$F'))['status'])")
  case "$st" in complete-review-required|failed-preserved) break;; esac
  sleep 30
done
echo "$(date -u +%FT%TZ) functional queue terminal: $st" >> ~/tnca_v3_campaign/supervisor.log
for p in $(pgrep -f 'tnca_src_r40_v3/' ); do echo "still live: $p" >> ~/tnca_v3_campaign/supervisor.log; done
while pgrep -f 'tnca_src_r40_v3/' >/dev/null; do sleep 10; done
cd ~/tnca_src_r40_v3x
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tnca/r40_conditional_6bd07b7_extensions MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec ~/tc_env/torchenv/bin/python -u tools/r40/v3_campaign.py run --out ~/tnca_v3_campaign/run \
  --xclbin ~/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin \
  --operating-clock-evidence ~/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json
