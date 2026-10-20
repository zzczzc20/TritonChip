#!/bin/bash
# Guarded OOC unit-timing queue for the final-image candidate (old A3 source vs d2cc850).
# Retries while the shared host lacks memory; never touches the card.
set -u
NEW=$HOME/tnca_src_r40_final_d2cc850
OLD=$HOME/tnca_src_r40_attempt03_54249f7
OUT=$HOME/tnca_ooc_final_d2cc850/units
source /tools/Xilinx/Vivado/2023.2/settings64.sh
for i in $(seq 1 200); do
  if OOC_MIN_AVAIL_GB=16 bash $NEW/tools/r40/final_image_ooc.sh units $OLD $NEW $OUT > $HOME/tnca_ooc_final_d2cc850/units.log 2>&1; then
    echo "$(date -u +%FT%TZ) UNITS-DONE" >> $HOME/tnca_ooc_final_d2cc850/queue.log; exit 0
  fi
  echo "$(date -u +%FT%TZ) attempt $i refused/failed: $(tail -1 $HOME/tnca_ooc_final_d2cc850/units.log)" >> $HOME/tnca_ooc_final_d2cc850/queue.log
  rm -rf $OUT
  sleep 600
done
