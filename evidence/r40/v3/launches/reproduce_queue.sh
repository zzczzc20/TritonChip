#!/bin/bash
# R40-10 item 5: independent reproduction from a clean archive of the delivery commit, queued behind the fe1b
# board queue (campaign -> re-collection -> R40-07.policy). Waits for QUEUE-X2-DONE and a free card, then runs
# tools/r40/v3_reproduce.sh from the archive's own copy (not from any working tree) against the fe1b native77/HBM14
# re-collection as the cycle reference. Fresh caches; never loads an image.
#
#   reproduce_queue.sh COMMIT
set -u
C=${1:?commit}
D=~/tnca_v3_reproduce
A=$D/r40_delivery_$C.tar.gz
L=$D/queue.log
until grep -q QUEUE-X2-DONE ~/tnca_v3_campaign_x2/queue.log 2>/dev/null; do sleep 60; done
while ps -eo args | grep -E 'archbench.run|v3_campaign.py run|v3_supplement.py run|v3_policy_board.py run' | grep -v grep >/dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) reproduce start $C $(sha256sum $A | cut -c1-64)" >> $L
mkdir -p $D/script && tar xzf $A -C $D/script tools/r40/v3_reproduce.sh
bash $D/script/tools/r40/v3_reproduce.sh $A $D/run_$C ~/tnca_runs_r40_v3x2 > $D/reproduce_$C.out 2>&1
echo "$(date -u +%FT%TZ) reproduce exit $? REPRODUCE-DONE" >> $L
