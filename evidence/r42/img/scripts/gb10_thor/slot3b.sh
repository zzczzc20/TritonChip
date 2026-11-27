#!/bin/bash
# rtl-1c: the whole tiny tier on the origin/R42 tree, the merged tree at default parameters, and the combined profile; then the small tier on the last two
while pgrep -f "[o]ut/slot3.sh" > /dev/null; do sleep 15; done
~/r42/img_out/img_1c.sh img_base core-vrf-sync-v5-x8-g2c4-dp tiny 2 tiny_origin
~/r42/img_out/img_1c.sh img_mc core-vrf-sync-v5-x8-g2c4-dp tiny 2 tiny_merged
~/r42/img_out/img_1c.sh img_mc core-r42-img tiny 2 tiny_lmf
~/r42/img_out/img_1c.sh img_mc core-vrf-sync-v5-x8-g2c4-dp small 2 small_merged
~/r42/img_out/img_1c.sh img_mc core-r42-img small 2 small_lmf
