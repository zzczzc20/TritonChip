#!/bin/bash
# after lane 1's idef tile: tnca_matrix out of context at the run-3 image's parameters (VRF_LAT 2, TAG_W 2): the base commit's unit, the merged unit at defaults, MAC_F32, OVL, both
while kill -0 1235342 2>/dev/null; do sleep 20; done
cd ~/r42img
BASE=$HOME/r42img/base/src; NEW=$HOME/r42img/lmf-a099878e/src
./unit_ooc.sh u_base $BASE
./unit_ooc.sh u_f $NEW MAC_F32=1
./unit_ooc.sh u_m $NEW OVL=1 VRF_LAT=2 TAG_W=2
./unit_ooc.sh u_mf $NEW OVL=1 VRF_LAT=2 TAG_W=2 MAC_F32=1
./unit_ooc.sh u_def $NEW
