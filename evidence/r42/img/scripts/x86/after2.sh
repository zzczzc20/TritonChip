#!/bin/bash
# lane 2 after its arms: kernel synthesis and tile 0 of LINE_HALF + MAC_F32 (the image without the overlapped sequencer)
while kill -0 1220667 2>/dev/null; do sleep 20; done
cd ~/r42img
./lane.sh lf
