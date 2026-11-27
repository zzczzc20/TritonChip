#!/bin/bash
# lane.sh: arms one after another (one lane). Each: img_chain.sh NAME TGZ IMAGE_TCL 0
cd ~/r42img
for arm in "$@"; do ./img_chain.sh $arm-a099878e src_img_a099878e.tgz evidence/r42/img/images/image_$arm.tcl 0; done
