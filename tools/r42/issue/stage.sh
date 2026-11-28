#!/bin/bash
# R42 issue: stage the source tree of a candidate image (nothing is linked or loaded).   stage.sh COMMIT OUTDIR
#   stage_<commit8>.tar.gz = git archive of COMMIT without evidence/, doc/, tutorial/, examples/, deliverables/, plus evidence/r42/issue/images/*.tcl from the same commit
#   and evidence/r41/hbm_bw/g4/run2/image.tcl (the run-3 image, for reference).
set -e
C=$(git rev-parse $1); OUT=$2
mkdir -p $OUT
T=$(mktemp -d)
git archive $C | tar -x -C $T
rm -rf $T/doc $T/tutorial $T/examples $T/deliverables $T/evidence
mkdir -p $T/evidence/r42/issue/images $T/evidence/r41/hbm_bw/g4/run2
git archive $C evidence/r42/issue/images 2>/dev/null | tar -x -C $T || true
git show origin/R42:evidence/r41/hbm_bw/g4/run2/image.tcl > $T/evidence/r41/hbm_bw/g4/run2/image.tcl
N=stage_$(echo $C | cut -c1-8).tar.gz
tar -C $T -czf $OUT/$N .
sha256sum $OUT/$N | tee $OUT/$N.sha256
rm -rf $T
