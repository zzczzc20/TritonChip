#!/bin/bash
# refresh.sh : fetch the rtl-3c records of GB10 and thor into the worktree evidence and print the combined table
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
L=/home/ubuntu/r42-wt/img/evidence/r42/img/rtl_3c
mkdir -p $L/gb10 $L/thor $S/img/all3c
rsync -a -e $S/gb_rsh gb10:r42/img_out/mc/ $L/gb10/ > /dev/null 2>&1
timeout 100 $S/t232 'cd ~/r42/img_out && tar czf - mc' < /dev/null 2>/dev/null | tar xz -C $L/thor --strip-components=1 2>/dev/null
rm -f $S/img/all3c/*
for h in gb10 thor; do for d in $L/$h/*_*; do b=$(basename $d); case $b in smoke*) continue;; esac; [ -d $d ] && ln -sfn $d $S/img/all3c/$b; done; done
cd /home/ubuntu/r42-wt/img; python3 tools/r42/img_table.py A $S/img/all3c --arms ${ARMS:-A,L,M,F,LM,LF,LMF}
# per-arm directories of symlinks to every record of the arm (all sets, both hosts), for img_predict.py
rm -rf $S/img/byarm; mkdir -p $S/img/byarm
for d in $S/img/all3c/*; do b=$(basename $d); arm=${b%%_*}; mkdir -p $S/img/byarm/$arm; for f in $d/*.jsonl; do [ -e "$f" ] || continue; case $f in *digest.jsonl) continue;; esac; ln -sf $(readlink -f $f) $S/img/byarm/$arm/${b}_$(basename $f); done; done
