#!/bin/bash
# usage: xf_jobs.sh JOBSFILE TREE CFG BIN IDFILE      JOBSFILE lines: TAG|ENV=V ENV2=V2   (ENV part may be empty)
JF=$1; TREE=$2; CFG=$3; BIN=$4; IDF=$5
while IFS='|' read -r TAG ENVS; do
  [ -z "$TAG" ] && continue
  XE=""; for kv in $ENVS; do XE="$XE --env $kv"; done
  XF_ENV="$XE" $HOME/r42/xfer_out/xf_tlm.sh $TREE $CFG $TAG $BIN $(cat $IDF) > $HOME/r42/xfer_out/log_$TAG.txt 2>&1
  python3 $HOME/r42/xfer_out/xf_collect.py $HOME/r42/xfer_out/collected_$(basename $JF .jobs).json $HOME/r42/xfer_out/tlm/$TAG
  echo "TAG $TAG done"
done < $JF
echo JOBS_DONE
