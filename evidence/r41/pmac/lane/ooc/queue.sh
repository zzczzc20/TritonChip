#!/bin/bash
# queue.sh JOBS OUTROOT [WORKERS] -- on the FPGA host: run the lane OOC jobs of
# JOBS (one per line: TREE TAG PERIOD [FILES=...] GENERICS...) with at most
# WORKERS (default 2) at a time, each through run_ooc.sh / vivado_slot.sh.
set -u
jobs=${1:?jobs}; outroot=${2:?outroot}; W=${3:-2}
here=$(cd $(dirname $0) && pwd)
mkdir -p $outroot
worker() {
    local k=$1 n=0
    while read -r tree tag period rest; do
        [ -z "$tree" ] && continue; [ "${tree:0:1}" = "#" ] && continue
        n=$((n + 1)); [ $(( (n - 1) % W )) -ne $k ] && continue
        files=""
        if [ "${rest%% *}" != "${rest#FILES=}" ] && [ "${rest:0:6}" = "FILES=" ]; then
            files=${rest%% *}; files=${files#FILES=}; files=${files//,/ }; rest=${rest#* }
        fi
        FILES="$files" bash -c '[ -z "$FILES" ] && unset FILES; exec "$@"' _ \
            $here/run_ooc.sh $tree $outroot $period $tag $rest
    done < $jobs
}
for k in $(seq 0 $((W - 1))); do worker $k & done
wait
echo "$(date -u +%FT%TZ) QUEUE DONE" >> $outroot/run.log
