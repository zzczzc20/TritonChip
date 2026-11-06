#!/bin/sh
# Lead-only launcher. It neither programs nor resets the shared U280.
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 IMMUTABLE_SOURCE_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
ROOT=$(cd "$1" && pwd)
OUT=$2
cd "$ROOT"
exec setsid nohup "$HOME/tc_env/torchenv/bin/python" \
    tools/performance_harness/launches/acceptance_queue.py \
    --root "$ROOT" --out "$OUT" \
    --python "$HOME/tc_env/torchenv/bin/python" \
    --budget-seconds 10800 \
    --manifest spec/r41/chat/s2-qwen-chat-v2.json \
    > "$OUT.log" 2>&1 < /dev/null &
