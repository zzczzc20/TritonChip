#!/bin/bash
# le_wait.sh FILE PATTERN [MAXMIN] : poll a GB10 file until PATTERN appears in it
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
F=$1; P=$2; M=${3:-120}; n=0
while [ $n -lt $M ]; do
  if timeout 30 $S/gb_rsh x "grep -q '$P' $F" < /dev/null 2>/dev/null; then echo "FOUND $P in $F"; exit 0; fi
  python3 -c "import time; time.sleep(60)"; n=$((n+1))
done
echo "TIMEOUT"; exit 1
