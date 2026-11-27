#!/bin/bash
# kern_seq.sh -- the kernel syntheses one after the other (each peaks near 17 GB: never two at once), waiting for a Vivado slot.
# Usage (x86 host, tree root): bash tools/r42/area/kern_seq.sh "NAME|FLATTEN|IMAGE" ...
cd $(cd $(dirname $0)/../../.. && pwd)
count() { pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true; }
for spec in "$@"; do
    IFS='|' read -r NAME FL IMG <<< "$spec"
    while [ "$(count)" -ge 2 ]; do sleep 20; done
    FLATTEN=$FL bash tools/r42/area/run_kernel.sh $NAME $IMG
done
