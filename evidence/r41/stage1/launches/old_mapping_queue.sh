#!/usr/bin/env bash
# One registered diagnostic arm on the already approved B40 image.
# Sources and manifests must be installed and reviewed before starting this.
set -euo pipefail
D=$HOME/r41s1_old_mapping_v1
R=$HOME/r41s1_old_mapping_control_v1
PY=$HOME/tc_env/torchenv/bin/python
K=/sys/bus/pci/devices/0000:04:00.1/kdsstat
U=/sys/bus/pci/devices/0000:04:00.1/xclbinuuid
C=/sys/bus/pci/devices/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs
UUID=1bada9db-78f3-5bc8-7ba6-cce062e7ade8
X=$HOME/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin
mkdir -p "$D"
exec 9>"$D/queue.lock"
flock -n 9 || { echo 'this diagnostic queue is already running'; exit 11; }
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$D/queue.log"; }
trap 'log "queue interrupted; no automatic retry"; exit 130' INT TERM

# Read-only source audit before any card operation. Neither source tree changes.
"$PY" - "$R" "$D" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,directory=map(Path,sys.argv[1:]);sys.path.insert(0,str(root))
from archbench.provenance import software_tree
proof=json.loads((directory/'preparation.json').read_text())
manifest=json.loads((directory/'board-manifest.json').read_text())
validation=json.loads((directory/'validation-v2.json').read_text())
preflight=json.loads((directory/'preflight-v2/preflight.json').read_text())
assert validation['status']=='old-mapping-control-validated'
assert preflight['status']=='registration-preflight-pass-no-device-access'
assert preflight['commands_executed']==0
assert preflight['manifest_sha256']==hashlib.sha256((directory/'board-manifest.json').read_bytes()).hexdigest()
assert hashlib.sha256((directory/'preparation.json').read_bytes()).hexdigest()==validation['preparation_sha256']
assert hashlib.sha256((directory/'validation-v2.json').read_bytes()).hexdigest()==manifest['registration']['validation_sha256']
assert software_tree(root,refresh=True)==proof['control_software_tree']==manifest['registration']['isolated_control']
assert hashlib.sha256((root/'compiler/python/tnca_cc/register_blocks.py').read_bytes()).hexdigest()==proof['control_file_sha256']
PY
log "queue start pid $$; require continuous 900-second idle interval even if B40 is loaded"
idle_since=-1
read -r up _ < /proc/uptime
waiting_start=${up%%.*}
while true; do
    snapshot=$(cat "$K")
    contexts=$(awk -F: '/^contexts:/{gsub(/[ \t]/,"",$2); print $2}' <<< "$snapshot")
    outstanding=$(awk -F: '/^outstanding execs:/{gsub(/[ \t]/,"",$2); print $2}' <<< "$snapshot")
    client=$(awk 'p && NF {print $1;exit} /^client pid:/ {p=1}' <<< "$snapshot")
    read -r up _ < /proc/uptime
    now=${up%%.*}
    log "idle probe contexts=$contexts outstanding=$outstanding client=${client:-none} uuid=$(cat "$U")"
    if [[ "$contexts" == 0 && "$outstanding" == 0 && -z "$client" ]]; then
        if (( idle_since < 0 )); then idle_since=$now; fi
        if (( now-idle_since >= 900 )); then break; fi
    else
        idle_since=-1
    fi
    if (( now-waiting_start >= 86400 )); then log 'waiting budget expired; no device change'; exit 12; fi
    sleep 30
done
log 'continuous 15-minute idle interval verified'
set +u
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
set -u
export TNCA_BOARD=u280
cd "$R"
if [[ $(cat "$U") != "$UUID" ]]; then
    bash flow/fpga/program.sh --check "$X" >> "$D/queue.log" 2>&1
    bash flow/fpga/program.sh --go "$X" >> "$D/queue.log" 2>&1
    for ((i=0;i<60;i++)); do
        [[ $(cat "$U") == "$UUID" ]] && break
        sleep 10
    done
fi
read -r mhz < "$C"
[[ $(cat "$U") == "$UUID" && "$mhz" == 90 ]] || { log 'wrong image/clock; stopping'; exit 13; }
log 'registered old-mapping diagnostic starts on B40 at 90 MHz'
set +e
"$PY" tools/performance_harness/experiment.py "$D/board-manifest.json" \
    --root "$R" --out "$D/out" --python "$PY" > "$D/experiment.log" 2>&1
rc=$?
set -e
log "experiment exit $rc; numerical/performance results still require review"
exit "$rc"
