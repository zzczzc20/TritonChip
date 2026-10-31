#!/bin/bash
#==============================================================================
# final_image_launch.sh -- the last R40 implementation, from a frozen commit.
#
#   tools/r40/final_image_launch.sh ARCHIVE.tar.gz COMMIT IMAGE_TCL
#
# Run on the FPGA host. ARCHIVE is `git archive COMMIT` of the candidate;
# IMAGE_TCL the attempt-3 image configuration (the candidate adds no image
# parameter). Unpacks a fresh ~/tnca_src_r40_attempt04_<commit>, records every
# input's sha256 (preflight.json), refuses to start without 35 GB available
# (routing peaked at 30.9 GB in attempt 3), then runs the unchanged attempt-3
# flow (flow/fpga/run_r40_attempt01.sh: SSI_SpreadSLLs, VPP_JOBS=2,
# --kernel_frequency 0:100) detached, followed by final_image_observe.sh.
# Never retries and never loads the card.
#==============================================================================
set -euo pipefail
ARCHIVE=${1:?archive}; COMMIT=${2:?commit}; IMAGE=${3:?image.tcl}
SRC=$HOME/tnca_src_r40_attempt04_$COMMIT
BUILD=$HOME/tnca_build/r40_attempt04_$COMMIT
[ ! -e "$SRC" ] && [ ! -e "$BUILD" ] || { echo "refusing: $SRC or $BUILD exists" >&2; exit 2; }
avail=$(free -g | awk '/^Mem:/{print $7}')
[ "$avail" -ge 35 ] || { echo "refusing: ${avail} GB available (< 35 GB)" >&2; exit 3; }
mkdir -p "$SRC"
tar xzf "$ARCHIVE" -C "$SRC"
mkdir -p "$SRC/evidence/r40/v2/convergence/attempt03"
cp "$IMAGE" "$SRC/evidence/r40/v2/convergence/attempt03/image.tcl"
python3 - "$SRC" "$COMMIT" "$ARCHIVE" <<'EOF'
import hashlib, json, sys
from pathlib import Path
src, commit, archive = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = sorted(p for d in ('rtl', 'flow', 'fpga') for p in (src/d).rglob('*') if p.is_file())
files.append(src/'evidence/r40/v2/convergence/attempt03/image.tcl')
doc = dict(commit=commit, archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
           inputs={str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(src/'preflight.json').write_text(json.dumps(doc, indent=1)+'\n')
print(len(doc['inputs']), 'inputs hashed')
EOF
cd "$SRC"
setsid nohup bash -c "bash flow/fpga/run_r40_attempt01.sh '$BUILD' '$SRC/evidence/r40/v2/convergence/attempt03/image.tcl'; \
  bash tools/r40/final_image_observe.sh '$SRC' '$BUILD'" > "$HOME/tnca_attempt04_$COMMIT.out" 2>&1 < /dev/null &
echo "launched build $BUILD from $SRC (pid $!)"
