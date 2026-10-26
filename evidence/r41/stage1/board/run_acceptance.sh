#!/usr/bin/env bash
# Local evidence evaluation only; does not connect to the FPGA host.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
frozen=${1:?usage: run_acceptance.sh /path/to/detached-frozen-worktree}
head_commit=$(git -C "$frozen" rev-parse HEAD)
expected_commit=$(git -C "$repo" rev-parse 25094197f6e6)
test "$head_commit" = "$expected_commit"
cp -a "$repo/evidence/r41/stage1/." "$frozen/evidence/r41/stage1/"
cp "$repo/doc/r41_stage1_report.md" "$frozen/doc/r41_stage1_report.md"
python3 - "$frozen" <<'PY'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from archbench.provenance import software_tree
assert software_tree(root)=='files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a'
assert hashlib.sha256((root/'doc/r41_tasks_and_acceptance.md').read_bytes()).hexdigest()=='2e0e598b9331dca5e922ce901ccd0d801792cefb41ca7fbf9b7bc4a41de3ddfd'
PY
tar xzf "$repo/evidence/r41/stage1/board/finished.tgz" -C "$frozen/evidence/r41/stage1/board"
set +e
make -C "$frozen" r41-stage1-acceptance PYTHON=python3 \
  R41_ACCEPTANCE_OUT="$repo/evidence/r41/stage1/board/acceptance.json" \
  > "$repo/evidence/r41/stage1/board/acceptance.log" 2>&1
rc=$?
set -e
cat "$repo/evidence/r41/stage1/board/acceptance.log"
printf '%s\n' "$rc" > "$repo/evidence/r41/stage1/board/acceptance.exit-code"
exit "$rc"
