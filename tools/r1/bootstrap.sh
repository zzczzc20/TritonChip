#!/usr/bin/env bash
# Bring up one R1 acceptance environment and run the gate in it.
#
# doc/phase1_tasks_and_acceptance.md R1-09 asks for two environments, A and B,
# that do not share a build, a venv or a generated artefact. This script is
# what makes one of them; running it twice with different roots makes two, and
# the fact that it takes the root as an argument rather than writing to a fixed
# place is the whole reason they can be independent.
#
#   usage: bootstrap.sh <env-root> <source-archive.tar.gz|->
#
# The archive is a clean `git archive` of the candidate commit. A checkout is
# not reused between environments: B copying A's tree would make "two
# environments" one environment with two names.
set -euo pipefail

ROOT_DIR=${1:?usage: bootstrap.sh <env-root> <archive>}
ARCHIVE=${2:?usage: bootstrap.sh <env-root> <archive>}
NAME=$(basename "$ROOT_DIR")

echo "=== $NAME: preparing $ROOT_DIR ==="
rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR/src" "$ROOT_DIR/evidence"
tar -xzf "$ARCHIVE" -C "$ROOT_DIR/src"

cd "$ROOT_DIR/src"

# A checkout extracted from an archive has no .git, so the gate cannot record a
# commit from it. The candidate commit is passed in explicitly and written
# where git_state() would have found it.
if [ -n "${R1_COMMIT:-}" ]; then
  printf '%s\n' "$R1_COMMIT" > .r1-commit
fi

# ---- this environment's own tools ------------------------------------------
#
# Everything below is built or installed INSIDE $ROOT_DIR. That is what makes
# A and B independent: R1-09 allows a shared, checksum-verified read-only
# download cache and nothing else, so pip may read from $R1_PIP_CACHE and no
# build directory, venv or generated file is shared.
#
# The versions come from spec/r1_toolchain.lock.json rather than from this
# script, because a second copy of a version number is a second thing to
# forget to update.
LOCK=spec/r1_toolchain.lock.json
jqf() { python3 -c "import json,sys;d=json.load(open('$LOCK'));print(eval(sys.argv[1],{'d':d}))" "$1"; }

MLIR_DIR_LOCKED=$(jqf "d['mlir']['mlir_dir']")
FILECHECK_LOCKED=$(jqf "d['mlir']['filecheck']")
TRITON_VERSION=$(jqf "d['triton']['version']")
TRITON_SHA=$(jqf "d['triton']['sha256']")

export LLVM_PREFIX=$(dirname "$(dirname "$MLIR_DIR_LOCKED")")
export LLVM_PREFIX=${LLVM_PREFIX%/lib}
export TNCA_LLVM=$LLVM_PREFIX
export TRITON_PYTHON="$ROOT_DIR/tritonenv/bin/python"

echo "=== $NAME: Triton $TRITON_VERSION into this environment's own venv ==="
python3 -m venv "$ROOT_DIR/tritonenv"
PIP_ARGS=(--disable-pip-version-check)
if [ -n "${R1_PIP_CACHE:-}" ]; then
  # Read-only by intent: a shared download cache is permitted, a shared
  # install is not.
  PIP_ARGS+=(--cache-dir "$R1_PIP_CACHE")
fi
"$ROOT_DIR/tritonenv/bin/pip" install -q "${PIP_ARGS[@]}" "triton==$TRITON_VERSION"
"$ROOT_DIR/tritonenv/bin/python" -c "import triton;assert triton.__version__=='$TRITON_VERSION', triton.__version__;print('  triton', triton.__version__)"
echo "  locked wheel sha256: $TRITON_SHA"

echo "=== $NAME: tnca-opt against the locked MLIR ==="
test -d "$MLIR_DIR_LOCKED" || {
  echo "the locked MLIR is not at $MLIR_DIR_LOCKED"; exit 1; }
test -x "$FILECHECK_LOCKED" || {
  echo "the locked FileCheck is not at $FILECHECK_LOCKED"; exit 1; }
make tnca-opt LLVM_PREFIX="$LLVM_PREFIX" >/dev/null
echo "  built $(./compiler/build/tools/tnca-opt/tnca-opt --version 2>/dev/null | head -1 || echo tnca-opt)"

# XLS, if this platform can run it. R1-09 wants rtl/gen re-generated from the
# DSLX rather than trusted, and the generator must not be somebody's ssh host.
# On a platform XLS does not publish binaries for, this says so and the
# generated check reports rtl/gen as not re-checked -- which is the honest
# outcome, not a silent pass.
if [ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ]; then
  echo "=== $NAME: XLS (pinned release, verified by checksum) ==="
  if XLS_CACHE=${R1_XLS_CACHE:-} bash tools/xls/fetch_xls.sh "$ROOT_DIR/xls"; then
    export XLS_HOME="$ROOT_DIR/xls/cur"
    echo "  XLS_HOME=$XLS_HOME"
  else
    echo "  XLS not available; rtl/gen will be reported as not re-checked"
  fi
fi

echo "=== $NAME: doctor ==="
make r1-doctor

echo "=== $NAME: generated artefacts (read-only check) ==="
# R1_XLS=1 only where XLS is actually present: asking for the rtl/gen check
# without a generator turns a missing tool into a failed check, which says
# something different from what happened.
if [ -n "${XLS_HOME:-}" ]; then
  make r1-check-generated R1_XLS=1
else
  make r1-check-generated
fi

echo "=== $NAME: gate self-test ==="
make r1-gate-selftest

echo "=== $NAME: acceptance ==="
export LLVM_PREFIX TNCA_LLVM TRITON_PYTHON
# The gate's own exit code is the verdict; `|| true` here would be the exact
# mistake this whole work package exists to fix, so it is absent on purpose.
set +e
$(command -v python3) tools/r1/gate.py --evidence "$ROOT_DIR/evidence"
RC=$?
set -e

echo "=== $NAME: verify the evidence it just wrote ==="
python3 tools/r1/verify_evidence.py "$ROOT_DIR/evidence" || true

echo "=== $NAME: gate exit=$RC ==="
exit $RC
