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

echo "=== $NAME: doctor ==="
make r1-doctor

echo "=== $NAME: generated artefacts (read-only check) ==="
make r1-check-generated

echo "=== $NAME: gate self-test ==="
make r1-gate-selftest

echo "=== $NAME: acceptance ==="
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
