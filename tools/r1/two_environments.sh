#!/usr/bin/env bash
# Run the R1 acceptance in two independent environments and compare them.
#
# doc/phase1_tasks_and_acceptance.md R1-09:
#
#   "两套相互独立的临时环境 A/B，从同一个干净候选 checkout bootstrap 和构建；
#    B 不可复制 A 的 build/venv/生成物，允许使用校验过的只读下载缓存。"
#
#   usage: two_environments.sh <root> <candidate.tar.gz> [pip-cache] [xls-cache]
#
# Everything each environment needs it builds itself, under its own directory.
# The two caches are downloads only: a wheel and a tarball, both verified by
# digest wherever they came from. Nothing else crosses between A and B, and in
# particular neither sees the other's build, venv or generated files.
#
# The final verdict is NOT either environment's. Each reports its own, leaving
# R1-09 and R1-12 open because one run cannot say whether two agree;
# compare_envs.py settles those over both.
set -euo pipefail

ROOT=${1:?usage: two_environments.sh <root> <candidate.tar.gz> [pip-cache] [xls-cache]}
ARCHIVE=${2:?usage: two_environments.sh <root> <candidate.tar.gz> [pip-cache] [xls-cache]}
PIP_CACHE=${3:-}
XLS_CACHE_DIR=${4:-}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$ROOT"
STAGE="$ROOT/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar xzf "$ARCHIVE" -C "$STAGE" 2>/dev/null

# Started in the CURRENT shell, not in a command substitution. `$(run_one A)`
# runs the function in a subshell, so the job it backgrounds is a child of
# that subshell and the main shell's `wait` cannot see it: "pid N is not a
# child of this shell", and the comparison then ran while both environments
# were still installing. A function body does not fork, so `$!` here is a
# child of this shell.
start_env() {
  local name=$1
  echo "################ $name ################"
  ( cd "$STAGE" \
    && R1_PIP_CACHE="$PIP_CACHE" R1_XLS_CACHE="$XLS_CACHE_DIR" \
       bash tools/r1/bootstrap.sh "$ROOT/$name" "$ARCHIVE" ) \
    > "$ROOT/$name.log" 2>&1 &
}

# In parallel. They share no state, so the only thing serialising them would
# buy is a quieter machine -- and on a runner below the reference spec, wall
# clock is the scarce thing.
start_env envA; PID_A=$!
start_env envB; PID_B=$!
echo "envA pid $PID_A, envB pid $PID_B; logs in $ROOT/env{A,B}.log"

RC_A=0; RC_B=0
wait "$PID_A" || RC_A=$?
wait "$PID_B" || RC_B=$?
echo "envA exit=$RC_A  envB exit=$RC_B"

echo "################ comparison ################"
for e in envA envB; do
  if [ ! -d "$ROOT/$e/evidence" ]; then
    echo "$e produced no evidence directory; see $ROOT/$e.log"
    exit 1
  fi
done
CMP="$ROOT/compare"
rm -rf "$CMP"; mkdir -p "$CMP"
cp -r "$ROOT/envA/evidence" "$CMP/A"
cp -r "$ROOT/envB/evidence" "$CMP/B"
python3 "$HERE/compare_envs.py" "$CMP" --json "$ROOT/compare.json"
