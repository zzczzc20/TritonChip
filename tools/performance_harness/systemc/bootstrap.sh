#!/usr/bin/env bash
# Run on the GB10; never invoke this on the FPGA implementation host.
set -euo pipefail
prefix=${1:?usage: bootstrap.sh ABSOLUTE_DEPENDENCY_DIRECTORY}
case "$prefix" in /*) ;; *) echo 'absolute dependency directory required' >&2; exit 2;; esac
mkdir -p "$prefix"
revision=70b0fc8e4a74acc677b0fc73cea08f940c2115d5 # Accellera SystemC 3.0.2
if [[ ! -d "$prefix/source/.git" ]]; then
  git clone https://github.com/accellera-official/systemc.git "$prefix/source"
fi
test -z "$(git -C "$prefix/source" status --porcelain)"
git -C "$prefix/source" checkout --detach "$revision"
test "$(git -C "$prefix/source" rev-parse HEAD)" = "$revision"
cmake -S "$prefix/source" -B "$prefix/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$prefix/install" \
  -DBUILD_SHARED_LIBS=ON -DENABLE_EXAMPLES=OFF
cmake --build "$prefix/build" --parallel 6
cmake --install "$prefix/build"
git -C "$prefix/source" rev-parse HEAD > "$prefix/revision.txt"
