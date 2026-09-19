#!/usr/bin/env bash
#
# fetch_toolchain.sh -- obtain the pinned LLVM/MLIR build.
#
# compiler_overview.md sec 15 C0 pins Triton and its matching LLVM revision.
# Triton requires its own pinned revision rather than an arbitrary release, so
# a system LLVM is not a substitute.
#
# Rather than build LLVM from source, this fetches the prebuilt package Triton
# itself downloads, into the very directory Triton's build looks in
# (~/.triton/llvm), so a later Triton build reuses it instead of fetching a
# second copy. The URL is not invented here: it is the scheme
# python/build_helpers.py constructs, and the sha256 is the one
# cmake/llvm-info.json publishes for this platform.
#
# Usage: compiler/pin/fetch_toolchain.sh [--force]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIN="$ROOT/compiler/pin/toolchain.json"

read -r LLVM_HASH < <(python3 -c "
import json; print(json.load(open('$PIN'))['llvm']['commit'])")

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  SUFFIX=macos-arm64 ;;
  Darwin-x86_64) SUFFIX=macos-x64 ;;
  Linux-aarch64) SUFFIX=ubuntu-arm64 ;;
  Linux-x86_64)  SUFFIX=ubuntu-x64 ;;
  *) echo "no prebuilt LLVM for $(uname -s)-$(uname -m); build from source" >&2
     exit 1 ;;
esac

read -r SHA256 < <(python3 -c "
import json
d = json.load(open('$PIN'))['llvm']['prebuilt_sha256']
print(d.get('$SUFFIX', ''))")

if [ -z "$SHA256" ]; then
  echo "compiler/pin/toolchain.json records no sha256 for $SUFFIX" >&2
  exit 1
fi

REV="${LLVM_HASH:0:8}"
BUILD=1                      # llvm-info.json build_number
NAME="llvm-$REV-$SUFFIX-$BUILD"
URL="https://oaitriton.blob.core.windows.net/public/llvm-builds/$NAME.tar.gz"

# The same location Triton's own build uses, so it is found rather than refetched.
DEST="${TRITON_HOME:-$HOME}/.triton/llvm"
LINK="$DEST/llvm-$SUFFIX"
TARGET="$DEST/$NAME"

if [ -d "$TARGET" ] && [ "${1:-}" != "--force" ]; then
  echo "already present: $TARGET"
  echo "MLIR_DIR=$LINK/lib/cmake/mlir"
  exit 0
fi

mkdir -p "$DEST"
ARCHIVE="$DEST/$NAME.tar.gz"

echo "fetching $NAME (about 355 MB)"
curl -L --fail --progress-bar -o "$ARCHIVE" "$URL"

echo "verifying sha256"
GOT="$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
if [ "$GOT" != "$SHA256" ]; then
  echo "sha256 mismatch for $NAME" >&2
  echo "  expected $SHA256" >&2
  echo "  got      $GOT" >&2
  echo "Refusing to unpack. The pin records what upstream published; a" >&2
  echo "mismatch means the download is not that." >&2
  rm -f "$ARCHIVE"
  exit 1
fi
echo "  ok: $GOT"

echo "unpacking"
rm -rf "$TARGET"
mkdir -p "$TARGET"
tar -xzf "$ARCHIVE" -C "$TARGET" --strip-components=1
rm -f "$ARCHIVE"

ln -sfn "$TARGET" "$LINK"

echo
echo "LLVM $REV ready at $TARGET"
echo "  MLIR_DIR=$LINK/lib/cmake/mlir"
echo "  LLVM_DIR=$LINK/lib/cmake/llvm"
echo
echo "For a Triton build: leave LLVM_SYSPATH unset. Triton looks for exactly"
echo "$NAME in this directory and will find it. Setting"
echo "LLVM_SYSPATH=$LINK makes Triton's setup.py"
echo "symlink that name onto itself, and every later configure then fails with"
echo "\"Too many levels of symbolic links\" while reporting only that it could"
echo "not find MLIR. If you must set it, give the versioned directory:"
echo "  LLVM_SYSPATH=$TARGET"
