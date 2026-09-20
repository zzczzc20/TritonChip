#!/usr/bin/env bash
# Fetch the pinned Google XLS release, by checksum, into a directory of ours.
#
# doc/phase1_tasks_and_acceptance.md R1-09:
#
#   "给 XLS 提供固定 revision/依赖的本地 Linux 或容器执行方式；从全部
#    DSLX/表/参数重新生成受控 RTL，不能依赖个人 SSH、私钥或远程 `cur` 目录。"
#   "允许下载固定 checksum 的公开/授权工具分发包；不能把未校验的本地缓存或
#    浮动 tag 当锁定版本。"
#
# So: a fixed release, a recorded sha256, verified before anything is
# unpacked, into `.xls/<version>` with `.xls/cur` pointing at it. `cur` here
# is ours and inside the tree -- the thing the contract rules out is a
# maintainer's remote `cur` that nothing can check.
#
#   usage: fetch_xls.sh [destination]           (default: <repo>/.xls)
#
# XLS publishes linux-x64 binaries only. On any other platform this says so
# and stops, rather than downloading something that cannot run: the
# acceptance environment is Linux x86_64 by contract, and the development
# platform keeps using the ssh runner, which `R1_STRICT=1` refuses.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CONFIG="$HERE/config.json"
DEST=${1:-$ROOT/.xls}

read -r VERSION URL SHA <<<"$(python3 - "$CONFIG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
r = c.get('release') or {}
missing = [k for k in ('version', 'url', 'sha256') if not r.get(k)]
if missing:
    sys.exit(f"tools/xls/config.json has no pinned release ({missing} missing)")
print(r['version'], r['url'], r['sha256'])
PY
)"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) ;;
  *)
    echo "XLS publishes linux-x64 binaries only; this is $(uname -s)/$(uname -m)."
    echo "The acceptance environment is Linux x86_64 (R1-09). On a development"
    echo "platform, xlsgen.py uses the ssh runner, and R1_STRICT=1 refuses it."
    exit 2 ;;
esac

mkdir -p "$DEST"
TARBALL="$DEST/xls-$VERSION.tar.gz"

# A shared read-only download cache is what R1-09 permits between the two
# acceptance environments ("允许使用校验过的只读下载缓存"). It saves a
# download and nothing else: the checksum below is verified whatever the
# file's provenance, so a poisoned cache is caught exactly like a poisoned
# download.
if [ ! -f "$TARBALL" ] && [ -n "${XLS_CACHE:-}" ] \
   && [ -f "$XLS_CACHE/xls-$VERSION.tar.gz" ]; then
  echo "from the shared cache: $XLS_CACHE/xls-$VERSION.tar.gz"
  cp "$XLS_CACHE/xls-$VERSION.tar.gz" "$TARBALL"
fi

if [ -f "$TARBALL" ]; then
  echo "already present: $TARBALL"
else
  echo "fetching $URL"
  curl -fL --retry 3 -o "$TARBALL.part" "$URL"
  mv "$TARBALL.part" "$TARBALL"
fi

# Before unpacking, not after. An unverified archive that has already been
# extracted has already had its way with the filesystem.
echo "verifying sha256"
GOT=$(sha256sum "$TARBALL" | cut -d' ' -f1)
if [ "$GOT" != "$SHA" ]; then
  echo "CHECKSUM MISMATCH"
  echo "  expected $SHA"
  echo "  got      $GOT"
  echo "The pinned release is what config.json says it is; a download that"
  echo "does not match it is not that release. Refusing to unpack."
  exit 1
fi

TARGET="$DEST/$VERSION"
if [ ! -d "$TARGET" ]; then
  echo "unpacking into $TARGET"
  mkdir -p "$TARGET"
  tar xzf "$TARBALL" -C "$TARGET" --strip-components=1
fi

ln -sfn "$VERSION" "$DEST/cur"
echo "  XLS $VERSION at $DEST/cur"
"$DEST/cur/interpreter_main" --version 2>&1 | head -1
echo
echo "xlsgen.py will use it: set XLS_HOME=$DEST/cur, or leave it -- the"
echo "default search includes <repo>/.xls/cur."
