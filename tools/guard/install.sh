#!/bin/sh
set -eu
root=$(git rev-parse --show-toplevel)
common=$(git rev-parse --path-format=absolute --git-common-dir)
dest="$common/guard"
mkdir -p "$dest/githooks"
for file in guard.py activity.py monitor.py watchdog.py; do
    install -m 0644 "$root/tools/guard/$file" "$dest/$file"
done
install -m 0755 "$root/tools/guard/supervisor.sh" "$dest/supervisor.sh"
for hook in pre-commit commit-msg post-commit pre-push; do
    install -m 0755 "$root/tools/guard/githooks/$hook" "$dest/githooks/$hook"
done
# An absolute path is necessary: older worktrees do not have the tracked hooks.
git config --file "$common/config" core.hooksPath "$dest/githooks"
git worktree list --porcelain | while IFS= read -r line; do
    case "$line" in
      'worktree '*)
        wt=${line#worktree }
        actual=$(git -C "$wt" config --get core.hooksPath || :)
        if [ "$actual" != "$dest/githooks" ]; then
            printf 'guard: conflicting hooksPath in %s: %s\n' "$wt" "$actual" >&2
            exit 1
        fi ;;
    esac
done
