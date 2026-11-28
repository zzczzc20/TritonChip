#!/bin/bash
# xsync_union.sh S : mirror rtl/ and tools/r42 of the CURRENT branch of the worktree to ~/r42_union on the x86 host (S = scratchpad holding vx)
S=$1
cd $(git rev-parse --show-toplevel) || exit 1
( git ls-files rtl tools/r42; git ls-files -o --exclude-standard rtl tools/r42 ) | sort -u | tar cz -T - | $S/vx 'mkdir -p ~/r42_union ~/r42_union_out && cd ~/r42_union && tar xz && echo synced $(find rtl -name "*.v" | wc -l) rtl files'
