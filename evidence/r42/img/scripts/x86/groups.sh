#!/bin/bash
# groups.sh DIR... : worst routed slack per endpoint group of tile 0 (top_paths_routed.rpt: one worst path per endpoint pin, most critical 2000)
for d in "$@"; do
  f=$HOME/r42img/$d/tile0/top_paths_routed.rpt; [ -f $f ] || { echo "$d: none"; continue; }
  awk -v d=$d '/^Slack/{s=$4; sub("ns","",s)} /^ *Destination: /{dst=$2;
      g="other"; if (dst ~ /x_valu_imm/) g="core x_valu_imm CE"; else if (dst ~ /g_lsu_hybrid.u_lsu.*out_vector/) g="lsu out_vector"; else if (dst ~ /u_mtx/) g="matrix unit"; else if (dst ~ /u_txf/) g="tensor engine"; else if (dst ~ /u_vfpu/) g="vector fpu"; else if (dst ~ /g_home_pipe/) g="home pipe"; else if (dst ~ /^u_core\//) g="core other";
      if (!(g in m) || s+0 < m[g]+0) m[g]=s; n[g]++}
    END{printf "%-12s", d; for (g in m) printf " | %s %.3f (%d)", g, m[g], n[g]; printf "\n"}' $f
done
