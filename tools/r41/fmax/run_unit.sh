#!/bin/bash
# run_unit.sh NAME TOP ROUTE FILE.v [FILE.v ...]  (x86 host; wraps TOP in registers)
set -u
name=$1 top=$2 route=$3; shift 3
files=(); for f in "$@"; do files+=("$(readlink -f "$f")"); done
d=~/fmax/runs/$name; rm -rf "$d"; mkdir -p "$d"
source ~/x86_link_env.sh >/dev/null 2>&1
python3 ~/fmax/fmax/regwrap.py "${files[0]}" "$top" > "$d/wrap.v"
cd "$d" && nice -n 5 vivado -mode batch -nojournal -log vivado.log \
  -source ~/fmax/fmax/unit_ooc.tcl -tclargs ${top}_rw "$d" "$route" "$d/wrap.v" "${files[@]}" > /dev/null 2>&1
python3 ~/fmax/fmax/unitsum.py "$d"
