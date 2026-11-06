#!/bin/bash
# R41 G2 FMA/MAC rebuild: DSLX -> combinational Verilog with the pinned XLS
# (the same steps as tools/xls/xlsgen.py's combinational modules).
#   xlsc.sh XLS_DIR SRC_DIR FILE.x TOP MODULE OUT_DIR
set -euo pipefail
x=$1; src=$2; file=$3; top=$4; mod=$5; out=$6
mkdir -p "$out"; cd "$src"
"$x/ir_converter_main" --dslx_stdlib_path="$x/xls/dslx/stdlib" --top="$top" "$file" > "$out/$mod.ir"
"$x/opt_main" "$out/$mod.ir" > "$out/$mod.opt.ir"
"$x/codegen_main" --generator=combinational --use_system_verilog=false --top="$top" \
    --module_name="$mod" --output_verilog_path="$out/$mod.v" "$out/$mod.opt.ir"
echo "XLSC $mod $(grep -c '^ ' "$out/$mod.ir") $(grep -c '^ ' "$out/$mod.opt.ir")"
