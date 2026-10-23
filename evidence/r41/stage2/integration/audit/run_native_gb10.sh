#!/bin/bash
set -eu
source "$HOME/tc_env/env.sh"
P="$HOME/tc_env/torchenv/bin/python"
R="$HOME/r41s2-int-audit-results"
I="$HOME/r41s2-int-audit-src"
B="$HOME/r41s2-int-base"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
check_memory() {
    free -g
    test "$(free -g | awk '/^Mem:/ {print $7}')" -ge 40
}
    if [ "$#" -eq 0 ]; then set -- base int; fi
for name in "$@"; do
        if [ "$name" = int ]; then tree="$I"; else tree="$B"; fi
        for row in features pair ports; do
            if [ -f "$R/$name-$row.exit" ] && [ "$(cat "$R/$name-$row.exit")" = 0 ]; then continue; fi
            check_memory > "$R/$name-$row.memory.log"
            cache="$R/cache-$name-$row"
            mkdir -p "$cache"
            export TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_HOME="$cache/home"
            case "$row" in
                features) args=(native_multicore_features.py --backend fpga-multicore-sim --cores 3 --pipeline 2 --board-top) ;;
                pair) args=(native_multicore_features.py --backend fpga-multicore-sim --cores 3 --pipeline 2 --board-top --pair) ;;
                ports) args=(native_memory_ports_tests.py --backend fpga-multicore-sim --cores 3 --ports 3 --pipeline 2 --board-top) ;;
            esac
            set +e
            (cd "$tree"; /usr/bin/time -v "$P" "$R/run_native.py" "$tree" "${args[@]}") > "$R/$name-$row.log" 2>&1
            code=$?
            set -e
            printf '%s\n' "$code" > "$R/$name-$row.exit"
        done
    done
