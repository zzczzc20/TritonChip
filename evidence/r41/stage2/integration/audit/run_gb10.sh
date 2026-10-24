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
check_memory > "$R/start-memory.log"
sha256sum "$I/tools/performance_harness/default_change_audit.py" "$B/tools/performance_harness/default_change_audit.py" > "$R/harness.sha256"
for name in int base; do
    if [ "$name" = int ]; then tree="$I"; else tree="$B"; fi
    (cd "$tree"; "$P" -c 'from archbench.provenance import software_tree; print(software_tree())') > "$R/$name.tree"
done
(
    for config in base-default int-default int-x8; do
        check_memory > "$R/$config.memory.log"
        case "$config" in
            base-default) tree="$B"; hash=$(cat "$R/base.tree"); profile=core-vrf-sync-v5 ;;
            int-default) tree="$I"; hash=$(cat "$R/int.tree"); profile=core-vrf-sync-v5 ;;
            int-x8) tree="$I"; hash=$(cat "$R/int.tree"); profile=core-vrf-sync-v5-x8 ;;
        esac
        /usr/bin/time -v "$P" "$R/run_audit.py" "$I" "$profile" run --source-root "$tree" --software-tree "$hash" --label "$config" --out "$R/$config" --jobs 2 > "$R/$config.log" 2>&1
    done
    "$P" "$I/tools/performance_harness/default_change_audit.py" compare --control "$R/base-default/report.json" --candidate "$R/int-default/report.json" --out "$R/default-compare.json" > "$R/default-compare.log" 2>&1
    "$P" "$I/tools/performance_harness/default_change_audit.py" compare --control "$R/int-default/report.json" --candidate "$R/int-x8/report.json" --out "$R/x8-compare.json" > "$R/x8-compare.log" 2>&1
) &
audit_pid=$!
(
    for name in base int; do
        if [ "$name" = int ]; then tree="$I"; else tree="$B"; fi
        for row in features pair ports; do
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
) &
native_pid=$!
wait "$audit_pid"
wait "$native_pid"
