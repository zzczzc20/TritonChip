#!/usr/bin/env bash
# Record core traces inside the evaluator's toolchain container, the way the
# agent harness measures a tree (tools/tritonchip_agent_harness/evaluator.py
# SCRIPT): the tree at /work, results at /out, the locked Icarus and the
# /opt/torchenv Torch/Triton environment. Nothing runs outside the container.
#
#   record_container.sh TREE OUT_DIR [record_rtl.py arguments...]
#
# e.g. record_container.sh ~/tritonchip-model-align/tree ~/tritonchip-model-align/run1 \
#          --suite spec/r40/suites/existing.json --tier tiny --jobs 8
set -euo pipefail
tree=$(cd "${1:?tree}" && pwd); out=${2:?out dir}; shift 2
image=${TNCA_TOOLCHAIN_IMAGE:-tritonchip-toolchain:dev}
mkdir -p "$out"; out=$(cd "$out" && pwd)
cat > "$out/record.sh" <<INNER
set -e
cd /work
make -s python/tnca/data/.staged > /out/stage.log 2>&1
/opt/torchenv/bin/python tools/performance_harness/align/record_rtl.py \\
    --python /opt/torchenv/bin/python --out /out $(printf '%q ' "$@") > /out/record.log 2>&1
INNER
exec docker run --rm --network none \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e TNCA_HOST_TIMEOUT="${TNCA_HOST_TIMEOUT:-7200}" \
    -v "$tree:/work" -v "$out:/out" -w /work "$image" sh /out/record.sh
