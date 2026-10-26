#!/usr/bin/env bash
# Record model core traces inside the evaluator's toolchain container: the
# tree at /work, results at /out, the model build (tnca-chip and its SystemC
# library, built on the host) mounted read-only at /model.
#
#   record_model_container.sh TREE OUT_DIR MODEL_DIR [record_model.py arguments...]
#
# MODEL_DIR holds model-build/tnca-chip and systemc/install/lib. With
# TNCA_REPLAY_DIR set to an RTL recording, it is mounted at /replay and every
# point replays its RTL trace (record_model.py --replay /replay).
set -euo pipefail
tree=$(cd "${1:?tree}" && pwd); out=${2:?out dir}; model=$(cd "${3:?model dir}" && pwd); shift 3
image=${TNCA_TOOLCHAIN_IMAGE:-tritonchip-toolchain:dev}
mkdir -p "$out"; out=$(cd "$out" && pwd)
cat > "$out/record.sh" <<INNER
set -e
cd /work
export LD_LIBRARY_PATH=/model/systemc/install/lib
make -s python/tnca/data/.staged > /out/stage.log 2>&1
/opt/torchenv/bin/python tools/performance_harness/align/record_model.py \\
    --python /opt/torchenv/bin/python --binary /model/model-build/tnca-chip \\
    --out /out/run ${TNCA_REPLAY_DIR:+--replay /replay} $(printf '%q ' "$@") > /out/record.log 2>&1
INNER
replay_mount=()
if [ -n "${TNCA_REPLAY_DIR:-}" ]; then replay_mount=(-v "$(cd "$TNCA_REPLAY_DIR" && pwd):/replay:ro"); fi
exec docker run --rm --network none \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$tree:/work" -v "$out:/out" -v "$model:/model:ro" "${replay_mount[@]}" -w /work "$image" sh /out/record.sh
