#!/bin/bash
# CPU-only, real native SystemC artifacts. No card or image commands.
set -euo pipefail
ROOT="$HOME/r41-accept-p1-v2"
OUT="$HOME/r41-accept-p1/candidate-sensitivity-v2"
mkdir -p "$OUT/configs"
export ROOT OUT
python3 - <<'PY'
import os,json,pathlib
root=pathlib.Path(os.environ['ROOT']);out=pathlib.Path(os.environ['OUT']);base={}
for line in (root/'tools/performance_harness/systemc/configs/r40-three-core.cfg').read_text().splitlines():
 if '=' in line and not line.startswith('#'):
  k,v=line.split('=',1);base[k]=int(v)
base['core_pipeline']=1
# These are one-factor uncertainty probes, NOT RTL counterfactual switches.
variants={'base':{},'F01':{'issue_cycles':1},'F02':{'scalar_latency':1},'F03':{'vector_fp_latency':1},'F04':{'vector_int_latency':1},'F05':{'vrf_bank_conflict_cycles':1},'F06':{'dispatch_cycles':4},'F07':{'lookup_cycles':1},'F08':{'mtx_pipe_cycles':1},'F09':{'vector_fp_latency':1},'F10':{'lsu_setup_cycles':1},'T2':{'dma_setup_cycles':1}}
registration=dict(scope='one-factor uncalibrated timing uncertainty probes; physical changes preserving architectural cycles have no counterfactual switch in this model',variants=variants,workload='standard Torch mm/addmm, aligned and tail, FP16, bitwise independent TNCA-order oracle',decision_rule='never infer a physical candidate gain from a proxy; preserve nonmonotonic effects and inconclusive coverage',budget_seconds=3600)
(out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
for tag,delta in variants.items():
 values=dict(base)
 for k,v in delta.items():values[k]+=v
 (out/'configs'/f'{tag}.cfg').write_text('\n'.join(f'{k}={v}' for k,v in values.items())+'\n')
PY
source "$HOME/tc_env/env.sh"
export PYTHONPATH="$ROOT:$ROOT/python:$ROOT/compiler/python:$ROOT/tools/asm:$ROOT/tools/performance_harness/systemc"
export TNCA_BACKEND=systemc-tlm-grid TNCA_PROFILE=core-vrf-sync-v5-x8 TNCA_TLM_BINARY="$HOME/r41-accept-p1/model-build/tnca-chip" TNCA_TLM_MEMORY_BYTES=0x200000000
export TORCH_EXTENSIONS_DIR="$HOME/r41-accept-p1/extensions" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TNCA_IEEE_DIV=1
for cfg in "$OUT"/configs/*.cfg; do
 tag=$(basename "$cfg" .cfg)
 export TNCA_TLM_CONFIG="$cfg" TNCA_TLM_OUT="$OUT/$tag/models"
 "$HOME/tc_env/torchenv/bin/python" "$ROOT/tools/performance_harness/fragment_case.py" --out "$OUT/$tag/case" --columns 32 --shapes 17x65x33,128x128x192 --bitwise --require-resident > "$OUT/$tag.log" 2>&1
done
