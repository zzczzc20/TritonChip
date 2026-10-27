# R41 Stage 2 FP8 Environment Repair and Board Execution Report

Date: 2026-10-26  
Host: `acf3019.ece.ust.hk` (Alveo U280 at PCI BDF `0000:04:00.1`)  
Batch Commit: `86a3d929cf3d` (run directory `~/r41s2_board_86a3d929cf3d_fp8`)  
Card Image: `~/tnca_images/r41s2_run1_c9f25787494a/tnca_u280.xclbin`  
- Image SHA256: `30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de`  
- Image UUID: `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`  
- Clock: 100 MHz DATA_CLK  
- Qualified Clock Evidence: `49868f45b5ced2f662e4f76bbb1bb4a2be166625915d869f20812afcee42f0d7`  
- Queue Log: `~/r41s2_board_86a3d929cf3d_fp8/queue.log`  

---

## 1. Executive Summary

In the initial Stage 2 board batch run (`~/r41s2_board_86a3d929cf3d`), two manifests—`s2-fp8-hub` and `s2-qwen-fp8`—were skipped because building the FP8 virtual environment failed.

This work repaired the FPGA-host Python environment without modifying the shared base `~/tc_env/torchenv`, proved the environment on CPU without touching the FPGA card, and subsequently executed both manifests on the approved 100 MHz U280 image following all card idle and loading governance rules.

Both manifests completed with status `completed-experiment` (exit code 0):
- **`s2-fp8-hub`**: 35 of 35 cases passed (**0 failures, 0 errors, 0 bit mismatches**). Every output across Hub v4, `torch.compile`, transformers `FP8Linear`, and Hub v1 matched the bit-exact reference.
- **`s2-qwen-fp8`**: 14 of 14 linear module evaluations passed (`bitwise_order: true` in all 14, `bitwise_cpu: true` for all 7 decode projections). The 2-layer decode sequence (`steps 0-4`) passed all tolerance and floor criteria with zero host transfers and zero recompilations during steady state.

| Manifest | Status | Verdict | Summary Basis |
|---|---|---|---|
| `s2-fp8-hub` | completed | **Passed** | 35 / 35 test cases pass with 0 bit mismatches against reference (20 Hub v4, 4 compile, 3 transformers `FP8Linear`, 8 Hub v1) |
| `s2-qwen-fp8` | completed | **Passed** | `linears`: 14 / 14 pass (7 decode projections bitwise equal to CPU; 7 prefill projections bitwise equal in execution order, rel L2 1.09e-7 to 5.87e-6 vs CPU); `decode`: 5 steps pass (`pass_: true`, `cpu_pair_reproduces_whole_model: true`), zero host transfers in steady state |

---

## 2. Diagnosis of the FP8 Environment Failure

The initial failure documented in `~/r41s2_board_86a3d929cf3d/fp8env.log` had four distinct root causes:

1. **Virtual Environment Isolation & Torch Package Masking**:
   The queue script invoked `$PY -m venv --system-site-packages $FP8ENV.tmp`. Because `$PY` was `~/tc_env/torchenv/bin/python`, `--system-site-packages` only exposed `/usr/lib/python3/dist-packages` and `/usr/local/lib/python3.10/dist-packages`, not the parent virtual environment's site-packages (`~/tc_env/torchenv/lib/python3.10/site-packages`).
   When `pip install` ran in `$FP8ENV.tmp`, pip did not find the installed Torch `2.14.0+cpu` and downloaded `torch-2.14.1` (555 MB wheel) from PyPI. The assertion `assert torch.__version__.startswith('2.14.0')` failed immediately.

2. **Backend Mismatch in `kernels` 0.17.2**:
   `kernels` selects its kernel implementation by checking `torch.version.cuda`. On CPU PyTorch (`2.14.0+cpu`), `torch.version.cuda` is `None`. `kernels` defaulted to backend `cpu`, which caused `kernels.get_kernel("kernels-community/finegrained-fp8")` to fail with:
   ```
   FileNotFoundError: backend (cuda) does not match system backend (cpu)
   ```
   The TNCA Triton backend targets the Triton CUDA interface, requiring `torch.version.cuda` to be present.

3. **`torchvision` Dynamic Link Failure**:
   Installing PyPI's standard `torchvision==0.29.0` wheel resulted in an import failure because `_C_stable.so` dynamically links against `libcudart.so` and `libtorch_cuda.so`, neither of which exists in the CPU-only host environment.

4. **Offline Hugging Face Hub Metadata Gaps**:
   In offline mode (`HF_HUB_OFFLINE=1`), `kernels.get_kernel` resolves version `4` by checking for `refs/v4` in `~/r41s2_fp8_inputs/hf/hub/kernels--kernels-community--finegrained-fp8/refs/`. This reference was missing. Furthermore, `trees/0e5764aa5c27d0f005117a92645156481540950a.json` listed ROCm and XPU blobs that were not staged on disk, causing `huggingface_hub` to raise `IncompleteSnapshotError`.

---

## 3. The Environment Fix & Pinned Package Manifest

### Layering via `.pth`
To preserve `~/tc_env/torchenv` unmodified while making all of its packages, editable installs (`tnca`), and entry points visible, the virtual environment `~/r41s2_fp8env` was created and layered with a `.pth` file (`zz_torchenv.pth`) placed in its `site-packages`:
```
/home/zzhouch/tc_env/torchenv/lib/python3.10/site-packages
import site; site.addsitedir("/home/zzhouch/tc_env/torchenv/lib/python3.10/site-packages")
import torch; hasattr(torch, "version") and setattr(torch.version, "cuda", "13.0")
```

### Pinned Dependencies
The missing dependencies were installed into `~/r41s2_fp8env` matching the versions used by GB10's `~/flux-venv`:
- `transformers == 5.20.2`
- `kernels == 0.17.2`
- `accelerate == 1.15.0`
- `safetensors == 0.8.0`
- `huggingface_hub == 1.33.0`
- `torchvision == 0.29.0+cpu` (sourced from PyTorch CPU wheel index `https://download.pytorch.org/whl/cpu`)
- `pillow == 10.2.0`

### Offline Hub Cache Fixes
1. Created `refs/v4` pointing to commit hash `0e5764aa5c27d0f005117a92645156481540950a`.
2. Pruned `trees/0e5764aa5c27d0f005117a92645156481540950a.json` so `_raise_if_incomplete_snapshot` only validates files physically present in the staged snapshot.
3. Updated and verified `~/r41s2_fp8_inputs/STAGED` (65 files verified).

### Queue Script Integration
The fix was integrated into `evidence/r41/stage2/board/launches/board_queue.sh` in the repository and deployed to `~/r41s2_board_queue.sh` on the host.

---

## 4. CPU-Side Validation (Zero Card Impact)

Before scheduling any FPGA access, the repaired environment was verified strictly on CPU:
1. **Module Imports**: Verified clean imports of `torch` (2.14.0+cpu), `triton` (3.8.0), `transformers` (5.20.2), `kernels` (0.17.2), `accelerate` (1.15.0), `safetensors` (0.8.0), `torchvision` (0.29.0+cpu), and `PIL` (10.2.0).
2. **Hub Kernels Loading**:
   - `kernels.get_kernel("kernels-community/finegrained-fp8", version=4)` loaded successfully offline.
   - `kernels.get_local_kernel(Path("~/r41s2_fp8_inputs/hub-main"), trust_remote_code=True)` (v1) loaded successfully.
3. **Reference Arithmetic Validation**:
   `tools/performance_harness/fp8_blockwise_ref.py validate` passed all 12 literal arithmetic checks.
4. **Qwen Runner CPU Validation**:
   `tools/models/qwen_tnca_layers.py linears` CPU validation against CPU reference and exact W8A8 reference passed with `passed: 2, total: 2` (both prefill and decode bitwise identical).

---

## 5. Board Run Execution Details

The board batch was launched detached using `board_queue.sh 86a3d929cf3d_fp8` under the approved constraints:
```bash
export TNCA_BOARD=u280
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export S2_MANIFESTS="s2-fp8-hub s2-qwen-fp8"
setsid nohup bash ~/r41s2_board_queue.sh 86a3d929cf3d_fp8 > ~/r41s2_board_86a3d929cf3d_fp8.nohup 2>&1 &
```

### Governance and Card State
- **Card Pre-Check**: Device was idle (`contexts: 0`, `outstanding: 0`, `client: ""`).
- **15-Minute Idle Rule**: Script observed 15 consecutive minutes of idle time (`13:53:13Z` to `14:08:14Z`).
- **Image Load**: Programmed 100 MHz run 1 image (`49964b17-e0b4-5bd2-dd40-7fc37b1205d3`).
- **Clock Counter**: Verified sysfs frequency counter at `100 MHz`.
- **Clock Evidence**: Qualified clock evidence verified (`49868f45b5ced2f6...`).
- **Thermals**: FPGA core 27°C - 32°C, well below thermal limits.
- **Card Exit State**: Left idle with 0 open contexts and no outstanding executions.

### Execution Timeline
- `13:53:09Z`: Queue start, archive extraction to `~/r41s2_board_86a3d929cf3d_fp8/src`.
- `14:08:14Z`: Card idle for 15 consecutive minutes confirmed.
- `14:08:28Z`: Image loaded, clock qualified, `s2-fp8-hub` start.
- `14:49:38Z`: `s2-fp8-hub` exit 0 (`status: completed-experiment`). Duration: 41 min 10 s.
- `14:49:38Z`: `s2-qwen-fp8` start.
- `14:57:53Z`: `s2-qwen-fp8` exit 0 (`status: completed-experiment`). Duration: 8 min 15 s.
- `14:57:53Z`: `S2-BOARD-QUEUE-DONE`. Total queue duration: 64 min 44 s.

---

## 6. Manifest Results

### Manifest `s2-fp8-hub`
Evaluated 35 test cases across 4 groups. All 35 cases matched reference outputs with **0 bit mismatches**:

#### 1. Group `v4` (Official Hub v4 Kernel)
- `fp8_act_quant`: 5 shapes (`4x256`, `7x384`, `129x256`, `16x2048`, `3x6144` bf16, block 128) -> **5/5 pass, 0 bit mismatches**.
- `matmul_2d block-dynamic`: 9 shapes (`1x256x256`, `7x136x144`, `16x256x256`, `33x256x384`, `129x200x328`, `7x6144x2048`, `7x2048x6144`, `16x9216x2560`, `129x2048x2048`) -> **9/9 pass, 0 bit mismatches**.
- `matmul_2d block-static`: 3 shapes (`7x136x144`, `129x200x328`, `16x2048x2048`) -> **3/3 pass, 0 bit mismatches**.
- `matmul_2d tensor-dynamic`: 3 shapes (`7x256x256`, `129x200x328`, `16x512x2048`) -> **3/3 pass, 0 bit mismatches**. Includes full Triton autotuning over 48 accelerator configurations for `16x512x2048`.

#### 2. Group `compile` (`torch.compile(fullgraph=True)`)
- `block-dynamic in a compiled graph 7x256x256`: **Pass, 0 bit mismatches** (1,792 elements).
- `block-dynamic in a compiled graph 33x136x384`: **Pass, 0 bit mismatches** (4,488 elements).
- `tensor-dynamic in a compiled graph 7x256x256`: **Pass, 0 bit mismatches** (1,792 elements).
- `fp8_act_quant in a compiled graph 7x384`: **Pass, 0 bit mismatches** (2,688 elements).

#### 3. Group `transformers` (Transformers 5.20.2 `FP8Linear`)
- `FP8Linear eager x(1,7,384) -> 320`: **Pass, 0 bit mismatches** (2,240 elements).
- `FP8Linear compiled x(1,7,384) -> 320`: **Pass, 0 bit mismatches** (2,240 elements).
- `FP8Linear eager x(1,16,2048) -> 2048`: **Pass, 0 bit mismatches** (32,768 elements).

#### 4. Group `v1` (Official Hub v1 Kernel from Main)
- `fp8_act_quant`: 2 shapes (`7x384`, `129x256` bf16, block 128) -> **2/2 pass, 0 bit mismatches**.
- `w8a8_block_fp8_matmul`: 4 shapes (`7x136x256`, `33x256x384`, `129x200x384`, `7x6144x2048`) -> **4/4 pass, 0 bit mismatches**.
- `w8a8_tensor_fp8_matmul`: 2 shapes (`7x256x256`, `129x200x384`) -> **2/2 pass, 0 bit mismatches**.

---

### Manifest `s2-qwen-fp8`

#### Step 1: `linears`
Evaluated Layer 13 linear modules in both decode (`M=1`) and prefill (`M=81`) modes using real activations captured from the CPU W8A8 run. All 14 tests passed:

| Module | Mode | M | K | N | Bitwise CPU | Bitwise Order | Relative L2 vs CPU | Status |
|---|---|---|---|---|---|---|---|---|
| `self_attn.q_proj` | decode | 1 | 2048 | 2048 | **true** | **true** | 0.0 | **pass** |
| `self_attn.q_proj` | prefill | 81 | 2048 | 2048 | **true** | **true** | 0.0 | **pass** |
| `self_attn.k_proj` | decode | 1 | 2048 | 1024 | **true** | **true** | 0.0 | **pass** |
| `self_attn.k_proj` | prefill | 81 | 2048 | 1024 | **true** | **true** | 0.0 | **pass** |
| `self_attn.v_proj` | decode | 1 | 2048 | 1024 | **true** | **true** | 0.0 | **pass** |
| `self_attn.v_proj` | prefill | 81 | 2048 | 1024 | false | **true** | 1.46e-6 | **pass** |
| `self_attn.o_proj` | decode | 1 | 2048 | 2048 | **true** | **true** | 0.0 | **pass** |
| `self_attn.o_proj` | prefill | 81 | 2048 | 2048 | false | **true** | 5.78e-7 | **pass** |
| `mlp.gate_proj` | decode | 1 | 2048 | 6144 | **true** | **true** | 0.0 | **pass** |
| `mlp.gate_proj` | prefill | 81 | 2048 | 6144 | false | **true** | 6.31e-7 | **pass** |
| `mlp.up_proj` | decode | 1 | 2048 | 6144 | **true** | **true** | 0.0 | **pass** |
| `mlp.up_proj` | prefill | 81 | 2048 | 6144 | false | **true** | 1.10e-7 | **pass** |
| `mlp.down_proj` | decode | 1 | 6144 | 2048 | **true** | **true** | 0.0 | **pass** |
| `mlp.down_proj` | prefill | 81 | 6144 | 2048 | false | **true** | 5.87e-6 | **pass** |

- Every decode projection (`M=1`) matched the CPU reference bit-for-bit (`rel_l2_vs_cpu: 0.0`).
- Every prefill projection (`M=81`) matched the exact device-order re-derivation bit-for-bit (`bitwise_order: true`), with relative L2 vs CPU well below acceptance thresholds.

#### Step 2: `decode`
Ran 2 consecutive decoder layers (layers 13 and 14) with device-resident KV cache across prompt prefill and 4 decode steps:
- **Graph Recompilation**: Step 0 compiled the initial graph, step 1 established the fixed decode bucket, and steps 2, 3, and 4 triggered **0 new graph compilations**.
- **Host Transfers**: In steady state (steps 2, 3, and 4), host read calls were **0**, host write calls were **0**, staging uploads/downloads were **0 bytes**, and resident bytes were constant at 101,659,648 bytes.
- **Performance**: Steady-state execution took 2.5 s per decode step (205,059,906 device cycles, 41 launches per step).
- **Numerical Conformance**: `cpu_pair_reproduces_whole_model: true`, `pass_: true`. All output tokens and KV cache tensors (`y0`, `y1`, `13.keys`, `13.values`, `14.keys`, `14.values`) satisfied the error floor criterion across all steps.

---

## 7. Conclusion

The FP8 Python environment on the FPGA host is fully repaired, reproducible, and layered cleanly over the shared `tc_env/torchenv`. Both previously skipped board manifests (`s2-fp8-hub` and `s2-qwen-fp8`) have now been executed on the approved 100 MHz U280 hardware and pass all bitwise and numerical criteria. Nothing remains unrun.
