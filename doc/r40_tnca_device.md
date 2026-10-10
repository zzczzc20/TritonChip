# The `tnca` PyTorch device (R40-01)

What makes `tnca` a PyTorch device, what each piece is registered through,
and what changes in Inductor once it is one. Implementation:
`python/tnca/torch_device/` (`csrc/tnca_device.cpp`, `__init__.py`,
`inductor_device.py`). Contract: `doc/r40_tasks_and_acceptance.md` R40-01.

## 1. The device

| Piece | Registered through | What it is |
|---|---|---|
| Name | `torch.utils.rename_privateuse1_backend("tnca")` | PrivateUse1 is `tnca`; `Tensor.tnca()`, `is_tnca` from `generate_methods_for_privateuse1_backend` |
| Storage | C++ `REGISTER_ALLOCATOR(PrivateUse1, ...)` | A DataPtr holds a **device address**: a buffer of the TNCA runtime's heap (`Session.allocate`), never dereferenced on the host. `tensor.data_ptr()` is where the bytes are on the device |
| Metadata operators | C++ `TORCH_LIBRARY_IMPL(aten, PrivateUse1)` | `empty.memory_format`, `empty_strided`, `as_strided`, `view`, `_reshape_alias`, `set_` (storage, storage+offset), `resize_`. The CPU's own code, since none reads data; a storage that must grow is reallocated and copied on the device, never `memcpy`'d |
| Device guard | C++ `C10_REGISTER_GUARD_IMPL(PrivateUse1, ...)` | Current device, one default stream per device |
| Hooks | `torch._C._acc.register_python_privateuseone_hook` | `is_available`, `has_primary_context`, `is_built` |
| Data operators | `torch.library` (Python) | `_copy_from`, `_copy_from_and_resize` (host↔device, device↔device, any strides and dtypes), `fill_`, `zero_`, `_local_scalar_dense`. Through the Session; a strided view stages exactly the span of storage it names and writes back only its own bytes |
| Device module | `torch._register_device_module("tnca", ...)` | `torch.tnca`: `is_available`, `device_count`, `current_device`, `set_device`, `device`, `synchronize`, `Stream`, `Event`, `memory_allocated` |

Everything that computes runs through `torch.compile`. An eager operator
with no `tnca` kernel is refused by the dispatcher (`NotImplementedError`) and
never runs on the host.

R40-01 launches synchronously, one stream per device: an event is always
complete. Ordering across streams, asynchronous copies and deferred frees
are R40-02's.

Why C++ at all: torch 2.14's Python-only route
(`torch.utils.backend_registration._setup_privateuseone_for_python_backend`)
creates PrivateUse1 tensors without storage (`create_empty_tensor`: zero
bytes, null data pointer). Views, offsets and aliases would then have to be
reimplemented in Python per operator. A real allocator gives them the CPU's
semantics for free.

## 2. Compilation for the device

| Piece | Registered through |
|---|---|
| Dynamo device interface | `register_interface_for_device("tnca", TncaInterface)`. Properties are the profile's target file, not placeholders: cores, Role Contexts, vector length, dtype capabilities |
| Scheduling and wrapper | `register_backend_for_device("tnca", TritonScheduling, PythonWrapperTnca)`; the wrapper's header registers the device in the process that runs it |
| Lowering policy | `tnca.torch_device.policy` (§3) |
| Wrapper operations | `register_device_op_overrides("tnca", ...)`: raw stream, `torch.tnca.set_device`, `torch.tnca.device`, `torch.tnca.synchronize` |
| Heuristics, templates, descriptors | `tnca.inductor`: the same pointwise/reduction caps, the single 16×16×16 GEMM/conv configuration and the descriptor rules as the CPU path, registered for device type `tnca` too (`TNCA_DEVICE_TYPES`) |
| Triton | Kernels compile for `GPUTarget("tnca", 0, 1)`, which TNCA's Triton backend (registry key `cpu`) serves. A second registry entry, `tnca`, carries the same, already active, driver and a compiler that claims no target. It exists because Inductor sets a Triton device's driver up by looking for an active backend not named `cpu`. It is added after Triton has created its one driver, since Triton counts active drivers and compatible compilers |
| Launch | `tnca.triton_backend.driver`: a `tnca` tensor's own buffer and byte offset are the argument, so nothing is copied (`resident_bytes` in the launch log). A launch mixing host and device tensors is refused |

## 3. Inductor's lowering policy for `tnca`

Inductor decides "a Triton GPU" by name (`torch._inductor.utils.is_gpu`:
`cuda`, `mps`, `xpu`, `mtia`), and a few more decisions test
`device.type == "cpu"`. Registering a backend changes neither. So both
choices for `tnca` were tried and measured on the tiny tier (RTL simulation,
device cycles against B39's CPU path).

**As a GPU** (`tnca` appended to `GPU_TYPES`), S_geo was 0.834 over 23
points. The GPU-side answers produced kernels the TNCA compiler runs worse:

| GPU-side answer | Effect on TNCA (tiny) |
|---|---|
| Graph inputs assumed aligned, so loads become tensor descriptors (`should_assume_input_aligned`) | One-row descriptor loads with reshapes in every reduction: rmsnorm 0.70, softmax 0.77–0.86, argmax/cumsum 0.85, transpose 0.68 |
| Welford for `var_mean` | layernorm 0.32, and an `scf.if` per step |
| Pointwise `cat` (masked loads at offsets) | rope 0.31: gathers, where the CPU path copies slices |
| GPU matmul decompositions and math SDPA | attention_decode 0.62 |

**As a non-GPU Triton device** (the choice made), wherever lowering asks
`is_gpu()` it gets the CPU path's answer: no input-alignment assumption,
no unrolling of one-element reductions, loops normalised and merged. The
decisions keyed on `device.type == "cpu"` are given the CPU path's answer by
`tnca.torch_device.policy`. Each has its own switch (on by default) so that
widening any of them to the GPU's choice is a measured ablation (R40-03/07)
rather than a side effect of registering a device:

| Switch | Decision (upstream code) | `tnca` gets |
|---|---|---|
| `TNCA_POLICY_TEMPLATES` | Triton GEMM/conv templates (`utils.use_triton_template`, rebound in `kernel.{mm,bmm,conv,mm_plus_mm,mm_grouped}`) | The CPU branch: templates for f16/bf16/f32. A non-GPU device other than `cpu` would get none, and there is no device ATen GEMM |
| `TNCA_POLICY_VARIANCE` | Two-step vs Welford variance (`lowering.use_two_step_variance`) | The CPU threshold (`config.cpp.use_two_step_variance_threshold`) |
| `TNCA_POLICY_CAT` | Pointwise cat vs `ConcatKernel` (`lowering.cat`) | `ConcatKernel` |
| `TNCA_POLICY_MM_DECOMP` | Small-matmul decompositions (`decomposition.{bmm,mm,addmm}`) | The CPU branches |
| `TNCA_POLICY_LAYOUT` | Channels-last for a graph with convolutions (`GraphLowering.decide_layout_opt`; the CPU path always chooses it, MKLDNN being present) | Channels-last |
| `TNCA_POLICY_REALIZE` | Realization and fusion heuristics (`ir.is_cpu`: storing a reused heavy result such as softmax's `exp`, opcount/read thresholds; `BaseSchedulerNode.is_cpu`: no loop reordering or index inversion in fusion) | The CPU's |
| (C++) | SDPA backend (`_fused_sdp_choice` stub) | The CPU selector's answer, so flash-attention-for-CPU is decomposed as on the B39 path; a device SDPA kernel is R40-08's |

The remaining device-type branches, and what they mean for `tnca`:

| Where | For `tnca` |
|---|---|
| `scheduler.create_backend` | `TritonScheduling` (registered) |
| Generated kernel header `set_driver_to_gpu()`; `triton_heuristics` precompile | Find the `tnca` registry entry, whose driver is TNCA's and already active |
| `codegen.triton` read-after-store in a reduction loop | `tl.debug_barrier()`, which the TNCA compiler lowers to a fence (ISA §9.2) |
| `utils.device_need_guard` | No device guard in the wrapper (one device per process in R40-01) |
| `scheduler` (0-d CPU tensors read by a device kernel) | Not converted to scalars for a non-GPU device; such an argument is refused by the launcher (mixed devices) |
| CUDA graphs, benchmarking, autotune subprocesses | Not used |

Every override is scoped to tensors on `tnca`; CPU and CUDA tensors in the
same process keep upstream's behaviour.

## 4. Descriptors on graph inputs (`TNCA_INPUTS_ALIGNED=1`, measured)

`tnca` storages are 64-byte aligned, so unlike host tensors the device's graph
inputs could be assumed aligned. Inductor would then read them through tensor
descriptors as on a GPU. The parity policy leaves that off. The opt-in switch
measures the alternative (RTL simulation, native device, device cycles):

| Point | Parity | Inputs aligned | Speedup |
|---|---|---|---|
| `op.add/medium` | 93 817 | 93 817 | 1.000 (no descriptor either way) |
| `op.matmul/medium` fp16 | 269 196 | 269 196 | 1.000 (the template's transfers are the TNCA compiler's) |
| `op.rmsnorm/medium` | 547 815 | 890 111 | 0.615 |
| `op.softmax/medium` | 564 395 | 613 172 | 0.920 |
| `op.transpose/medium` fp32 | 91 440 | 109 984 | 0.831 |

The tiny tier said the same. Inductor's reductions run one row per program,
and a descriptor load of a one-row block costs the TNCA lowering more than a
vector load of that row. So the switch stays off. Making descriptors pay on
these kernels is a matter of wider, multi-row blocks (R40-03), measured then.

Where the standard path does move data through the transfer engine on `tnca`:
- the GEMM template's tiles: TensorMaps built by the TNCA compiler, transfers and MMAs;
- an intermediate realised and read back (softmax's stored `exp`), through a descriptor.

Inductor generates no descriptor for a one-dimensional pointwise kernel such as
`add`, with or without the switch.
