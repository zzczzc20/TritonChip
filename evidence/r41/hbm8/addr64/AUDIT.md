# Address-width audit

Scope: the native tnca Torch -> Inductor/Triton -> compiler -> runtime -> LSU/descriptor/DMA -> SystemC/AXI path. Repeated address uses are grouped with their defining calculation/field. `raw/address-sites.txt` contains the expanded source-site inventory. Python `int` is unbounded; physical scalar ISA registers and address buses are 64 bits. “Correct” describes width/range handling, not an assertion that every possible Torch/Triton operation is implemented.

| Computation or held field | Width | Assessment | Source |
|---|---|---|---|
| Torch allocation request, DataPtr and storage context | size_t/u64 (64 on tested hosts) | Correct; device address is never a host dereference | `python/tnca/torch_device/csrc/tnca_device.cpp:79` |
| Torch as_strided sizes, strides, storage_offset | signed i64 | Correct; native Torch view/bounds metadata | `python/tnca/torch_device/csrc/tnca_device.cpp:189` |
| Torch resize_/set_ required storage bytes | size_t; offsets i64 | Correct; computeStorageNbytes; device copy on growth | `python/tnca/torch_device/csrc/tnca_device.cpp:206` |
| Storage addresses exported to Python | u64 | Correct; base distinct from data_ptr/view offset | `python/tnca/torch_device/csrc/tnca_device.cpp:317` |
| Python storage allocation rounding and launch reserve | Python int | Correct; no int32 cast; reported device capacity | `python/tnca/torch_device/__init__.py:78` |
| Live storage address lookup and buffer-relative offset | Python int | Correct; live storage ownership checked | `python/tnca/torch_device/__init__.py:133` |
| View first byte, signed stride extent and byte span | Python int | Correct; storage_base + (storage_offset + lo)*itemsize | `python/tnca/torch_device/__init__.py:185` |
| CPU upload/download staging lengths, aligned byte addresses | Python int | Correct for host transfers; kernel numerics remain device-side | `python/tnca/torch_device/__init__.py:155` |
| Triton native argument byte offset | Python int -> device_ptr64 | Correct; base_off + storage_offset*element_size | `python/tnca/triton_backend/driver.py:330` |
| Storage resize byte-copy count, loop offsets and sublaunch bases | i32 per launch / u64 bases / Python total | Fixed: split counts below 2^31; no total truncation | `python/tnca/torch_device/copy.py:26` |
| Tensor copy count, source/destination spans and split views | Python int; i32 local kernel indices | Fixed: recursively split largest byte-span dimension | `python/tnca/torch_device/copy.py:131` |
| Copy overlap and disjoint views sharing storage | Python integer byte intervals | Correct; overlapping copy refused, disjoint copy via device temporary | `python/tnca/torch_device/copy.py:116` |
| Runtime buffer address, size, rounded allocation length | Python int | Correct; logical and reserved bytes distinct | `python/tnca/runtime.py:66` |
| Heap/code base, limit, cursor, capacity, free intervals, align-up | Python int | Correct; first fit, aligned range end checked before mutation | `python/tnca/runtime.py:183` |
| Reserved TMU hole and allocation end | Python int; [0x7fff0000,0x7fff1000) | Correct; no allocation may straddle it, gap reused | `python/tnca/runtime.py:58` |
| Backend-negotiated memory window and heap limit | Python int | Correct; uses actual device memory_bytes | `python/tnca/runtime.py:442` |
| Buffer upload/download offsets and lengths | Python int | Correct; bounds checked against logical size | `python/tnca/runtime.py:546` |
| Argument buffer+offset address, leases and alias spans | Python int -> u64 | Correct; offset in [0,size); conservative span to buffer end | `python/tnca/runtime.py:850` |
| Argument, descriptor, scratch layout and physical-core strides | Python int -> u64 | Correct; aligned slab includes per-core/per-role scratch | `python/tnca/runtime.py:789` |
| Grid dimensions and product | Python int; negotiated finite grid capability | Correctly refuses excessive grids; not an address truncation | `python/tnca/runtime.py:746` |
| ABI pointer slots and integer scalar range checks | 8-byte slots; pointer u64, signed i32/i64 as declared | Correct; over-wide scalar/pointer refused | `python/tnca/abi.py:199` |
| Host descriptor shape/stride/span | Python int; dimensions bounded signed i32 | Correct; required span checked against buffer, byte stride <2^63 | `python/tnca/tensor.py:59` |
| Host TensorMap base/extents/byte strides/block sizes | u64/u64/i64/u32, 256 bytes | Correct; struct Q/q/I, no base truncation | `python/tnca/tensor.py:67` |
| Compiler scalar SSA address and immediate arithmetic | physical 64 bits (IR dtype may be i32) | Correct; s.add/mul/shl operate on all 64 bits | `compiler/python/tnca_cc/ir.py:176` |
| Compiler pointer argument and descriptor ABI loads | 64-bit scalar load | Correct; ARG_BASE and pointer fields read with .d | `compiler/python/tnca_cc/frontend.py:1483` |
| Uniform pointer chains: constants, scalar offsets, byte scaling | Python constants; runtime scalar u64 | Correct; combines whole addptr chain in scalar registers | `compiler/python/tnca_cc/frontend.py:4429` |
| Vector local/gather element indices | signed i32, sign-extended by LSU | Bounded path; does not carry the full tensor byte address | `compiler/python/tnca_cc/frontend.py:4494` |
| Wide tensor values and index arithmetic | lo/hi pair of 32-bit lanes; scalar reconstruction 64 | Correct within supported block forms; unsupported operations refused | `compiler/python/tnca_cc/wide.py:82` |
| Selection of Inductor index64 addressing | PID sign-extension to i64 detected in TTIR | Conservative pattern gate, not a general pointer-range proof; other wide forms retain checked narrowing | `compiler/python/tnca_cc/wide.py:424` |
| Contiguous index64 pointer invariant and lane residual | u64 scalar invariant + signed i32 lane | Fixed: invariant joins scalar base before byte scaling | `compiler/python/tnca_cc/wide.py:536` |
| Independent 64-bit gather/scatter lane offsets | lo/hi -> scalar u64 -> byte address | Fixed: one masked access per lane; no high-bit loss | `compiler/python/tnca_cc/wide.py:638` |
| 2D wide tile anchor and relative gather indices | u64 scalar anchor; checked signed i32 residuals | Fixed: rebase first element; tile span outside signed i32 traps | `compiler/python/tnca_cc/wide.py:576` |
| Narrowing from wide offsets to bounded gather offsets | hi must equal sign(lo) | Correct; TRAP_USER 0xa5564 rather than wrap | `compiler/python/tnca_cc/wide.py:703` |
| Vector i64 division/remainder by positive constants | scalar 64-bit reciprocal multiply-high | Fixed: exact truncating division/remainder; no float conversion | `compiler/python/tnca_cc/wide.py:283` |
| Scalar i64 GEMM grid division/remainder | 64-bit scalar | Fixed during review: reciprocal constant divider or full 64-step runtime divider | `compiler/python/tnca_cc/frontend.py:1903` |
| Row-pointer broadcast of narrowed/rebased [R,1] offsets | i32 residual, splat of lane 0 per row | Fixed: wide.py keeps one-column offset rows lane-uniform (the importer's column convention); the first review put this in tt.broadcast, which changed 8 default conv points, and was moved | `compiler/python/tnca_cc/wide.py:568` |
| Hoisted DMA map origin for sliced pointers | u64 base + Python constant bytes | Fixed: reconstruct from entry argument; do not use a later pointer definition | `compiler/python/tnca_cc/frontend.py:2667` |
| Tile descriptor base/stride and dynamic descriptor coordinates | u64 scalar; Python static affine coefficients | Correct for supported geometry; map origin calculated in scalar arithmetic | `compiler/python/tnca_cc/frontend.py:2681` |
| TensorMap compile-time packing | u64 base/extents/i64 strides; u32 box | Correct; geometry checks bound box/window sizes | `compiler/python/tnca_cc/tiles.py:60` |
| Runtime descriptor field loads and origin=sum(coord*stride) | 64-bit scalar | Correct; loads .d and scalar multiplications | `compiler/python/tnca_cc/frontend.py:3106` |
| Small explicit descriptor-box absolute offsets | i32 only when whole descriptor span proven bounded; otherwise scalar64 | Fixed during review; unknown/large strides use per-lane scalar addresses | `compiler/python/tnca_cc/descriptors.py:266` |
| Descriptor row transfers and descriptor gather/scatter row selection | scalar u64 base + row*byte_stride | Correct; no full address in vector index | `compiler/python/tnca_cc/descriptors.py:149` |
| Role-context scratch displacement | 64-bit scalar role_id * byte size | Correct; per-role scratch offset in scalar registers | `compiler/python/tnca_cc/rctx_spmd.py:282` |
| Codegen large scalar immediates and memory ops | 64-bit scalar; signed immediate expanded as needed | Correct; large values materialized instead of low-word encoding | `compiler/python/tnca_cc/ir.py:181` |
| Transport upload/download address and byte/word count | Python int -> protocol 64-bit fields | Correct; transfers checked/aligned before serialization | `python/tnca/transport.py:334` |
| MMIO host address/window size and DMA relative offset | 64-bit registers; Python int byte counts | Correct; per-transfer window bounds, DMA offsets not u32 | `python/tnca/backends/u50.py:572` |
| AXI external address translation | 64-bit base, size, offset, AR/AW | Correct; axi_base + aligned(req_addr - LOCAL_LIMIT) | `rtl/fpga/tnca_axi_bridge.v:178` |
| LSU effective and per-lane addresses | 64-bit; signed ELEN=32 gather index | Correct; sign-extend before shift and base addition | `rtl/core/tnca_lsu.v:404` |
| RTL descriptor snapshots, outer strides and DMA address accumulation | 64-bit base/extent/stride/coordinate/accumulator | Correct; local window offsets/row-byte counts bounded 32-bit | `rtl/core/tnca_tensor.v:195` |
| RTL DMA row address and memory request address | 64-bit | Correct; local window indexing is not an external address | `rtl/core/tnca_tensor.v:600` |
| SystemC frame scalar registers, LSU effective addresses | u64; signed i32 gathered index | Correct; matches RTL index sign extension and scaling | `tools/performance_harness/systemc/src/chip.cpp:580` |
| SystemC TensorMap fields, coordinate*stride and final bytes | u64 modulo 2^64; unsigned bounded local flat index | Correct; window bounds checked independently | `tools/performance_harness/systemc/include/tnca_tensor_model.hpp:7` |
| SystemC memory capacity, bounds, reserved hole, paged-store addresses | u64; subtraction-form bounds check | Correct; out-of-window/hole rejected before backing-store access | `tools/performance_harness/systemc/include/tnca_model.hpp:133` |
| SystemC host transfer addresses and lengths | Python int decimal protocol -> model u64 | Correct; TNCA_TLM_MEMORY_BYTES sent without 32-bit cast | `tools/performance_harness/systemc/tlm_backend.py:252` |
| XRT BO properties and physical base | u64 size/paddr; BO handle u32 | Correct; handle is not an address | `python/tnca/backends/u50.py:831` |
| XRT allocation, DMA transfer length and BO-relative byte offset | ctypes c_size_t (64 on tested hosts) | Correct declarations for AllocBO/ReadBO/WriteBO/SyncBO; bounds checked before calls; board not exercised | `python/tnca/backends/u50.py:1018` |
| XCLBIN memory topology capacity/base | u64 -> Python int | Correct; size_kb multiplied in Python without narrowing | `python/tnca/backends/u50.py:895` |
| Physical MMIO register access | two u32 halves -> u64 | Correct; high half at offset+4 is preserved | `python/tnca/backends/u50.py:1276` |
| RTL host memory/DMA window guard and address cursor | u64 address, size and word count | Correct for negotiated windows; host address is not the register-file index | `rtl/fpga/tnca_mmio.v:424` |
| Affine i64 tile indices and lazy data rows | symbolic Python coefficients + u64 scalar origins; checked i32 local span | Fixed during review: avoid eager full-tile low/high pairs; materialize data a row at a time | `compiler/python/tnca_cc/wide.py:337` |
| Affine i64 row materialization for data/mask consumers | 64-bit scalar arithmetic -> lo/hi vectors | Correct; no narrowing when an affine value is consumed as data | `compiler/python/tnca_cc/wide.py:325` |
