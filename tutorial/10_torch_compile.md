# 10 — PyTorch: `torch.compile` onto TNCA

[`examples/10_torch_compile.py`](examples/10_torch_compile.py)

Everything so far started from a Triton kernel you wrote. This page starts
from ordinary PyTorch: `torch.compile` captures the function, Inductor fuses
it and generates Triton, and the kernels run on TNCA. Nothing in the model
code knows about TNCA.

## Set up

This needs PyTorch at the locked version (`spec/r38/torch.lock.json`) in its
own environment, with this package installed into it:

```sh
python3 -m venv ~/torchenv
~/torchenv/bin/pip install "torch==2.14.0" "triton==$(python3 -c "import json;print(json.load(open('spec/r1_toolchain.lock.json'))['triton']['version'])")" numpy
make python/tnca/data/.staged
~/torchenv/bin/pip install -e .
~/torchenv/bin/tnca-sim prepare          # build the simulated device, once
~/torchenv/bin/python tutorial/examples/10_torch_compile.py
```

`tnca-sim prepare` exists because a launch never builds the simulator it runs
on: an iverilog compile hidden inside a kernel call would be the wrong place
to spend minutes.

## The whole integration is one call

```python
import tnca.inductor
tnca.inductor.configure()

compiled = torch.compile(mlp_block)      # CPU tensors in, CPU tensors out
```

`configure()` sets upstream options through upstream extension points and
returns what it set:

```text
Inductor, as configured for TNCA:
  cpu_backend                  triton
  max_block                    16
  persistent_reductions        False
  max_autotune_gemm_backends   TRITON
```

- `cpu_backend = triton`: Inductor generates Triton, not C++, for CPU
  tensors, and the tnca Triton backend serves that target.
- `max_block = 16`: every block dimension is capped at one 16-lane vector
  register per tile row. A smaller block only means more programs; the
  kernels mask by their extents.
- `persistent_reductions = False`: a persistent reduction needs one block to
  span the whole reduced dimension, which the cap forbids.
- `max_autotune_gemm_backends = TRITON`: a matmul goes through Inductor's
  Triton template, with one 16×16×16 configuration (the MMA fragment), never
  through the ATen kernel that would run on the host.

## What it prints

```text
mlp_block, compiled:
  Triton kernels run on TNCA  ['triton_red_fused_add_mean_mul_pow_rsqrt_silu_1', 'triton_tem_fused_mm_0']
  host fallbacks              none
  max |compiled - eager|      9.54e-07  (tolerance 0.000366)

with_sort, compiled:
  host fallbacks              ['ATen op: torch.ops.aten.sort']
  equal to eager              True

OK
```

`mlp_block` is a matmul, an RMS norm and a SiLU. Inductor made two kernels:
the matmul from its GEMM template, and one fused reduction kernel for the
norm and the activation. Both ran on the device, and the result matches
eager PyTorch on the CPU.

`with_sort` is the control. `torch.sort` has no Triton lowering on this path,
so it stays on the host, and the script **reports** that rather than letting
it pass as offloaded. A path that silently computed part of a graph on the
host would make every performance number about it wrong.

## How the check works

`torch._inductor.utils.run_and_get_code` is the helper Inductor's own tests
use. It returns the generated wrapper source alongside the result, and the
two `tnca.inductor` functions read that source:

- `triton_kernels(code)` lists the Triton kernels the wrapper launches.
- `host_fallbacks(code)` lists anything computed on the host: an extern
  kernel, an ATen call, a C++ kernel.

The repository runs 24 graphs this way (`tools/r38/torch_tests.py`, `make
r38-torch`), from elementwise and softmax through `bmm`, an embedding,
`argmax`, `cumsum` and causal and decode attention. Every one must run with no fallback,
except the sort control, which must report one. With
`TNCA_TENSOR_DESCRIPTORS=1`, Inductor also loads and stores through tensor
descriptors where its rules allow; the same 24 graphs pass that way too.

## Tensor descriptors from PyTorch

```sh
TNCA_TENSOR_DESCRIPTORS=1 ~/torchenv/bin/python your_program.py
```

That turns on Inductor's tensor-descriptor code generation (the path that
targets TMA on NVIDIA GPUs) for CPU tensors. The kernels then use
`tl.make_tensor_descriptor`, which [09](09_tensor_descriptors.md) covers from
the host side.

Next: [11 — running on the U280](11_running_on_the_u280.md).
