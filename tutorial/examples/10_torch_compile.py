"""Tutorial 10: a PyTorch function through torch.compile -> Inductor -> TNCA.

`tnca.inductor.configure()` points Inductor's code generation for CPU tensors
at Triton and the tnca Triton backend; after that, `torch.compile` is plain
PyTorch. Every kernel Inductor generates runs on the simulated device, and
anything it leaves on the host is a fallback this script reports rather than
hides. The answer is checked against eager PyTorch on the CPU.

    ~/torchenv/bin/tnca-sim prepare          # once: build the simulated device
    ~/torchenv/bin/python tutorial/examples/10_torch_compile.py
"""
import sys

import torch

import tnca.inductor

config = tnca.inductor.configure()

from torch._inductor.utils import run_and_get_code   # noqa: E402  (after configure)


def mlp_block(x, w, g):
    """A matmul, an RMS norm and a SiLU: three of the shapes a transformer has."""
    h = x @ w
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * g
    return torch.nn.functional.silu(h)


def with_sort(x):
    """torch.sort has no Triton lowering on this path, so it stays on the host."""
    return torch.sort(x, dim=-1).values


def main():
    print("Inductor, as configured for TNCA:")
    for k in ('cpu_backend', 'max_block', 'persistent_reductions',
              'max_autotune_gemm_backends'):
        print(f"  {k:<28s} {config[k]}")

    torch.manual_seed(0)
    x, w, g = torch.randn(16, 48), torch.randn(48, 16), torch.randn(16)
    want = mlp_block(x, w, g)
    got, code = run_and_get_code(torch.compile(mlp_block), x, w, g)
    kernels = tnca.inductor.triton_kernels(code)
    fallbacks = tnca.inductor.host_fallbacks(code)
    err = (got - want).abs().max().item()
    tol = 2 ** -14 * max(want.abs().max().item(), 1.0)
    print("\nmlp_block, compiled:")
    print(f"  Triton kernels run on TNCA  {kernels}")
    print(f"  host fallbacks              {fallbacks or 'none'}")
    print(f"  max |compiled - eager|      {err:.3g}  (tolerance {tol:.3g})")
    ok = (err <= tol) and not fallbacks and bool(kernels)

    torch._dynamo.reset()
    x = torch.randn(3, 8)
    got, code = run_and_get_code(torch.compile(with_sort), x)
    fallbacks = tnca.inductor.host_fallbacks(code)
    print("\nwith_sort, compiled:")
    print(f"  host fallbacks              {fallbacks}")
    print(f"  equal to eager              {torch.equal(got, torch.sort(x, dim=-1).values)}")
    ok &= bool(fallbacks) and torch.equal(got, torch.sort(x, dim=-1).values)

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
