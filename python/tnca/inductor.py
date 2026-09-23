"""PyTorch Inductor on TNCA: the integration, and only the integration.

    import tnca.inductor
    tnca.inductor.configure()
    compiled = torch.compile(model)      # CPU tensors; kernels run on TNCA

Everything upstream stays upstream: Dynamo captures, AOTAutograd decomposes,
Inductor fuses and generates Triton, Triton specialises and caches. What
this module sets, each through an upstream extension point:

  * `cpu_backend = "triton"`: Inductor generates Triton, not C++, for CPU
    tensors. The tnca Triton backend (tnca.triton_backend) serves that target
    and offloads the host tensors to the device.
  * Codegen heuristics registered for device type "cpu" through
    `torch._inductor.heuristics.registry.register_codegen_heuristic` -- the
    mechanism ROCm and XPU use for their own configs. They are Inductor's
    default pointwise and reduction heuristics with every block dimension
    capped at STAGE1_MAX_BLOCK: TNCA's compiler lowers one 16-lane vector
    register per tile row and a wider tile is not built yet
    (compiler_overview.md sec 12). A smaller block is only more programs --
    Inductor's kernels mask by xnumel/rnumel -- so this is the small-shape
    limit of doc/r38_pytorch_p1plus.md, not a numerical trade.
  * `triton.persistent_reductions = False`: a persistent reduction needs one
    block to span the whole reduced dimension, which the cap forbids; the
    looped form Inductor also generates has no such requirement.
  * GEMM through Inductor's Triton mm template, never through the ATen
    extern kernel (that would run on the host): `max_autotune_gemm` with
    `max_autotune_gemm_backends = "TRITON"`, and a template heuristic for
    device "cpu" registered through `register_template_heuristic` that offers
    exactly one configuration, 16x16x16 -- the MMA fragment (ISA sec 9C). One
    candidate means Inductor has nothing to benchmark, so the choice does not
    depend on simulator wall time.
  * `compile_threads = 1`: Inductor's parallel compile runs kernels in worker
    processes that never executed this function, so they would use the
    default heuristics. Compiling in this process keeps one configuration;
    the simulator, not compilation, is where the time goes.
"""
STAGE1_MAX_BLOCK = 16
_registered = False


def _clamp(configs, cap):
    out, seen = [], set()
    for c in configs:
        for k, v in list(c.kwargs.items()):
            if k.endswith('BLOCK') and isinstance(v, int):
                c.kwargs[k] = min(v, cap)
        key = tuple(sorted(c.kwargs.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _register(cap):
    global _registered
    if _registered:
        return
    from torch._inductor.heuristics.registry import register_codegen_heuristic
    from torch._inductor.heuristics.triton_codegen.pointwise import PointwiseHeuristic
    from torch._inductor.heuristics.triton_codegen.reduction import ReductionHeuristic

    @register_codegen_heuristic('pointwise', 'cpu')
    class TNCAPointwiseHeuristic(PointwiseHeuristic):
        """Inductor's default pointwise configs, block dimensions capped."""

        def get_configs(self, *args, **kwargs):
            return _clamp(super().get_configs(*args, **kwargs), cap)

    @register_codegen_heuristic('reduction', 'cpu')
    class TNCAReductionHeuristic(ReductionHeuristic):
        """Inductor's default looped-reduction configs, block dimensions capped."""

        def get_configs(self, **kwargs):
            return _clamp(super().get_configs(**kwargs), cap)

    import torch._inductor.heuristics.template.triton as tt
    from torch._inductor.heuristics.registry import register_template_heuristic
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.kernel.bmm import bmm_template

    def one_fragment(self):
        self.mm_configs = [tt.GemmConfig(16, 16, 16, 1, 1, group_m=1)]
        self.exhaustive_configs = list(self.mm_configs)

    @register_template_heuristic(mm_template.uid, 'cpu')
    @register_template_heuristic(bmm_template.uid, 'cpu')
    class TNCAMMTemplateConfigHeuristic(tt.CPUMMTemplateConfigHeuristic):
        """One 16x16x16 configuration: the MMA fragment."""

        def __init__(self):
            super().__init__()
            one_fragment(self)

    @register_template_heuristic(mm_template.uid, 'cpu', op_name='addmm')
    @register_template_heuristic(bmm_template.uid, 'cpu', op_name='baddbmm')
    class TNCAAddmmTemplateConfigHeuristic(tt.CPUAddmmTemplateConfigHeuristic):
        """addmm/baddbmm: the same single configuration."""

        def __init__(self):
            super().__init__()
            one_fragment(self)

    _registered = True


def configure(max_block=STAGE1_MAX_BLOCK):
    import os
    os.environ.setdefault('TNCA_TRITON', '1')
    import torch._inductor.config as config
    config.cpu_backend = 'triton'
    config.triton.persistent_reductions = False
    config.compile_threads = 1
    config.max_autotune_gemm = True
    config.max_autotune_gemm_backends = 'TRITON'
    config.autotune_fallback_to_aten = False
    _register(max_block)
    return dict(cpu_backend=config.cpu_backend, max_block=max_block,
                persistent_reductions=config.triton.persistent_reductions,
                compile_threads=config.compile_threads,
                max_autotune_gemm_backends=config.max_autotune_gemm_backends)
