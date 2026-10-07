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
  * Reductions run ONE ROW PER PROGRAM (XBLOCK = 1): a row is one vector
    register here, and it keeps every per-row value -- an argmax index, an
    embedding's token id -- a one-row block the importer holds in a register.
  * Convolution through Inductor's Triton conv template, with one 16x16x16
    configuration registered on the device's config heuristic.
  * `scaled_dot_product_attention` on a CPU tensor is decomposed (upstream's
    own math decomposition of `_scaled_dot_product_flash_attention_for_cpu`)
    instead of being kept as a host call.
  * No compiled-graph cache (`fx_graph_cache`, AOTAutograd cache): their keys
    do not cover the registrations above. No autotuning benchmarks
    (`benchmark_epilogue_fusion`): the compiled graph never depends on host
    timing.

`host_fallbacks(code)` and `triton_kernels(code)` read Inductor's generated
wrapper: what stayed on the host, and which kernels were generated.
"""
STAGE1_MAX_BLOCK = 16
_registered = False


def _clamp(configs, cap, rows=None):
    """ONE configuration: Inductor's candidates with every block dimension
    capped at `cap` (non-reduction dimensions at `rows` when given), and of
    those the one with the largest blocks.

    One, because Inductor chooses among several by launching each and timing
    it. On this device a launch is a simulation (or a board run), so the
    choice would depend on host speed, compile time would grow with every
    candidate, and the benchmark launches would sit in the same launch log
    as the graph's own -- a compile-only sweep counted 39 338 of them for one
    attention call. With one candidate there is nothing to benchmark.
    """
    best, best_size = None, -1
    for c in configs:
        for k, v in list(c.kwargs.items()):
            if k.endswith('BLOCK') and isinstance(v, int):
                lim = rows if (rows is not None and k[0] in 'XYZ') else cap
                c.kwargs[k] = min(v, lim)
        size = 1
        for k, v in c.kwargs.items():
            if k.endswith('BLOCK') and isinstance(v, int):
                size *= v
        if size > best_size:
            best, best_size = c, size
    return [best] if best is not None else []


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
        """Inductor's default looped-reduction configs: one row per program
        (XBLOCK = 1), the reduced dimension in blocks of `cap`.

        One row is one vector register wide on this machine, and a program
        reduces it while a grid of programs covers the rows. Several rows
        per program would only stack registers the core works through one
        after another anyway, and it makes every per-row value -- an
        argmax's index, an embedding row's id -- a 2-D block where one
        element per program is what the computation has."""

        def get_configs(self, **kwargs):
            return _clamp(super().get_configs(**kwargs), cap, rows=1)

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

    # Convolution through Inductor's Triton conv template, with the one
    # configuration a 16-lane machine can build (the template's own list
    # starts at 64x256 blocks). Through the device's config heuristic, the
    # object `V.choices.get_conv_configs("cpu")` reads.
    from torch._inductor.heuristics.template.triton import ConvConfig
    from torch._inductor.choices import InductorChoices
    cpu_heuristic = InductorChoices().get_config_heuristics('cpu')
    cpu_heuristic.conv_configs = [ConvConfig(cap, cap, cap, 1, 1)]

    _registered = True


def _decompose_cpu_attention():
    """scaled_dot_product_attention on a CPU tensor reaches Inductor as
    `aten._scaled_dot_product_flash_attention_for_cpu`, which Inductor
    keeps as a host call. Upstream's decomposition of that op (the one
    torch.export uses) is the math formulation -- matmul, softmax, matmul
    -- which Inductor then generates as Triton like any other graph. Added
    to Inductor's own decomposition table, so it applies to CPU tensors
    only: a CUDA tensor never reaches the `_for_cpu` op."""
    import torch
    from torch._decomp import decomposition_table
    from torch._inductor.decomposition import decompositions
    op = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default
    if op not in decompositions and op in decomposition_table:
        decompositions[op] = decomposition_table[op]


def configure(max_block=STAGE1_MAX_BLOCK):
    import os
    os.environ.setdefault('TNCA_TRITON', '1')
    import torch._inductor.config as config
    config.cpu_backend = 'triton'
    config.triton.persistent_reductions = False
    config.compile_threads = 1
    config.max_autotune_gemm = True
    config.max_autotune_gemm_backends = 'TRITON'
    # Convolution likewise: the Triton template is the only candidate, so
    # there is no ATen (host) choice to fall back to and nothing to time.
    config.max_autotune_conv_backends = 'TRITON'
    config.autotune_fallback_to_aten = False
    # No benchmarking. With max-autotune on, Inductor times a template with
    # and without a fused epilogue before deciding; on this device that
    # would time the simulator (and the CPU device interface cannot time at
    # all). One configuration per template and no timing means the compiled
    # graph never depends on how fast the host was.
    config.benchmark_epilogue_fusion = False
    # No compiled-graph cache. Its key covers Inductor's config but not the
    # decompositions and heuristics registered above, so a graph compiled
    # before configure() -- or before a change here -- would be served
    # again with its host fallbacks in it. Kernels are still cached, by
    # Triton, under a key that covers the TNCA compiler sources.
    config.fx_graph_cache = False
    config.fx_graph_remote_cache = False
    import torch._functorch.config as functorch_config
    functorch_config.enable_autograd_cache = False
    _register(max_block)
    _decompose_cpu_attention()
    return dict(cpu_backend=config.cpu_backend, max_block=max_block,
                persistent_reductions=config.triton.persistent_reductions,
                compile_threads=config.compile_threads,
                max_autotune_gemm_backends=config.max_autotune_gemm_backends)


# ---------------------------------------------------------------------------
# What ran where. Inductor's generated wrapper (torch._inductor.utils.
# run_and_get_code) is the record of a compiled graph: every Triton kernel it
# launches is a TNCA launch, and any computation it leaves on the host -- an
# extern kernel, an ATen call, a C++ kernel -- is a FALLBACK. Buffer
# allocation, reinterpretation and views are not computation and are not
# listed.
# ---------------------------------------------------------------------------
import re as _re

FALLBACK_PATTERNS = [
    (_re.compile(r'\bextern_kernels\.\w+'), 'extern kernel'),
    (_re.compile(r'\btorch\.ops\.aten\.(?!empty|empty_strided|as_strided|reinterpret)\w+'), 'ATen op'),
    (_re.compile(r'\baten\.(?!empty|empty_strided|as_strided)\w+\('), 'ATen op'),
    (_re.compile(r'\bcpp_fused\w*'), 'C++ kernel'),
]


def _executed(src):
    """The wrapper as it runs: no comments (Inductor annotates each kernel
    launch with the ATen ops it replaced), and nothing from the benchmark
    harness Inductor appends after `def benchmark_compiled_module`."""
    src = src.split('def benchmark_compiled_module', 1)[0]
    return '\n'.join(l for l in src.splitlines() if not l.lstrip().startswith('#'))


def host_fallbacks(code):
    """Host-side computation in the generated wrapper source(s), sorted."""
    found = []
    for src in map(_executed, code):
        for pat, what in FALLBACK_PATTERNS:
            for m in pat.finditer(src):
                found.append(f'{what}: {m.group(0)}')
    return sorted(set(found))


def triton_kernels(code):
    """Names of the Triton kernels the generated code defines."""
    return sorted({m.group(1) for src in code
                   for m in _re.finditer(r'def (triton_\w+)\(', src)})
