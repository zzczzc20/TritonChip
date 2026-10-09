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


_NO_X_DIM = [False]


def _install_no_x_dim():
    """Let Inductor drop the x dimension of a looped reduction.

    Inductor generates a reduction without its x dimension -- 1-D blocks
    `[R0_BLOCK]` instead of `[XBLOCK, R0_BLOCK]` -- only for a persistent
    reduction whose configuration fixes XBLOCK = 1 (`want_no_x_dim`). Every
    TNCA reduction runs XBLOCK = 1 (TNCAReductionHeuristic), looped or not,
    so the x dimension of a looped one is a unit dimension too. Without it
    the row is an ordinary 1-D vector to the compiler: a contiguous load and
    lane arithmetic, where `[1, R0_BLOCK]` went through the 2-D tile path
    (an index tile per use, a gather, a pass through scratch).
    """
    import torch._inductor.codegen.triton as T
    if getattr(T.TritonKernel.want_no_x_dim, '_tnca', False):
        return
    orig = T.TritonKernel.want_no_x_dim

    def want_no_x_dim(self):
        if orig(self):
            return True
        from torch._inductor.virtualized import V
        return bool(_NO_X_DIM[0] and V.graph.get_current_device_or_throw().type == 'cpu'
                    and self.inside_reduction
                    and len(self.numels) == self.num_reduction_dims + 1)
    want_no_x_dim._tnca = True
    T.TritonKernel.want_no_x_dim = want_no_x_dim


def configure(max_block=None, tensor_descriptors=None, no_x_dim=None):
    """Point Inductor's CPU code generation at TNCA.

    `tensor_descriptors` turns on Inductor's `use_tensor_descriptor` code
    generation, so its kernels load and store through
    `tl.make_tensor_descriptor` (rank 1-5) instead of pointer arithmetic --
    the same code Inductor generates for TMA on NVIDIA GPUs. It is ON by
    default: since R39 this is the standard path every performance number
    comes from (doc/r39_tasks_and_acceptance.md sec 0.1).
    TNCA_TENSOR_DESCRIPTORS=0 turns it off; off, kernels use pointers.

    `max_block` caps every block dimension (default: TNCA_MAX_BLOCK, else
    STAGE1_MAX_BLOCK). Both are knobs an ablation sets through the
    environment, and both are in the dict this returns, which ArchBench
    records with every point.

    `no_x_dim` (TNCA_NO_X_DIM) generates looped reductions without their
    unit x dimension (R39-02, _install_no_x_dim).
    """
    import os
    os.environ.setdefault('TNCA_TRITON', '1')
    if tensor_descriptors is None:
        tensor_descriptors = os.environ.get('TNCA_TENSOR_DESCRIPTORS', '1') != '0'
    if max_block is None:
        max_block = int(os.environ.get('TNCA_MAX_BLOCK', STAGE1_MAX_BLOCK))
    if no_x_dim is None:
        # On for the standard path since R39-02a's ablation passed.
        no_x_dim = os.environ.get('TNCA_NO_X_DIM', '1') != '0'
    _NO_X_DIM[0] = bool(no_x_dim)
    _install_no_x_dim()
    import torch._inductor.config as config
    config.cpu_backend = 'triton'
    if tensor_descriptors:
        _enable_tensor_descriptors(config)
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
    out = dict(cpu_backend=config.cpu_backend, max_block=max_block,
               persistent_reductions=config.triton.persistent_reductions,
               compile_threads=config.compile_threads,
               max_autotune_gemm_backends=config.max_autotune_gemm_backends,
               tensor_descriptors=bool(tensor_descriptors),
               no_x_dim=bool(no_x_dim))
    return out


def _enable_tensor_descriptors(config):
    """Inductor's tensor-descriptor path for CPU tensors (TMA on NVIDIA).

    Inductor offers it for device "cpu" when a Triton backend is registered
    under the name `cpu` and the stable descriptor API exists
    (TMACompatibilityChecker.can_use_tma). TNCA's plugin is registered under
    that name (pyproject.toml). Both checks are functools-cached, so a value
    computed before this ran is cleared.
    """
    import triton
    if 'cpu' not in triton.backends.backends:
        raise RuntimeError(
            "TNCA's Triton plugin is not registered as Triton's `cpu` backend, "
            "so Inductor will not generate tensor descriptors for it. Reinstall "
            "the package (pip install -e .) so its entry point is current.")
    import torch.utils._triton as tu
    for name in ('has_triton_stable_tma_api', 'has_triton_tma_device',
                 'has_triton_tensor_descriptor_host_tma'):
        getattr(getattr(tu, name, None), 'cache_clear', lambda: None)()
    config.triton.use_tensor_descriptor = True
    # A descriptor's base is the buffer pointer; Inductor makes every input
    # 16-byte aligned (copying one that is not), which is the TMA contract
    # and more than the transfer engine needs.
    config.assume_aligned_inputs = True
    _apply_tma_rules()


# Element types a descriptor may carry into the TNCA compiler
# (tnca_cc.descriptors): the value model's 1, 2 and 4-byte types.
def _descriptor_dtypes():
    import torch
    return {torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int8,
            torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}


def _apply_tma_rules():
    """Decide descriptor eligibility for TNCA by Inductor's own CUDA rules.

    Inductor's TMACompatibilityChecker returns early for device "cpu"
    ("the 16-byte rules do not apply"), but Triton's make_tensor_descriptor
    enforces at least 16 bytes in the last block dimension on every backend
    -- so that branch generates kernels Triton then refuses (seen: a
    reduction's output store with XBLOCK = 1). This subclass, installed
    through TritonKernel.tma_compatibility_checker_cls, evaluates the
    CUDA branch instead: the same alignment, minimum-block and store rules,
    so a graph gets descriptors on TNCA exactly where it would on an NVIDIA
    GPU. It also keeps to the element types the TNCA compiler takes through
    a descriptor. Installed only by configure(tensor_descriptors=True).
    """
    from torch._inductor.codegen import triton as T
    base = T.TritonKernel.tma_compatibility_checker_cls
    if getattr(base, '_tnca_rules', False):
        return
    import torch
    from torch._inductor.virtualized import V

    def on_cpu():
        return V.graph.get_current_device_or_throw().type == 'cpu'

    import os
    bisect = os.environ.get('TNCA_TDESC_BISECT')
    lo, hi = (int(x) for x in bisect.split(':')) if bisect else (0, None)

    class TNCATMACompatibilityChecker(base):
        _tnca_rules = True
        # TNCA_TDESC_BISECT=lo:hi -- a debugging knob, not a configuration:
        # the eligible loads and stores of a graph are numbered in the order
        # Inductor asks about them (deterministic for a given graph), and
        # only those numbered lo..hi-1 get a descriptor. A binary search over
        # the range finds the one access whose descriptor lowering is wrong.
        _seen = 0

        def can_use_tma(self):
            if self.force or not on_cpu():
                return super().can_use_tma()
            if not super().can_use_tma():
                return False
            if self.dtype not in _descriptor_dtypes():
                return False
            multirow = getattr(self.kernel.features, 'has_strict_sum_multirow_reduction',
                               lambda: False)
            if self.for_store and (self.kernel.no_x_dim or multirow()):
                return False
            if bisect:
                n = TNCATMACompatibilityChecker._seen
                TNCATMACompatibilityChecker._seen += 1
                if n < lo or (hi is not None and n >= hi):
                    return False
            return True

        def are_block_parameters_compatible(self, block_params, constant_offset=0):
            if not on_cpu():
                return super().are_block_parameters_compatible(block_params, constant_offset)
            # One range tree split over two descriptor dimensions -- Inductor
            # matching `y // 4` and `y % 4` as dimensions of their own, a
            # block shape [(YBLOCK + 3)//4, XBLOCK, Min(4, YBLOCK)] -- makes
            # the kernel reshape its tile to rank 4 and permute it around the
            # access. The importer has no lowering for that, and the box rows
            # it would move are a quarter or half a vector wide, where the
            # pointer access Inductor falls back to moves one tile row per
            # instruction.
            seen = set()
            for dim in block_params.block_shape:
                names = {x.name for x in getattr(dim, 'free_symbols', ())}
                if names & seen:
                    return False
                seen |= names
            # TNCA runs a reduction one row per program (XBLOCK = 1, see
            # _register): Inductor's no_x_dim in all but name. A descriptor
            # whose innermost block is XBLOCK can then never span 16 bytes,
            # and the minimum Inductor would record for it is clamped away.
            from torch.utils._sympy.symbol import SymT, symbol_is_type
            inner = block_params.block_shape[-1]
            inner_syms = getattr(inner, 'free_symbols', ())
            if any(t.is_reduction for t in self.kernel.range_trees) and \
                    any(symbol_is_type(x, SymT.XBLOCK) for x in inner_syms):
                return False
            graph = V.graph
            graph.get_current_device_or_throw = lambda: torch.device('cuda')
            try:
                return super().are_block_parameters_compatible(block_params, constant_offset)
            finally:
                del graph.get_current_device_or_throw

    # TritonKernel's hook for exactly this (a class attribute subclasses set).
    T.TritonKernel.tma_compatibility_checker_cls = TNCATMACompatibilityChecker


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
