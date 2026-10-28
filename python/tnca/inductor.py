"""PyTorch Inductor on TNCA: the integration, and only the integration.

    import tnca.inductor
    tnca.inductor.configure()
    compiled = torch.compile(model)      # CPU tensors; kernels run on TNCA

Everything upstream stays upstream: Dynamo captures, AOTAutograd decomposes,
Inductor fuses and generates Triton, Triton specialises and caches. What
this module sets, each through an upstream extension point:

  * `cpu_backend = "triton"`: Inductor generates Triton, not C++, for CPU
    tensors. The tnca Triton backend (tnca.triton_backend) serves that target
    and offloads the host tensors to the device. On the native `tnca` device
    (TNCA_TORCH_DEVICE=native) CPU tensors are host data instead and keep
    Inductor's C++ backend (see configure()).
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
  * No decompose-K GEMM on TNCA device types (`TNCAChoices`, through
    upstream's `inductor_choices_class`): for a skinny GEMM (K >= 64 *
    max(M, N), e.g. a 1 x 2048 x 16 decode projection) Inductor offers only
    its decompose-K subgraph, which splits K into a batched GEMM plus a sum.
    TNCA has no configuration for it, so the graph had no choice at all
    (NoValidChoicesError), and a split K would change the per-K FP32 FMA
    order every TNCA GEMM keeps (R41 sec 2.2). The mm template, whose own
    conditions admitted the shape, is offered instead.
  * No split scans on TNCA device types (`_install_unsplit_scan`): Inductor's
    split scan is a decoupled look-back across programs (cross-program
    atomics and spin waits) whose R0_BLOCK is raised to
    `min_split_scan_rblock` = 256 after the device heuristic, which TNCA
    refuses; it also combines block partials in another order. A scan stays
    one looped scan per row, the conservative schedule.
  * No compiled-graph cache (`fx_graph_cache`, AOTAutograd cache): their keys
    do not cover the registrations above. No autotuning benchmarks
    (`benchmark_epilogue_fusion`): the compiled graph never depends on host
    timing.

`host_fallbacks(code)` and `triton_kernels(code)` read Inductor's generated
wrapper: what stayed on the host, and which kernels were generated.
"""
import os
import time

STAGE1_MAX_BLOCK = 16
# With TNCA_TILED_POINTWISE (default 0), a pointwise kernel's block may grow to
# this many lanes; tnca_cc/pointwise_tiles.py runs it as a loop over register
# tiles, and a kernel it cannot tile falls back to the 16-lane configuration.
TILED_POINTWISE_BLOCK = 1024
# The device types whose Triton kernels TNCA runs: `cpu` (CPU tensors, the
# R39 migration path, B39) and `tnca` (the native device, R40-01). Both get
# the same heuristics, templates and descriptor rules.
TNCA_DEVICE_TYPES = ('cpu', 'tnca')
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


def _with_fallback(configs, cap, rows=None):
    """_clamp's one configuration, followed, when it is wider than the
    standard 16 lanes, by the 16-lane configuration as a fallback.

    Inductor compiles every configuration it is given; _install_block_fallback
    keeps exactly one -- the first that compiles and whose code fits the
    device code store -- so nothing is ever benchmarked. At the default cap
    there is one configuration, as before."""
    import copy
    narrow = _clamp(copy.deepcopy(configs), STAGE1_MAX_BLOCK, rows=rows) if cap > STAGE1_MAX_BLOCK else []
    wide = _clamp(configs, cap, rows=rows)
    if narrow and wide and narrow[0].kwargs != wide[0].kwargs:
        return wide + narrow
    return wide


def _register(cap, reduction_rows=1, gemm_tile=(16,16,16)):
    global _registered
    if _registered:
        return
    from torch._inductor.heuristics.registry import register_codegen_heuristic
    from torch._inductor.heuristics.triton_codegen.pointwise import PointwiseHeuristic
    from torch._inductor.heuristics.triton_codegen.reduction import ReductionHeuristic

    @register_codegen_heuristic('pointwise', 'tnca')
    @register_codegen_heuristic('pointwise', 'cpu')
    class TNCAPointwiseHeuristic(PointwiseHeuristic):
        """Inductor's default pointwise configs, block dimensions capped."""

        def get_configs(self, *args, **kwargs):
            import copy
            original = super().get_configs(*args, **kwargs)
            configs = _with_fallback(copy.deepcopy(original), _POINTWISE_CAP[0] or cap)
            # Padding and multi-axis gathers can exhaust scalar/predicate
            # registers even at VLEN. Retile the same pointwise program,
            # keeping the first legal configuration without benchmarking.
            for limit in (8, 4, 1):
                smaller = _clamp(copy.deepcopy(original), min(cap, limit))
                if smaller and not any(c.kwargs == smaller[0].kwargs for c in configs):
                    configs.extend(smaller)
            return configs

    @register_codegen_heuristic('reduction', 'tnca')
    @register_codegen_heuristic('reduction', 'cpu')
    class TNCAReductionHeuristic(ReductionHeuristic):
        """Looped reductions with independently bounded row and lane blocks."""

        def get_configs(self, **kwargs):
            # A scan or sort has no wide register-block lowering
            # (tnca_cc.register_blocks): it keeps the standard 16 lanes,
            # the semantically equivalent retile, instead of failing.
            narrow = (kwargs.get('inductor_meta') or {}).get('tnca_narrow_block')
            lanes = min(cap, STAGE1_MAX_BLOCK) if narrow else cap
            return _with_fallback(super().get_configs(**kwargs), lanes, rows=reduction_rows)

    import torch._inductor.heuristics.template.triton as tt
    from torch._inductor.heuristics.registry import register_template_heuristic
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.kernel.bmm import bmm_template

    def one_fragment(self):
        # These dimensions describe legal TNCA fragments. The inherited
        # CPU heuristic halves them for CPU compile cost, which would make
        # the requested wide logical tile disappear before TTIR generation.
        self.should_scale_configs = False
        self.mm_configs = [tt.GemmConfig(*gemm_tile, 1, 1, group_m=1)]
        self.exhaustive_configs = list(self.mm_configs)

    @register_template_heuristic(mm_template.uid, 'tnca')
    @register_template_heuristic(bmm_template.uid, 'tnca')
    @register_template_heuristic(mm_template.uid, 'cpu')
    @register_template_heuristic(bmm_template.uid, 'cpu')
    class TNCAMMTemplateConfigHeuristic(tt.CPUMMTemplateConfigHeuristic):
        """One legal logical tile, lowered into physical MMA fragments."""

        def __init__(self):
            super().__init__()
            one_fragment(self)

    @register_template_heuristic(mm_template.uid, 'tnca', op_name='addmm')
    @register_template_heuristic(bmm_template.uid, 'tnca', op_name='baddbmm')
    @register_template_heuristic(mm_template.uid, 'cpu', op_name='addmm')
    @register_template_heuristic(bmm_template.uid, 'cpu', op_name='baddbmm')
    class TNCAAddmmTemplateConfigHeuristic(tt.CPUAddmmTemplateConfigHeuristic):
        """addmm/baddbmm: the same single configuration."""

        def __init__(self):
            super().__init__()
            one_fragment(self)

    @register_template_heuristic(mm_template.uid, 'tnca', op_name='scaled_mm')
    @register_template_heuristic(mm_template.uid, 'cpu', op_name='scaled_mm')
    class TNCAScaledMMTemplateConfigHeuristic(tt.CPUScaledMMTemplateConfigHeuristic):
        """`aten._scaled_mm` through Inductor's own lowering (tuned_scaled_mm).

        Upstream's scaled-mm mixin, unchanged: the same mm template with
        its epilogue scaling (`acc * (scale_a * scale_b)`, then the bias)
        and the scales and bias as suffix arguments, for tensorwise and
        rowwise scales. Without this registration the device's plain mm
        heuristic answered for op `scaled_mm`, so the template was
        rendered without the suffix arguments and failed. Its tile is the
        same single legal configuration as mm's; the FP8 operands reach
        `tl.dot` as they are stored, which the compiler lowers onto the
        FP8 MMA (ISA sec 9C, CAPABILITY0.MATRIX_FP8).
        """

        def __init__(self):
            super().__init__()
            one_fragment(self)

        def _valid(self, kernel_inputs):
            # Upstream offers no configuration when K <= 16 ("Triton
            # crashes"), and B200's tcgen05 needs K >= 32: both are CUDA
            # code-generation limits. A K of 16 or less is one masked
            # 16-deep MMA fragment here, which the compiler lowers like
            # any other K tail; refusing it would leave the graph with no
            # choice at all (there is no ATen fallback on this device).
            return True

    # Convolution through Inductor's Triton conv template, with the one
    # configuration a 16-lane machine can build (the template's own list
    # starts at 64x256 blocks). Through the device's config heuristic, the
    # object `V.choices.get_conv_configs("cpu")` reads.
    from torch._inductor.heuristics.template.triton import ConvConfig
    from torch._inductor.choices import InductorChoices
    for dev in ('cpu', 'tnca'):
        InductorChoices().get_config_heuristics(dev).conv_configs = [ConvConfig(cap, cap, cap, 1, 1)]

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


def _choices_class():
    """InductorChoices for TNCA device types: no decompose-K GEMM.

    `tuned_mm` offers decompose-K (a K-split batched GEMM and a sum) when
    K >= 32 * max(M, N), and from 64 * max(M, N) it offers decompose-K
    INSTEAD of the mm template. No TNCA heuristic serves decompose-K, so
    above that ratio the GEMM had no choice, and in between it was offered
    and dropped. Here the decompose-K subgraph is removed for TNCA device
    types and the mm template takes its place when it was left out. The mm
    template was inside the same `use_triton_template` condition, so this
    only undoes the "Triton is slow for large K" preference -- a CUDA
    performance rule -- and keeps the per-K FP32 FMA accumulation of every
    other TNCA GEMM. Other devices get upstream's choices unchanged.
    """
    from torch._inductor.choices import InductorChoices

    class TNCAChoices(InductorChoices):
        def uuid(self):
            return 'tnca-choices-1'

        def get_template_configs(self, kernel_inputs, templates, op_name, kwarg_overrides=None):
            if getattr(kernel_inputs, 'device_type', None) in TNCA_DEVICE_TYPES:
                templates = _without_decompose_k(templates)
            return super().get_template_configs(kernel_inputs, templates, op_name, kwarg_overrides)

    return TNCAChoices


def _without_decompose_k(templates):
    from torch._inductor.kernel.mm import decompose_k_subgraph_template, mm_template
    uids = [getattr(t, 'uid', None) for t in templates]
    if decompose_k_subgraph_template.uid not in uids:
        return templates
    at = uids.index(decompose_k_subgraph_template.uid)
    rest = [t for t in templates if getattr(t, 'uid', None) != decompose_k_subgraph_template.uid]
    if mm_template.uid not in uids:
        rest.insert(at, mm_template)
    return rest


def _install_choices(config):
    """Install TNCAChoices through upstream's `inductor_choices_class`."""
    current = config.inductor_choices_class
    if current is not None and getattr(current, '_tnca', False):
        return
    if current is not None:
        raise RuntimeError(f'inductor_choices_class is already set ({current!r}); '
                           'TNCA needs its own to keep decompose-K off its devices')
    cls = _choices_class()
    cls._tnca = True
    config.inductor_choices_class = cls


def _install_unsplit_scan():
    """One looped scan per row on TNCA device types: never a split scan.

    Inductor's TritonSplitScanKernel splits a long scan across programs that
    publish block sums and wait on their predecessors (decoupled look-back:
    cross-program atomics and spin loops), and raises its R0_BLOCK to
    `min_split_scan_rblock` (256) after the device heuristic, wider than any
    TNCA register block. It also adds the block partials in a different
    order from the sequential scan. With one split the scan is Inductor's
    ordinary looped scan, which the reduction heuristic sizes like any other
    row. Only the split count of TNCA devices changes.
    """
    import torch._inductor.ir as ir
    upstream = ir.Scan.num_splits.__func__
    if getattr(upstream, '_tnca', False):
        return

    def num_splits(cls, *args, **kwargs):
        hint, splits = upstream(cls, *args, **kwargs)
        device = kwargs.get('device', args[0] if args else None)
        if getattr(device, 'type', None) in TNCA_DEVICE_TYPES:
            return hint, 1
        return hint, splits
    num_splits._tnca = True
    ir.Scan.num_splits = classmethod(num_splits)


_NO_X_DIM = [False]
_BLOCK_CAP = [STAGE1_MAX_BLOCK]
_POINTWISE_CAP = [0]   # 0: the block cap; set by configure() from TNCA_TILED_POINTWISE


def _install_narrow_blocks():
    """Mark kernels whose operations have no wide register-block lowering.

    tnca_cc.register_blocks splits wide values only for its listed operations;
    tt.scan and tt.sort are not among them. A kernel that generates one
    carries `tnca_narrow_block` in its inductor_meta, and the reduction
    heuristic gives it the standard 16 lanes. The key is emitted only when
    the block cap exceeds 16, so the generated source at the default cap is
    unchanged.
    """
    import torch._inductor.codegen.triton as T
    if getattr(T.TritonKernel.inductor_meta_per_kernel, '_tnca', False):
        return

    def marking(name):
        orig = getattr(T.TritonKernel, name)

        def op(self, *args, **kwargs):
            self.__dict__.setdefault('_tnca_narrow_ops', set()).add(name)
            return orig(self, *args, **kwargs)
        op._tnca = True
        setattr(T.TritonKernel, name, op)
    for name in ('scan', 'sort'):
        marking(name)
    # Index-carrying reductions fold a value and its index together; their
    # wide lowering is slower than 16 lanes (op.argmax/medium 0.61x on the
    # U280 at TNCA_MAX_BLOCK=64), so they keep the standard width too.
    orig_reduction = T.TritonKernel.reduction

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if reduction_type in ('argmax', 'argmin'):
            self.__dict__.setdefault('_tnca_narrow_ops', set()).add(reduction_type)
        return orig_reduction(self, dtype, src_dtype, reduction_type, value)
    reduction._tnca = True
    T.TritonKernel.reduction = reduction
    orig_meta = T.TritonKernel.inductor_meta_per_kernel

    def inductor_meta_per_kernel(self):
        out = orig_meta(self)
        ops = self.__dict__.get('_tnca_narrow_ops')
        if ops and _BLOCK_CAP[0] > STAGE1_MAX_BLOCK:
            out['tnca_narrow_block'] = sorted(ops)
        return out
    inductor_meta_per_kernel._tnca = True
    T.TritonKernel.inductor_meta_per_kernel = inductor_meta_per_kernel


# The code store of the device: 16,384 words (rtl/tnca_multicore_shell.v
# CODE_WORDS; the runtime's region below heap_base). A wide kernel must leave
# room for the runtime's own modules, so a fallback is taken above 112 KiB.
CODE_STORE_BYTES = 0x20000
WIDE_CODE_LIMIT = CODE_STORE_BYTES - 0x4000


def _resource_plan(result):
    """The execution resource plan sealed with a compiled TNCA kernel, or None.

    The common resource description (tnca_cc.execution_resources): retile,
    refusal and tuning read their numbers here instead of recomputing them.
    """
    import os
    import tempfile
    blob = (getattr(getattr(result, 'kernel', None), 'asm', None) or {}).get('tnca')
    if not blob:
        return None
    from . import artifact
    from .aot import plan_of
    fd, path = tempfile.mkstemp(suffix='.tnca')
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(blob)
        manifest, _segments = artifact.load(path)
    finally:
        os.unlink(path)
    return plan_of(manifest)


def _code_bytes(result):
    """The code bytes of a compiled TNCA kernel, from its resource plan, or None."""
    plan = _resource_plan(result)
    return None if plan is None else plan['resources']['code_bytes']


def _install_block_fallback():
    """Keep the first legal TNCA configuration, including resource retiles.

    The first is used if it compiles and its code fits WIDE_CODE_LIMIT;
    otherwise the 16-lane one -- the attributable retile v1 sec 2.2 asks for
    instead of a launch that cannot be loaded. Only TNCA kernels with more
    than one configuration are touched; every other autotuner (CUDA in the
    same process) runs Inductor's own code."""
    from torch._inductor.runtime import triton_heuristics as H
    if getattr(H.CachingAutotuner._precompile_worker, '_tnca', False):
        return
    orig = H.CachingAutotuner._precompile_worker

    def _precompile_worker(self):
        configs = self.configs
        started = time.monotonic()
        if (os.environ.get('TNCA_TUNE', '0') == '1' and configs and
                getattr(self.device_props, 'type', None) == 'tnca'):
            self._tnca_tuning_configs = list(configs)
            if all(c.kwargs and set(c.kwargs) <= {'XBLOCK', 'YBLOCK', 'ZBLOCK'} for c in configs):
                # A maximum width is a search bound, not permission to use a
                # wide baseline on budget expiry or an inconclusive result.
                conservative = [c for c in configs if all(v <= STAGE1_MAX_BLOCK for v in c.kwargs.values())]
                if conservative:
                    configs = self.configs = conservative
        if (self.compile_results or not configs or len(configs) < 2
                or getattr(self.device_props, 'type', None) not in TNCA_DEVICE_TYPES):
            try:
                return orig(self)
            finally:
                self._tnca_baseline_compile_seconds = time.monotonic()-started
        reasons = []
        for i, config in enumerate(configs):
            last = i == len(configs) - 1
            try:
                result = self._precompile_config(config)
            except Exception as e:                           # noqa: BLE001
                if last:
                    raise
                reasons.append(f'{config.kwargs}: {type(e).__name__}: {str(e)[:200]}')
                continue
            plan = _resource_plan(result)
            size = None if plan is None else plan['resources']['code_bytes']
            if not last and size is not None and size > WIDE_CODE_LIMIT:
                reasons.append(f'{config.kwargs}: code {size} B exceeds {WIDE_CODE_LIMIT} B '
                               f'(execution resource plan)')
                continue
            self.compile_results = [result]
            self.configs = None
            self._tnca_baseline_compile_seconds = time.monotonic()-started
            if reasons:
                _BLOCK_FALLBACKS.append(dict(kernel=self.inductor_meta.get('kernel_name'),
                                             used=dict(config.kwargs), reasons=reasons,
                                             used_resources=None if plan is None else plan['resources']))
            return
    _precompile_worker._tnca = True
    H.CachingAutotuner._precompile_worker = _precompile_worker
    upstream_cached_autotune = H.cached_autotune
    def cached_autotune(*call_args, **call_kwargs):
        triton_meta = call_args[2] if len(call_args) > 2 else call_kwargs.get('triton_meta', {})
        if (os.environ.get('TNCA_TUNE', '0') == '1' and
                getattr(triton_meta.get('device'), 'type', None) == 'tnca'):
            # Upstream's winner cache lacks TNCA image/clock/memory-map
            # identity. Keep the complete candidate set for our own cache.
            if len(call_args) > 5:
                call_args = list(call_args)
                call_args[5] = dict(call_args[5] or {}, force_disable_caches=True)
            else:
                call_kwargs['inductor_meta'] = dict(call_kwargs.get('inductor_meta') or {}, force_disable_caches=True)
        return upstream_cached_autotune(*call_args, **call_kwargs)
    H.cached_autotune = cached_autotune
    upstream_run = H.CachingAutotuner.run
    def run(self, *args, stream, benchmark_run=False, **kwargs):
        if (os.environ.get('TNCA_TUNE', '0') != '1' or benchmark_run or
                getattr(self.device_props, 'type', None) != 'tnca'):
            return upstream_run(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
        from .inductor_tuning import choose
        choose(self, args, stream=stream, kwargs=kwargs)
        return upstream_run(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
    H.CachingAutotuner.run = run


_BLOCK_FALLBACKS = []


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
        return bool(_NO_X_DIM[0] and V.graph.get_current_device_or_throw().type in TNCA_DEVICE_TYPES
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

    `TNCA_REDUCTION_ROWS` independently caps the row block (1/2/4/8/16,
    default 1). `no_x_dim` (TNCA_NO_X_DIM) removes the unit row dimension
    only when this cap is 1 (R39-02, _install_no_x_dim).
    """
    import os
    # TNCA's Triton driver is THE active driver only where there is no CUDA
    # device. With CUDA present (R40-01: CPU, CUDA and tnca tensors in one
    # process) it stays out of Triton's count of active drivers, and Inductor
    # selects it per kernel -- the `cpu` registry entry for CPU tensors and
    # tnca ones (tnca.torch_device.inductor_device) -- as it does for any
    # second Triton backend. TNCA_TRITON=1 set by the caller still forces it.
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:                                        # noqa: BLE001
        cuda = False
    if not cuda:
        os.environ.setdefault('TNCA_TRITON', '1')
    if tensor_descriptors is None:
        tensor_descriptors = os.environ.get('TNCA_TENSOR_DESCRIPTORS', '1') != '0'
    if max_block is None:
        max_block = int(os.environ.get('TNCA_MAX_BLOCK', STAGE1_MAX_BLOCK))
    if no_x_dim is None:
        # On for the standard path since R39-02a's ablation passed.
        no_x_dim = os.environ.get('TNCA_NO_X_DIM', '1') != '0'
    gemm_tile = tuple(int(n) for n in os.environ.get('TNCA_GEMM_TILE', '16,16,16').split(','))
    window_reuse = int(os.environ.get('TNCA_WINDOW_REUSE', '0'))
    if window_reuse not in (0, 1):
        raise ValueError('TNCA_WINDOW_REUSE must be 0 or 1')
    supported_tiles = ((16,16,16), (16,32,16), (16,64,16), (16,16,32), (16,32,32))
    if window_reuse:
        supported_tiles += ((16,64,32), (16,16,64))
    if gemm_tile not in supported_tiles:
        raise ValueError('TNCA_GEMM_TILE supports K16 with N16/32/64, or K32 with N16/32; '
                         'window reuse additionally supports N64 K32 and N16 K64')
    reduction_rows = int(os.environ.get('TNCA_REDUCTION_ROWS', '1'))
    if reduction_rows not in (1, 2, 4, 8, 16):
        raise ValueError('TNCA_REDUCTION_ROWS must be 1, 2, 4, 8 or 16')
    sdpa_mode = os.environ.get('TNCA_SDPA', 'decomposed')
    if sdpa_mode not in ('decomposed', 'fused'):
        raise ValueError('TNCA_SDPA must be decomposed or fused')
    # Dropping the row dimension is valid only for a single-row program.
    no_x_dim = bool(no_x_dim) and reduction_rows == 1
    _NO_X_DIM[0] = bool(no_x_dim)
    _install_no_x_dim()
    _BLOCK_CAP[0] = max_block
    tiled = int(os.environ.get('TNCA_TILED_POINTWISE', '0'))
    if tiled not in (0, 1, 2):
        raise ValueError('TNCA_TILED_POINTWISE must be 0, 1 or 2')
    _POINTWISE_CAP[0] = max(max_block, TILED_POINTWISE_BLOCK) if tiled else 0
    _install_narrow_blocks()
    _install_block_fallback()
    import torch._inductor.config as config
    # The B39 path (CPU tensors, every launch staging its arguments) runs
    # CPU-tensor kernels on TNCA. On the native device (R40-01) TNCA's
    # tensors are `tnca` tensors and a CPU tensor is host data the caller
    # placed there -- integer metadata such as packed-sequence offsets --
    # so Inductor compiles its kernels for the host, as it does for CPU
    # tensors anywhere, and never stages them to the device implicitly.
    # host_fallbacks() tells such host integer-metadata kernels from a
    # numerical one.
    native = os.environ.get('TNCA_TORCH_DEVICE', 'cpu') == 'native'
    config.cpu_backend = 'cpp' if native else 'triton'
    if tensor_descriptors:
        _enable_tensor_descriptors(config)
    config.triton.persistent_reductions = False
    # Mix-order split reductions (a reduction over a non-innermost axis
    # fused with its transpose) fail in Inductor's lowering for the tnca
    # device -- x.sum(0), x.t().sum(-1) -- while the ordinary reduction
    # lowering is correct for them (tools/r40/v3_repro_reduce_axis.py).
    config.triton.mix_order_reduction = False
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
    _install_choices(config)
    _install_unsplit_scan()
    _register(max_block, reduction_rows, gemm_tile)
    _decompose_cpu_attention()
    from . import linalg_decomp
    linalg_decomp.register()
    from . import mask_index
    mask_index.register()
    from . import online_attention
    if sdpa_mode == 'fused':
        online_attention.enable()
    else:
        online_attention.disable()
    out = dict(cpu_backend=config.cpu_backend, max_block=max_block,
               persistent_reductions=config.triton.persistent_reductions,
               compile_threads=config.compile_threads,
               max_autotune_gemm_backends=config.max_autotune_gemm_backends,
               tensor_descriptors=bool(tensor_descriptors),
               no_x_dim=bool(no_x_dim), reduction_rows=reduction_rows, gemm_tile=gemm_tile)
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
        return V.graph.get_current_device_or_throw().type in TNCA_DEVICE_TYPES

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
            # Inductor's own first test knows `cpu` (a Triton CPU backend and
            # the stable descriptor API) but not `tnca`, which it would send
            # down the CUDA-capability branch. The tnca device asks the CPU
            # question -- is TNCA's plugin there, is the API -- and then the
            # rules below, the same for both.
            graph = V.graph
            if graph.get_current_device_or_throw().type != 'cpu':
                graph.get_current_device_or_throw = lambda: torch.device('cpu')
                try:
                    ok = super().can_use_tma()
                finally:
                    del graph.get_current_device_or_throw
            else:
                ok = super().can_use_tma()
            if not ok:
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
            # The same rule for the REDUCTION tree, judged on the block shape
            # before its stride-0 dimensions were removed
            # (_record_full_block_shape): a periodic access -- a 64-value
            # table read at `r % 64` over a 128-wide reduction, as a rotary
            # table read inside a fused attention-score reduction is --
            # splits the reduction tree into a stride-1 dimension and a
            # stride-0 one, and once the stride-0 one is dropped (to be
            # restored by tl.broadcast_to, a rank-raising expand the importer
            # has no lowering for) the split is invisible above. A split of
            # the x tree there is degenerate on TNCA -- a reduction program
            # has one row (XBLOCK = 1), so both pieces have extent 1 -- and
            # lowers: it keeps its descriptor.
            full = getattr(block_params, '_tnca_full_block_shape', None)
            if full is not None:
                seen = set()
                for dim in full:
                    names = {x.name for x in getattr(dim, 'free_symbols', ())
                             if x.name.upper().startswith(('R0_', 'R1_', 'RBLOCK'))}
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
    _record_full_block_shape()


def _record_full_block_shape():
    """Keep, on a descriptor's block parameters, its block shape before
    BlockDescriptorOptions.create removed the stride-0 dimensions.

    The TNCA checker above refuses a descriptor whose range tree is split over
    two of its dimensions; that rule must see the stride-0 ones too. Only
    adds an attribute: what Inductor generates is unchanged unless the TNCA
    checker refuses on it, and other devices never read it.
    """
    from torch._inductor.codegen import triton as T
    options = T.TensorDescriptorOptions
    if getattr(options.create, '_tnca_full_shape', False):
        return
    upstream = options.create.__func__

    def create(cls, **kwargs):
        out = upstream(cls, **kwargs)
        try:
            out.params._tnca_full_block_shape = list(out.broadcast_shape)
        except AttributeError:
            pass
        return out
    create._tnca_full_shape = True
    options.create = classmethod(create)


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


# A C++ kernel's argument types, from `name = async_compile.cpp_pybinding([...]`.
_CPP_SIGNATURE = _re.compile(r"\b(cpp_fused\w*)\s*=\s*async_compile\.cpp_pybinding\(\s*\[([^\]]*)\]")
_INTEGER_C_TYPES = _re.compile(r"^(const\s+)?(bool|u?int(8|16|32|64)_t)\s*\*?$")


def host_metadata_kernels(code):
    """C++ kernels whose every argument is an integer or bool (pointer or
    scalar): computation on host integer metadata the caller placed on the
    CPU (configure(): on the native device CPU tensors are host data), e.g.
    `cu_seqlens[1:] - cu_seqlens[:-1]` before a split. No floating-point
    value is computed on the host by these. Sorted names."""
    names = set()
    for src in code:
        for m in _CPP_SIGNATURE.finditer(src):
            types = [t.strip().strip("'\"") for t in m.group(2).split(',') if t.strip()]
            if types and all(_INTEGER_C_TYPES.match(t) for t in types):
                names.add(m.group(1))
    return sorted(names)


_ITEM = _re.compile(r'\b(buf\d+|arg\d+_\d+)\b[^\n]*\.item\(\)')


def scalar_reads(code):
    """`.item()` reads in the wrapper (a data-dependent size or a check's
    value), as dict(host=, device=): one of a buffer allocated on `tnca` is a
    device readback of one element; one of host memory (empty_strided_cpu,
    or a buffer a host C++ kernel reads or writes) is a host read. A graph input is counted as a
    device read (its device is not in the wrapper)."""
    host = device = 0
    for src in map(_executed, code):
        cpu = set(_re.findall(r'\b(buf\d+) = empty_strided_cpu\(', src))
        # A C++ kernel reads and writes host memory only: its buffers are
        # CPU ones, whatever reuse named them.
        for call in _re.findall(r'\bcpp_fused\w*\(([^()]*)\)', src):
            cpu.update(_re.findall(r'\b(buf\d+|arg\d+_\d+)\b', call))
        for m in _ITEM.finditer(src):
            if m.group(1) in cpu:
                host += 1
            else:
                device += 1
    return dict(host=host, device=device)


def host_fallbacks(code):
    """Host-side computation in the generated wrapper source(s), sorted.
    Host integer-metadata kernels (host_metadata_kernels) are not listed."""
    found = []
    metadata = set(host_metadata_kernels(code))
    for src in map(_executed, code):
        for pat, what in FALLBACK_PATTERNS:
            for m in pat.finditer(src):
                if m.group(0) in metadata:
                    continue
                found.append(f'{what}: {m.group(0)}')
    return sorted(set(found))


def triton_kernels(code):
    """Names of the Triton kernels the generated code defines."""
    return sorted({m.group(1) for src in code
                   for m in _re.finditer(r'def (triton_\w+)\(', src)})
