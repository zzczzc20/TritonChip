"""Where Inductor's lowering policy depends on the device, what `tnca` gets.

`tnca` is a Triton device that is not on Inductor's list of GPUs
(`torch._inductor.utils.GPU_TYPES`). Inductor's lowering then takes the
non-GPU answer wherever it asks `is_gpu()` -- no input-alignment assumption,
no reduction unrolling for one-element outputs, loops normalised and merged
as on the CPU -- which is what B39's CPU path compiled with and what R39 tuned
the TNCA compiler against. A few decisions ask `device.type == "cpu"`
instead, and for those this module gives `tnca` the CPU path's answer too.
Each is one switch, on by default, so R40-01's device time starts from
parity with B39 and widening any of them to the GPU's choice is an ablation
of its own (doc/r40_tasks_and_acceptance.md sec 3.2), not a side effect of
registering a device:

  TNCA_POLICY_TEMPLATES   Triton GEMM/conv templates for tnca, as for CPU
                          tensors (`use_triton_template`: a non-GPU device
                          other than `cpu` would get none; there is no device
                          ATen GEMM to fall back to)
  TNCA_POLICY_VARIANCE    var/var_mean two-step (sum, then sum of squares)
                          below the CPU threshold, instead of Welford
                          (`lowering.use_two_step_variance`)
  TNCA_POLICY_CAT         `cat` as a ConcatKernel (copies into slices), not a
                          pointwise kernel of masked loads (`lowering.cat`)
  TNCA_POLICY_MM_DECOMP   the CPU's small-matmul decompositions for bmm, mm
                          and addmm (`decomposition.{bmm,mm,addmm}`), not the
                          GPU's
  TNCA_POLICY_LAYOUT      channels-last layout for a graph with convolutions,
                          as the CPU path always chooses (with MKLDNN present)
                          (`GraphLowering.decide_layout_opt`)
  TNCA_POLICY_REALIZE     the CPU's realization and fusion heuristics: a
                          reused result of a heavy op (exp, log, ...) is
                          stored rather than recomputed, the CPU's opcount and
                          read thresholds, and no loop reordering or index
                          inversion during fusion (`ir.is_cpu`,
                          `BaseSchedulerNode.is_cpu`, used only there)

Every override is scoped to tensors on `tnca`; CPU and CUDA in the same
process keep upstream's behaviour (R40-01 acceptance 2).

One switch goes the other way, off by default, for the measured question of
whether the GPU's answer pays on TNCA:

  TNCA_INPUTS_ALIGNED=1   graph inputs on tnca are assumed aligned when they
                          are (`utils.should_assume_input_aligned`: storages
                          are 64-byte aligned, so a view is when its offset
                          is), which makes them eligible for tensor
                          descriptors as on a GPU. Off, they are treated as
                          the CPU path treats host tensors: never assumed
                          aligned, so no input is read through a descriptor.
"""
import os

NAME = 'tnca'
_installed = False


def _on(name):
    return os.environ.get(name, '1') != '0'


def _rebind(old, new, modules):
    """Point every listed module's name for `old` at `new` (they import the
    function by name, so patching only its home module would miss them)."""
    for mod in modules:
        for attr, val in list(vars(mod).items()):
            if val is old:
                setattr(mod, attr, new)


def _templates():
    import torch
    import torch._inductor.utils as iu
    import torch._inductor.kernel.mm as k_mm
    import torch._inductor.kernel.bmm as k_bmm
    import torch._inductor.kernel.conv as k_conv
    import torch._inductor.kernel.mm_plus_mm as k_mmpmm
    import torch._inductor.kernel.mm_grouped as k_grouped
    from torch._inductor import config
    upstream = iu.use_triton_template
    if getattr(upstream, '_tnca', False):
        return

    def use_triton_template(layout, *, enable_int32=False, enable_float8=False,
                            check_max_autotune=True):
        if layout.device.type != NAME:
            return upstream(layout, enable_int32=enable_int32, enable_float8=enable_float8,
                            check_max_autotune=check_max_autotune)
        # The CPU branch of upstream's rule, for the tnca device.
        from torch._inductor.codegen.common import BackendFeature, has_backend_feature
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if enable_int32:
            dtypes.append(torch.int32)
        if enable_float8:
            dtypes += [torch.float8_e4m3fn, torch.float8_e5m2]
        return (layout.dtype in dtypes
                and (config.max_autotune or config.max_autotune_gemm or not check_max_autotune)
                and iu._use_autotune_backend('TRITON')
                and has_backend_feature(layout.device, BackendFeature.TRITON_TEMPLATES))
    use_triton_template._tnca = True
    _rebind(upstream, use_triton_template, (iu, k_mm, k_bmm, k_conv, k_mmpmm, k_grouped))


def _device_of(x):
    try:
        return x.get_device()
    except Exception:                                        # noqa: BLE001
        return None


def _variance():
    import torch._inductor.lowering as L
    from torch._inductor import config
    upstream = L.use_two_step_variance
    if getattr(upstream, '_tnca', False):
        return

    def use_two_step_variance(x, *args, **kwargs):
        d = _device_of(x)
        if d is None or d.type != NAME:
            return upstream(x, *args, **kwargs)
        # Upstream's non-CPU, non-CUDA branch with the CPU's threshold is
        # exactly its CPU branch: two-step below the threshold unless the
        # output is one element.
        with config.patch(unroll_reductions_threshold=config.cpp.use_two_step_variance_threshold):
            return upstream(x, *args, **kwargs)
    use_two_step_variance._tnca = True
    L.use_two_step_variance = use_two_step_variance


def _cat():
    import torch
    import torch._inductor.lowering as L
    from torch._inductor import config
    aten = torch.ops.aten
    for key in (aten.cat, aten.cat.default):
        upstream = L.lowerings.get(key)
        if upstream is None or getattr(upstream, '_tnca', False):
            continue

        def cat(inputs, dim=0, upstream=upstream):
            d = _device_of(inputs[0]) if inputs else None
            if d is None or d.type != NAME:
                return upstream(inputs, dim)
            # With no input count allowed for a pointwise cat, upstream's
            # lowering ends in the ConcatKernel the CPU path takes.
            with config.patch(max_pointwise_cat_inputs=0, max_complex_pointwise_cat_inputs=0,
                              force_pointwise_cat=False):
                return upstream(inputs, dim)
        cat._tnca = True
        L.lowerings[key] = cat


def _depthwise_convolution():
    """Small 1-D depthwise kernels as generic Torch slices and pointwise ops.

    The existing Triton conv templates do not cover this grouped 1-D case.
    Bound the unrolled kernel width, retain the input dtype and strides, and
    keep low-precision accumulation in FP32. Other convolution geometries
    and other devices retain upstream lowering.
    """
    import torch
    from torch._inductor.decomposition import decompositions
    from torch.fx.experimental.symbolic_shapes import guard_or_false
    op = torch.ops.aten.convolution.default
    upstream = decompositions.get(op)
    if getattr(upstream, '_tnca_depthwise', False):
        return

    def convolution(input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups):
        supported = (input.device.type == NAME and input.ndim == weight.ndim == 3
                     and not transposed and len(stride) == len(padding) == len(dilation) == 1
                     and not any(output_padding) and input.dtype == weight.dtype
                     and input.is_floating_point()
                     and guard_or_false(groups == input.shape[1])
                     and guard_or_false(weight.shape[0] == input.shape[1])
                     and guard_or_false(weight.shape[1] == 1)
                     and isinstance(weight.shape[2], int) and 1 <= weight.shape[2] <= 16
                     and guard_or_false(stride[0] > 0) and guard_or_false(dilation[0] > 0)
                     and guard_or_false(padding[0] >= 0))
        if not supported:
            return (upstream(input, weight, bias, stride, padding, dilation,
                             transposed, output_padding, groups) if upstream else NotImplemented)
        width = weight.shape[2]
        step, pad, dil = stride[0], padding[0], dilation[0]
        length = (input.shape[-1] + 2 * pad - dil * (width - 1) - 1) // step + 1
        if not guard_or_false(length > 0):
            return NotImplemented
        dtype = torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
        padded = torch.nn.functional.pad(input.to(dtype), (pad, pad))
        weights = weight.to(dtype)
        result = None
        for tap in range(width):
            values = padded[:, :, tap * dil:tap * dil + length * step:step]
            term = values * weights[:, 0, tap].view(1, -1, 1)
            result = term if result is None else result + term
        if bias is not None:
            result = result + bias.to(dtype).view(1, -1, 1)
        return result.to(input.dtype)

    convolution._tnca_depthwise = True
    decompositions[op] = convolution


def _mm_decompositions():
    import torch
    from torch._decomp.decompositions import pw_cast_for_opmath
    from torch._inductor import decomposition as dec
    from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true
    aten = torch.ops.aten
    table = dec.decompositions

    def wrap(op, tnca_rule):
        upstream = table.get(op)
        if upstream is None or getattr(upstream, '_tnca', False):
            return

        def fn(*args, **kwargs):
            if args and isinstance(args[0], torch.Tensor) and args[0].device.type == NAME:
                return tnca_rule(*args, **kwargs)
            return upstream(*args, **kwargs)
        fn._tnca = True
        table[op] = fn

    # The CPU branches of torch/_inductor/decomposition.py, for tnca.
    @pw_cast_for_opmath
    def bmm(self, batch2, out_dtype=None):
        if statically_known_true(self.shape[2] == 1) and statically_known_true(batch2.shape[1] == 1):
            return (self * batch2).contiguous()
        if statically_known_true(self.size(1) == 1) and statically_known_true(batch2.size(-1) == 1):
            return torch.sum(self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True).unsqueeze(1)
        return NotImplemented

    @pw_cast_for_opmath
    def addmm(self, mat1, mat2, out_dtype=None, beta=1, alpha=1):
        if statically_known_true(mat1.size(0) == 1) and statically_known_true(mat2.size(-1) == 1):
            out = torch.sum(mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True).unsqueeze(0)
            return alpha * out + beta * self
        if (statically_known_true(mat1.size(0) == 1) and guard_or_false(mat2.size(0) <= 16)
                and guard_or_false(mat2.size(1) <= 16)):
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            return alpha * out + beta * self
        return NotImplemented

    @pw_cast_for_opmath
    def mm(self, input2, out_dtype=None):
        if (statically_known_true(self.size(-1) == 1) and statically_known_true(self.size(0) != 1)
                and statically_known_true(input2.size(1) != 1)
                and statically_known_true(self.size(0) > 0)
                and statically_known_true(input2.size(0) == 1)
                and self.dtype == input2.dtype
                and guard_or_false((torch.numel(self) + torch.numel(input2)) <= 32)):
            return self * input2
        if statically_known_true(self.size(0) == 1) and statically_known_true(input2.size(-1) == 1):
            return torch.sum(self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True).unsqueeze(0)
        return NotImplemented

    wrap(aten.bmm.default, bmm)
    wrap(aten.addmm.default, addmm)
    wrap(aten.mm.default, mm)

    # And the CPU branch of the bmm lowering (kernel/bmm.py tuned_bmm): a
    # batched matmul with M or N of 1 is memory bound and becomes a multiply
    # and a sum, which fuse with what is around them -- attention decode's
    # q @ k^T and p @ v.
    import torch._inductor.lowering as L
    for key in (aten.bmm, aten.bmm.default):
        upstream = L.lowerings.get(key)
        if upstream is None or getattr(upstream, '_tnca', False):
            continue

        def tuned_bmm(mat1, mat2, *args, upstream=upstream, **kwargs):
            d1, d2 = _device_of(mat1), _device_of(mat2)
            if (d1 is not None and d2 is not None and d1.type == d2.type == NAME
                    and not args and not kwargs.get('out_dtype')
                    and (mat1.get_size()[1] == 1 or mat2.get_size()[2] == 1)):
                return L.sum_(L.mul(L.unsqueeze(mat1, -1), L.unsqueeze(mat2, 1)), axis=2)
            return upstream(mat1, mat2, *args, **kwargs)
        tuned_bmm._tnca = True
        L.lowerings[key] = tuned_bmm


def _realize_and_fusion():
    import torch._inductor.ir as ir
    import torch._inductor.scheduler as sch
    upstream = ir.is_cpu
    if getattr(upstream, '_tnca', False):
        return

    def is_cpu(x):
        return upstream(x) or ir.get_device_type(x) == NAME
    is_cpu._tnca = True
    ir.is_cpu = is_cpu

    up_node = sch.BaseSchedulerNode.is_cpu

    def node_is_cpu(self):
        if up_node(self):
            return True
        d = self.get_device()
        return d is not None and d.type == NAME
    sch.BaseSchedulerNode.is_cpu = node_is_cpu


def _layout():
    import torch
    from torch._inductor import config
    from torch._inductor.graph import GraphLowering
    upstream = GraphLowering.decide_layout_opt
    if getattr(upstream, '_tnca', False):
        return

    def decide_layout_opt(gm, *, is_inference):
        convs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.convolution.default]
        if (config.layout_optimization and convs
                and all(getattr(n.args[i].meta.get('val'), 'device', None) is not None
                        and n.args[i].meta['val'].device.type == NAME
                        for n in convs for i in (0, 1))):
            # The CPU path's answer (with MKLDNN present, always channels-last).
            return True
        return upstream(gm, is_inference=is_inference)
    decide_layout_opt._tnca = True
    GraphLowering.decide_layout_opt = staticmethod(decide_layout_opt)


def _inputs_aligned():
    import torch._inductor.utils as iu
    import torch._inductor.graph as ig
    import torch._inductor.compile_fx as cf
    upstream = iu.should_assume_input_aligned
    if getattr(upstream, '_tnca', False):
        return

    def should_assume_input_aligned(example_input):
        if example_input.device.type == NAME:
            return iu.tensor_is_aligned(example_input)
        return upstream(example_input)
    should_assume_input_aligned._tnca = True
    _rebind(upstream, should_assume_input_aligned, (iu, ig, cf))


def install():
    """Install every policy whose switch is on (idempotent)."""
    global _installed
    if _installed:
        return
    if _on('TNCA_POLICY_TEMPLATES'):
        _templates()
    if _on('TNCA_POLICY_VARIANCE'):
        _variance()
    if _on('TNCA_POLICY_CAT'):
        _cat()
    if _on('TNCA_POLICY_MM_DECOMP'):
        _mm_decompositions()
    if _on('TNCA_POLICY_DEPTHWISE'):
        _depthwise_convolution()
    if _on('TNCA_POLICY_REALIZE'):
        _realize_and_fusion()
    if _on('TNCA_POLICY_LAYOUT'):
        _layout()
    if os.environ.get('TNCA_INPUTS_ALIGNED') == '1':
        _inputs_aligned()
    _installed = True
