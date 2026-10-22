"""Tensor-descriptor eligibility, judged on the access before Inductor strips its broadcast dimensions.

tnca.inductor._apply_tma_rules refuses a descriptor when one range tree is
split over two descriptor dimensions (`r // 64` and `r % 64` as dimensions of
their own): Inductor then rebuilds the kernel's tile from the descriptor box
with a rank-raising `[:, None, None]`, `tl.broadcast_to`, `tl.trans` and
`tl.reshape`, which the TNCA importer has no lowering for ("tt.expand_dims of
something that is not a range"), and falls back to the pointer access the
importer does lower.

That rule reads the block parameters Inductor hands to
`are_block_parameters_compatible`. But BlockDescriptorOptions.create removes
the dimensions of stride 0 first (they are restored with tl.broadcast_to). A
periodic access -- a buffer of 64 values read at `r % 64` over a 128-wide
reduction, as a rotary table read inside a fused attention-score reduction is
-- splits the range tree into a stride-1 dimension and a stride-0 one, and
after the stride-0 one is removed the split is invisible: the descriptor is
accepted and the kernel fails to lower.

`install()` records, on the block parameters a TensorDescriptorOptions is
created with, the block shape before the stride-0 dimensions were removed, and
applies the existing rule to that shape for TNCA device types. Anything the
existing rule accepts on the full shape is unchanged; CUDA and other devices
are untouched.
"""


def install():
    from torch._inductor.codegen import triton as T
    from torch._inductor.virtualized import V
    from .inductor import TNCA_DEVICE_TYPES

    options = T.TensorDescriptorOptions
    if not getattr(options.create, '_tnca_full_shape', False):
        upstream_create = options.create.__func__

        def create(cls, **kwargs):
            out = upstream_create(cls, **kwargs)
            try:
                out.params._tnca_full_block_shape = list(out.broadcast_shape)
            except AttributeError:
                pass
            return out
        create._tnca_full_shape = True
        options.create = classmethod(create)

    base = T.TritonKernel.tma_compatibility_checker_cls
    if getattr(base, '_tnca_full_shape_rule', False):
        return

    class TNCABroadcastAwareChecker(base):
        _tnca_full_shape_rule = True

        def are_block_parameters_compatible(self, block_params, constant_offset=0):
            full = getattr(block_params, '_tnca_full_block_shape', None)
            if full is not None and V.graph.get_current_device_or_throw().type in TNCA_DEVICE_TYPES:
                seen = set()
                for dim in full:
                    names = {x.name for x in getattr(dim, 'free_symbols', ())}
                    if names & seen:
                        return False
                    seen |= names
            return super().are_block_parameters_compatible(block_params, constant_offset)

    T.TritonKernel.tma_compatibility_checker_cls = TNCABroadcastAwareChecker
