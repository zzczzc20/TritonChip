"""`aten._is_all_true` on TNCA as a device reduction instead of a host call.

`torch._check(t)` on a tensor `t` -- what model code writes as a compilable
assert, e.g. "the placeholder token count equals the number of features" --
reaches Inductor as `aten._is_all_true(t)` followed by `.item()`. Inductor has
no lowering for `_is_all_true` and keeps it as an ATen call in the wrapper, so
the tnca dispatcher is asked for `aten::all.all_out` on device memory, which it
does not have: the graph cannot run.

`_is_all_true(t)` is defined as `t.all()` returning a 0-d bool tensor
(aten/src/ATen/native/TensorCompare.cpp). Inductor lowers `aten.all` itself, as
a reduction, so decomposing to it keeps the check on the device; the `.item()`
that follows is the same device-to-host read of one value the check always
needed. Registered in Inductor's decomposition table for TNCA device types
only; every other device keeps upstream's handling (NotImplemented).
"""


def install():
    import torch
    from torch._inductor.decomposition import decompositions
    from .inductor import TNCA_DEVICE_TYPES

    op = torch.ops.aten._is_all_true.default
    if getattr(decompositions.get(op), '_tnca', False):
        return

    def is_all_true(t):
        if t.device.type not in TNCA_DEVICE_TYPES:
            return NotImplemented
        return torch.all(t)
    is_all_true._tnca = True
    decompositions[op] = is_all_true
