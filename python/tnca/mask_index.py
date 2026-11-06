"""Boolean-mask indexing and tensor checks on TNCA, as device operations.

Inductor has no lowering for three operations model code uses on masks it
computes on the device -- a placeholder-token mask from `input_ids == id`,
say -- and keeps them as ATen calls in the wrapper, which the tnca
dispatcher does not run (there is no host numerical fallback):

* `aten.index_put` with one boolean mask and one row of values per selected
  position (`x[mask] = v`, `x[mask, :] = v`). That is exactly
  `x.masked_scatter(mask, v)`: the k-th selected position takes the k-th
  row of v, in row-major order. Inductor lowers masked_scatter on the
  device (an int64 cumsum of the mask gives each position its source row;
  no count is read back). Values that broadcast over the selected rows (one
  row) become `where(mask, v, x)`.
* `aten.index` with one boolean mask (`x[mask]`). Its result has as many
  rows as the mask has True values, which only the data says: the count is
  a device reduction read back once (`.item()`, one integer, the same sync
  every PyTorch device makes for a boolean index -- CUDA's nonzero), and the
  rows are gathered on the device: each selected position scatters its
  index to its rank (cumsum - 1), the gather reads those rows.
* `aten._is_all_true` (`torch._check(t)` on a tensor, i.e.
  `torch._check_tensor_all`): `t.all()`, the device reduction it is
  defined as (aten/src/ATen/native/TensorCompare.cpp); the `.item()` that
  follows is the check's own read of one boolean.

Registered in Inductor's decomposition table by tnca.inductor.configure()
for TNCA device types only; anything else (integer indices, several
indices, other devices) returns NotImplemented and keeps upstream's
handling.
"""
_registered = False


def _tnca(t):
    from .inductor import TNCA_DEVICE_TYPES
    return t.device.type in TNCA_DEVICE_TYPES


def _single_mask(x, indices):
    """The boolean mask when `indices` is one leading boolean mask over x."""
    import torch
    if len(indices) != 1 or indices[0] is None or indices[0].dtype not in (torch.bool, torch.uint8):
        return None
    m = indices[0]
    if m.dim() == 0 or m.dim() > x.dim() or tuple(m.shape) != tuple(x.shape[:m.dim()]):
        return None
    return m.bool()


def is_all_true(t):
    import torch
    if not _tnca(t):
        return NotImplemented
    return torch.all(t)


def index_put(x, indices, values, accumulate=False):
    import torch
    m = _single_mask(x, indices)
    if m is None or accumulate or not _tnca(x):
        return NotImplemented
    k = m.dim()
    rest = tuple(x.shape[k:])
    v = values.to(x.dtype)
    while v.dim() < 1 + len(rest):
        v = v.unsqueeze(0)
    mb = m.reshape(tuple(m.shape) + (1,) * len(rest)).expand(x.shape)
    if v.dim() == 1 + len(rest) and v.shape[0] == 1:
        # One row for every selected position.
        return torch.where(mb, v[0].expand(rest) if rest else v.reshape(()), x)
    v = v.reshape((-1,) + rest) if v.dim() > 1 + len(rest) else v
    return torch.masked_scatter(x, mb, v.expand((v.shape[0],) + rest).contiguous())


def index_put_(x, indices, values, accumulate=False):
    out = index_put(x, indices, values, accumulate)
    if out is NotImplemented:
        return out
    return x.copy_(out)


def index(x, indices):
    import torch
    m = _single_mask(x, indices)
    if m is None or not _tnca(x):
        return NotImplemented
    k = m.dim()
    rest = tuple(x.shape[k:])
    flat = m.reshape(-1)
    n = flat.numel()
    count = flat.sum().item()
    torch._check(count >= 0)
    torch._check(count <= n)
    rank = torch.cumsum(flat, 0) - 1
    slot = torch.where(flat, rank, torch.full_like(rank, count))
    order = torch.arange(n, dtype=torch.int64, device=x.device)
    rows = torch.zeros(count + 1, dtype=torch.int64, device=x.device).index_put((slot,), order)
    return x.reshape((n,) + rest)[rows[:count]]


def register():
    """Add the decompositions to Inductor's own table (idempotent)."""
    global _registered
    if _registered:
        return
    import torch
    from torch._inductor.decomposition import decompositions
    aten = torch.ops.aten
    for op, fn in ((aten._is_all_true.default, is_all_true),
                   (aten.index_put.default, index_put),
                   (aten.index_put_.default, index_put_),
                   (aten.index.Tensor, index)):
        if op in decompositions:
            raise RuntimeError(f'{op} already has an Inductor decomposition')
        decompositions[op] = fn
    _registered = True
