"""`torch.linalg.solve_triangular` on TNCA, as standard Torch operations.

Inductor keeps `aten.linalg_solve_triangular` as a host call
(lowering.py: `make_fallback`), and the tnca device refuses host ATen
calls. This module gives Inductor a decomposition of the operator into
operations it already generates Triton for -- batched matmuls through the
mm/bmm template, and pointwise kernels -- so a solve runs on the device
like any other graph. It is added to Inductor's decomposition table by
`tnca.inductor.configure()` and applies only to tensors on a TNCA device
type (`tnca.inductor.TNCA_DEVICE_TYPES`). On any other device it returns
`NotImplemented`, which keeps upstream's behaviour (the ATen fallback).

The algorithm is a blocked substitution with inverted diagonal blocks,
the formulation GPU BLAS libraries use for TRSM (MAGMA, cuBLAS):

1. Unless `unitriangular`, scale the rows of A and B by the reciprocal of
   A's diagonal: D^-1 A X = D^-1 B is a unit-triangular system with the
   same solution. Every later step is a unit solve.
2. Pad n with zeros up to a multiple of the block size b. A unit solve
   never reads the diagonal, so the padded unknowns are exactly zero and
   do not touch the real ones.
3. Invert every b x b diagonal block at once. With S a block's strict
   triangle and N = -S, which is nilpotent (N^b = 0),
   (I - N)^-1 = (I + N)(I + N^2)(I + N^4)...(I + N^(b/2)):
   log2(b) - 1 squarings and as many multiply-adds, batched over every
   block of every matrix.
4. Block substitution, right-looking: X_i = T_i^-1 R_i, then the rows not
   yet solved lose the block column times X_i (a matmul and a subtract).
   Lower triangles go top-down, upper triangles bottom-up.

Only the triangle the flags select is read, as torch does: diagonal
blocks through `tril`/`triu`, off-diagonal blocks from the selected side,
and the diagonal not at all when `unitriangular`. `left=False` solves the
transposed system. Batch dimensions broadcast as in torch.

Arithmetic is binary32 throughout (the TNCA matmul accumulates per K in
binary32, ISA sec 9C). Other dtypes are refused with an error: torch's CPU
operator takes float64 and complex, which TNCA has no arithmetic for.
Numerics: this is not substitution's operation order, so results are not
bitwise equal to LAPACK. tools/performance_harness/triangular_solve_cases.py
measures the forward error against a float64 reference.
"""
import math

#: Diagonal block size: the 16x16 MMA fragment (ISA sec 9C).
BLOCK = 16
_registered = False


def _block_size(n):
    """b: the fragment size, or the next power of two at or above a smaller n."""
    return min(BLOCK, 1 << max(0, math.ceil(math.log2(max(n, 1)))))


def _on_tnca(*tensors):
    from .inductor import TNCA_DEVICE_TYPES
    return all(t.device.type in TNCA_DEVICE_TYPES for t in tensors)


def _inverse_unit_blocks(T, upper):
    """(I + S)^-1 for a batch of b x b blocks T [..., b, b] whose strict
    triangle S is selected by `upper` (the diagonal is taken as one)."""
    import torch
    b = T.shape[-1]
    N = -(T.triu(1) if upper else T.tril(-1))
    P = torch.eye(b, dtype=T.dtype, device=T.device) + N         # (I + N)
    Q = N
    for _ in range(max(0, math.ceil(math.log2(b)) - 1)):
        Q = Q @ Q                                                # N^(2^k)
        P = P + P @ Q                                            # P (I + N^(2^k))
    return P


def _solve_left(A, B, upper, unitriangular):
    """X with A X = B; A [..., n, n] and B [..., n, k] share batch dims."""
    import torch
    import torch.nn.functional as F
    n = A.shape[-1]
    if not unitriangular:
        # D^-1 A is unit triangular: scale the rows of A and of B once,
        # then every step below is a unit solve. (The other triangle may
        # overflow here; it is never read.)
        inv_d = 1.0 / torch.diagonal(A, dim1=-2, dim2=-1)
        A = A * inv_d.unsqueeze(-1)
        B = B * inv_d.unsqueeze(-1)
    b = _block_size(n)
    nb = -(-n // b)
    m = nb * b
    if m != n:
        # Zero padding: a unit solve never reads the diagonal, so the padded
        # unknowns solve to zero and do not touch the real ones.
        pad = m - n
        A = F.pad(A, (0, pad, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
    # The diagonal blocks, [..., nb, b, b].
    tiles = A.unflatten(-1, (nb, b)).unflatten(-3, (nb, b))     # [..., nb, b, nb, b]
    diag = torch.diagonal(tiles, dim1=-4, dim2=-2).movedim(-1, -3)
    inv = _inverse_unit_blocks(diag, upper)
    xs = [None] * nb
    rest = B                                                     # rows still to solve
    order = range(nb - 1, -1, -1) if upper else range(nb)
    for i in order:
        if upper:
            here = rest[..., i * b:(i + 1) * b, :]
        else:
            here = rest[..., :b, :]
        x = inv[..., i, :, :] @ here
        xs[i] = x
        if upper and i > 0:
            rest = rest[..., :i * b, :] - A[..., :i * b, i * b:(i + 1) * b] @ x
        elif not upper and i + 1 < nb:
            rest = rest[..., b:, :] - A[..., (i + 1) * b:, i * b:(i + 1) * b] @ x
    X = xs[0] if nb == 1 else torch.cat(xs, dim=-2)
    return X[..., :n, :] if m != n else X


def solve_triangular(A, B, *, upper, left=True, unitriangular=False):
    """Decomposition of aten.linalg_solve_triangular.default for TNCA."""
    import torch
    if not _on_tnca(A, B):
        return NotImplemented
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise RuntimeError(
            f'linalg.solve_triangular on {A.device.type}: {A.dtype}/{B.dtype} is not supported; '
            'TNCA solves in binary32 only (no float64 or complex arithmetic)')
    if A.dim() < 2 or B.dim() < 2 or A.shape[-1] != A.shape[-2]:
        return NotImplemented          # upstream's checks raise the right error
    if not left:
        # X A = B  <=>  A^T X^T = B^T, with the other triangle.
        return solve_triangular(A.mT, B.mT, upper=not upper, left=True,
                                unitriangular=unitriangular).mT
    n, k = A.shape[-1], B.shape[-1]
    if B.shape[-2] != n:
        return NotImplemented
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A = A.expand(*batch, n, n)
    B = B.expand(*batch, n, k)
    if n == 0 or k == 0:
        return B.new_zeros((*batch, n, k))
    return _solve_left(A, B, upper, unitriangular)


def register():
    """Add the decomposition to Inductor's own table (idempotent)."""
    global _registered
    if _registered:
        return
    import torch
    from torch._inductor.decomposition import decompositions
    decompositions[torch.ops.aten.linalg_solve_triangular.default] = solve_triangular
    _registered = True
