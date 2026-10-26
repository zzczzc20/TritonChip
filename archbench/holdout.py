"""The 4 sealed R41 holdout points (spec/r41/holdout.json).

These points are sealed for final policy evaluation only; never used for
fitting or candidate development.
"""
import torch
import torch.nn.functional as F

from .registry import Demand, Point, register_holdout


class Matmul(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y)


class Linear(torch.nn.Module):
    def forward(self, x, weight):
        return F.linear(x, weight)


class Softmax(torch.nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1)


class RMSNorm(torch.nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randn(g, shape, dtype='bfloat16'):
    return (torch.randn(shape, generator=g) * 0.1).to(getattr(torch, dtype))


def _make_mm(seed=20261020):
    g = _gen(seed)
    return Matmul(), (_randn(g, (19, 47)), _randn(g, (47, 37)))


def _make_linear(seed=20261020):
    g = _gen(seed)
    return Linear(), (_randn(g, (23, 61)), _randn(g, (43, 61)))


def _make_softmax(seed=20261020):
    g = _gen(seed)
    return Softmax(), (_randn(g, (7, 113)),)


def _make_rmsnorm(seed=20261020):
    g = _gen(seed)
    return RMSNorm(), (_randn(g, (11, 79)),)


# Register the 4 sealed holdout points from spec/r41/holdout.json
register_holdout(Point(
    id='H41-mm', kind='op', family='matmul', tier='holdout', dtype='bf16',
    params={'m': 19, 'k': 47, 'n': 37}, make=_make_mm,
    demand=Demand(macs=19 * 47 * 37, read_bytes=2 * (19 * 47 + 47 * 37), write_bytes=2 * (19 * 37)),
    tol=0.02, note='Sealed R41 holdout matmul'
))

register_holdout(Point(
    id='H41-linear', kind='op', family='linear', tier='holdout', dtype='bf16',
    params={'b': 23, 'in': 61, 'out': 43}, make=_make_linear,
    demand=Demand(macs=23 * 61 * 43, read_bytes=2 * (23 * 61 + 43 * 61), write_bytes=2 * (23 * 43)),
    tol=0.02, note='Sealed R41 holdout linear'
))

register_holdout(Point(
    id='H41-softmax', kind='op', family='softmax', tier='holdout', dtype='bf16',
    params={'b': 7, 'n': 113}, make=_make_softmax,
    demand=Demand(fp_ops=7 * 113 * 3, sfu_ops=7 * 113, read_bytes=2 * (7 * 113), write_bytes=2 * (7 * 113)),
    tol=0.02, note='Sealed R41 holdout softmax'
))

register_holdout(Point(
    id='H41-rmsnorm', kind='op', family='rmsnorm', tier='holdout', dtype='bf16',
    params={'b': 11, 'n': 79}, make=_make_rmsnorm,
    demand=Demand(fp_ops=11 * 79 * 4, sfu_ops=11, read_bytes=2 * (11 * 79), write_bytes=2 * (11 * 79)),
    tol=0.02, note='Sealed R41 holdout rmsnorm'
))
