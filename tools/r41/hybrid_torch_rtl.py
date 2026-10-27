#!/usr/bin/env python3
"""Standard torch numerics through the coupled wide RTL MMIO stand-in."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--hybrid', action='store_true')
p.add_argument('--trace', action='store_true')
p.add_argument('--size', type=int, default=33)
p.add_argument('--warmup', action='store_true')
p.add_argument('--max-block', type=int, choices=(16,32,64,128), default=16)
p.add_argument('--dtype', choices=('float32','float16'), default='float32')
a = p.parse_args()
if a.size < 1:
    p.error('size must be positive')
os.environ.update(TNCA_PROFILE='core-vrf-sync-v5', TNCA_BACKEND='fpga-multicore-sim',
    TNCA_BACKEND_ARGS=json.dumps(dict(core_count=3, memory_ports=3, hybrid_lsu=a.hybrid, trace=a.trace,
        sfu_lanes=4 if a.hybrid else 1, noc_link_stages=2, dispatch_stages=2, home_stages=2)),
    TNCA_RCTX_SPMD='1', TNCA_HOST_TIMEOUT='1800', TNCA_MAX_BLOCK=str(a.max_block))
import torch
import triton
import tnca.torch_device as D
from tnca.triton_backend.driver import launch_log

D.register()
torch.manual_seed(41)
dtype=getattr(torch,a.dtype)
x, y = torch.randn(a.size,dtype=dtype), torch.randn(a.size,dtype=dtype)
dx, dy = x.to('tnca'), y.to('tnca')
operations = dict(add=lambda x,y:x+y, mul=lambda x,y:x*y,
    silu=lambda x,y:torch.nn.functional.silu(x),
    gelu=lambda x,y:torch.nn.functional.gelu(x), copy=lambda x,y:x.clone())
for name, operation in operations.items():
    fn = torch.compile(operation, fullgraph=True, dynamic=False)
    if a.warmup:
        warm = fn(dx, dy)
        torch.testing.assert_close(warm.cpu(), operation(x,y))
        del warm
    with launch_log() as launches:
        result = fn(dx, dy)
    torch.testing.assert_close(result.cpu(), operation(x,y))
    assert launches, (name, 'no TNCA launch')
    print(json.dumps(dict(operator=name, shape=[a.size], dtype=a.dtype,
        hybrid=a.hybrid, numerics='pass', launches=len(launches),
        rtl_device_cycles=[row.get('device_cycles') for row in launches],
        warmed=a.warmup, max_block=a.max_block, grids=[row.get('grid') for row in launches], upload_bytes=[row.get('upload_bytes') for row in launches],
        download_bytes=[row.get('download_bytes') for row in launches],
        evidence='coupled-RTL-functional', board_speedup_measured=False)), flush=True)
print('HYBRID TORCH COUPLED RTL ALL 5 OPERATORS PASS', flush=True)
