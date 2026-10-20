"""Repro: int64 transposed pointwise at TNCA_MAX_BLOCK=32 exceeds the 128 KiB code region.

    python3 tools/r40/v3_repro_int64_code_region.py ROOT [width]

Expected on the delivery tree: width 16 PASS; width 32/64 TNCALaunchError
'capacity-exceeded: code region' (text segment ~250 KB > runtime.Session
code region 0..heap_base=0x20000) after a 238-slot / 5265-reload vector spill;
no compile-time retile to a smaller block.
"""
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
os.environ['TNCA_MAX_BLOCK'] = sys.argv[2] if len(sys.argv) > 2 else '32'
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
sys.path[:0] = [str(root), str(root / 'python'), str(root / 'compiler/python'), str(root / 'tools/asm')]
import torch  # noqa: E402
import triton  # noqa: E402,F401
import tnca.torch_device as D  # noqa: E402
from tnca import device  # noqa: E402

device.prepare(os.environ['TNCA_PROFILE'])
D.register()
x = torch.arange(65 * 33, dtype=torch.int64).reshape(65, 33)
x = torch.where(x % 2 == 0, x + (1 << 35), -x - (1 << 37))
fn = lambda t: (t.t() * 3 + 7).contiguous()  # noqa: E731
try:
    out = torch.compile(fn, fullgraph=True, dynamic=False)(x.to('tnca'))
    print('PASS' if torch.equal(out.cpu(), fn(x)) else 'WRONG', 'width', os.environ['TNCA_MAX_BLOCK'])
except Exception as e:  # noqa: BLE001
    print('FAIL', type(e).__name__, str(e).splitlines()[0][:200], 'width', os.environ['TNCA_MAX_BLOCK'])
