"""Extra single-program proxies of the Hub FP8 kernel for the fp8sw study (measurement proxies, not registered in any suite).

N = 128 is one column block, so with M <= 16 the grid is one program: a clean instruction-level profile. K = 256 and 512
give the cost of one K block by difference. Registered next to the kfuse-speed proxies (tools/r42/kfuse_speed_xl_points.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kfuse_speed_xl_points as P  # noqa: E402

for _m, _n, _k in ((16, 128, 256), (16, 128, 512), (1, 128, 256), (1, 128, 512), (16, 256, 512), (1, 256, 512)):
    try:
        P._add(_m, _n, _k)
    except Exception:   # already registered
        pass
