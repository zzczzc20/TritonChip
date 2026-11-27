#!/usr/bin/env python3
"""tools/r42/lineelig_mc.py (link_budget_rtl.py plus an output digest) with the FP8 Hub-kernel proxy points of the R42 studies registered
(tools/r42/fp8sw/points.py: op.fp8_linear/xl/fp8-m16-k256-n128 and the other reduced shapes, measurement proxies, not registered in
any suite), so that they can run on the three-core stand-in. Same arguments as link_budget_rtl.py. Domain: rtl-3c, never board."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'tools/r42/fp8sw', ROOT / 'tools/r42', ROOT / 'tools/r41', ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
import points  # noqa: E402,F401  registers the proxies
import lineelig_mc  # noqa: E402

if __name__ == '__main__':
    lineelig_mc.main()
