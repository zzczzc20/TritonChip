#!/usr/bin/env python3
"""Print a profile as the Verilog header its device is compiled with.

    tools/profile_header.py core-vrf-sync-v1 > sim/build/params/core-vrf-sync-v1.vh

The same text python/tnca/device.py writes when it builds a device: one
`define per parameter the profile states (tnca.profiles.uarch_header).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from tnca import profiles  # noqa: E402

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.stdout.write(profiles.uarch_header(sys.argv[1]))
