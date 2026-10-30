#!/usr/bin/env python3
"""Extract the candidate xclbin HBM map without opening an FPGA device."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'python'))
from tnca.backends.u50 import IP_NAME, _xclbin_bank, _xclbin_hbm_allocation
import auto


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    groups = [_xclbin_bank(str(a.image), IP_NAME, arg_index=i) for i in range(3)]
    allocation = _xclbin_hbm_allocation(str(a.image), 0, 8 << 30)
    result = dict(image_sha256=auto.sha(a.image), groups=groups, allocation_8gib=allocation,
                  scope='static xclbin parse; no device opened and no board result')
    with a.out.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print('static HBM map extracted:', result['image_sha256'])


if __name__ == '__main__':
    main()
