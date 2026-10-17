#!/usr/bin/env python3
"""Profile and actual heap properties across profile changes in one process."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
import triton  # noqa: F401
from tnca import device
from tnca.adapters import r1
from tnca.torch_device.inductor_device import properties
from tnca.triton_backend.driver import TNCAUtils, _Sessions


def main():
    utils = TNCAUtils()
    for profile in ('core-vrf-sync-v5', 'core-vrf-sync-v5/hbm', 'core-vrf-sync-v4'):
        os.environ['TNCA_PROFILE'] = profile
        device.prepare(profile)
        p = properties(0)
        target, _ = r1.target_for(device.base_profile(profile))
        heap = _Sessions.get(profile)._heap
        assert p.profile == profile
        assert p.total_memory == heap.limit - heap.base
        assert p.regs_per_multiprocessor == target.phys_vregs
        t = utils.get_device_properties(0)
        assert t['max_threads_per_block'] == target.vlen
        assert t['multiprocessor_count'] == target.num_cores
        assert t['max_num_regs'] == target.phys_vregs
        assert t['max_shared_mem'] == 0 and t['warpSize'] == 1
        print(f'PASS {profile}: {p.total_memory} heap bytes, {target.phys_vregs} registers', flush=True)
    try:
        properties(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError('nonexistent device accepted')
    print('PASS nonexistent device refused')
    print('r40-properties-tests PASS (4/4)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
