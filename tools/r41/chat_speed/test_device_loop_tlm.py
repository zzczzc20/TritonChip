#!/usr/bin/env python3
"""Check device-controlled loop support without reading a TNCA tensor on the host."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm'), str(ROOT / 'tools/models'),
                str(ROOT / 'tools/performance_harness/systemc')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
                      TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp', TNCA_BACKEND='systemc-tlm-grid')
    result = dict(status='fail', domain='DP SystemC/TLM, not board')
    try:
        import torch
        import triton  # Load backend plugins before the TNCA driver.
        import tlm_backend
        import tnca.torch_device as device
        from tnca.inductor import host_fallbacks
        from chat import run_with_code_capture, transfer_meter

        tlm_backend.register()
        device.register()
        torch.set_num_threads(1)
        initial = torch.tensor(0, dtype=torch.int32).to('tnca')

        def loop(start):
            def cond(value):
                return value < 3
            def body(value):
                return (value + 1,)
            return torch.while_loop(cond, body, (start,))[0]

        compiled = torch.compile(loop, fullgraph=True)
        with transfer_meter(True) as warm:
            first, code = run_with_code_capture(lambda: compiled(initial))
            warm.mem.synchronize()
        with transfer_meter(True) as steady:
            second, reused = run_with_code_capture(lambda: compiled(initial))
            steady.mem.synchronize()
        warm_transfers, steady_transfers = warm.summary(), steady.summary()
        values = [int(first.cpu().item()), int(second.cpu().item())]
        fallbacks = host_fallbacks(code + reused)
        problems = []
        if values != [3, 3]:
            problems.append('loop result differs from the exact reference')
        if not code or fallbacks:
            problems.append('compiled wrapper is missing or performs host numerical work')
        if steady_transfers['host_read_calls'] or steady_transfers['staging_download_bytes']:
            problems.append('loop control read TNCA memory on the host')
        result.update(status='fail' if problems else 'pass', problems=problems, values=values,
                      host_fallbacks=fallbacks, wrappers=len(code + reused),
                      warmup_host_reads=warm_transfers['host_read_calls'],
                      steady_host_reads=steady_transfers['host_read_calls'],
                      steady_launches=steady_transfers['launches'],
                      steady_device_cycles=steady_transfers['device_cycles'])
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result['status'] != 'pass'


if __name__ == '__main__':
    sys.exit(main())
