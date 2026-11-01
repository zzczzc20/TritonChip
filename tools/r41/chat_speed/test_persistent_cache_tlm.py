#!/usr/bin/env python3
"""One process of a cold/warm persistent graph cache test on DP TLM only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm'), str(ROOT / 'tools/models'),
                str(ROOT / 'tools/performance_harness/systemc')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--expect-hit', action='store_true')
    parser.add_argument('--expect-miss', action='store_true')
    parser.add_argument('--shift', type=int, default=0,
                        help='change the graph and exact result to exercise cache invalidation')
    args = parser.parse_args()
    if args.expect_hit and args.expect_miss:
        parser.error('choose at most one cache expectation')
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
                      TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp', TNCA_BACKEND='systemc-tlm-grid')
    import torch
    import triton  # Discover plugins before importing the TNCA driver.
    import tlm_backend
    import tnca.torch_device as device
    from tnca.inductor import host_fallbacks
    from tnca.triton_backend import driver
    from torch._dynamo.utils import counters
    from chat import run_with_code_capture, transfer_meter

    tlm_backend.register()
    device.register()
    torch.set_num_threads(1)
    a = (torch.arange(256, dtype=torch.int32).reshape(16, 16) % 7 - 3).float()
    b = (torch.arange(256, dtype=torch.int32).reshape(16, 16) % 5 - 2).float()
    expected = a @ b + args.shift
    a_device, b_device = a.to('tnca'), b.to('tnca')
    shift = args.shift
    fn = torch.compile(lambda x, y: x @ y + shift, fullgraph=True)
    started = time.perf_counter()
    with transfer_meter(True) as meter, driver.launch_log() as launches:
        result, code = run_with_code_capture(lambda: fn(a_device, b_device))
        meter.mem.synchronize()
    seconds = time.perf_counter() - started
    transfers = meter.summary()
    got = result.cpu()
    mismatches = int((got.view(torch.int32) != expected.view(torch.int32)).sum())
    fallback = host_fallbacks(code)
    hits = int(counters['inductor']['fxgraph_cache_hit'])
    misses = int(counters['inductor']['fxgraph_cache_miss'])
    measured = [row for row in launches if not row.get('skipped')]
    problems = []
    if mismatches or not code or fallback:
        problems.append('output or wrapper fallback proof failed')
    if transfers['host_read_calls'] or transfers['staging_download_bytes']:
        problems.append('host read during compiled device work')
    if not measured:
        problems.append('no device launches were measured')
    if args.expect_hit and not hits:
        problems.append('no persistent FX graph cache hit')
    if args.expect_miss and not misses:
        problems.append('changed graph did not produce an FX graph cache miss')
    summary = dict(status='fail' if problems else 'pass', domain='DP SystemC/TLM, not board',
                   problems=problems, elapsed_seconds=seconds, fx_graph_cache_hits=hits,
                   fx_graph_cache_misses=misses, graph_shift=shift, bit_mismatches=mismatches,
                   inspected_wrappers=len(code), host_fallbacks=fallback,
                   host_read_calls=transfers['host_read_calls'], launches=len(measured),
                   device_cycles=sum(row.get('device_cycles') or 0 for row in measured))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return bool(problems)


if __name__ == '__main__':
    sys.exit(main())
