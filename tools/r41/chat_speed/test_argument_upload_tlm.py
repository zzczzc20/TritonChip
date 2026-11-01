#!/usr/bin/env python3
"""Compare repeated stable ABI blocks on DP TLM; no board is accessed."""
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
    parser.add_argument('--cache', choices=('0', '1'), required=True)
    args = parser.parse_args()
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
                      TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp', TNCA_BACKEND='systemc-tlm-grid',
                      TNCA_ARG_UPLOAD_CACHE=args.cache)
    import torch
    import triton  # Discover plugins before importing the TNCA driver.
    import tlm_backend
    import tnca.torch_device as device
    from tnca.inductor import host_fallbacks
    from tnca.runtime import Session
    from chat import run_with_code_capture, transfer_meter

    tlm_backend.register()
    device.register()
    torch.set_num_threads(1)
    a = (torch.arange(256, dtype=torch.int32) % 7 - 3).float()
    b = (torch.arange(256, dtype=torch.int32) % 5 - 2).float()
    expected = a + b
    a_device, b_device = a.to('tnca'), b.to('tnca')
    output = torch.empty_like(a_device)

    def add_into(x, y, out):
        torch.add(x, y, out=out)
        return out

    compiled = torch.compile(add_into, fullgraph=True)
    session = device.MEMORY.session()
    loads = 0
    original_load = Session.load

    def load(self, *pos, **kw):
        nonlocal loads
        module = original_load(self, *pos, **kw)
        loads += 1
        return module

    Session.load = load
    try:
        _, code = run_with_code_capture(lambda: compiled(a_device, b_device, output))
        session.drain()
        before, load_before = session.argument_upload_stats, loads
        before_counters = session.dev.counters_snapshot()
        started = time.perf_counter()
        with transfer_meter(True) as meter:
            for _ in range(4):
                compiled(a_device, b_device, output)
                session.drain()
        elapsed = time.perf_counter() - started
        after_counters = session.dev.counters_snapshot()
        after = session.argument_upload_stats
        transfers = meter.summary()
        value = output.cpu()
        mismatches = int((value.view(torch.int32) != expected.view(torch.int32)).sum())
        fallback = host_fallbacks(code)
        instr = sum(value - before_counters.get(key, 0) for key, value in after_counters.items()
                    if key.startswith('core_') and key.endswith('_instructions'))
        delta = {key: after[key] - before[key] for key in
                 ('uploads_avoided', 'bytes_avoided', 'uploads_attempted', 'bytes_attempted')}
        problems = []
        if mismatches or not code or fallback:
            problems.append('bitwise output or wrapper fallback proof failed')
        if transfers['host_read_calls'] or transfers['staging_download_bytes']:
            problems.append('host read during repeated device work')
        if args.cache == '1' and not delta['uploads_avoided']:
            problems.append('no argument upload reuse was observed')
        if args.cache == '0' and delta['uploads_avoided']:
            problems.append('disabled argument cache reported reuse')
        result = dict(status='fail' if problems else 'pass', domain='DP SystemC/TLM, not board',
                      problems=problems, cache_enabled=args.cache == '1', bit_mismatches=mismatches,
                      host_fallbacks=fallback, host_read_calls=transfers['host_read_calls'],
                      wall_seconds=elapsed, launches=transfers['launches'], device_cycles=transfers['device_cycles'],
                      instructions=instr, instructions_per_mac=None,
                      code_loads=loads - load_before, argument_uploads=delta)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return bool(problems)
    finally:
        Session.load = original_load


if __name__ == '__main__':
    sys.exit(main())
