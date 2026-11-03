#!/usr/bin/env python3
"""One small DP TLM batch for device tail clearing and queued feedback.

This is not a model-quality or board-speed measurement. It checks exact
device bytes and host-read counters before larger model or board testing.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm'), str(ROOT / 'tools/models'),
                str(ROOT / 'tools/performance_harness/systemc')]
# Triton requires exactly one active backend; this script targets the TLM.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
                  TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp', TNCA_BACKEND='systemc-tlm-grid',
                  TNCA_SUBMIT='batched', TNCA_TRANSFER_POLICY='resident')

import torch
import triton  # Discover backend plugins before importing the driver.
import tlm_backend
import tnca.torch_device as device
from tnca.inductor import host_fallbacks
from tnca.torch_device.copy import zero_view_tensor
from chat import run_with_code_capture, transfer_meter


def measured_sequence(compiled, value, *, queued, count=3):
    meters = []
    started = time.perf_counter()
    for _ in range(count):
        with transfer_meter(True, deferred=queued) as meter:
            value = compiled(value)
            if not queued:
                meter.mem.synchronize()
        meters.append(meter)
    device.MEMORY.synchronize()
    wall = time.perf_counter() - started
    totals = [meter.summary() for meter in meters]
    return value, dict(wall_seconds=wall,
                       launches=sum(x['launches'] for x in totals),
                       device_cycles=sum(x['device_cycles'] or 0 for x in totals),
                       missing_cycles=any(x['device_cycles'] is None for x in totals),
                       host_reads=sum(x['host_read_calls'] for x in totals),
                       staging_download_bytes=sum(x['staging_download_bytes'] for x in totals))


def main():
    tlm_backend.register()
    device.register()
    torch.set_num_threads(1)
    src = torch.arange(256, dtype=torch.float32).reshape(1, 2, 16, 8).to(torch.bfloat16)
    expected = src.clone()
    expected[:, :, 3:6, :] = 0
    resident = src.to('tnca')
    with transfer_meter(True) as meter:
        zero_view_tensor(resident[:, :, 3:6, :])
        meter.mem.synchronize()
    zero_stats = meter.summary()
    actual = resident.cpu()
    zero_ok = torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    initial = torch.arange(16, dtype=torch.float32)
    first, second = initial.to('tnca'), initial.to('tnca')
    compiled = torch.compile(lambda x: x * 2 + 1, fullgraph=True, dynamic=False)
    warm, wrappers = run_with_code_capture(lambda: compiled(first))
    device.MEMORY.synchronize()
    del warm
    fallback = host_fallbacks(wrappers) if wrappers else ['uninspected wrapper']
    control, control_stats = measured_sequence(compiled, first, queued=False)
    candidate, candidate_stats = measured_sequence(compiled, second, queued=True)
    reference = ((initial * 2 + 1) * 2 + 1) * 2 + 1
    control_bits = control.cpu().view(torch.int32)
    candidate_bits = candidate.cpu().view(torch.int32)
    expected_bits = reference.view(torch.int32)
    exact = torch.equal(control_bits, expected_bits) and torch.equal(candidate_bits, expected_bits)
    no_reads = all(x['host_reads'] == 0 and x['staging_download_bytes'] == 0
                   for x in (control_stats, candidate_stats))
    ok = zero_ok and not zero_stats['host_read_calls'] and not zero_stats['staging_download_bytes'] \
        and exact and no_reads and not fallback
    result = dict(status='pass' if ok else 'fail', scope='small DP TLM batch, not board or full model',
                  tail_clear_bitwise=zero_ok, tail_host_reads=zero_stats['host_read_calls'],
                  feedback_bitwise=exact, compiled_wrapper_fallbacks=fallback,
                  control=control_stats, queued=candidate_stats)
    print(json.dumps(result), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
