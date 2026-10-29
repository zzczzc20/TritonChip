#!/usr/bin/env python3
"""Functional dynamic-SDPA check on the SystemC/TLM model, not a board benchmark."""
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm'), str(ROOT / 'tools/performance_harness/systemc')]


def main():
    import torch
    import torch.nn.functional as F
    import triton  # noqa: F401
    import tnca.inductor  # noqa: F401
    import tnca.torch_device as device
    import tlm_backend
    from archbench.provenance import software_tree

    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'domain': 'SystemC/TLM functional test, not board', 'status': 'running',
              'source_tree': software_tree(ROOT, refresh=True),
              'profile': os.environ.get('TNCA_PROFILE'), 'atol': 0.00001, 'rtol': 0.0001,
              'cases': []}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        tlm_backend.register()
        device.register()
        torch.manual_seed(20261029)

        def attention(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        compiled = torch.compile(attention, backend='inductor', fullgraph=True)
        for length in (1, 16, 128):
            case = {'query_length': length, 'key_length': 128, 'heads': 8, 'head_width': 64,
                    'status': 'running'}
            report['cases'].append(case)
            save()
            q = torch.randn(1, 8, length, 64, dtype=torch.float32)
            k = torch.randn(1, 8, 128, 64, dtype=torch.float32)
            v = torch.randn(1, 8, 128, 64, dtype=torch.float32)
            reference = attention(q, k, v)
            actual = compiled(q.to('tnca'), k.to('tnca'), v.to('tnca')).cpu()
            case['max_abs_error'] = (actual - reference).abs().max().item()
            torch.testing.assert_close(actual, reference, rtol=report['rtol'], atol=report['atol'])
            case['status'] = 'pass'
            save()
        report['status'] = 'pass'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
