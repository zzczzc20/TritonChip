#!/usr/bin/env python3
"""Native residency/staging controls through the standard Torch path.

This capability checkpoint checks transfer counts, artifact identity and
alias semantics. Frozen latency, batching and asynchronous stress gates
remain separate measurements; this tool does not mark them passing.
"""
import os
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca.triton_backend.driver import launch_log


def measured(fn, policy):
    os.environ['TNCA_TRANSFER_POLICY'] = policy
    with torch.no_grad(), launch_log() as log:
        out = fn()
        host = out.cpu()
    assert log and all(not e.get('skipped') for e in log), 'graph did not execute'
    up = sum(e['upload_bytes'] for e in log)
    down = sum(e['download_bytes'] for e in log)
    if policy == 'resident':
        assert up == down == 0, (up, down)
    else:
        assert up == down and up > 0, (up, down)
    return host, log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence', action='store_true')
    args = ap.parse_args()
    if os.environ.get('TNCA_BACKEND', 'sim-file') in ('sim-file', 'sim-pipe'):
        from tnca import device
        device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(40)
    h = torch.randn(16, 16)
    x = h.to('tnca')
    graph = torch.compile(lambda t: (t * 2 + 1).softmax(-1), fullgraph=True)
    ref = (h * 2 + 1).softmax(-1)
    # Compile first. Inductor's one-time SDPA pattern initialization
    # (fuse_attention._sfdp_init) traces example tensors on the device and
    # launches native copies inside the first call; the policies are compared
    # on the steady graph, not on that one-time compilation work.
    os.environ['TNCA_TRANSFER_POLICY'] = 'resident'
    with torch.no_grad():
        graph(x).cpu()
    artifacts = []
    for policy in ('resident', 'per-kernel'):
        for _ in range(2):
            host, log = measured(lambda: graph(x), policy)
            torch.testing.assert_close(host, ref, rtol=1e-4, atol=1e-5)
            artifacts.append([e['artifact'] for e in log])
        print(f'PASS {policy}: graph correctness and transfer audit', flush=True)
    assert all(a == artifacts[0] for a in artifacts), 'policy changed compiled artifacts'
    print('PASS residency/staging artifact identity', flush=True)
    # Two arguments share one backing storage and have different offsets.
    # Staging each argument independently would break this mutation's alias.
    mutate = torch.compile(lambda a, b: a.add_(b), fullgraph=True)
    for policy in ('resident', 'per-kernel'):
        dx = h.to('tnca')
        a, b = dx[1:9, :8], dx[2:10, 8:]
        want = h.clone()
        want[1:9, :8].add_(want[2:10, 8:])
        _, log = measured(lambda: mutate(a, b), policy)
        torch.testing.assert_close(dx.cpu(), want)
        if policy == 'per-kernel':
            # Exactly one original -> shadow -> original round trip.
            assert len(log) == 1 and log[0]['upload_bytes'] == 2 * dx.untyped_storage().nbytes(), log
        print(f'PASS {policy}: strided offset aliases and in-place update', flush=True)
    os.environ['TNCA_TRANSFER_POLICY'] = 'invalid-policy'
    try:
        graph(x)
    except Exception as e:
        assert 'TNCA_TRANSFER_POLICY' in str(e), e
    else:
        raise AssertionError('unknown policy accepted')
    print('PASS unsupported transfer policy refused', flush=True)
    print('r40-residency-tests PASS (6/6)', flush=True)
    if args.evidence:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import evidence
        checks = [dict(name=name, ok=True) for name in (
            'resident:zero-kernel-transfers', 'staging:actual-round-trips',
            'policy:artifact-identity', 'resident:offset-alias',
            'staging:offset-alias', 'policy:unknown-refused')]
        evidence.record('R40-02.zero-transfer', 'not-measured', checks,
                        detail='control and alias smoke passed; frozen chained graphs, '
                               'multi-step decode and steady weight audit remain')
    return 0


if __name__ == '__main__':
    sys.exit(main())
