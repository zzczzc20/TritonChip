#!/usr/bin/env python3
"""Runner for the four sealed R41 holdout points (spec/r41/holdout.json).

Validates holdout integrity, runs operations against independent CPU float32
references, measures tolerances (rtol <= 0.02, atol <= 0.02) and emits
structured reports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm')]


def digest_point(point):
    blob = json.dumps({k: v for k, v in point.items() if k != 'sha256'},
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


def verify_holdout_spec(spec_path):
    doc = json.loads(Path(spec_path).read_text())
    points = doc.get('points', [])
    if len(points) != 4:
        raise ValueError(f'Expected exactly 4 sealed holdout points, got {len(points)}')
    seen = set()
    for p in points:
        pid = p['id']
        if pid in seen:
            raise ValueError(f'Duplicate holdout point {pid}')
        seen.add(pid)
        expected = p.get('sha256')
        actual = digest_point(p)
        if actual != expected:
            raise ValueError(f'Point {pid} hash mismatch: expected {expected}, got {actual}')
    return doc


def run_point_eager(point):
    """Run point in PyTorch CPU eager mode with independent float32 reference."""
    import torch
    import torch.nn.functional as F

    seed = point.get('seed', 20261020)
    g = torch.Generator().manual_seed(seed)
    pid = point['id']
    op = point['operation']
    shape = point['shape']

    t0 = time.perf_counter()
    if op == 'matmul':
        m, k, n = shape
        a = (torch.randn((m, k), generator=g) * 0.1).to(torch.bfloat16)
        b = (torch.randn((k, n), generator=g) * 0.1).to(torch.bfloat16)
        got = torch.mm(a, b)
        ref = torch.mm(a.float(), b.float())
    elif op == 'linear':
        b_sz, in_f, out_f = shape
        x = (torch.randn((b_sz, in_f), generator=g) * 0.1).to(torch.bfloat16)
        w = (torch.randn((out_f, in_f), generator=g) * 0.1).to(torch.bfloat16)
        got = F.linear(x, w)
        ref = F.linear(x.float(), w.float())
    elif op == 'softmax':
        b_sz, n = shape
        x = (torch.randn((b_sz, n), generator=g) * 0.1).to(torch.bfloat16)
        got = F.softmax(x, dim=-1)
        ref = F.softmax(x.float(), dim=-1)
    elif op == 'rmsnorm':
        b_sz, n = shape
        x = (torch.randn((b_sz, n), generator=g) * 0.1).to(torch.bfloat16)
        got = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    else:
        raise ValueError(f'Unsupported operation: {op}')
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    diff = (got.float() - ref).abs()
    max_abs = float(diff.max().item())
    ref_scale = float(max(ref.abs().max().item(), 1.0))
    max_rel = max_abs / ref_scale

    rtol = point.get('rtol', 0.02)
    atol = point.get('atol', 0.02)
    passed = (max_rel <= rtol) and (max_abs <= atol)

    return {
        'id': pid,
        'operation': op,
        'shape': shape,
        'dtype': point['dtype'],
        'status': 'passed' if passed else 'failed',
        'max_rel_error': max_rel,
        'max_abs_error': max_abs,
        'rtol': rtol,
        'atol': atol,
        'wall_ms': elapsed_ms
    }


def run_holdout(spec_path, out_path=None, check_only=False):
    spec = verify_holdout_spec(spec_path)
    if check_only:
        report = {
            'schema': 'tnca-r41-holdout-run/1',
            'status': 'verified-only',
            'spec': str(spec_path),
            'points': len(spec['points']),
            'results': [{'id': p['id'], 'sha256': p['sha256'], 'status': 'hash-verified'}
                        for p in spec['points']]
        }
    else:
        results = []
        all_passed = True
        for p in spec['points']:
            res = run_point_eager(p)
            if res['status'] != 'passed':
                all_passed = False
            results.append(res)
        report = {
            'schema': 'tnca-r41-holdout-run/1',
            'status': 'passed' if all_passed else 'failed',
            'spec': str(spec_path),
            'points': len(results),
            'results': results
        }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, default=ROOT / 'spec/r41/holdout.json',
                    help='Holdout spec path')
    ap.add_argument('--out', type=Path, help='Output path for JSON report')
    ap.add_argument('--check-only', action='store_true',
                    help='Verify spec hashes only without numerical execution')
    args = ap.parse_args()

    report = run_holdout(args.spec, out_path=args.out, check_only=args.check_only)
    print(json.dumps(report, indent=2))
    if report['status'] not in ('passed', 'verified-only'):
        sys.exit(1)


if __name__ == '__main__':
    main()
