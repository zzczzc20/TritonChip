#!/usr/bin/env python3
"""Bounded standard-Torch pilot for the frozen XL41 BF16 GEMV point."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen35_stage2_pilot as board  # noqa: E402
import qwen_fp8_lib as runtime  # noqa: E402


def pattern(shape):
    count = 1
    for n in shape:
        count *= n
    index = torch.arange(count, dtype=torch.int64)
    return (((index*17+3).remainder(257)-128).float()/128).to(torch.bfloat16).reshape(shape)


class Matmul(nn.Module):
    def forward(self, left, right):
        return torch.matmul(left, right)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    a = p.parse_args()
    root = a.root.resolve()
    spec_path = root/'spec/r41/stage2/xl_core.json'
    points = json.loads(spec_path.read_text())['points']
    point = next(x for x in points if x['id'] == 'XL41-gemv')
    definition = {k: v for k, v in point.items() if k != 'sha256'}
    point_sha = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (point_sha != point['sha256'] or point['operation'] != 'matmul' or point['shape'] != [1, 2048, 6144]
            or point['dtype'] != 'bfloat16' or point['rtol'] != 0.02 or point['atol'] != 0.02):
        raise ValueError('frozen XL41 GEMV definition differs')
    a.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='r41-xl41-gemv-pilot/1', status='running', device=a.device, point=point['id'],
                  point_sha256=point_sha, spec_sha256=board.sha(spec_path), script_sha256=board.sha(Path(__file__)),
                  pilot_only=True, torch_version=str(torch.__version__))
    out = a.out/'report.json'
    def save():
        out.write_text(json.dumps(result, indent=2)+'\n')
    save()
    guard = contextlib.ExitStack()
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(root))
        from archbench.provenance import software_tree
        result['software_tree'] = software_tree(root, refresh=True)
        if a.device == 'tnca':
            result['board'] = board.board_preflight(root, os.environ, guard)
            save()
        left, right = pattern((1, 2048)), pattern((2048, 6144))
        reference = left.float() @ right.float()
        stage = Matmul().eval()
        if a.device == 'tnca':
            result['board']['before_device_open'] = board.check_card(root)
            save()
            device = runtime.use_tnca(root)
            runtime.prepare_for_compile(stage)
            stage = torch.compile(stage, fullgraph=True, dynamic=False)
            left, right = left.to(device), right.to(device)
        t0 = time.monotonic()
        with torch.inference_mode():
            candidate = stage(left, right)
        if a.device == 'tnca' and candidate.device.type != 'tnca':
            raise ValueError('candidate output did not remain on TNCA')
        if candidate.dtype != torch.bfloat16 or tuple(candidate.shape) != (1, 6144):
            raise ValueError('candidate output shape or dtype differs')
        actual = candidate.detach().cpu().float()
        finite = bool(torch.isfinite(actual).all())
        difference = (actual-reference).abs()
        allowed = point['atol']+point['rtol']*reference.abs()
        failed = int((difference > allowed).sum())
        from safetensors.torch import save_file
        data_path = a.out/'values.safetensors'
        save_file(dict(reference=reference.contiguous(), candidate=actual.contiguous()), str(data_path))
        result.update(status='pass' if finite and failed == 0 else 'fail', finite=finite,
                      mismatches=failed, max_abs_error=float(difference.max()),
                      max_relative_error=float(difference.max()/reference.abs().max().clamp_min(1)),
                      output_values_sha256=board.sha(data_path), wall_seconds=time.monotonic()-t0)
        if result['status'] != 'pass':
            raise ValueError('XL41 GEMV numeric comparison failed')
    except Exception as exc:
        result['status'] = 'fail'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['total_wall_seconds'] = time.monotonic()-started
        save()
        guard.close()
    print(json.dumps(dict(status=result['status'], point=point['id'], pilot_only=True)))


if __name__ == '__main__':
    main()
