#!/usr/bin/env python3
"""Bounded standard-Torch pilots for frozen XL41 transpose and matrix points."""
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

CHUNK = 1 << 20
DEFINITIONS = {
    'XL41-transpose': ('transpose', [16384, 16384]),
    'XL41-gemm': ('matmul', [512, 2048, 6144]),
    'XL41-projection': ('matmul', [1, 2048, 248320]),
}


def pattern(shape):
    count = 1
    for n in shape:
        count *= n
    data = torch.empty(count, dtype=torch.bfloat16)
    for begin in range(0, count, CHUNK):
        end = min(count, begin+CHUNK)
        index = torch.arange(begin, end, dtype=torch.int64)
        data[begin:end] = (((index*17+3).remainder(257)-128).float()/128).to(torch.bfloat16)
    return data.reshape(shape)


class Matmul(nn.Module):
    def forward(self, left, right):
        return torch.matmul(left, right)


class Transpose(nn.Module):
    def forward(self, source):
        return source.transpose(0, 1).contiguous()


def compare(reference, candidate, point):
    if reference.shape != candidate.shape or candidate.dtype != torch.bfloat16:
        raise ValueError('output shape or dtype differs')
    # Flattening materializes the transposed CPU reference; every output is
    # checked, and the full candidate checksum is retained.
    expected = reference.reshape(-1)
    actual = candidate.reshape(-1)
    output_hash = hashlib.sha256()
    mismatches, finite, max_abs, max_ref = 0, True, 0.0, 0.0
    for begin in range(0, expected.numel(), CHUNK):
        end = min(expected.numel(), begin+CHUNK)
        x, y = expected[begin:end].float(), actual[begin:end]
        output_hash.update(memoryview(y.view(torch.uint8).numpy()))
        yf = y.float()
        finite = finite and bool(torch.isfinite(yf).all())
        delta = (yf-x).abs()
        mismatches += int((delta > point['atol']+point['rtol']*x.abs()).sum())
        max_abs = max(max_abs, float(delta.max()))
        max_ref = max(max_ref, float(x.abs().max()))
    offsets = sorted({0, expected.numel()//2, expected.numel()-1})
    samples = [dict(index=i, expected=float(expected[i]), actual=float(actual[i])) for i in offsets]
    return dict(finite=finite, mismatches=mismatches, max_abs_error=max_abs,
                max_relative_error=max_abs/max(max_ref, 1), output_sha256=output_hash.hexdigest(),
                checked_elements=expected.numel(), samples=samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    p.add_argument('--point', choices=tuple(DEFINITIONS), required=True)
    a = p.parse_args()
    root = a.root.resolve()
    spec = root/'spec/r41/stage2/xl_core.json'
    point = next(x for x in json.loads(spec.read_text())['points'] if x['id'] == a.point)
    definition = {k: v for k, v in point.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    operation, shape = DEFINITIONS[a.point]
    if (digest != point['sha256'] or point['operation'] != operation or point['shape'] != shape
            or point['dtype'] != 'bfloat16' or point['rtol'] != 0.02 or point['atol'] != 0.02
            or (a.point == 'XL41-transpose' and point.get('materialized') is not True)):
        raise ValueError('frozen XL41 point definition differs')
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='r41-xl41-matrix-pilot/1', status='running', device=a.device, point=point['id'],
                  point_sha256=digest, spec_sha256=board.sha(spec), script_sha256=board.sha(Path(__file__)),
                  pilot_only=True, torch_version=str(torch.__version__))
    out = a.out/'report.json'
    def save():
        out.write_text(json.dumps(report, indent=2)+'\n')
    save()
    guard = contextlib.ExitStack()
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(root))
        from archbench.provenance import software_tree
        report['software_tree'] = software_tree(root, refresh=True)
        if a.device == 'tnca':
            report['board'] = board.board_preflight(root, os.environ, guard)
            save()
        if operation == 'transpose':
            operands = [pattern(tuple(shape))]
            reference = operands[0].transpose(0, 1)
            stage = Transpose().eval()
        else:
            m, k, n = shape
            operands = [pattern((m, k)), pattern((k, n))]
            reference = operands[0].float() @ operands[1].float()
            stage = Matmul().eval()
        if a.device == 'tnca':
            report['board']['before_device_open'] = board.check_card(root)
            save()
            device = runtime.use_tnca(root)
            runtime.prepare_for_compile(stage)
            stage = torch.compile(stage, fullgraph=True, dynamic=False)
            operands = [x.to(device) for x in operands]
        t0 = time.monotonic()
        with torch.inference_mode():
            candidate = stage(*operands)
        if candidate.device.type != ('tnca' if a.device == 'tnca' else 'cpu'):
            raise ValueError('output is not on the requested device')
        if operation == 'transpose':
            if not candidate.is_contiguous():
                raise ValueError('transpose output was not materialized')
            if a.device == 'tnca':
                from tnca.torch_device import storage_address
                inp, outp = int(storage_address(operands[0])), int(storage_address(candidate))
                size = operands[0].untyped_storage().nbytes()
                if not (inp+size <= outp or outp+candidate.untyped_storage().nbytes() <= inp):
                    raise ValueError('transpose output aliases its input')
                report['device_ranges'] = dict(input_base=inp, input_bytes=size, output_base=outp,
                                               output_bytes=candidate.untyped_storage().nbytes())
            elif candidate.untyped_storage().data_ptr() == operands[0].untyped_storage().data_ptr():
                raise ValueError('transpose output aliases its input')
        actual = candidate.detach().cpu()
        checks = compare(reference, actual, point)
        report.update(checks=checks, wall_seconds=time.monotonic()-t0,
                      status='pass' if checks['finite'] and checks['mismatches'] == 0 else 'fail')
        if report['status'] != 'pass':
            raise ValueError('XL41 numerical comparison failed')
    except Exception as exc:
        report['status'] = 'fail'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['total_wall_seconds'] = time.monotonic()-started
        save()
        guard.close()
    print(json.dumps(dict(status=report['status'], point=point['id'], pilot_only=True)))


if __name__ == '__main__':
    main()
