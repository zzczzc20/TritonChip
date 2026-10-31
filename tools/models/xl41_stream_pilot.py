#!/usr/bin/env python3
"""Bounded standard-Torch pilot for the frozen >2 GiB XL41 stream point."""
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


class MaterializedCopy(nn.Module):
    def forward(self, source):
        return source.clone()


def make_input(count):
    # Chunking only constructs host inputs; the device operation below is one
    # full materialized clone over the registered single storage.
    data = torch.empty(count, dtype=torch.bfloat16)
    for begin in range(0, count, CHUNK):
        end = min(count, begin+CHUNK)
        index = torch.arange(begin, end, dtype=torch.int64)
        data[begin:end] = (((index*17+3).remainder(257)-128).float()/128).to(torch.bfloat16)
    return data


def check_all(expected, actual, point):
    if expected.shape != actual.shape or actual.dtype != torch.bfloat16:
        raise ValueError('materialized output shape or dtype differs')
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    mismatches = 0
    exact_mismatches = 0
    max_abs = 0.0
    finite = True
    for begin in range(0, expected.numel(), CHUNK):
        end = min(expected.numel(), begin+CHUNK)
        x, y = expected[begin:end], actual[begin:end]
        source_hash.update(memoryview(x.view(torch.uint8).numpy()))
        output_hash.update(memoryview(y.view(torch.uint8).numpy()))
        xf, yf = x.float(), y.float()
        finite = finite and bool(torch.isfinite(yf).all())
        delta = (yf-xf).abs()
        mismatches += int((delta > point['atol']+point['rtol']*xf.abs()).sum())
        exact_mismatches += int((x != y).sum())
        max_abs = max(max_abs, float(delta.max()))
    boundary = (1 << 31)//expected.element_size()
    offsets = [0, 1, boundary-1, boundary, boundary+1, expected.numel()-1]
    samples = [dict(index=i, expected=float(expected[i]), actual=float(actual[i])) for i in offsets]
    return dict(finite=finite, mismatches=mismatches, exact_mismatches=exact_mismatches,
                max_abs_error=max_abs, input_sha256=source_hash.hexdigest(),
                output_sha256=output_hash.hexdigest(), boundary_samples=samples,
                checked_elements=expected.numel())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    a = p.parse_args()
    root = a.root.resolve()
    spec = root/'spec/r41/stage2/xl_core.json'
    point = next(x for x in json.loads(spec.read_text())['points'] if x['id'] == 'XL41-stream')
    definition = {k: v for k, v in point.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest != point['sha256'] or point['operation'] != 'stream' or point['dtype'] != 'bfloat16'
            or point['shape'] != [1073741888] or point['storage_bytes'] != 2147483776
            or point['materialized'] is not True or point['rtol'] != 0.02 or point['atol'] != 0.02):
        raise ValueError('frozen XL41 stream definition differs')
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='r41-xl41-stream-pilot/1', status='running', device=a.device, point=point['id'],
                  point_sha256=digest, spec_sha256=board.sha(spec), script_sha256=board.sha(Path(__file__)),
                  pilot_only=True, torch_version=str(torch.__version__))
    out = a.out/'report.json'
    def save():
        out.write_text(json.dumps(report, indent=2)+'\n')
    save()
    guard = contextlib.ExitStack()
    start = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(root))
        from archbench.provenance import software_tree
        report['software_tree'] = software_tree(root, refresh=True)
        if a.device == 'tnca':
            report['board'] = board.board_preflight(root, os.environ, guard)
            save()
        source = make_input(point['shape'][0])
        if source.untyped_storage().nbytes() != point['storage_bytes']:
            raise ValueError('source storage is not the registered >2 GiB size')
        stage = MaterializedCopy().eval()
        if a.device == 'tnca':
            report['board']['before_device_open'] = board.check_card(root)
            save()
            device = runtime.use_tnca(root)
            runtime.prepare_for_compile(stage)
            stage = torch.compile(stage, fullgraph=True, dynamic=False)
            device_source = source.to(device)
            from tnca.torch_device import storage_address
            source_address = int(storage_address(device_source))
        else:
            device_source = source
            source_address = None
        t0 = time.monotonic()
        with torch.inference_mode():
            candidate = stage(device_source)
        if candidate.device.type != ('tnca' if a.device == 'tnca' else 'cpu'):
            raise ValueError('copy output is not on the requested device')
        if candidate.untyped_storage().nbytes() != point['storage_bytes']:
            raise ValueError('output storage is not the registered >2 GiB size')
        if a.device == 'tnca':
            output_address = int(storage_address(candidate))
            size = point['storage_bytes']
            if (not (source_address+size <= output_address or output_address+size <= source_address)
                    or source_address+size <= 1 << 31):
                raise ValueError('output aliases input or input does not reach high addresses')
            report['device_ranges'] = dict(input_base=source_address, input_bytes=point['storage_bytes'],
                                           output_base=output_address, output_bytes=point['storage_bytes'])
        elif candidate.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
            raise ValueError('output aliases input')
        actual = candidate.detach().cpu()
        checks = check_all(source, actual, point)
        report.update(checks=checks, wall_seconds=time.monotonic()-t0,
                      status='pass' if checks['finite'] and checks['mismatches'] == 0 else 'fail')
        if report['status'] != 'pass':
            raise ValueError('XL41 stream numeric comparison failed')
    except Exception as exc:
        report['status'] = 'fail'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['total_wall_seconds'] = time.monotonic()-start
        save()
        guard.close()
    print(json.dumps(dict(status=report['status'], point=point['id'], pilot_only=True)))


if __name__ == '__main__':
    main()
