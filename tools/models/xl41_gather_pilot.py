#!/usr/bin/env python3
"""Bounded XL41 gather pilot preserving the formula and probing boundary rows."""
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
from xl41_matrix_pilot import compare, pattern  # noqa: E402


class Gather(nn.Module):
    def forward(self, source, indices):
        return torch.index_select(source, 0, indices)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    a = p.parse_args()
    root = a.root.resolve()
    spec = root/'spec/r41/stage2/xl_core.json'
    point = next(x for x in json.loads(spec.read_text())['points'] if x['id'] == 'XL41-gather')
    definition = {k: v for k, v in point.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest != point['sha256'] or point['operation'] != 'gather' or point['shape'] != [65536, 2048, 8193]
            or point['dtype'] != 'bfloat16' or point['indices_dtype'] != 'int64'
            or point['rtol'] != 0.02 or point['atol'] != 0.02 or point['strides'] != [2048, 1]):
        raise ValueError('frozen XL41 gather definition differs')
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='r41-xl41-gather-pilot/1', status='running', device=a.device, point=point['id'],
                  point_sha256=digest, spec_sha256=board.sha(spec), script_sha256=board.sha(Path(__file__)),
                  pilot_only=True, formal_single_vector_unresolved=True, torch_version=str(torch.__version__), runs=[])
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
        source = pattern((65536, 2048))
        formula = ((torch.arange(8193, dtype=torch.int64)//2)*65521+7).remainder(65536)
        boundary = torch.tensor([0, 32767, 32768, 65535], dtype=torch.int64)
        absent = [int(x) for x in boundary if not bool((formula == x).any())]
        report['formula_missing_required_rows'] = absent
        if absent != [0, 32767, 65535]:
            raise ValueError('frozen formula boundary inventory differs')
        stage = Gather().eval()
        if a.device == 'tnca':
            report['board']['before_device_open'] = board.check_card(root)
            save()
            device = runtime.use_tnca(root)
            runtime.prepare_for_compile(stage)
            stage = torch.compile(stage, fullgraph=True, dynamic=False)
            device_source = source.to(device)
            from tnca.torch_device import storage_address
            base = int(storage_address(device_source))
            report['source_device_range'] = dict(base=base, bytes=device_source.untyped_storage().nbytes(),
                boundary_row_addresses={str(int(i)):base+int(i)*2048*2 for i in boundary})
        else:
            device_source = source
            device = torch.device('cpu')
        with torch.inference_mode():
            for name, indices in (('formula', formula), ('boundary', boundary)):
                reference = torch.index_select(source.float(), 0, indices)
                t0 = time.monotonic()
                candidate = stage(device_source, indices.to(device))
                if candidate.device.type != ('tnca' if a.device == 'tnca' else 'cpu'):
                    raise ValueError('gather output is not on the requested device')
                checks = compare(reference, candidate.detach().cpu(), point)
                passed = checks['finite'] and checks['mismatches'] == 0
                report['runs'].append(dict(name=name, index_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                                           count=indices.numel(), passed=passed, checks=checks,
                                           wall_seconds=time.monotonic()-t0))
                save()
                if not passed:
                    raise ValueError('XL41 gather numerical comparison failed: '+name)
        report['status'] = 'pass'
    except Exception as exc:
        report['status'] = 'fail'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['total_wall_seconds'] = time.monotonic()-started
        save()
        guard.close()
    print(json.dumps(dict(status=report['status'], point=point['id'], pilot_only=True,
                          formal_single_vector_unresolved=True)))


if __name__ == '__main__':
    main()
