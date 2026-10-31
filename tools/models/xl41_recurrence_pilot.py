#!/usr/bin/env python3
"""Exploratory standard-Torch FP32 gated recurrence under the XL41 shape."""
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen35_stage2_pilot as board  # noqa: E402
import qwen_fp8_lib as runtime  # noqa: E402


class Recurrence(nn.Module):
    def forward(self, state, values, gates):
        for i in range(values.shape[2]):
            g = gates[:, :, i, :]
            state = state * torch.exp(-F.softplus(g)) + torch.sigmoid(g) * values[:, :, i, :]
        return state


def inputs():
    index = torch.arange(1*2*65*128, dtype=torch.int64).reshape(1, 2, 65, 128)
    values = ((index*17+3).remainder(257)-128).float()/128
    gates = values.clone()
    gates[:, :, 0, 0] = -20
    gates[:, :, 0, 1] = 0
    gates[:, :, 0, 2] = 20
    return values, gates


def references(values, gates):
    x, g = values.double(), gates.double()
    gate = torch.sigmoid(g)
    decay = torch.exp(-F.softplus(g))
    state = torch.zeros((1, 2, 128), dtype=torch.float64)
    result = []
    for _ in range(4):
        for i in range(65):
            state = state*decay[:, :, i, :] + gate[:, :, i, :]*x[:, :, i, :]
        result.append(state.clone())
    return result


def compare(reference, candidate, point):
    if candidate.dtype != torch.float32 or candidate.shape != reference.shape:
        raise ValueError('FP32 recurrence state shape or dtype differs')
    x, y = reference.double(), candidate.double()
    finite = bool(torch.isfinite(y).all())
    delta = (y-x).abs()
    mismatches = int((delta > point['atol']+point['rtol']*x.abs()).sum())
    return dict(finite=finite, mismatches=mismatches, checked_elements=y.numel(),
                max_abs_error=float(delta.max()), max_relative_error=float(delta.max()/x.abs().max().clamp_min(1)),
                output_sha256=hashlib.sha256(memoryview(candidate.contiguous().view(torch.uint8).numpy())).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    a = p.parse_args()
    root = a.root.resolve()
    spec = root/'spec/r41/stage2/xl_core.json'
    point = next(x for x in json.loads(spec.read_text())['points'] if x['id'] == 'XL41-recurrence')
    definition = {k: v for k, v in point.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest != point['sha256'] or point['operation'] != 'gated_recurrence' or point['shape'] != [1, 2, 65, 128]
            or point['dtype'] != 'float32' or point['state_dtype'] != 'float32'
            or point['chunk'] != 32 or point['updates'] != 4 or point['rtol'] != 0.0001 or point['atol'] != 1e-05):
        raise ValueError('frozen XL41 recurrence definition differs')
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='r41-xl41-recurrence-pilot/1', status='running', device=a.device, point=point['id'],
                  point_sha256=digest, spec_sha256=board.sha(spec), script_sha256=board.sha(Path(__file__)),
                  pilot_only=True, formal_update_rule_unresolved=True,
                  update_rule='state * exp(-softplus(gate)) + sigmoid(gate) * input',
                  torch_version=str(torch.__version__), updates=[])
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
        values, gates = inputs()
        expected = references(values, gates)
        stage = Recurrence().eval()
        if a.device == 'tnca':
            report['board']['before_device_open'] = board.check_card(root)
            save()
            device = runtime.use_tnca(root)
            runtime.prepare_for_compile(stage)
            stage = torch.compile(stage, fullgraph=True, dynamic=False)
        else:
            device = torch.device('cpu')
        values, gates = values.to(device), gates.to(device)
        state = torch.zeros((1, 2, 128), dtype=torch.float32).to(device)
        with torch.inference_mode():
            for update in range(4):
                t0 = time.monotonic()
                for begin in (0, 32, 64):
                    state = stage(state, values[:, :, begin:begin+32, :], gates[:, :, begin:begin+32, :])
                if state.device.type != ('tnca' if a.device == 'tnca' else 'cpu'):
                    raise ValueError('recurrence state is not on the requested device')
                checks = compare(expected[update], state.detach().cpu(), point)
                passed = checks['finite'] and checks['mismatches'] == 0
                report['updates'].append(dict(update=update, chunks=[32, 32, 1], passed=passed, checks=checks,
                                              wall_seconds=time.monotonic()-t0))
                save()
                if not passed:
                    raise ValueError('XL41 recurrence numerical comparison failed at update '+str(update))
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
                          formal_update_rule_unresolved=True)))


if __name__ == '__main__':
    main()
