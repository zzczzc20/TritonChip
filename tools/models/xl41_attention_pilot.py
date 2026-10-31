#!/usr/bin/env python3
"""Bounded standard-Torch pilot for frozen XL41 causal ragged GQA."""
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
from xl41_matrix_pilot import compare, pattern  # noqa: E402


class Attention(nn.Module):
    def forward(self, query, key, value, mask, valid_query):
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                 dropout_p=0.0, is_causal=False, enable_gqa=True)
        return torch.where(valid_query, result, torch.zeros_like(result))


def masks(lengths, size):
    pos = torch.arange(size)
    valid_query = pos[None, None, :, None] < lengths[:, None, None, None]
    valid_key = pos[None, None, None, :] < lengths[:, None, None, None]
    causal = pos[None, None, None, :] <= pos[None, None, :, None]
    # Give a masked query one valid key to avoid an all-masked softmax;
    # the query's output is then explicitly masked to zero.
    mask = (valid_key & causal & valid_query) | (~valid_query & (pos[None, None, None, :] == 0))
    return mask, valid_query


def independent_reference(q, k, v, mask, valid_query):
    k32 = k.float().repeat_interleave(4, dim=1)
    v32 = v.float().repeat_interleave(4, dim=1)
    scores = (q.float() @ k32.transpose(-1, -2)) / 16.0
    probabilities = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1)
    result = probabilities @ v32
    return torch.where(valid_query, result, torch.zeros_like(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    a = p.parse_args()
    root = a.root.resolve()
    spec = root/'spec/r41/stage2/xl_core.json'
    point = next(x for x in json.loads(spec.read_text())['points'] if x['id'] == 'XL41-attention')
    definition = {k: v for k, v in point.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest != point['sha256'] or point['operation'] != 'attention' or point['shape'] != [2, 8, 2, 129, 256]
            or point['dtype'] != 'bfloat16' or point['lengths'] != [129, 127]
            or point['causal'] is not True or point['gqa'] is not True or point['masked_tail'] is not True
            or point['rtol'] != 0.02 or point['atol'] != 0.02):
        raise ValueError('frozen XL41 attention definition differs')
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='r41-xl41-attention-pilot/1', status='running', device=a.device, point=point['id'],
                  point_sha256=digest, spec_sha256=board.sha(spec), script_sha256=board.sha(Path(__file__)),
                  pilot_only=True, masked_query_policy='zero output', torch_version=str(torch.__version__))
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
        q = pattern((2, 8, 129, 256))
        k, v = pattern((2, 2, 129, 256)), pattern((2, 2, 129, 256))
        mask, valid = masks(torch.tensor(point['lengths'], dtype=torch.int64), 129)
        reference = independent_reference(q, k, v, mask, valid)
        stage = Attention().eval()
        operands = [q, k, v, mask, valid]
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
            raise ValueError('attention output is not on the requested device')
        actual = candidate.detach().cpu()
        checks = compare(reference, actual, point)
        tail_zero = bool((actual[1, :, 127:, :] == 0).all())
        report.update(checks=checks, tail_zero=tail_zero, wall_seconds=time.monotonic()-t0,
                      status='pass' if checks['finite'] and checks['mismatches'] == 0 and tail_zero else 'fail')
        if report['status'] != 'pass':
            raise ValueError('XL41 attention numerical or tail check failed')
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
