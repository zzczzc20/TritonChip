#!/usr/bin/env python3
"""Grow resident LLaMA KV state through standard fullgraph Torch calls.

This correctness audit starts from the unchanged frozen tiny decode point.
It appends the device's new entries and feeds its own argmax token back.
Output downloads are explicit validation outside each transfer audit.
It measures neither latency nor online attention's memory complexity.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm',
          ROOT / 'tools/r40'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5/hbm')
import torch
import triton  # noqa: F401
from archbench import registry
from archbench.ops import rope_tables
from archbench.platforms.common import compare
import tnca.torch_device as D
from tnca import device
from tnca.triton_backend.driver import launch_log
from frozen_residency_tests import transfers


class DecodeStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token, keys, values, position):
        logits, new_k, new_v = self.model(token, keys[..., :position, :],
                                         values[..., :position, :])
        keys[..., position:position + 1, :].copy_(new_k)
        values[..., position:position + 1, :].copy_(new_v)
        next_token = logits.argmax(dim=-1)
        return logits, new_k, new_v, next_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--bucket', action='store_true',
                        help='fixed capacity with resident tensor position; require one graph')
    args = parser.parse_args()
    if not 2 <= args.steps <= 64:
        parser.error('steps must be between 2 and 64')
    if os.environ.get('TNCA_LAUNCH') == 'skip':
        raise RuntimeError('this audit requires actual execution')
    if os.environ.get('TNCA_TRANSFER_POLICY', 'resident') != 'resident':
        raise RuntimeError('this audit requires resident transfers')
    from tnca.backend import needs_elaborator
    if needs_elaborator(os.environ.get('TNCA_BACKEND', 'sim-file')):
        device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    point = registry.get('model.llama_decode/tiny/fp16-b1-p16')
    frozen = json.loads((ROOT / 'spec/r40/suites/R40-02.graphs3.json').read_text())
    expected = next(p['sha256'] for p in frozen['points'] if p['id'] == point.id)
    assert point.definition_sha256() == expected, 'frozen starting point changed'
    model, (token, initial_k, initial_v) = point.make(0)
    reference = copy.deepcopy(model).float()
    prefix = initial_k.shape[-2]
    capacity = prefix + args.steps
    shape = list(initial_k.shape)
    shape[-2] = capacity
    keys, values = torch.zeros(shape, dtype=initial_k.dtype), torch.zeros(shape, dtype=initial_v.dtype)
    keys[..., :prefix, :].copy_(initial_k)
    values[..., :prefix, :].copy_(initial_v)
    ref_k, ref_v = keys.float(), values.float()
    dkeys, dvalues, dtoken = keys.to('tnca'), values.to('tnca'), token.to('tnca')
    model = model.eval().to('tnca')
    # Each position selects a view of this one resident table. There is no
    # per-step host upload of RoPE state or replacement of model weights.
    cos, sin = rope_tables(capacity, model.c.hd, 'fp16')
    dcos, dsin = cos.to('tnca'), sin.to('tnca')
    ref_cos, ref_sin = cos.float(), sin.float()
    weights = tuple((p.data_ptr(), p.untyped_storage().nbytes()) for p in model.parameters())
    weight_names = {D.MEMORY.locate(p.data_ptr())[0].name for p in model.parameters()}
    cache_addresses = (dkeys.data_ptr(), dvalues.data_ptr())
    # Different prefix lengths deliberately use guarded specializations.
    # Bound their cache budget to this explicit sequence length.
    torch._dynamo.config.recompile_limit = args.steps + 2
    if args.bucket:
        from bucket_decode import BucketDecodeStep
        step = BucketDecodeStep(model, dcos, dsin, capacity)
        dposition = torch.tensor([prefix], dtype=torch.int64).to('tnca')
    else:
        step = DecodeStep(model)
    from torch._dynamo.utils import counters
    graphs_before = counters['stats']['unique_graphs']
    compiled = torch.compile(step, fullgraph=True, dynamic=False)
    session = D.MEMORY.session()
    with torch.no_grad():
        for position in range(prefix, capacity):
            if not args.bucket:
                model.cos = dcos[position:position + 1, :]
                model.sin = dsin[position:position + 1, :]
            reference.cos = ref_cos[position:position + 1, :]
            reference.sin = ref_sin[position:position + 1, :]
            want = reference(token, ref_k[..., :position, :], ref_v[..., :position, :])
            ref_k[..., position:position + 1, :].copy_(want[1])
            ref_v[..., position:position + 1, :].copy_(want[2])
            # Compile and initialize this guarded prefix specialization before
            # auditing a steady call. Repeating the append at this position
            # is idempotent: the model reads only the preceding prefix.
            if not args.bucket or position == prefix:
                with transfers(session) as initialization:
                    warm = compiled(dtoken, dkeys, dvalues,
                                    dposition if args.bucket else position)
                assert not [e for e in initialization
                            if e['direction'] == 'h2d' and e['name'] in weight_names], \
                    'prefix specialization retransferred model weights'
                assert compare(tuple(t.cpu() for t in warm[:3]), want, point.tolerance)['ok']
                del warm
            with transfers(session, stacks=True) as events, launch_log() as launches:
                out = compiled(dtoken, dkeys, dvalues,
                               dposition if args.bucket else position)
            assert launches and len(launches) >= 3
            tensor = [e for e in events if e['tensor'] or e['staging']]
            if tensor:
                path = ROOT / 'sim/build/r40' / f'decode_transfer_failure_{position}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(tensor, indent=2) + '\n')
                raise AssertionError(f'{len(tensor)} implicit tensor transfers; see {path}')
            assert all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches)
            assert tuple((p.data_ptr(), p.untyped_storage().nbytes())
                         for p in model.parameters()) == weights
            assert (dkeys.data_ptr(), dvalues.data_ptr()) == cache_addresses
            # Explicit validation downloads. Only the small public outputs
            # cross this boundary; the full KV state remains resident.
            got = tuple(t.cpu() for t in out)
            verdict = compare(got[:3], want, point.tolerance)
            assert verdict['ok'], verdict
            assert torch.equal(got[3], got[0].argmax(dim=-1))
            token, dtoken = got[3], out[3]
            if args.bucket:
                assert got[4].item() == position + 1
                dposition = out[4]
                assert counters['stats']['unique_graphs'] - graphs_before == 1, \
                    'fixed bucket recompiled for a new tensor position'
            print(f'PASS position={position} KV={position + 1}/{capacity}: '
                  f'{len(launches)} kernels, tensor transfers=0, '
                  f'max_rel_err={verdict["max_rel_err"]:.6g}', flush=True)
            del out
    # Final explicit state check proves that the in-place appends survived
    # beyond the per-step result tensors and the prefix was preserved.
    verdict = compare((dkeys.cpu(), dvalues.cpu()), (ref_k, ref_v), point.tolerance)
    assert verdict['ok'], verdict
    assert not session.unfinished and all(not b.leases for b in session._buffers.values())
    if args.bucket:
        print('AUDIT fixed bucket: one compiled graph, resident tensor position', flush=True)
    print(f'AUDIT {session.memory_stats}', flush=True)
    print(f'r40-decode-residency-tests PASS ({args.steps}/{args.steps}), '
          f'KV {prefix}->{capacity}, fixed cache addresses, weights unchanged', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
