#!/usr/bin/env python3
"""Audit actual tensor transfers on an unchanged frozen multi-kernel point.

This is a correctness checkpoint, not a latency measurement or a KV growth
test. Each point runs in its own process. Output copies are explicit and
outside the steady graph audit; runtime transfers inside it are observed.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
from archbench import registry
from archbench.platforms.common import compare, flatten, reference
import tnca.torch_device as D
from tnca import device
from tnca.triton_backend.driver import launch_log


@contextmanager
def transfers(session):
    """Observe successful runtime copies without changing their payloads."""
    events = []
    upload, download = session.upload, session.download

    def record(direction, buf, size):
        events.append(dict(direction=direction, bytes=size, name=buf.name,
                           tensor=D.MEMORY.buffers.get(buf.addr) is buf,
                           staging=buf.name.startswith('staging:')))

    def put(buf, data, offset=0):
        count = upload(buf, data, offset)
        record('h2d', buf, count)
        return count

    def get(buf, nbytes=None, offset=0):
        data = download(buf, nbytes, offset)
        record('d2h', buf, len(data))
        return data

    session.upload, session.download = put, get
    try:
        yield events
    finally:
        # Restore normal method lookup as well as behavior.
        del session.upload, session.download


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--point', required=True)
    ap.add_argument('--steps', type=int, default=2)
    args = ap.parse_args()
    if args.steps < 2:
        ap.error('--steps must be at least 2')
    subset = json.loads((ROOT/'spec/r40/suites/R40-02.graphs3.json').read_text())
    frozen = {p['id']: p['sha256'] for p in subset['points']}
    if args.point not in frozen:
        ap.error('point must belong to the frozen R40-02.graphs3 subset')
    point = registry.get(args.point)
    assert point.definition_sha256() == frozen[point.id], 'frozen definition changed'
    if os.environ.get('TNCA_LAUNCH') == 'skip':
        raise RuntimeError('this audit requires actual kernel execution')
    if os.environ.get('TNCA_BACKEND', 'sim-file') in ('sim-file', 'sim-pipe'):
        device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    want = reference(point, 0)
    model, inputs = point.make(0)
    model = model.eval().to('tnca')
    inputs = tuple(t.to('tnca') if isinstance(t, torch.Tensor) else t for t in inputs)
    weights = tuple((p.data_ptr(), p.untyped_storage().nbytes()) for p in model.parameters())
    compiled = torch.compile(model, fullgraph=True)
    session = D.MEMORY.session()
    os.environ['TNCA_TRANSFER_POLICY'] = 'resident'
    with torch.no_grad():
        warm = compiled(*inputs)
        assert compare(warm, want, point.tolerance)['ok']
        del warm
        identity = None
        for policy in ('resident', 'per-kernel'):
            os.environ['TNCA_TRANSFER_POLICY'] = policy
            for step in range(args.steps):
                with transfers(session) as events, launch_log() as launches:
                    out = compiled(*inputs)
                assert len(launches) >= 3, 'the graph must actually launch >=3 kernels'
                artifacts = [e['artifact'] for e in launches]
                identity = artifacts if identity is None else identity
                assert artifacts == identity, 'transfer policy changed the artifacts'
                tensor = [e for e in events if e['tensor'] or e['staging']]
                actual = {d: sum(e['bytes'] for e in tensor if e['direction'] == d)
                          for d in ('h2d', 'd2h')}
                reported = dict(h2d=sum(e['upload_bytes'] for e in launches),
                                d2h=sum(e['download_bytes'] for e in launches))
                assert actual == reported, (actual, reported, events)
                assert (actual == dict(h2d=0, d2h=0) if policy == 'resident'
                        else actual['h2d'] == actual['d2h'] and actual['h2d'] > 0)
                assert compare(out, want, point.tolerance)['ok']
                assert tuple((p.data_ptr(), p.untyped_storage().nbytes())
                             for p in model.parameters()) == weights
                # Positive control: an explicit tensor download must be seen.
                with transfers(session) as explicit:
                    tuple(t.cpu() for t in flatten(out))
                assert any(e['tensor'] and e['direction'] == 'd2h' and e['bytes'] > 0
                           for e in explicit), explicit
                print(f'PASS {point.id} {policy} step={step}: '
                      f'{len(launches)} kernels, actual tensor bytes={actual}', flush=True)
                del out
    print(f'r40-frozen-residency-tests PASS ({args.steps * 2}/{args.steps * 2})', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
