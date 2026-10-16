#!/usr/bin/env python3
"""Audit real ring buffers in standard native Torch GEMM; no perf sign-off."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--stages', type=int, choices=(1, 2, 3), required=True)
ap.add_argument('--trace-dir', type=Path, help='retain engine trace and measured launch identity')
ap.add_argument('--require-overlap', action='store_true')
args = ap.parse_args()
if args.require_overlap and args.trace_dir is None:
    ap.error('--require-overlap needs --trace-dir')
if args.trace_dir:
    if os.environ.get('TNCA_BACKEND', 'sim-file') != 'sim-file':
        ap.error('engine trace audit requires the isolated single-core file binding')
    if args.trace_dir.exists() and any(args.trace_dir.iterdir()):
        ap.error('--trace-dir must be empty; preserve previous attempts')
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TNCA_DEVICE_PLUSARGS'] = os.environ.get('TNCA_DEVICE_PLUSARGS', '') + ' +engtrace +trace'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm')]
os.environ.update(TNCA_RCTX_ROLES='2', TNCA_RCTX_STAGES=str(args.stages))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')

import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import artifact, device
from tnca.triton_backend.driver import launch_log


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(47)
    count = 0
    measured = []
    for name, fn in [('mm', lambda a,b,c: a@b),
                     ('addmm', lambda a,b,c: torch.addmm(c,a,b))]:
        graph = torch.compile(fn, fullgraph=True, dynamic=False)
        for m,n,k in ((16,16,48), (17,31,65)):
            a,b,c = [torch.randn(*shape,dtype=torch.float16)*.1
                     for shape in ((m,k),(k,n),(n,))]
            da,db,dc = [x.to('tnca') for x in (a,b,c)]
            warm = graph(da,db,dc)
            torch.testing.assert_close(warm.cpu(), fn(a,b,c), rtol=2e-3, atol=2e-3)
            del warm
            session = D.MEMORY.session()
            history_start = len(session.dev.history)
            with launch_log() as log:
                out = graph(da,db,dc)
            sequences = [e['seq'] for e in session.dev.history[history_start:]
                         if e['op'] in (3, 9)]
            torch.testing.assert_close(out.cpu(), fn(a,b,c), rtol=2e-3, atol=2e-3)
            assert log and all(e['upload_bytes']==e['download_bytes']==0 for e in log)
            rings = []
            for entry in log:
                manifest,_ = artifact.load(entry['artifact_path'])
                prov = manifest['optional_metadata']['provenance']
                ring = prov.get('rctx_roles_ring')
                if ring:
                    assert ring['stages'] == args.stages, ring
                    assert ring['ring_windows'] >= args.stages, ring
                    assert len(manifest['execution']['program_to_role']['0']) == 2
                    rings.append(ring)
            assert rings, [(artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance'])
                           for e in log]
            measured.append(dict(name=name, shape=[m,n,k], stages=args.stages,
                                 sequences=sequences, rings=rings))
            if args.trace_dir:
                for entry in log:
                    raw = Path(entry['artifact_path']).read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    (args.trace_dir / (digest + '.tnca')).write_bytes(raw)
            count += 1
            print(f'PASS {name} {m}x{n}x{k}, stages={args.stages}, rings={rings}',flush=True)
    print(f'r40-native-pipeline PASS ({count}/{count})',flush=True)
    if args.trace_dir:
        from engine_overlap import selected
        trace_path = Path(session.dev.log)
        session.close()  # Flush the trace after all measured commands finish.
        raw_trace = trace_path.read_text()
        (args.trace_dir / 'engine.log').write_text(raw_trace)
        sequences = [seq for row in measured for seq in row['sequences']]
        if not sequences:
            raise AssertionError('no measured runtime launch sequences')
        audit = selected(raw_trace, sequences)
        report = dict(stages=args.stages, cases=measured, launches=audit,
                      overlap_cycles=sum(row['overlap_cycles'] for row in audit.values()))
        (args.trace_dir / 'overlap.json').write_text(json.dumps(report,indent=2)+'\n')
        if args.require_overlap:
            assert report['overlap_cycles'] > 0, 'no actual DMA/MMA overlap in measured launches'
        print(f"AUDIT measured DMA/MMA overlap={report['overlap_cycles']} cycles",flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
