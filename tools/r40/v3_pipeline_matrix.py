#!/usr/bin/env python3
"""R40-07 v3 directed matrix: standard native Torch ring pipelines.

    v3_pipeline_matrix.py --memory {fast,slow,hbm,hbm-adverse} --stages {1,2,3}
        --json OUT [--trace-dir DIR]

Standard `torch.compile(fullgraph=True)` mm / addmm / linear (transposed
weight) graphs on the native `tnca` device, with the producer/consumer ring
forced legal (TNCA_RCTX_ROLES=2, as tools/r40/native_pipeline_tests.py) and
TNCA_RCTX_STAGES set, over a short odd-step K, a long K (256) and a long K with
M/N/K tails (17x31x257, masked edge tiles). The memory is the v5 device's
fixed-latency memory (fast), its slow random-stall reordering memory (slow:
MEM_LATENCY=7, RANDOM_STALL, MEM_REORDER), or the AXI HBM path of the /hbm
profile at default timing (hbm) and late / out-of-order / backpressured
(hbm-adverse: +axi_lat_min=16 +axi_lat_max=96 +axi_reorder +axi_bp=30).

Each case is checked against the CPU result on warm and measured calls, with
zero steady tensor transfers, and the executed artifacts must carry the
requested ring (stages, windows, two roles per program). With --trace-dir,
the device's +engtrace service intervals of the measured launches give the
DMA/MMA overlap per case; every ring case with two or three stages must
overlap. This is functional evidence on simulated RTL, not a speed result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--memory', choices=('fast', 'slow', 'hbm', 'hbm-adverse'), required=True)
ap.add_argument('--stages', type=int, choices=(1, 2, 3), required=True)
ap.add_argument('--trace-dir', type=Path)
ap.add_argument('--json', type=Path, required=True)
ap.add_argument('--shapes', default='16x16x48,16x16x256,17x31x257')
args = ap.parse_args()
PROFILE = {'fast': 'core-vrf-sync-v5', 'slow': 'core-vrf-sync-v5/slow',
           'hbm': 'core-vrf-sync-v5/hbm', 'hbm-adverse': 'core-vrf-sync-v5/hbm'}[args.memory]
AXI = '+axi_seed=7 +axi_lat_min=16 +axi_lat_max=96 +axi_reorder +axi_bp=30' \
    if args.memory == 'hbm-adverse' else ''
if args.trace_dir:
    if os.environ.get('TNCA_BACKEND', 'sim-file') != 'sim-file':
        ap.error('engine trace audit requires the isolated single-core file binding')
    if args.trace_dir.exists() and any(args.trace_dir.iterdir()):
        ap.error('--trace-dir must be empty; preserve previous attempts')
    args.trace_dir.mkdir(parents=True, exist_ok=True)
plus = ' '.join(x for x in (os.environ.get('TNCA_DEVICE_PLUSARGS', ''), AXI,
                            '+engtrace +trace' if args.trace_dir else '') if x)
os.environ['TNCA_DEVICE_PLUSARGS'] = plus
os.environ['TNCA_PROFILE'] = PROFILE
os.environ.update(TNCA_RCTX_ROLES='2', TNCA_RCTX_STAGES=str(args.stages))
os.environ.setdefault('TNCA_HOST_TIMEOUT', '172800')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm'),
                str(ROOT / 'tools/r40')]

import torch  # noqa: E402
import triton  # noqa: E402,F401
import tnca.torch_device as D  # noqa: E402
from tnca import artifact, device  # noqa: E402
from tnca.triton_backend.driver import launch_log  # noqa: E402

GRAPHS = [('mm', lambda a, w, b, c: a @ b),
          ('addmm', lambda a, w, b, c: torch.addmm(c, a, b)),
          ('linear', lambda a, w, b, c: torch.nn.functional.linear(a, w, c))]


def main():
    device.prepare(PROFILE)
    D.register()
    torch.manual_seed(4007)
    shapes = [tuple(int(v) for v in s.split('x')) for s in args.shapes.split(',')]
    cases, failures = [], []
    session = None
    for name, fn in GRAPHS:
        graph = torch.compile(fn, fullgraph=True, dynamic=False)
        for m, n, k in shapes:
            a = torch.randn(m, k, dtype=torch.float16) * .1
            b = torch.randn(k, n, dtype=torch.float16) * .1
            w = b.t().contiguous()            # linear's weight is [n, k]: B transposed
            c = torch.randn(n, dtype=torch.float16) * .1
            want = fn(a, w, b, c)
            dev = [x.to('tnca') for x in (a, w, b, c)]
            row = dict(graph=name, shape=[m, n, k], stages=args.stages, memory=args.memory)
            try:
                warm = graph(*dev)
                torch.testing.assert_close(warm.cpu(), want, rtol=2e-3, atol=2e-3)
                del warm
                session = D.MEMORY.session()
                start = len(session.dev.history)
                with launch_log() as log:
                    out = graph(*dev)
                seqs = [e['seq'] for e in session.dev.history[start:] if e['op'] in (3, 9)]
                got = out.cpu()
                err = (got.float() - want.float()).abs().max().item()
                torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)
                assert log and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in log), \
                    'steady launches moved tensor bytes'
                rings, refused, digests = [], [], []
                for entry in log:
                    raw = Path(entry['artifact_path']).read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    digests.append(digest)
                    if args.trace_dir:
                        (args.trace_dir / (digest + '.tnca')).write_bytes(raw)
                    manifest, _ = artifact.load(entry['artifact_path'])
                    prov = manifest['optional_metadata']['provenance']
                    ring = prov.get('rctx_roles_ring')
                    if ring:
                        assert ring['stages'] == args.stages, ring
                        assert ring['ring_windows'] >= args.stages, ring
                        assert len(manifest['execution']['program_to_role']['0']) == 2
                        rings.append(ring)
                    elif prov.get('rctx_roles_refused'):
                        refused.append(prov['rctx_roles_refused'])
                assert rings, f'no ring in the executed GEMM; refusals: {refused}'
                row.update(ok=True, max_abs_err=err, sequences=seqs, rings=rings,
                           refusals=refused, artifacts=digests,
                           launches=len(log))
                print(f'PASS {name} {m}x{n}x{k} memory={args.memory} stages={args.stages} '
                      f'err={err:.3g} rings={rings}', flush=True)
            except Exception as error:                       # noqa: BLE001
                row.update(ok=False, error=f'{type(error).__name__}: {error}'[:2000])
                failures.append(row)
                print(f'FAIL {name} {m}x{n}x{k} memory={args.memory} stages={args.stages}: '
                      f'{row["error"]}', flush=True)
            cases.append(row)
    report = dict(schema='tnca-r40-v3-pipeline-matrix/1', profile=PROFILE,
                  memory=args.memory, plusargs=plus, stages=args.stages,
                  roles='2 (forced legal split)', cases=cases)
    if args.trace_dir and session is not None:
        from engine_overlap import parse
        trace_path = Path(session.dev.log)
        session.close()                          # flush the trace
        raw = trace_path.read_text()
        (args.trace_dir / 'engine.log').write_text(raw)
        report['engine_log_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        try:
            launches = parse(raw)
        except ValueError as error:
            launches = {}
            report['trace_error'] = str(error)
            failures.append(dict(trace_error=str(error)))
        for row in cases:
            if not row.get('ok'):
                continue
            missing = [s for s in row['sequences'] if s not in launches]
            per = [dict(seq=s, **{k: launches[s][k] for k in
                                  ('mma_cycles', 'dma_cycles', 'overlap_cycles')})
                   for s in row['sequences'] if s in launches]
            row['trace'] = per
            row['overlap_cycles'] = sum(p['overlap_cycles'] for p in per)
            if missing:
                row.update(ok=False, error=f'measured sequences missing from trace: {missing}')
                failures.append(row)
            elif args.stages >= 2 and row['overlap_cycles'] <= 0:
                row.update(ok=False, error='ring launch shows no DMA/MMA service overlap')
                failures.append(row)
            print(f"AUDIT {row['graph']} {'x'.join(map(str, row['shape']))} "
                  f"overlap={row['overlap_cycles']} cycles over {len(per)} launch(es)", flush=True)
    report['ok'] = not failures and all(r.get('ok') for r in cases)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1) + '\n')
    passed = sum(bool(r.get('ok')) for r in cases)
    print(f"r40-v3-pipeline-matrix {'PASS' if report['ok'] else 'FAIL'} "
          f"({passed}/{len(cases)}) memory={args.memory} stages={args.stages}", flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
