#!/usr/bin/env python3
"""Standard Torch GEMM residency case, with artifacts and native model counters.

Run in a fresh process per configuration. CPU is an independent reference;
the tested graphs always execute on tnca through Inductor and the Session.
This produces development evidence, never a board acceptance record.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--columns', type=int, choices=(16, 32), default=32)
    ap.add_argument('--dtype', choices=('float16', 'bfloat16'), default='float16')
    ap.add_argument('--strided', action='store_true')
    ap.add_argument('--require-resident', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'artifacts').mkdir()
    os.environ.update(TNCA_GEMM_TILE=f'16,{args.columns},16', TNCA_TORCH_DEVICE='native',
                      TNCA_TENSOR_DESCRIPTORS='1', TNCA_INDEX_OPT='1',
                      TRITON_CACHE_DIR=str(out/'triton-cache'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    report = dict(schema='tnca-fragment-case/1', status='running',
                  execution='torch.compile(backend=inductor,fullgraph=True) on tnca',
                  evidence='development-only', columns=args.columns, dtype=args.dtype,
                  strided=args.strided, rows=[])
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        import torch
        import triton
        import tnca
        from tnca import artifact
        from tnca.triton_backend.driver import launch_log
        from archbench.provenance import software_tree
        import tnca.torch_device as device
        for module in (tnca,):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise ValueError('module imported outside the requested source tree')
        backend = os.environ.get('TNCA_BACKEND', 'sim-file')
        if backend.startswith('systemc-tlm'):
            import tlm_backend
            model_class = tlm_backend.register()
        else:
            model_class = None
        report.update(software_tree=software_tree(ROOT, refresh=True), backend=backend,
                      toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
        device.register()
        torch.manual_seed(47)
        dtype = getattr(torch, args.dtype)
        for name, fn in [('mm', lambda a, b, c: a @ b),
                         ('addmm', lambda a, b, c: torch.addmm(c, a, b))]:
            compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
            for m, n, k in ((17, 65, 33), (33, 31, 65)):
                if args.strided:
                    a = (torch.randn(k, m).to(dtype) * .1).t()
                    b = (torch.randn(n, k).to(dtype) * .1).t()
                else:
                    a = torch.randn(m, k).to(dtype) * .1
                    b = torch.randn(k, n).to(dtype) * .1
                c = torch.randn(n).to(dtype) * .1
                # Copy the complete bases before making device views, retaining
                # the input strides instead of silently making them contiguous.
                if args.strided:
                    da, db = a.t().to('tnca').t(), b.t().to('tnca').t()
                else:
                    da, db = a.to('tnca'), b.to('tnca')
                dc = c.to('tnca')
                assert da.stride() == a.stride() and db.stride() == b.stride()
                reference = fn(a.float(), b.float(), c.float())
                cold = compiled(da, db, dc)
                torch.testing.assert_close(cold.cpu().float(), reference, rtol=2e-3, atol=2e-3)
                snapshots = ({str(i): model.counters_snapshot()
                              for i, model in enumerate(model_class.instances) if model.proc is not None}
                             if model_class else {})
                start = time.monotonic()
                with launch_log(isolated=True) as launches:
                    result = compiled(da, db, dc)
                wall = time.monotonic() - start
                after = ({str(i): model.counters_snapshot()
                          for i, model in enumerate(model_class.instances) if model.proc is not None}
                         if model_class else {})
                got = result.cpu().float()
                torch.testing.assert_close(got, reference, rtol=2e-3, atol=2e-3)
                assert launches and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches)
                resources = []
                for entry in launches:
                    manifest, segments = artifact.load(entry['artifact_path'])
                    prov = manifest['optional_metadata']['provenance']
                    plan = prov.get('execution_resources', {})
                    if args.require_resident:
                        loops = [loop for role in plan.get('roles', []) for loop in role['loops']
                                 if loop['static_mma_commands'] == args.columns // 16]
                        assert any(loop['static_acc_reads'] == loop['static_acc_writes'] ==
                                   loop['static_acc_zeros'] == 0 for loop in loops), plan
                    digest = hashlib.sha256(segments['text']).hexdigest()
                    shutil.copyfile(entry['artifact_path'], out/'artifacts'/f'{digest}.tnca')
                    resources.append(dict(code_sha256=digest, code_bytes=len(segments['text']),
                                          manifest_resources=manifest['resources'], plan=plan))
                deltas = {i: {key: value - snapshots.get(i, {}).get(key, 0)
                              for key, value in values.items()}
                          for i, values in after.items()}
                row = dict(graph=name, shape=[m, n, k], strides=[list(a.stride()), list(b.stride())],
                           status='pass', max_abs_error=float((got-reference).abs().max()),
                           output_f32_sha256=hashlib.sha256(got.contiguous().numpy().tobytes()).hexdigest(),
                           rtol=2e-3, atol=2e-3, resident_host_wall_seconds=wall,
                           device_cycles=sum(e['device_cycles'] for e in launches),
                           launches=launches, artifacts=resources, model_counter_deltas=deltas)
                if model_class:
                    from prediction_domain import timing_status, VALID_STATUSES
                    qualifications = [timing_status(after[str(i)], slow=model.identity['dram_path']=='slow')
                                      for i, model in enumerate(model_class.instances) if str(i) in after]
                    row['model_timing_status'] = ('inconclusive-fixed-point' if not qualifications or
                        any(q not in VALID_STATUSES for q in qualifications) else
                        'uncalibrated-fixed-point-checked' if 'uncalibrated-fixed-point-checked' in qualifications
                        else 'uncalibrated')
                report['rows'].append(row)
                save()
        if software_tree(ROOT, refresh=True) != report['software_tree']:
            raise ValueError('source changed during the experiment')
        report['status'] = 'functional-pass'
    except Exception:
        report.update(status='fail', error=traceback.format_exc())
        raise
    finally:
        save()
    print(f'fragment case: {report["status"]} ({len(report["rows"])} graphs/shapes)', flush=True)


if __name__ == '__main__':
    main()
