#!/usr/bin/env python3
"""Model-independent attention/conv/recurrent-state feasibility via standard Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', choices=('attention', 'depthwise', 'recurrence'), action='append')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TRITON_CACHE_DIR=str(out/'triton-cache'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'),
                      TNCA_TENSOR_DESCRIPTORS='1', TNCA_TORCH_DEVICE='native')
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    report = dict(schema='tnca-primitive-feasibility/1', status='running', cases=[],
                  scope='representative primitives; not complete model execution',
                  execution='torch.compile(backend=inductor,fullgraph=True) on tnca')
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        import torch
        import triton
        import torch.nn.functional as F
        import tnca.inductor
        import tnca.aot
        import tnca.dump
        from tnca import artifact
        from tnca.triton_backend.driver import launch_log
        import tnca.torch_device as device
        from archbench.provenance import software_tree
        if os.environ.get('TNCA_BACKEND', '').startswith('systemc-tlm'):
            import tlm_backend
            model_class = tlm_backend.register()
        else:
            model_class = None
        device.register()
        report.update(software_tree=software_tree(ROOT, refresh=True),
                      toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
        torch.manual_seed(20261020)
        def attention(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        def depthwise(x, w):
            return F.conv1d(x, w, padding=3, groups=x.shape[1])[..., :x.shape[-1]]
        def recurrence(state, q, k, v, log_decay, beta):
            # One delta-rule state transition, all arithmetic and state FP32.
            decayed = state * torch.exp(log_decay)[..., None, None]
            predicted = torch.matmul(k.unsqueeze(-2), decayed).squeeze(-2)
            delta = (v - predicted) * beta[..., None]
            updated = decayed + k[..., :, None] * delta[..., None, :]
            result = torch.matmul(q.unsqueeze(-2), updated).squeeze(-2)
            return updated, result
        definitions = [
            ('attention', attention, ((1, 8, 5, 256), (1, 2, 5, 256), (1, 2, 5, 256)), torch.bfloat16, 2e-3, 2e-3),
            ('depthwise', depthwise, ((1, 8, 7), (8, 1, 4)), torch.bfloat16, 2e-3, 2e-3),
            ('recurrence', recurrence, ((1, 1, 128, 128), (1, 1, 128), (1, 1, 128),
                                       (1, 1, 128), (1, 1), (1, 1)), torch.float32, 2e-4, 2e-4)]
        for name, fn, shapes, dtype, rtol, atol in definitions:
            if args.only and name not in args.only:
                continue
            row = dict(name=name, shapes=shapes, dtype=str(dtype), rtol=rtol, atol=atol, status='running')
            report['cases'].append(row)
            save()
            ir_dir = out/name/'ir'
            os.environ.update(TNCA_KERNEL_DUMP='1', TNCA_DUMP_DIR=str(ir_dir))
            cpu = tuple(torch.randn(shape).to(dtype)*.02 for shape in shapes)
            reference = fn(*(t.float() for t in cpu))
            dev = tuple(t.to('tnca') for t in cpu)
            session = device.MEMORY.session()
            row['memory_before_compile'] = session.memory_stats
            compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
            start = time.monotonic()
            from torch._inductor.utils import run_and_get_code
            lower = tnca.aot._lower_ttir
            write_dump = tnca.dump.Dump.write
            compiler_calls = []
            def distinct_dump(stages, directory, **keywords):
                digest = hashlib.sha256(json.dumps(stages.stages, sort_keys=True).encode()).hexdigest()
                return write_dump(stages, Path(directory)/digest, **keywords)
            def timed_lower(*arguments, **keywords):
                started = time.monotonic()
                try:
                    return lower(*arguments, **keywords)
                finally:
                    compiler_calls.append(time.monotonic()-started)
            with patch('tnca.aot._lower_ttir', side_effect=timed_lower), \
                    patch.object(tnca.dump.Dump, 'write', distinct_dump), \
                    launch_log(isolated=True) as cold_launches:
                cold, wrappers = run_and_get_code(compiled, *dev)
            row['cold_compile_and_execute_seconds'] = time.monotonic()-start
            row['tnca_backend_lowering_seconds'] = compiler_calls
            row['compiler_time_scope'] = 'TNCA lowering calls only; cold wall includes Dynamo/Inductor/Triton and execution'
            if tnca.inductor.host_fallbacks(wrappers):
                raise ValueError('standard path emitted a numerical host fallback')
            for i, wrapper in enumerate(wrappers):
                (out/f'{name}-wrapper-{i}.py').write_text(wrapper)
            def check(result):
                values = result if isinstance(result, tuple) else (result,)
                refs = reference if isinstance(reference, tuple) else (reference,)
                hashes = []
                for value, ref in zip(values, refs, strict=True):
                    actual = value.cpu().float()
                    torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol)
                    hashes.append(hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())
                return hashes
            check(cold)
            with launch_log(isolated=True) as launches:
                result = compiled(*dev)
            row['output_sha256'] = check(result)
            kernels = {}
            for entry in cold_launches+launches:
                path = Path(entry['artifact_path'])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in kernels:
                    continue
                manifest, segments = artifact.load(path)
                destination = out/name/'artifacts'/f'{digest}.tnca'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
                kernels[digest] = dict(kernel=entry['kernel'], artifact=str(destination.relative_to(out)),
                    code_bytes=len(segments['text']), resources=manifest['resources'],
                    execution_resources=manifest['optional_metadata']['provenance'].get('execution_resources', {}))
            stages = []
            for metadata_path in sorted(ir_dir.rglob('*.json')):
                metadata = json.loads(metadata_path.read_text())
                if not {'ttir', 'tnir', 'asm', 'hex'} <= set(metadata['stages']):
                    raise ValueError('representative Triton compile lacks mandatory raw IR stages')
                files = {}
                for stage, filename in metadata['stages'].items():
                    path = metadata_path.parent/filename
                    data = path.read_bytes()
                    files[stage] = dict(path=str(path.relative_to(out)), bytes=len(data),
                                        lines=len(data.splitlines()), sha256=hashlib.sha256(data).hexdigest())
                stages.append(dict(kernel=metadata['kernel_name'], stages=files))
            if not stages or not compiler_calls:
                raise ValueError('fresh representative compilation did not retain compiler-stage evidence')
            row['compiled_kernel_inventory'] = kernels
            row['ir_inventory'] = stages
            row['memory_after_execution'] = session.memory_stats
            row['memory_scope'] = 'actual Session heap/code peaks since process start, including native helper copies; not complete-model peak'
            if model_class:
                from prediction_domain import timing_status, VALID_STATUSES
                qualifications = [timing_status(model.counters_snapshot(), slow=model.identity['dram_path']=='slow')
                                  for model in model_class.instances if model.proc is not None]
                row['model_timing_status'] = ('inconclusive-fixed-point' if not qualifications or
                    any(q not in VALID_STATUSES for q in qualifications) else
                    'uncalibrated-fixed-point-checked' if 'uncalibrated-fixed-point-checked' in qualifications
                    else 'uncalibrated')
            if name == 'recurrence' and (result[0].dtype != torch.float32 or dev[0].dtype != torch.float32):
                raise ValueError('FP32 recurrent state was narrowed')
            assert launches and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches)
            row.update(status='pass', launches=launches,
                       device_cycles=sum(e['device_cycles'] for e in launches),
                       code_bytes=[len(artifact.load(e['artifact_path'])[1]['text']) for e in launches],
                       host_max_rss_kib_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            print(name, 'pass', row['device_cycles'], flush=True)
            save()
        if software_tree(ROOT, refresh=True) != report['software_tree']:
            raise ValueError('source changed during feasibility execution')
        report['status'] = 'functional-pass'
    except Exception:
        report.update(status='failed-preserved', error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
