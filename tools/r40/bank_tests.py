#!/usr/bin/env python3
"""Native Torch correctness and physical-bank ledger for one allocation mode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--enabled', type=int, choices=(0, 1), required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
os.environ['TNCA_BANK_ALLOC'] = str(args.enabled)
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
os.environ['TNCA_MAX_BLOCK'] = '64'
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device, artifact
from tnca.triton_backend.driver import launch_log
from tnca.triton_backend.compiler import TNCAOptions
from tnca_cc.bankcost import conflicts, moves, geometry
from tnca_cc.target import TargetInfo
from tnca_cc.diagnostics import Diagnostic


def main():
    assert TNCAOptions(bank_alloc=0).hash() != TNCAOptions(bank_alloc=1).hash()
    try:
        geometry(TargetInfo.default())
    except Diagnostic as error:
        assert 'declared banked VRF' in str(error)
    else:
        raise AssertionError('non-banked profile accepted for bank allocation')
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(51)
    cases = [
        ('triad', lambda a,b,c: torch.where(a < .25, a*b+c, c*b-a),
         tuple(torch.randn(127) for _ in range(3))),
        ('layernorm', lambda a: torch.nn.functional.layer_norm(a, (65,)),
         (torch.randn(3,65),)),
        ('f32-dot', lambda a,b: a @ b,
         (torch.randn(16,16)*.1, torch.randn(16,16)*.1)),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix='r40-bank-asm-') as dump:
        os.environ['TRITON_CACHE_DIR'] = os.path.join(dump, 'cache')
        os.environ['TNCA_KERNEL_DUMP'] = '1'
        os.environ['TNCA_DUMP_DIR'] = dump
        for name, fn, inputs in cases:
            compiled = torch.compile(fn, fullgraph=True)
            native = tuple(t.to('tnca') for t in inputs)
            # Compilation can launch native copy kernels (autotuner inputs, as
            # gemm_tiles_tests.py notes), and two specializations of the same
            # copy kernel share one dump directory. Ledger only the graph's own
            # steady launches.
            with torch.no_grad():
                warm = compiled(*native)
            torch.testing.assert_close(warm.cpu(), fn(*inputs), rtol=2e-4, atol=2e-4)
            del warm
            with torch.no_grad(), launch_log() as log:
                out = compiled(*native)
            torch.testing.assert_close(out.cpu(), fn(*inputs), rtol=2e-4, atol=2e-4)
            assert log and all(e['upload_bytes']==e['download_bytes']==0 for e in log)
            ledgers = []
            for launch in log:
                manifest, _ = artifact.load(launch['artifact_path'])
                prov = manifest['optional_metadata']['provenance']
                paths = list(Path(dump).rglob(launch['kernel']+'.s'))
                assert len(paths) == 1, (launch['kernel'], paths)
                asm = paths[0].read_text()
                assert hashlib.sha256(asm.encode()).hexdigest() == prov['assembly_sha256']
                actual = conflicts(asm, 4)
                report = prov.get('bank_allocation')
                if args.enabled:
                    assert report == dict(banks=4, static_extra_reads=actual['extra_cycles'],
                                          sites=actual['sites']), (report, actual)
                else:
                    assert report is None, report
                ledgers.append(dict(static_extra_reads=actual['extra_cycles'],
                                    sites=actual['sites'], moves=moves(asm)['count'],
                                    device_cycles=launch['device_cycles'],
                                    resources=manifest['resources'],
                                    asm_sha256=prov['assembly_sha256']))
            results.append(dict(name=name, ledgers=ledgers))
            print(f'PASS {name}: bank_alloc={args.enabled}, physical ledgers={ledgers}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(enabled=args.enabled, cases=results), indent=2)+'\n')
    print('r40-bank-tests PASS (3/3)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
