"""Exercise TNCA_TUNE through torch.compile's actual Inductor runtime hook.

--functional-rtl replaces only physical-image qualification with an explicit
test identity. Measurements still come from the real MMIO RTL device counter;
this test does not certify a physical image, clock, timing or performance gate.
Without that flag SystemC must retain the conservative native configuration.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--functional-rtl', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TNCA_TUNE='1', TNCA_TUNE_CANDIDATES='2', TNCA_TUNE_SECONDS='300',
                      TNCA_TUNE_DIR=str(out/'tune'), TRITON_CACHE_DIR=str(out/'triton-cache'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    import torch
    import triton
    if os.environ.get('TNCA_BACKEND', '').startswith('systemc'):
        import tlm_backend
        tlm_backend.register()
    import tnca.torch_device as device
    device.register()
    from tnca.inductor_tuning import SELECTIONS
    from tnca.triton_backend.driver import launch_log
    from tnca.errors import SessionBusy
    torch.manual_seed(3108)
    a, b = torch.randn(17), torch.randn(17)
    da, db = a.to('tnca'), b.to('tnca')
    session = device.MEMORY.session()
    graph = torch.compile(lambda a, b: (a+b)*.5, backend='inductor',
                          fullgraph=True, dynamic=False)
    def functional_identity(s):
        assert args.functional_rtl and s.backend_name == 'fpga-multicore-sim'
        digest = hashlib.sha256()
        for base in (ROOT/'rtl', ROOT/'tb/fpga'):
            for path in sorted(base.rglob('*.v*')):
                digest.update(path.read_bytes())
        return dict(image='functional-rtl:'+digest.hexdigest(),
                    clock=dict(clock_hz=100000000, operating_clock_evidence_sha256='functional-test-nominal-normalizer'),
                    memory_map=s.dev.axi_window, capabilities=s.dev.capabilities())
    context = patch('tnca.inductor_tuning._hardware_identity', functional_identity) if args.functional_rtl else nullcontext()
    with context:
        with launch_log(isolated=True) as cold:
            result = graph(da, db)
        torch.testing.assert_close(result.cpu(), (a+b)*.5)
        assert SELECTIONS, 'TNCA_TUNE did not enter the public Inductor hook'
        if args.functional_rtl:
            selected = [r for r in SELECTIONS if r['status'] == 'selected']
            assert selected, SELECTIONS
            first = selected[-1]['selection']
            assert len([r for r in first['rows'] if r['status'] == 'measured']) == 2, first
            assert not first['cache_hit']
            assert all(r['device_records'][0]['clock'] == 'mmio-device-clock/1' for r in first['rows'])
            # Cold compilation itself emits native helper copies. Isolate
            # the graph kernel by the actual selected tuner's identity:
            # its tuning warmups/candidates must not enter the caller log.
            kernels = {r['kernel'] for r in selected}
            assert len([e for e in cold if e['kernel'] in kernels]) == 1, cold
            limits = (session.deadline_cycles, session.dev.deadline_cycles, session.dev.host_timeout)
            rejected = []
            with session.measurement_scope(wall_seconds=30, clock_hz=100000000):
                def other_thread():
                    try:
                        session.submit(None, {})
                    except SessionBusy:
                        rejected.append(True)
                thread = threading.Thread(target=other_thread)
                thread.start(); thread.join()
                assert rejected == [True], 'measurement launch limits leaked to another thread'
            assert limits == (session.deadline_cycles, session.dev.deadline_cycles, session.dev.host_timeout)
        else:
            assert all(r['status'] == 'conservative' for r in SELECTIONS), SELECTIONS
            assert any('model wall time' in r.get('reason', '') for r in SELECTIONS), SELECTIONS
            assert not (out/'tune/runs').exists(), 'an unqualified domain spawned tuning work'
        with launch_log(isolated=True) as formal:
            result = graph(da, db)
        torch.testing.assert_close(result.cpu(), (a+b)*.5)
        assert len(formal) == 1 and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in formal)
    report = dict(schema='tnca-public-inductor-tuning-test/1', status='functional-pass',
                  domain='functional-RTL-test-with-injected-image-qualification' if args.functional_rtl else 'uncalibrated-SystemC-conservative',
                  physical_qualification='not-measured', selections=SELECTIONS, cold_launches=cold,
                  formal_launches=formal)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], domain=report['domain'],
                          selections=[r['status'] for r in SELECTIONS]), indent=2))


if __name__ == '__main__':
    main()
