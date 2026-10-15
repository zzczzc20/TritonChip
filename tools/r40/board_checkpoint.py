#!/usr/bin/env python3
"""Unscaled frozen add graph on the physical three-core image; no gate waiver.

Run from an immutable source archive. Required hashes must be recorded before
the experiment, not filled in from its output. No simulator or size override.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xclbin', type=Path, required=True)
    ap.add_argument('--image-sha256', required=True)
    ap.add_argument('--software-tree-sha256', required=True,
                    help='exact ArchBench files:<sha256> identity of the immutable archive')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    # Exclusive creation preserves failed attempts as well as successful ones.
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='fail', gate_status='not-measured', performance='not-measured')
    try:
        import contract_v2
        contract_v2.verify()
        from archbench.provenance import software_tree
        tree = software_tree()
        image_sha = hashlib.sha256(args.xclbin.read_bytes()).hexdigest()
        assert image_sha == args.image_sha256, 'image differs from preregistered snapshot'
        assert tree == args.software_tree_sha256, 'software differs from preregistered snapshot'
        config = dict(xclbin=str(args.xclbin.resolve()), image_profile='core-vrf-sync-v5',
                      core_count=3, memory_ports=3, hbm_dma=True,
                      hbm_bytes=1 << 30, device_clock=True)
        report['identity'] = dict(software_tree=tree, image_sha256=image_sha,
                                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                contract_sha256=contract_v2.LOCK_SHA256,
                                backend='fpga-u280-multicore', backend_args=config,
                                clock_policy='100MHz-undivided', clock_hz=100000000,
                                clock_qualification='requires matching route/timing and board clock reports')
        os.environ.update(TNCA_BACKEND='fpga-u280-multicore',
                          TNCA_BACKEND_ARGS=json.dumps(config),
                          TNCA_PROFILE='core-vrf-sync-v5', TNCA_TORCH_DEVICE='native',
                          TNCA_ACTIVE_CORES='3',
                          CUDA_VISIBLE_DEVICES='', R40_CONTRACT='v2')
        os.environ.setdefault('TNCA_HOST_TIMEOUT', '7200')
        os.environ.setdefault('TNCA_DEADLINE', '4000000000')
        import torch
        import triton
        report['identity']['compiler'] = dict(torch=torch.__version__, triton=triton.__version__)
        import tnca.torch_device as D
        from tnca.triton_backend.driver import launch_log
        from archbench import registry
        from archbench.platforms.common import reference, compare
        from hbm_address import topology_probe_offsets
        point = registry.get('op.add/hbm/fp16-n67108864')
        frozen = json.loads((ROOT/'spec/r40/suites/R40-04.streaming.json').read_text())
        expected_sha = next(p['sha256'] for p in frozen['points'] if p['id'] == point.id)
        assert point.definition_sha256() == expected_sha, 'frozen point changed'
        report['point'] = dict(id=point.id, sha256=expected_sha, seed=0,
                               definition=point.definition())
        D.register()
        session = D.MEMORY.session()
        dev = session.dev
        assert dev.name == 'fpga-u280-multicore'
        report['capabilities'] = dev.capabilities()
        assert dev.physical_grid_capability()[0] == 3
        assert dev.memory_port_capability() == 3
        assert dev.capabilities()['cache_ownership_handoff']
        report['window'] = dev.axi_window
        assert dev.axi_window['size'] >= 1 << 30
        topology_probe_offsets(dev.axi_window)
        assert len(dev.axi_window['allocation']['physical_banks']) >= 2
        ref = reference(point, 0)
        model, inputs = point.make(0)
        report['input_bytes'] = sum(x.numel()*x.element_size() for x in inputs)
        assert report['input_bytes'] >= 64 << 20
        model = model.eval().to('tnca')
        inputs = tuple(x.to('tnca') for x in inputs)
        compiled = torch.compile(model, fullgraph=True, dynamic=False)
        with torch.no_grad():
            warm = compiled(*inputs)
        report['warm_correctness'] = compare(warm, ref, point.tolerance)
        assert report['warm_correctness']['ok'], report['warm_correctness']
        del warm
        events = []
        original_wait = session.wait

        def capture_wait(event, *a, **kw):
            result = original_wait(event, *a, **kw)
            events.append(dict(job_id=event.job_id, state=event.state,
                               history=list(event.history), device_timing=event.device_timing))
            return result

        before = dev.memory_port_metrics()
        start = len(dev.history)
        session.wait = capture_wait
        try:
            with torch.no_grad(), launch_log() as launches:
                output = compiled(*inputs)
        finally:
            session.wait = original_wait
        # Record device completion and actual outlet traffic BEFORE the CPU
        # read's ownership maintenance can contribute any traffic.
        report['events'] = events
        report['launches'] = launches
        report['cores'] = dev.physical_grid_metrics()
        after = dev.memory_port_metrics()
        report['outlets'] = [dict(port=a['port'],
                                 read_lines=a['read_lines']-b['read_lines'],
                                 write_lines=a['write_lines']-b['write_lines'])
                             for a,b in zip(after,before)]
        report['completion_history'] = dev.history[start:]
        report['memory'] = dict(tensor_peak_bytes=D.MEMORY.peak,
                              session=session.memory_stats,
                              live_buffers=[dict(address=b.addr, bytes=b.size)
                                            for b in D.MEMORY.buffers.values() if b.alive])
        banks = dev.axi_window['allocation']['physical_banks']
        # Virtual heap addresses include the local prefix; DMA maps the rest
        # relative to the BO's actual physical base.
        pcs = set()
        for b in report['memory']['live_buffers']:
            lo = dev.axi_window['base'] + b['address'] - dev.axi_window['local_limit']
            hi = lo + b['bytes']
            pcs.update(bank['pc'] for bank in banks
                       if lo < bank['allocated_base']+bank['allocated_size']
                       and hi > bank['allocated_base'])
        report['tensor_physical_pcs'] = sorted(pcs)
        assert len(pcs) >= 2, 'large graph did not span two actual physical PCs'
        assert launches and all(l['upload_bytes'] == l['download_bytes'] == 0 for l in launches)
        assert events and all(e['state'] == 'SUCCEEDED' and
                              'DRAINING' in [s for s,t in e['history']] for e in events)
        assert all(l['device_timing'] is not None for l in launches)
        assert any(r['op'] == 4 and r['status'] == 0 for r in report['completion_history'])
        assert report['cores']['peak_active'] == 3
        assert all(c['issued'] == c['completed'] > 0 and c['occupied_cycles'] > 0
                   for c in report['cores']['cores'])
        grid = launches[-1]['grid']
        assert sum(c['issued'] for c in report['cores']['cores']) == grid[0]*grid[1]*grid[2]
        assert all(p['read_lines'] > 0 and p['write_lines'] > 0 for p in report['outlets'])
        assert not any(dev.axi_status().values()), dev.axi_status()
        # Host visibility is checked through the standard ownership handoff,
        # after drained device completion; preserve both sides of that boundary.
        start = len(dev.history)
        report['correctness'] = compare(output, ref, point.tolerance)
        report['host_visibility_history'] = dev.history[start:]
        assert report['correctness']['ok'], report['correctness']
        handoffs = [r for r in report['host_visibility_history'] if r['op'] == 18]
        assert handoffs and all(r['status'] == 0 for r in handoffs)
        report['visibility'] = dict(device_completion='DRAINING then quiesce then SUCCEEDED',
                                   host_read='successful ownership handoff then numerical comparison',
                                   raw_dma_without_handoff='not-certified')
        from archbench.provenance import _tree_cache
        _tree_cache.pop(ROOT, None)
        assert software_tree() == tree, 'software changed during experiment'
        report['status'] = 'checkpoint-pass'
    except Exception:
        report['error'] = traceback.format_exc()
    (args.out/'report.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps(dict(status=report['status'], report=str(args.out/'report.json'))), flush=True)
    return 0 if report['status'] == 'checkpoint-pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
