#!/usr/bin/env python3
"""R40-06.semantics: one standard Torch graph across the core/port/latency matrix.

    python3 tools/r40/v3_semantics_matrix.py --config mc4 --out DIR

Each configuration runs in its own process against the real multicore RTL
(Icarus), through the standard path: tnca tensors, torch.compile, Inductor,
Triton, the TNCA compiler and the physical-grid runtime. Only the backend
selection changes between configurations; the model source, the frontend
operator rules, the ISA and every compile option are the product defaults.

Two graphs, both resident (zero tensor staging in the measured call):
  pointwise  (a+b)*0.5 over 4097 fp32 elements (a ragged grid tail)
  softmax    torch.softmax(x, -1) over a 33x100 fp32 tensor (row reductions,
             33 programs: not divisible by 2, 4 or 8 cores)

The record keeps numerical results, the output bytes' SHA256 (identical bits
are expected in every configuration: each program's arithmetic does not
depend on which core runs it), the compiled kernel artifact hashes, the
physical-grid telemetry and each memory outlet's line traffic. This is RTL
functional evidence; cycle counts are not converted to time or performance.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# name -> (backend, backend args). sim-multicore: tb/device/tnca_multicore_device
# (DRAM model latency 11+7p cycles, random stalls, reordering);
# fpga-multicore-sim: the U280 board XO top behind the MMIO v2 register path,
# AXI line models with latency 25+19p and the image's two-stage links.
CONFIGS = {
    'mc1': ('sim-multicore', dict(core_count=1)),
    'mc2': ('sim-multicore', dict(core_count=2)),
    'mc3': ('sim-multicore', dict(core_count=3)),
    'mc4': ('sim-multicore', dict(core_count=4)),
    'mc8': ('sim-multicore', dict(core_count=8)),
    'mc3p3': ('sim-multicore', dict(core_count=3, memory_ports=3)),
    'mc4p2': ('sim-multicore', dict(core_count=4, memory_ports=2)),
    # The image's options: two-stage links, cache maintenance + DMA ownership,
    # runtime memory controls. The mailbox journal it captures is replayed by
    # tools/r40/replay_mmio_journal.py (fixed-input transaction/completion replay).
    'bt3p3s2': ('fpga-multicore-sim', dict(core_count=3, memory_ports=3, noc_link_stages=2,
                                           dispatch_stages=2, home_stages=2, code_stages=2,
                                           board_top=True, maintenance=True, hbm_dma=True,
                                           memory_controls=True, journal=True)),
    'regs3p1s0': ('fpga-multicore-sim', dict(core_count=3, memory_ports=1)),
}


def child(name, out_dir):
    backend, args = CONFIGS[name]
    os.environ.update(TNCA_PROFILE='core-vrf-sync-v5', TNCA_BACKEND=backend,
                      TNCA_BACKEND_ARGS=json.dumps(args))
    os.environ.setdefault('TNCA_HOST_TIMEOUT', '7200')
    for p in (ROOT, ROOT / 'python', ROOT / 'compiler' / 'python', ROOT / 'tools' / 'asm'):
        sys.path.insert(0, str(p))
    import torch
    import triton
    import tnca.torch_device as D
    from tnca.triton_backend.driver import launch_log
    from archbench import provenance
    tree = provenance.software_tree(refresh=True)
    D.register()
    torch.manual_seed(40)
    a, b = torch.randn(4097), torch.randn(4097)
    x = torch.randn(33, 100)
    graphs = {
        'pointwise': (lambda p, q: (p + q) * 0.5, (a, b)),
        'softmax': (lambda t: torch.softmax(t, -1), (x,)),
    }
    session = D.MEMORY.session()
    result = dict(config=name, backend=backend, backend_args=args, software_tree=tree,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, triton=triton.__version__,
                  env={k: v for k, v in os.environ.items() if k.startswith('TNCA_')},
                  graphs={})
    ok = True
    for gname, (fn, host_inputs) in graphs.items():
        t0 = time.time()
        dev_inputs = tuple(t.to('tnca') for t in host_inputs)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        warm = compiled(*dev_inputs)
        torch.testing.assert_close(warm.cpu(), fn(*host_inputs))
        del warm
        ports0 = session.dev.memory_port_metrics()
        start = len(session.dev.history)
        with torch.no_grad(), launch_log() as launches:
            out = compiled(*dev_inputs)
        ports1 = session.dev.memory_port_metrics()
        grid = session.dev.physical_grid_metrics()
        history = session.dev.history[start:]
        host = out.cpu()
        ref = fn(*host_inputs)
        checks = dict(
            numerical=bool(torch.allclose(host, ref, rtol=1.3e-6, atol=1e-5)),
            device_output=out.device.type == 'tnca',
            zero_tensor_staging=all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches),
            whole_grid_launch=sum(r['op'] == 11 for r in history) == len(launches) and len(launches) > 0,
            no_per_program_launch=not any(r['op'] in (3, 9) for r in history),
            physical_grid=bool(session.physical_grid),
            programs_exactly_once=(sum(c['issued'] for c in grid['cores']) ==
                                   launches[-1]['grid'][0] * launches[-1]['grid'][1] * launches[-1]['grid'][2]
                                   and all(c['issued'] == c['completed'] for c in grid['cores'])),
        )
        delta = [dict(port=q['port'], read_lines=q['read_lines'] - p['read_lines'],
                      write_lines=q['write_lines'] - p['write_lines']) for p, q in zip(ports0, ports1)]
        checks['every_outlet_traffic'] = bool(delta) and all(r['read_lines'] > 0 and r['write_lines'] > 0 for r in delta)
        result['graphs'][gname] = dict(
            checks=checks, ok=all(checks.values()),
            output_sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest(),
            max_abs_err=float((host - ref).abs().max()),
            artifacts=[e.get('artifact') for e in launches],
            kernels=[e['kernel'] for e in launches],
            grids=[list(e['grid']) for e in launches],
            device_cycles=[e.get('device_cycles') for e in launches],
            physical_grid=grid, outlet_traffic=delta, wall_s=round(time.time() - t0, 1))
        ok = ok and result['graphs'][gname]['ok']
        print(f"  {'PASS' if result['graphs'][gname]['ok'] else 'FAIL'}  {name} {gname} "
              f"{json.dumps(checks)} peak={grid.get('peak_active')} outlets={delta}", flush=True)
    result['journal_dir'] = getattr(getattr(session.dev, 'link', None), 'journal_dir', None)
    result['link_dir'] = getattr(getattr(session.dev, 'link', None), 'dir', None)
    result['software_tree_unchanged'] = provenance.software_tree(refresh=True) == tree
    result['ok'] = ok and result['software_tree_unchanged']
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f'{name}.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"SEMANTICS {'PASS' if result['ok'] else 'FAIL'} {name}", flush=True)
    return 0 if result['ok'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', choices=sorted(CONFIGS), required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if (Path(args.out) / f'{args.config}.json').exists():
        ap.error('result exists; preserve earlier attempts')
    return child(args.config, args.out)


if __name__ == '__main__':
    sys.exit(main())
