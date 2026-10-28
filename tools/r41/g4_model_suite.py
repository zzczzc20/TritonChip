#!/usr/bin/env python3
"""Run the registered memory-bound suite through the SystemC/TLM model (G4).

    $TORCH_PYTHON tools/r41/g4_model_suite.py --binary BUILD/tnca-chip --out DIR \
        [--size small] [--variant run1 --variant g4 --variant ideal] [--jobs 3]

Every point of archbench/suites/r41_membound.json of the chosen size runs
through tools/performance_harness/systemc/run_graphs.py: the standard path
(torch.compile -> Inductor -> Triton -> TNCA compiler -> Session) with the
compiler's default settings, three active cores, into the model configured
as one of these memory systems (all at 100 MHz):

  run1   the delivered run-1 geometry (configs/r41-hbm8-device-service.cfg,
         clock 100 MHz; SFU occupancy 22/23 as corrected in
         evidence/r41/hbm_bw/vector_schedule/README.md);
  g4     the G4_DESIGN.md memory parameters: 64 B core beat and NoC payload,
         two memory lanes per home and link, home service of one line per
         cycle per lane (all tl_slice_* occupancies 1), 128 MSHRs and 256
         targets per home, 64 LSU slots, six outlets (two per core) with 64
         lines outstanding and single-cycle issue, lines striped across the 32
         pseudo-channels in 4 KiB pages;
  ideal  g4 with every memory latency set to its minimum, to isolate the
         core-side (issue, dependency, SFU) limit of the compiled code.

The model has no key for a full-line write without fill or for a wider
transfer engine, so those parts of the design are not represented (stated
in G4_DESIGN.md). Output: per variant the run_graphs records, and
`suite_model.json` with, per point, device cycles and
B/cycle/core = (ArchBench demand reads + writes) / cycles / 3, computed here
from the records. Domain: model-prediction-uncalibrated.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CORES = 3


def configs():
    cfg = ROOT/'tools/performance_harness/systemc/configs/r41-hbm8-device-service.cfg'
    base = dict(line.split('#')[0].strip().split('=', 1) for line in cfg.read_text().splitlines()
                if '=' in line.split('#')[0])
    base.update(clock_hz='100000000', sfu_occupancy='22', sfu_latency='23')
    g4 = dict(base)
    g4.update(tl_core_beat_bytes='64', tl_noc_payload_bytes='64', tl_memory_lanes='2',
              tl_lsu_units='8', tl_lsu_shared_issue='0', lsu_slots='64', lsu_beat_slots='8',
              tl_slice_hit_occupancy='1', tl_slice_miss_occupancy='1', tl_slice_mreq_occupancy='1',
              tl_slice_fill_occupancy='1', tl_slice_wake_occupancy='1', tl_slice_writeback_occupancy='1',
              mshrs='128', targets='256', outlets='6', active_outlets='6', outstanding='64',
              tl_outlet_read_issue_cycles='1', tl_outlet_write_issue_cycles='1',
              tl_pc_stripe_bytes='4096')
    ideal = dict(g4)
    ideal.update(tl_memory_lanes='8', tl_noc_inject_cycles='1', tl_noc_eject_cycles='1', noc_stages='1',
                 tl_slice_hit_latency='1', tl_slice_to_outlet_cycles='1', tl_outlet_to_slice_cycles='1',
                 hbm_latency_cycles='1', tl_hbm_write_latency_cycles='1', outlets='32', active_outlets='32')
    return dict(run1=(base, 3), g4=(g4, 6), ideal=(ideal, 32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', default='small', choices=['small', 'medium', 'large'])
    ap.add_argument('--variant', action='append')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--point', action='append', help='restrict to these suite ids (smoke tests only)')
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--torch-extensions-dir', type=Path)
    ap.add_argument('--env', action='append', default=[],
                    help='NON-DEFAULT compiler/runtime setting KEY=VALUE for an exploratory arm; recorded and labelled')
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    suite_path = ROOT/'archbench/suites/r41_membound.json'
    suite = json.loads(suite_path.read_text())
    ids = [p['id'] for p in suite['points'] if f'-{a.size}-' in p['id']]
    if a.point:
        ids = [i for i in ids if i in a.point]
    import archbench.registry as reg
    summary = dict(domain='model-prediction-uncalibrated', clock_hz_configured=100000000, cores=CORES,
                   suite=str(suite_path.relative_to(ROOT)), suite_sha256=suite.get('sha256'),
                   size=a.size, binary=str(a.binary), variants={},
                   non_default_settings=a.env,
                   counts_for_target=not a.env)
    for name in a.variant or ['run1', 'g4', 'ideal']:
        values, ports = configs()[name]
        cfg = a.out/f'{name}.cfg'
        cfg.write_text(''.join(f'{k}={v}\n' for k, v in values.items()))
        command = [sys.executable, str(ROOT/'tools/performance_harness/systemc/run_graphs.py'),
                   '--source-root', str(ROOT), '--binary', str(a.binary.resolve()), '--config', str(cfg),
                   '--out', str(a.out/name), '--active-cores', str(CORES),
                   '--env', f'TNCA_RTL_MEM_PORTS={ports}', '--env', f'TNCA_RTL_OUTSTANDING={values["outstanding"]}',
                   '--jobs', str(a.jobs), '--timeout', str(a.timeout)]
        for e in a.env:
            command += ['--env', e]
        if a.torch_extensions_dir:
            command += ['--torch-extensions-dir', str(a.torch_extensions_dir)]
        for pid in ids:
            command += ['--point', pid]
        (a.out/f'{name}.command.json').write_text(json.dumps(command, indent=1)+'\n')
        log = a.out/f'{name}.log'
        with log.open('w') as fh:
            rc = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')).returncode
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
        rows = {}
        for r in recs:
            pid = r.get('key')
            if pid not in ids:
                continue
            d = reg.get(pid).demand
            cyc = r.get('device_cycles')
            rows[pid] = dict(status=r.get('status'), device_cycles=cyc,
                             demand_bytes=d.read_bytes + d.write_bytes,
                             b_per_cycle_per_core=((d.read_bytes + d.write_bytes)/cyc/CORES) if cyc else None,
                             gb_per_s_at_100mhz=((d.read_bytes + d.write_bytes)/cyc*0.1) if cyc else None,
                             max_rel_err=r.get('max_rel_err'))
        missing = [p for p in ids if p not in rows]
        summary['variants'][name] = dict(returncode=rc, config=values, points=rows, missing=missing)
        (a.out/'suite_model.json').write_text(json.dumps(summary, indent=1)+'\n')
        print(name, 'rc', rc, 'points', len(rows), 'missing', len(missing), flush=True)


if __name__ == '__main__':
    main()
