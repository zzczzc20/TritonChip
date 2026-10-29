#!/usr/bin/env python3
"""Stage counters of standard-path ArchBench points on the run-1 image RTL.

    link_budget_rtl.py --point ID [--point ID ...] --out FILE.jsonl [--memory-bytes N]

Each point runs exactly as ArchBench runs it on the board (archbench.platforms.
tnca.run: torch.compile on device "tnca", default compiler settings, a cold
call, then one timed call), but on the RTL stand-in of the board kernel
(`fpga-multicore-sim`, board top tnca_multicore_hbm_xo) elaborated at the
parameters of the run-1 image (evidence/r41/stage2/integration/image.json).
Every image parameter is either passed to the bench or equal to the RTL
default, otherwise the run refuses (the rule of tools/r40/image_geometry_tests.py).

The memory behind the three AXI outlets is the testbench line model
(tb/common/tnca_axi_line_model.v: 25/44/63-cycle latency per outlet, eight IDs,
random ready stalls), not HBM. Results are RTL simulation, never board.

tb/fpga/tnca_stage_probes.vh prints cumulative stage counters at every grid
start and completion; the record keeps, for the timed call's launches, the
differences per tile (core port, NI, home slice, home memory side, home engine
state cycles, core pipeline counters) and per outlet.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT/'evidence/r41/stage2/integration/image.json'
PROFILE = 'core-vrf-sync-v5-x8'
# image parameter -> board-bench parameter passed through rtl_params
BENCH = ('WAYS', 'WAY_W', 'DCO_REGISTERED_RANGES', 'NUCA_REGISTER_VICTIM_SELECT', 'NUCA_CACHE_DATA_PIPELINE',
         'NUCA_REGISTER_VICTIM_RANK', 'DCO_TLW_PIPE_STAGES', 'DCO_REGISTER_WRITES', 'DCO_LOCAL_REPLICAS',
         'HAS_TENSOR_XPOSE', 'MTX_VRF_REG',
         # R41 G2 sizing and section-4 changes, bench parameters of tnca_multicore_board_regs
         'NUM_RCTX', 'RCTX_W', 'PHYS_VREGS', 'PHYS_W', 'ACC_N', 'ACC_W', 'BAR_N', 'BAR_W', 'BAR_EPOCH_W',
         'PHYS_ADDR_W', 'NUCA_SHARED_DATA_READ', 'AXI_LINE_LUTRAM',
         # R41 G4 one-lane path options
         'HOME_LINE_WRITE_NOFILL', 'HOME_LINE_PIPE', 'HOME_MSHR_N', 'LINE_SLOTS', 'AXI_OUT_N', 'AXI_PIPELINED_ISSUE',
         'TENSOR_LINE')
# image parameter -> backend keyword (the opt-in wide path, G4)
KWARG = dict(HYBRID_LSU=('hybrid_lsu', bool), HOME_BACKGROUND_READS=('home_background_reads', bool),
             SFU_LANES=('sfu_lanes', int))
LINK = dict(MEM_PORTS='memory_ports', LINK_STAGES='link_stages', NOC_LINK_STAGES='noc_link_stages',
            DISPATCH_STAGES='dispatch_stages', MEM_HOME_STAGES='home_stages', CODE_LINK_STAGES='code_stages',
            CACHE_MAINTENANCE='maintenance', MEMORY_CONTROLS='memory_controls')


def rtl_defaults():
    src = (ROOT/'rtl/board/tnca_multicore_hbm_xo.v').read_text()
    start = src.index('module tnca_multicore_hbm_xo')
    head = src[start:src.index('\n) (', start)]
    out = {k: int(v) for k, v in re.findall(r'\b([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*[,)\n]', head)}
    out['AXI_ID_W'] = max(1, (out['AXI_OUT_N'] - 1).bit_length())
    return out


def image_parameters(path):
    """An image's parameters: image.json's `parameters`, or a Tcl `dict create`."""
    text = Path(path).read_text()
    if str(path).endswith('.json'):
        return json.loads(text)['parameters']
    m = re.search(r'dict create ([^\]]+)\]', text)
    toks = m.group(1).split()
    return {toks[i]: int(toks[i + 1]) for i in range(0, len(toks), 2)}


def backend_args(memory_bytes, image_path=IMAGE, profile=None):
    image = image_parameters(image_path)
    defaults = rtl_defaults()
    rtl, link, checked, kw = {}, {}, {}, {}
    for name, value in sorted(image.items()):
        if name in KWARG:
            kw[KWARG[name][0]] = KWARG[name][1](value)
        elif name in BENCH:
            rtl[name] = value
        elif name in LINK:
            link[LINK[name]] = value
        elif name == 'AXI_ID_W':
            if value != max(1, (image.get('AXI_OUT_N', defaults['AXI_OUT_N']) - 1).bit_length()):
                raise SystemExit('image AXI_ID_W does not follow its AXI_OUT_N')
            checked[name] = value          # derived from AXI_OUT_N in the bench and the top
        elif name in ('SETS', 'SET_W'):
            checked[name] = value          # cache_sets below
        elif name in defaults:
            if defaults[name] != value:
                raise SystemExit(f'image {name}={value} differs from the RTL default {defaults[name]} '
                                 'and has no bench parameter')
            checked[name] = value
        else:
            raise SystemExit(f'image parameter {name} unknown to the bench and the RTL top')
    if 1 << image['SET_W'] != image['SETS']:
        raise SystemExit('inconsistent SETS/SET_W')
    args = dict(core_count=3, memory_ports=link['memory_ports'], memory_bytes=memory_bytes,
                link_stages=link['link_stages'], noc_link_stages=link['noc_link_stages'],
                dispatch_stages=link['dispatch_stages'], home_stages=link['home_stages'],
                code_stages=link['code_stages'], board_top=True, maintenance=link['maintenance'],
                memory_controls=bool(link['memory_controls']), cache_sets=image['SETS'], rtl_params=rtl,
                stage_probes=True, hbm_dma=True, host_timeout=86400, **kw)
    if profile:
        args['image_profile'] = profile
    return args, dict(image=image, image_file=str(image_path), defaults_equal=checked)


LINE = re.compile(r'^STAGE (TILE|PORT|CYCLE) (.*)$')


def stage_blocks(log):
    """[(phase, cycle, {tiles}, {ports})] in log order."""
    blocks = {}
    order = []
    for line in Path(log).read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        kv = dict(item.split('=', 1) for item in m.group(2).split())
        phase = kv.pop('at')
        cycle = int(kv.pop('c', kv.get('cycle', 0)))
        key = (phase, cycle)
        if key not in blocks:
            blocks[key] = dict(phase=phase, cycle=cycle, tiles={}, ports={})
            order.append(key)
        if m.group(1) == 'TILE':
            index = int(kv.pop('t'))
            blocks[key]['tiles'][index] = {k: int(v) for k, v in kv.items()}
        elif m.group(1) == 'PORT':
            index = int(kv.pop('p'))
            blocks[key]['ports'][index] = {k: int(v) for k, v in kv.items()}
    return [blocks[k] for k in order]


def launches(blocks):
    """BEGIN/END pairs -> per-launch differences."""
    out, begin = [], None
    for b in blocks:
        if b['phase'] == 'BEGIN':
            begin = b
        elif begin is not None:
            out.append(dict(cycles=b['cycle'] - begin['cycle'],
                            tiles={t: {k: v - begin['tiles'][t][k] for k, v in row.items()}
                                   for t, row in b['tiles'].items()},
                            ports={p: {k: v - begin['ports'][p][k] for k, v in row.items()}
                                   for p, row in b['ports'].items()}))
            begin = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--point', action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--memory-bytes', type=int, default=2 << 20)
    ap.add_argument('--image', default=str(IMAGE),
                    help='image parameters (image.json or an image .tcl); default: the run-1 image')
    ap.add_argument('--profile', default=PROFILE, help='the profile the image implements')
    ap.add_argument('--set', action='append', default=[], metavar='TNCA_KEY=VALUE',
                    help='a NON-DEFAULT compiler setting for a default-off candidate (G4); '
                         'recorded in every record, which then counts for no target')
    a = ap.parse_args()
    extra = dict(kv.split('=', 1) for kv in a.set)
    if any(not k.startswith('TNCA_') for k in extra):
        ap.error('--set takes TNCA_* settings only')
    args, identity = backend_args(a.memory_bytes, a.image, a.profile if a.profile != PROFILE else None)
    for k in [k for k in os.environ if k.startswith('TNCA_')]:
        del os.environ[k]
    os.environ.update(TNCA_BACKEND='fpga-multicore-sim', TNCA_BACKEND_ARGS=json.dumps(args),
                      TNCA_TORCH_DEVICE='native', TNCA_HOST_TIMEOUT='86400', TNCA_DEADLINE='200000000000')
    os.environ.update(extra)
    for p in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
        sys.path.insert(0, str(p))
    import triton                                    # noqa: F401  (Triton first: backend discovery)
    from archbench import registry
    from archbench.platforms import tnca as platform
    from tnca.triton_backend import driver
    for pid in a.point:
        point = registry.get(pid)
        started = time.time()
        rec = platform.run(point, profile=a.profile, warmup=0, timed=1)
        logs = [s.dev.link.log for s in driver._Sessions.open.values()
                if getattr(getattr(s, 'dev', None), 'link', None) is not None]
        per_launch = []
        for log in logs:
            per_launch += launches(stage_blocks(log))
        # the timed call: the last launches, as many as the measured call made
        kernels = rec.get('kernels') or []
        count = (rec.get('time') or {}).get('launches') or 0
        timed = per_launch[-count:] if count else []
        t = (rec.get('timings') or {}).get('device-cycles-serial-grid') or {}
        out = dict(schema='tnca-link-budget-rtl/1', evidence='rtl-simulation', board=False,
                   point=pid, point_sha256=point.definition_sha256(), status=rec.get('status'),
                   problems=rec.get('problems'), correctness=rec.get('correctness'), demand=point.demand.as_dict(),
                   kernels=kernels, time=rec.get('time'), device_cycles_samples=t.get('samples'), profile=a.profile,
                   backend='fpga-multicore-sim', backend_args=args, image=identity,
                   memory_model='tb/common/tnca_axi_line_model.v (latency 25+19p cycles, 8 IDs, random ready stalls)',
                   software_tree=(rec.get('platform') or {}).get('software_tree'),
                   all_launches=len(per_launch), timed_launches=timed, logs=logs,
                   non_default_settings=extra, counts_for_target=not extra,
                   wall_seconds=round(time.time() - started, 1))
        with a.out.open('a') as fh:
            fh.write(json.dumps(out) + '\n')
        print(pid, rec.get('status'), t.get('samples'), [l['cycles'] for l in timed], flush=True)
        for s in list(driver._Sessions.open.values()):
            try:
                s.close()
            except Exception:                                           # noqa: BLE001
                pass
        driver._Sessions.open.clear()


if __name__ == '__main__':
    main()
