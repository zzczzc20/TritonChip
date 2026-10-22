#!/usr/bin/env python3
"""The registered R41 hbm8 board experiments, and their simulator twins.

    hbm8_manifest.py --check                 # registered manifests == generated
    hbm8_manifest.py --write                 # (re)generate spec/r41/hbm8/experiments/*.json
    hbm8_manifest.py --sim OUT_DIR [--rtl-root R] [--hbm-base B]
                                             # same steps as kind=rtl on the sparse stand-in

Two board manifests, so that a failure in the >2 GiB single-storage compiler
path cannot stop the window qualification (experiment.py stops a manifest at
its first failed step) and vice versa:

  board-hbm8        window, negatives, samples, alias, crossing, dma, clamp
  board-hbm8-big    bigoffset, stream

Every board step runs tools/performance_harness/hbm8_case.py once, in its own
process, on the already-loaded B40 image a88ebdf5 at 90 MHz; experiment.py
probes the loaded UUID, the clock counter and kdsstat before and after each
step and refuses a different image or a busy card. Nothing here loads,
resets or programs a card.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'spec/r41/hbm8/experiments'
MAP = 'evidence/r41/hbm8/xclbin_hbm_map.json'
IMAGE = '{home}/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin'
GIB = 1 << 30

GROUPS = {
    'board-hbm8': ('window', 'negatives', 'samples', 'alias', 'crossing', 'dma', 'clamp'),
    'board-hbm8-big': ('bigoffset', 'stream'),
}
# Minutes per case process on the FPGA host (compile + DMA + device), from the
# R40 fe1b records: ~13-21 s per small process, ~3-5 s per 64 MiB of DMA
# through the Python transfer path, 8 GiB BO allocation per process.
MINUTES = dict(window=2, negatives=2, samples=6, alias=5, crossing=4, dma=4, clamp=2, bigoffset=4, stream=12)
WHAT = dict(
    window='8 GiB XRT BO in MBG[0:31], R_AXI_BASE/SIZE/STAT readback, heap = window, PC0-PC31 contiguous',
    negatives='allocator/host/MMIO refusals at the 8 GiB end, host window last word, reserved TMU page',
    samples='34 probes: every PC boundary, both BO edges, virtual 4 GiB; DMA in, 3-core device add, DMA out',
    alias='bit 30/31/32/32+31 address pairs and TMU page + 2/4/6 GiB: distinct device load/store values',
    crossing='device store-load across virtual 4 GiB, physical 4 GiB and the window end; DMA out and in',
    dma='256 MiB DMA in at the window top, device copy to 6 GiB, 2 x 256 MiB DMA out',
    clamp='hbm_bytes 16 GiB exposes exactly the 8 GiB group',
    bigoffset='2.25 GiB storage: strided device read/write at byte offsets above 2^31',
    stream='2.25 GiB storage: 36 x 64 MiB DMA in, device row reduction over all of it, DMA out',
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def board_block(hbm_bytes):
    return dict(
        image=IMAGE, image_sha256='a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083',
        image_uuid='1bada9db-78f3-5bc8-7ba6-cce062e7ade8',
        clock_evidence='{home}/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json',
        clock_evidence_sha256='8ad4b3bd30a3f5fe8819fe59ef8ad160fd0de1478aaa506dc33d82785fb9f487',
        clock_hz=90000000, device='0000:04:00.1', mgmt_device='0000:04:00.0',
        backend='fpga-u280-multicore',
        backend_args=dict(xclbin=IMAGE, image_profile='core-vrf-sync-v5', core_count=3, memory_ports=3,
                          hbm_dma=True, hbm_bytes=hbm_bytes, device_clock=True),
        profile='core-vrf-sync-v5',
        transport=dict(deadline_cycles=200000000000, host_timeout_seconds=28800,
                       source='evidence/r40/v3/campaign_fe1b/campaign.json transport'),
        memory_map=dict(window_base=0, window_bytes=8 * GIB, local_limit=131072, bank='MBG[0:31]',
                        physical_pcs='PC0-PC31', source=MAP, source_sha256=file_sha(ROOT / MAP)),
        load_policy='never loads: runs only on the B40 image a88ebdf5 already loaded by the Stage 1 queue; '
                    'every fpga step probes the loaded UUID, the clock counter and kdsstat first and refuses otherwise')


def step(case):
    hbm = 2 * 8 * GIB if case == 'clamp' else 8 * GIB
    board = board_block(hbm)
    args = json.dumps({k: v for k, v in board['backend_args'].items()}, separators=(',', ':'))
    return dict(
        id=case, kind='fpga', timeout_seconds=7200 if case == 'stream' else 3600,
        env=dict(TNCA_BACKEND='fpga-u280-multicore', TNCA_BACKEND_ARGS=args, TNCA_PROFILE='core-vrf-sync-v5',
                 TNCA_OPERATING_CLOCK_EVIDENCE='{home}/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json',
                 TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='28800'),
        argv=['{python}', 'tools/performance_harness/hbm8_case.py', '--case', case, '--out', '{out}/' + case],
        board=dict(backend_args=board['backend_args']) if case == 'clamp' else None,
        timing_boundary='functional: hbm8_case.py report.json checks; device cycles recorded per launch, '
                        'diagnostic only, no performance gate',
        cache_state='fresh process and session per case; experiment-local Triton/Inductor/extension caches under {out}/caches',
        outputs=['{out}/' + case + '/report.json'],
        what=WHAT[case])


def manifest(name):
    cases = GROUPS[name]
    steps = [{k: v for k, v in step(c).items() if v is not None} for c in cases]
    steps.append(dict(id='summary', kind='report', timeout_seconds=120,
                      argv=['{python}', 'tools/performance_harness/hbm8_summary.py', '--group', name,
                            *[f'{c}={{out}}/{c}/report.json' for c in cases], '--out', '{out}/summary.json'],
                      inputs=[f'{{out}}/{c}/report.json' for c in cases], outputs=['{out}/summary.json']))
    return dict(
        schema='tnca-experiment-spec/1', id=name, allowed_hosts=['acf3019'],
        hypothesis=('The delivered three-core B40 image a88ebdf5 at 90 MHz serves an 8 GiB device window over '
                    'HBM[0:31] with the current runtime: the RTL address path is 64-bit end to end '
                    '(evidence/r41/hbm8 audit) and the xclbin connects all three outlets to the 8 GiB group, so '
                    'the R40 4 GiB window was an allocation choice, not a limit.'
                    if name == 'board-hbm8' else
                    'A single storage larger than 2 GiB in the 8 GiB window is written by DMA, read entirely '
                    'by a standard Torch device reduction and read back, and strided kernels whose element '
                    'offsets scale past 2^31 bytes address it correctly.'),
        competing_explanations=[
            'XRT refuses or places differently an 8 GiB BO in MBG[0:31] on this image (host memory, group map)',
            'a platform/interconnect address-map segment narrower than the xclbin connectivity drops PC16-PC31',
            'a high address bit is lost in RTL that the audit and RTL simulation missed (aliasing probes)',
            'generated kernels scale int32 element offsets in 32-bit arithmetic (bigoffset/stream only)',
            'host transfer path cost or memory for multi-GiB transfers (diagnostic, not a window limit)'],
        expected_counter_changes=[
            'outlet read/write line deltas on all three outlets for the probe kernels',
            'each launch issues and completes programs on all three cores (peak_active 3)',
            'stream: outlet read lines x 64 B >= the 2.25 GiB storage for one row reduction',
            'R_AXI_STAT error/stray bits stay clear after every case'],
        expected_observations=[f'{c}: {WHAT[c]}' for c in cases],
        board=board_block(8 * GIB),
        sources=['tools/performance_harness', 'tools/r40/hbm_address.py', 'archbench', 'python/tnca',
                 'compiler/python', 'compiler/targets', 'tools/asm', 'isa', MAP, f'spec/r41/hbm8/experiments/{name}.json'],
        env=dict(MAX_JOBS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                 PYTHONFAULTHANDLER='1', PYTHONPATH='{root}:{root}/python:{root}/compiler/python:{root}/tools/asm',
                 TRITON_CACHE_DIR='{out}/caches/triton', TORCHINDUCTOR_CACHE_DIR='{out}/caches/inductor',
                 TORCH_EXTENSIONS_DIR='{out}/caches/torch-extensions'),
        registration=dict(
            status='registered-not-run', registered_utc='2026-10-22', tool='tools/performance_harness/hbm8_case.py',
            cases=list(cases), order='as listed; one process per case',
            expected_board_hours=round(sum(MINUTES[c] for c in cases) / 60, 3),
            estimate_method='per-case minutes in tools/performance_harness/hbm8_manifest.py MINUTES',
            launch='evidence/r41/hbm8/launches/hbm8_queue.sh (after the Stage 1 queue; never loads)',
            simulator_twin='hbm8_manifest.py --sim: the same steps on fpga-multicore-sparse-sim'),
        conclusion_source='{out}/summary.json',
        steps=steps)


def sim_twin(name, rtl_root=None, hbm_base=0):
    spec = copy.deepcopy(manifest(name))
    spec['id'] = name + '-sim'
    spec['allowed_hosts'] = ['gx10-6646']
    spec.pop('board')
    spec['sources'] = [s for s in spec['sources'] if not s.startswith('spec/r41/hbm8/experiments/')]
    spec['registration'] = dict(status='simulator twin of ' + name, board_manifest=f'spec/r41/hbm8/experiments/{name}.json')
    for s in spec['steps']:
        if s['kind'] == 'fpga':
            s['kind'] = 'rtl'
            s.pop('board', None)
            s['env'] = dict(TNCA_PROFILE='core-vrf-sync-v5', TNCA_HOST_TIMEOUT='14400')
            s['argv'] = s['argv'] + ['--sim', '--hbm-base', hex(hbm_base)] + \
                (['--rtl-root', str(rtl_root)] if rtl_root else [])
            s['timeout_seconds'] = 6 * 3600
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--sim', type=Path)
    ap.add_argument('--rtl-root', type=Path)
    ap.add_argument('--hbm-base', type=lambda v: int(v, 0), default=0)
    a = ap.parse_args()
    if a.write:
        SPEC.mkdir(parents=True, exist_ok=True)
        for name in GROUPS:
            (SPEC / f'{name}.json').write_text(json.dumps(manifest(name), indent=1) + '\n')
    if a.check:
        bad = [n for n in GROUPS if json.loads((SPEC / f'{n}.json').read_text()) != manifest(n)]
        if bad:
            sys.exit(f'registered manifests differ from the generator: {bad}')
        print('hbm8 manifests: registered == generated')
    if a.sim:
        a.sim.mkdir(parents=True, exist_ok=True)
        for name in GROUPS:
            (a.sim / f'{name}-sim.json').write_text(json.dumps(sim_twin(name, a.rtl_root, a.hbm_base), indent=1) + '\n')


if __name__ == '__main__':
    main()
