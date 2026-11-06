#!/usr/bin/env python3
"""R41 Stage 2 board batch on the run-1 100 MHz image (30e79769..., UUID 49964b17...).

    stage2_board.py --write | --check        # spec/r41/stage2/board/*.json == generated
    stage2_board.py --image-source           # (re)write evidence/r41/stage2/board/image_source.json from c9f2578
    stage2_board.py source-check --root R --out F       # board tree == image source (rtl, flow, xls, ...)
    stage2_board.py fp8-summary --out F --group G=REPORT... [--dump DIR...] [--require-x8]

The registered manifests, in launch order (ORDER). Every fpga step runs on
the image loaded by evidence/r41/stage2/board/launches/board_queue.sh;
experiment.py probes the loaded UUID, the clock counter (100 MHz) and kdsstat
before and after each step and refuses anything else. The qualified clock
evidence is produced at load time (clock_qualify.py), so the manifests name
its path and every record binds its sha256.

The batch manifests (identity smoke, anchors, original suite, endurance mix)
reuse the Stage 1 plan's frozen measurement rules (spec/r41/stage1/plan.json)
and suites; only the board block differs: image, UUID, clock, profile
core-vrf-sync-v5-x8 and the memory map parsed from this xclbin.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = ROOT / 'spec/r41/stage2/board'
S1 = ROOT / 'spec/r41/stage1/experiments'
EVID = 'evidence/r41/stage2/board'
MAP = EVID + '/xclbin_hbm_map.json'
SOURCE = EVID + '/image_source.json'
GIB = 1 << 30

TAG = 'r41s2_run1_c9f25787494a'
IMAGE = '{home}/tnca_images/' + TAG + '/tnca_u280.xclbin'
IMAGE_SHA256 = '30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
UUID = '49964b17-e0b4-5bd2-dd40-7fc37b1205d3'
CLOCK = '{home}/tnca_build/' + TAG + '/r41s2-board/qualified_clock.json'
CLOCK_HZ = 100_000_000
PROFILE = 'core-vrf-sync-v5-x8'
SOURCE_COMMIT = 'c9f25787494a8414f9ab19e03bea6fb3fa794d20'
IMAGE_SOURCE_PATHS = ('rtl', 'flow/fpga', 'flow/synth', 'xls', 'spec/r1_profiles.json', 'compiler/targets',
                      'evidence/r41/stage2/integration/image.tcl', 'evidence/r41/stage2/integration/image.json')
FP8_ENV = '{home}/r41s2_fp8env/bin/python'
FP8_IN = '{home}/r41s2_fp8_inputs'
KNOBS = dict(TNCA_ACTIVE_CORES='3', TNCA_TORCH_DEVICE='native', TNCA_TRANSFER_POLICY='resident')
X8_LOAD = 't.ld.async.b.t'

ORDER = ('s2-identity', 's2-anchors', 's2-fp8-cases', 's2-fp8-blockwise', 's2-hbm8', 's2-hbm8-big',
         's2-original-suite', 's2-fp8-hub', 's2-qwen-fp8', 's2-endurance')
CRITICAL = ('s2-identity',)          # the queue stops when one of these fails

# The endurance mix: one ArchBench process (= one runtime session) each, drawn
# in a seeded order per round: HBM streaming, compute GEMM, multi-launch
# decode, attention, reductions, conv, a whole block, plus the FP8 matmul.
MIX = ('op.add/hbm/fp16-n67108864', 'op.matmul/medium/fp16-m128-k192-n128', 'model.llama_decode/hbm/fp16-b1-p512',
       'op.attention_prefill/medium/fp16-b1-h4-s96-d32', 'op.softmax/medium/fp16-r80-c512',
       'op.conv2d/medium/fp32-n1-cin24-h16-w16-cout24', 'model.gpt2_block/medium/fp16-d160-h5-b1-s64',
       'op.layernorm/medium/fp16-r128-d512')
SMOKE = ('op.add/tiny/fp16-n1000', 'op.matmul/tiny/fp16-m16-k32-n16', 'model.llama_decode/tiny/fp16-b1-p16')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_shas():
    out = {}
    for name in ('existing', 'hbm-tier', 'large'):
        for p in json.loads((ROOT / f'spec/r40/suites/{name}.json').read_text())['points']:
            out[p['id']] = p['sha256']
    return out


def backend_args(hbm_bytes):
    return dict(xclbin=IMAGE, image_profile=PROFILE, core_count=3, memory_ports=3, hbm_dma=True,
                hbm_bytes=hbm_bytes, device_clock=True)


def board_block(hbm_bytes=4 * GIB):
    pcs = 'PC0-PC15' if hbm_bytes <= 4 * GIB else 'PC0-PC31'
    return dict(
        image=IMAGE, image_sha256=IMAGE_SHA256, image_uuid=UUID, clock_evidence=CLOCK, clock_hz=CLOCK_HZ,
        device='0000:04:00.1', mgmt_device='0000:04:00.0', backend='fpga-u280-multicore',
        backend_args=backend_args(hbm_bytes), profile=PROFILE,
        transport=dict(deadline_cycles=200000000000, host_timeout_seconds=28800,
                       source='evidence/r40/v3/campaign_fe1b/campaign.json transport'),
        memory_map=dict(window_base=0, window_bytes=min(hbm_bytes, 8 * GIB), local_limit=131072,
                        bank='MBG[0:31]', physical_pcs=pcs, source=MAP, source_sha256=file_sha(ROOT / MAP)),
        image_source=dict(commit=SOURCE_COMMIT, branch='r41s2-t2', record=SOURCE,
                          parameters='evidence/r41/stage2/integration/image.json',
                          run='evidence/r41/stage2/run1/README.md'),
        load_policy='loaded only by evidence/r41/stage2/board/launches/board_queue.sh after the Stage 1 and hbm8 '
                    'queues finished and the card was idle 15 consecutive minutes (user authorization for this '
                    'image only); every fpga step probes the loaded UUID, the 100 MHz clock counter and kdsstat '
                    'first and refuses otherwise; the clock evidence is captured after that load')


def fpga_env(hbm_bytes, **extra):
    env = dict(TNCA_BACKEND='fpga-u280-multicore',
               TNCA_BACKEND_ARGS=json.dumps(backend_args(hbm_bytes), separators=(',', ':')),
               TNCA_PROFILE=PROFILE, TNCA_OPERATING_CLOCK_EVIDENCE=CLOCK,
               TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='28800')
    env.update(extra)
    return env


BASE_ENV = dict(MAX_JOBS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                PYTHONFAULTHANDLER='1', PYTHONPATH='{root}:{root}/python:{root}/compiler/python:{root}/tools/asm',
                TRITON_CACHE_DIR='{out}/caches/triton', TORCHINDUCTOR_CACHE_DIR='{out}/caches/inductor',
                TORCH_EXTENSIONS_DIR='{out}/caches/torch-extensions')
SOURCES = ['tools/performance_harness', 'archbench', 'python/tnca', 'compiler/python', 'compiler/targets',
           'tools/asm', 'isa', MAP, SOURCE]
FUNCTIONAL = ('functional: bitwise/tolerance comparisons in the tool report; device cycles recorded per launch, '
              'diagnostic only, no performance gate')
CACHE = 'fresh process and session per step; experiment-local Triton/Inductor/extension caches under {out}/caches'
BATCH_TIMING = ('ArchBench protocol 2: cold call untimed; registered warm-up/timed steady calls; '
                'device-cycles-serial-grid and device-graph-makespan from first accepted launch to last device '
                'completion (MMIO common-clock endpoints); request-latency recorded, not ranked')
BATCH_CACHE = 'fresh experiment-local Triton/Inductor/extension caches under {out}/caches; one ArchBench process per run'


def base(name, hypothesis, competing, counters, observations, registration, steps, *, hbm_bytes=4 * GIB,
         extra_sources=(), conclusion=None):
    spec = dict(schema='tnca-experiment-spec/1', id=name, allowed_hosts=['acf3019'], hypothesis=hypothesis,
                competing_explanations=competing, expected_counter_changes=counters,
                expected_observations=observations, board=board_block(hbm_bytes),
                sources=SOURCES + list(extra_sources) + [f'spec/r41/stage2/board/{name}.json'],
                env=dict(BASE_ENV),
                registration=dict(dict(status='registered-not-run', registered_utc='2026-10-23',
                                       contract='doc/r41_tasks_and_acceptance.md S2-01 (S2-04/S2-05 board items)',
                                       launch='evidence/r41/stage2/board/launches/board_queue.sh',
                                       generator='tools/performance_harness/stage2_board.py'), **registration),
                steps=steps)
    if conclusion:
        spec['conclusion_source'] = conclusion
    return spec


def batch_spec(name, stage1, batch_name, batch, hypothesis, observations, registration, extra_sources=()):
    """A board_batch manifest: Stage 1's frozen plan and suites, the run-1 board."""
    s1 = json.loads((S1 / f'{stage1}.json').read_text()) if stage1 else \
        json.loads((S1 / 'board-anchors.json').read_text())
    spec = base(name, hypothesis, list(s1['competing_explanations']) if stage1 else [
        'image or clock differs from the registered run-1 image (checked by identity, not by timing)',
        'the X8 profile or 100 MHz image changes a numeric result'],
        list(s1['expected_counter_changes']) if stage1 else ['no counter change: identity smoke only'],
        observations, registration, [dict(
            id=batch_name, kind='fpga',
            argv=['{python}', 'tools/performance_harness/board_batch.py', '--manifest',
                  f'spec/r41/stage2/board/{name}.json', '--batch', batch_name, '--root', '{root}',
                  '--out', '{out}/' + batch_name, '--python', '{python}'],
            timeout_seconds=259200 if batch_name == 'original' else 21600,
            timing_boundary=BATCH_TIMING, cache_state=BATCH_CACHE,
            outputs=['{out}/' + batch_name + '/batch.json', '{out}/' + batch_name + '/formal.jsonl'])],
        extra_sources=['archbench', 'spec/r40/suites', 'spec/r41/stage1/plan.json', *extra_sources])
    for key in ('plan', 'plan_sha256', 'measurement', 'suites', 'estimates'):
        spec[key] = copy.deepcopy(s1[key])
    spec['board_batches'] = {batch_name: batch}
    spec['registration'].update(knobs=dict(KNOBS),
                                reference='Stage 1 B40 board records of the same points and knobs '
                                          '(evidence/r41/stage1, evidence/r40/v3/R40-04.hbm/board_run_fe1b)')
    return spec


def configs(ids):
    shas = point_shas()
    return [dict(id=p, point=p, point_sha256=shas[p], knobs=dict(KNOBS)) for p in ids]


def m_identity():
    spec = batch_spec(
        's2-identity', None, 'identity',
        dict(sampling='fixed', fixed=dict(warmup=0, timed=1), configurations=configs(SMOKE)),
        'The card runs the R41 Stage 2 run-1 image (xclbin 30e79769..., UUID 49964b17..., source c9f2578, '
        'profile core-vrf-sync-v5-x8) at an actual 100 MHz: the loaded UUID, the xclbin DATA_CLK, the board clock '
        'counter and the post-route evidence agree, the board software tree carries the image RTL unchanged, and '
        'three small ArchBench points pass numerically on three cores with clock_hz 100 MHz in their records.',
        ['identity: loaded UUID 49964b17..., counter 100 MHz before and after every step',
         'source: rtl, flow/fpga, flow/synth, xls, profiles and targets byte-identical to c9f2578',
         'clock: qualified_clock.json passes tnca.operating_clock.qualified_clock; xclbinutil DATA_CLK 100 MHz',
         'smoke: 3 points x 1 process, status pass, evidence_level measured-board, clock_hz 100000000'],
        dict(expected_board_hours=0.1, purpose='S2-01 items 3 and 6: actual 100 MHz, image hash/source/'
                                                'parameters/clock bound together'))
    spec['steps'] = [
        dict(id='source', kind='report', timeout_seconds=600,
             argv=['{python}', 'tools/performance_harness/stage2_board.py', 'source-check', '--root', '{root}',
                   '--out', '{out}/source.json'], outputs=['{out}/source.json']),
        dict(id='clock', kind='fpga', timeout_seconds=600,
             argv=['{python}', 'tools/performance_harness/clock_qualify.py', 'verify', '--evidence', CLOCK,
                   '--image', IMAGE, '--uuid', UUID, '--out', '{out}/clock.json'],
             timing_boundary='read-only sysfs counter and xclbinutil; no device work',
             cache_state='none', outputs=['{out}/clock.json'])] + spec['steps']
    return spec


def m_anchors():
    s1 = json.loads((S1 / 'board-anchors.json').read_text())
    return batch_spec('s2-anchors', 'board-anchors', 'anchors', copy.deepcopy(s1['board_batches']['anchors']),
                      'On the run-1 image at 100 MHz the three frozen anchors pass numerically in three '
                      'independent sessions, and their device-cycle spread is recorded against the B40 anchors.',
                      ['3 anchors x 3 interleaved processes pass on image 30e79769 at 100 MHz',
                       'per-anchor graph-makespan spread and cycles vs the B40 Stage 1 anchors (S2-01: report '
                       'cycles and wall time separately; 90->100 MHz alone is ~11.1 %)'],
                      dict(expected_board_hours=0.13, anchors=list(s1['registration']['anchors'])))


def m_original():
    s1 = json.loads((S1 / 'board-original-suite.json').read_text())
    return batch_spec('s2-original-suite', 'board-original-suite', 'original',
                      copy.deepcopy(s1['board_batches']['original']),
                      'The original R40 full suite (113 unique points, hbm.json) passes its numeric references on '
                      'the run-1 image at 100 MHz under the X8 profile, with no fallback.',
                      ['113 configurations pass numerically (max_rel_err <= tol), evidence_level measured-board',
                       'device cycles per point recorded for the S2-01 cycles/wall-time report vs B40'],
                      dict(expected_board_hours=2.0, coverage=s1['registration']['coverage'],
                           order=s1['registration']['order'], sampling_rule=s1['registration']['sampling_rule']))


def tool_step(sid, argv, outputs, timeout, hbm_bytes, **env):
    return dict(id=sid, kind='fpga', timeout_seconds=timeout, env=fpga_env(hbm_bytes, **env), argv=argv,
                timing_boundary=FUNCTIONAL, cache_state=CACHE, outputs=outputs)


def summary_step(groups, dumps, require_x8):
    argv = ['{python}', 'tools/performance_harness/stage2_board.py', 'fp8-summary', '--out', '{out}/summary.json']
    for g, report in groups:
        argv += ['--group', f'{g}={report}']
    for d in dumps:
        argv += ['--dump', d]
    if require_x8:
        argv.append('--require-x8')
    return dict(id='summary', kind='report', timeout_seconds=600, argv=argv,
                inputs=[r for _, r in groups], outputs=['{out}/summary.json'])


FP8_COMPETING = ['the 100 MHz image or the X8 transfer engine changes an FP8 result the simulators gave '
                 '(bitwise compare to the per-K / blockwise references)',
                 'the board runtime path (HBM DMA, 3-core launch map) differs from the simulator stand-ins',
                 'a kernel falls back to the gather path, so X8 is not exercised (asm census)']


def m_fp8_cases():
    steps = [tool_step('cases', ['{python}', 'tools/performance_harness/fp8_cases.py', '--out', '{out}/cases'],
                       ['{out}/cases/report.json'], 4 * 3600, 8 * GIB, TNCA_IEEE_DIV='1'),
             summary_step([('fp8_cases', '{out}/cases/report.json')], ['{out}/cases/dump'], True)]
    return base('s2-fp8-cases',
                'The Torch-level FP8 matrix (fp8_cases.py, full: casts, plumbing, _scaled_mm per-tensor/rowwise '
                'incl. e4m3 x e5m2, W8A8 and weight-only linears) passes bitwise on the run-1 image, and the '
                'compiled FP8 dots load B with the X8 transposed 1-byte write (t.ld.async.b.t).',
                FP8_COMPETING,
                ['m.mma.async FP8 MMAs in the _scaled_mm/linear kernels (mma census per case)',
                 f'{X8_LOAD} present in the compiled FP8 dot kernels (X8, HAS_TENSOR_XPOSE=2)'],
                ['fp8_cases full: every case pass (simulator reference: 100/100 on systemc-tlm-grid)',
                 f'summary: {X8_LOAD} count > 0 over the dumped scaled_mm/linear kernels'],
                dict(expected_board_hours=1.0, tool='tools/performance_harness/fp8_cases.py',
                     reference='evidence/r41/fp8/README.md (100/100 systemc-tlm-grid; 8/8 fpga-multicore-sim)'),
                steps, hbm_bytes=8 * GIB, conclusion='{out}/summary.json')


def m_fp8_blockwise():
    steps = [tool_step('blockwise', ['{python}', 'tools/performance_harness/fp8_blockwise_cases.py', '--out',
                                     '{out}/blockwise'], ['{out}/blockwise/report.json'], 4 * 3600, 8 * GIB,
                       TNCA_IEEE_DIV='1', TNCA_KERNEL_DUMP='1', TNCA_DUMP_DIR='{out}/blockwise/dump'),
             summary_step([('fp8_blockwise_cases', '{out}/blockwise/report.json')], ['{out}/blockwise/dump'], False)]
    return base('s2-fp8-blockwise',
                'Blockwise-scaled _scaled_mm / _scaled_mm_v2 (K-block Triton template, E4M3 and E4M3 x E5M2 in '
                '16-deep steps) passes bitwise against fp8_block_reference on the run-1 image.',
                FP8_COMPETING, ['m.mma.async FP8 MMAs per K block; per-block scale multiply in the epilogue'],
                ['fp8_blockwise_cases full: every case pass'],
                dict(expected_board_hours=0.5, tool='tools/performance_harness/fp8_blockwise_cases.py',
                     reference='evidence/r41/fp8/blockwise/README.md'),
                steps, hbm_bytes=8 * GIB, conclusion='{out}/summary.json')


def m_fp8_hub():
    hub = dict(TNCA_IEEE_DIV='1', HF_HOME=FP8_IN + '/hf', HF_HUB_OFFLINE='1')
    steps = [tool_step('hub', [FP8_ENV, '-u', '{root}/tools/performance_harness/fp8_kernels_cases.py', '--out',
                               '{out}/hub', '--group', 'v4', '--group', 'compile', '--group', 'transformers'],
                       ['{out}/hub/report.json'], 4 * 3600, 8 * GIB, **hub),
             tool_step('hubv1', [FP8_ENV, '-u', '{root}/tools/performance_harness/fp8_kernels_cases.py', '--out',
                                 '{out}/hubv1', '--group', 'v1', '--v1-path', FP8_IN + '/hub-main'],
                       ['{out}/hubv1/report.json'], 2 * 3600, 8 * GIB, **hub),
             summary_step([('hub', '{out}/hub/report.json'), ('hub_v1', '{out}/hubv1/report.json')],
                          ['{out}/hub/dump', '{out}/hubv1/dump'], False)]
    return base('s2-fp8-hub',
                'The official Hub FP8 kernels (kernels-community/finegrained-fp8 v4 and v1, loaded with '
                'kernels.get_kernel) and transformers 5.20.2 FP8Linear (eager and compiled) run unmodified on the '
                'run-1 image and match their references bitwise.',
                FP8_COMPETING, ['FP8 MMAs and per-block scaling in the Hub kernels (code facts)'],
                ['hub (v4, compile, transformers): every case pass', 'hub v1: every case pass'],
                dict(expected_board_hours=1.0, tool='tools/performance_harness/fp8_kernels_cases.py',
                     python=FP8_ENV + ' (venv over ~/tc_env/torchenv: transformers 5.20.2, kernels 0.17.2, '
                                      'built by the queue before loading)',
                     inputs=FP8_IN + ' (Hub kernel snapshot and main checkout staged from GB10, '
                                     'evidence/r41/stage2/board/launches/stage_inputs.sh)',
                     reference='evidence/r41/fp8/kernels/README.md'),
                steps, hbm_bytes=8 * GIB, conclusion='{out}/summary.json')


QWEN_MODULES = ['^model.language_model.layers.13.(self_attn.(q|k|v|o)_proj|mlp.(gate|up|down)_proj)$']


def m_qwen():
    env = dict(TNCA_IEEE_DIV='1', HF_HOME=FP8_IN + '/hf', HF_HUB_OFFLINE='1',
               QWEN_FP8_VIEW_DIR=FP8_IN + '/snapview', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
               TNCA_TORCH_DEVICE='native', TNCA_KERNEL_DUMP='1')
    tool = '{root}/tools/models/qwen_tnca_layers.py'
    steps = [tool_step('linears', [FP8_ENV, '-I', '-u', tool, 'linears', '--model', 'vl', '--teacher-force',
                                   FP8_IN + '/qwen/vl/ref_fp8', '--modules', *QWEN_MODULES, '--tnca-tree', '{root}',
                                   '--threads', '4', '--out', '{out}/linears/result.json'],
                       ['{out}/linears/result.json'], 3 * 3600, 8 * GIB, TNCA_DUMP_DIR='{out}/linears/dump', **env),
             tool_step('decode', [FP8_ENV, '-I', '-u', tool, 'decode', '--model', 'vl', '--index', '13',
                                  '--bucket', '32', '--steps', '4', '--tnca-tree', '{root}', '--threads', '4',
                                  '--out', '{out}/decode/result.json'],
                       ['{out}/decode/result.json'], 3 * 3600, 8 * GIB, TNCA_DUMP_DIR='{out}/decode/dump', **env)]
    return base('s2-qwen-fp8',
                'Qwen3-VL-2B-Instruct-FP8 layer 13: its seven FP8Linear modules (official Hub-kernel route, '
                'M = prefill and M = 1) and the decoder pair 13-14 with device KV cache over 4 decode steps '
                'reproduce the simulator results on the run-1 image (bitwise to the TNCA-order re-derivation; '
                'decode against the CPU W8A8 model run).',
                FP8_COMPETING + ['host memory or wall time of the model load (RSS limit 22 GiB)'],
                [f'{X8_LOAD} in the FP8 dot kernels (X8 dot, evidence/r41/fp8/models_tnca)'],
                ['linears: 14/14 (7 modules x 2 M) pass', 'decode: 4/4 steps within the recorded floors'],
                dict(expected_board_hours=0.7, tool='tools/models/qwen_tnca_layers.py',
                     simulator_wall='SystemC/TLM: linears 1.5-75 s per module, decode pair 6.6 min',
                     inputs=FP8_IN + ' (HF snapshot, snapview, CPU reference ref_fp8, staged from GB10)',
                     reference='evidence/r41/fp8/models_tnca/README.md'),
                steps, hbm_bytes=8 * GIB, extra_sources=['tools/models'])


def m_hbm8(name):
    import hbm8_manifest
    spec = hbm8_manifest.manifest(name)
    spec['id'] = 's2-' + name[len('board-'):]
    spec['board'] = board_block(8 * GIB)
    spec['hypothesis'] = spec['hypothesis'].replace('delivered three-core B40 image a88ebdf5 at 90 MHz',
                                                    'R41 Stage 2 run-1 image 30e79769 at 100 MHz')
    spec['sources'] = [s for s in spec['sources'] if not s.startswith(('spec/r41/hbm8', 'evidence/r41/hbm8'))] + \
        [MAP, f'spec/r41/stage2/board/{spec["id"]}.json']
    for s in spec['steps']:
        if s['kind'] == 'fpga':
            hbm = 16 * GIB if s['id'] == 'clamp' else 8 * GIB
            s['env'] = fpga_env(hbm)
            if 'board' in s:
                s['board'] = dict(backend_args=backend_args(hbm))
    spec['registration'] = dict(spec['registration'], registered_utc='2026-10-23',
                                launch='evidence/r41/stage2/board/launches/board_queue.sh',
                                generator='tools/performance_harness/stage2_board.py (from hbm8_manifest.py)',
                                xclbin_map='evidence/r41/stage2/board/xclbin_hbm_map.json: identical HBM topology, '
                                           'group connectivity and 4/8 GiB allocation to the B40 map')
    spec['registration'].pop('simulator_twin', None)
    return spec


def m_endurance():
    batch = dict(sampling='fixed', fixed=dict(warmup=0, timed=1), configurations=configs(MIX))
    spec = batch_spec(
        's2-endurance', None, 'mix', batch,
        'The run-1 image at 100 MHz sustains a two-hour continuous mixed run: every session (one ArchBench '
        'process or one FP8 matmul process) passes numerically, the card identity never changes, and neither the '
        'per-session peak RSS of a repeated configuration nor host memory grows over the run.',
        ['every session pass; >= 2 h of continuous sessions', 'per-configuration peak RSS and device cycles flat '
         'over rounds (least-squares slope recorded); MemAvailable trend recorded'],
        dict(expected_board_hours=2.1, duration_seconds=7200, purpose='S2-01 item 8',
             mix=list(MIX) + ['fp8_cases --quick --group scaled_mm']))
    spec['competing_explanations'] = ['host runtime leaks (XRT BOs, mapped buffers, compile caches) grow per session',
                                      'device state left by one session changes a later session (numeric failure)',
                                      'thermal or shared-card interference over two hours']
    spec['expected_counter_changes'] = ['no counter change between rounds of the same configuration']
    spec['steps'][0].update(
        id='mix', timeout_seconds=4 * 3600,
        argv=['{python}', 'tools/performance_harness/endurance_mixed.py', '--manifest',
              'spec/r41/stage2/board/s2-endurance.json', '--batch', 'mix', '--root', '{root}', '--out', '{out}/mix',
              '--python', '{python}', '--duration', '7200', '--fp8-every', '3'],
        outputs=['{out}/mix/endurance.json', '{out}/mix/sessions.jsonl'])
    spec['steps'][0]['env'] = fpga_env(4 * GIB, TNCA_IEEE_DIV='1')
    spec['conclusion_source'] = '{out}/mix/endurance.json'
    return spec


BUILDERS = {'s2-identity': m_identity, 's2-anchors': m_anchors, 's2-fp8-cases': m_fp8_cases,
            's2-fp8-blockwise': m_fp8_blockwise, 's2-hbm8': lambda: m_hbm8('board-hbm8'),
            's2-hbm8-big': lambda: m_hbm8('board-hbm8-big'), 's2-original-suite': m_original,
            's2-fp8-hub': m_fp8_hub, 's2-qwen-fp8': m_qwen, 's2-endurance': m_endurance}


def manifest(name):
    return BUILDERS[name]()


# ----------------------------------------------------------- image source ----
def tracked_files(commit, paths):
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', commit, '--', *paths],
                         capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.splitlines():
        meta, path = line.split('\t', 1)
        blob = meta.split()[2]
        data = subprocess.run(['git', '-C', str(ROOT), 'cat-file', 'blob', blob], capture_output=True,
                              check=True).stdout
        rows[path] = hashlib.sha256(data).hexdigest()
    return rows


def image_source():
    files = tracked_files(SOURCE_COMMIT, IMAGE_SOURCE_PATHS)
    return dict(schema='tnca-r41s2-image-source/1', commit=SOURCE_COMMIT, branch='r41s2-t2',
                archive_sha256='4a2dcaf62b7b3c9d6ff94144dc51793003f8e49e51c9817a19361738abfadd48',
                image_sha256=IMAGE_SHA256, image_uuid=UUID, profile=PROFILE, clock_hz=CLOCK_HZ,
                image_tcl_sha256='50843bee977ebad5712205d6a4a8ca4827477d3c5ceafd5b5ecaa93f9bb1b58b',
                image_json_sha256='1d185bbefaf9e1d6f7394763263ece23289ccecdb2192085d230c74ac760318c',
                launch_record='~/tnca_images/r41s2_run1_c9f25787494a/r41s2_run1_c9f25787494a.launch.json',
                paths=list(IMAGE_SOURCE_PATHS), files=files,
                files_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def source_check(root, out):
    rec = json.loads((ROOT / SOURCE).read_text())
    root = Path(root)
    have = {}
    for p in rec['paths']:
        base = root / p
        items = [base] if base.is_file() else sorted(x for x in base.rglob('*') if x.is_file())
        for f in items:
            have[str(f.relative_to(root))] = file_sha(f)
    missing = sorted(set(rec['files']) - set(have))
    extra = sorted(set(have) - set(rec['files']))
    changed = sorted(k for k in set(have) & set(rec['files']) if have[k] != rec['files'][k])
    problems = []
    if missing or changed or extra:
        problems.append(f'image source differs: {len(missing)} missing, {len(changed)} changed, {len(extra)} extra')
    for key, path in (('image_tcl_sha256', 'evidence/r41/stage2/integration/image.tcl'),
                      ('image_json_sha256', 'evidence/r41/stage2/integration/image.json')):
        if file_sha(root / path) != rec[key]:
            problems.append(f'{path} is not the image parameter file')
    params = json.loads((root / 'evidence/r41/stage2/integration/image.json').read_text())
    if params.get('profile') != PROFILE or params['parameters'].get('HAS_TENSOR_XPOSE') != 2:
        problems.append('image parameters are not core-vrf-sync-v5-x8 / HAS_TENSOR_XPOSE=2')
    target = json.loads((root / 'compiler/targets/tnca-sim-vrf-v5-x8.json').read_text())
    caps = json.dumps(target)
    if 'tensor_transpose_8bit' not in caps:
        problems.append('compiler target lacks tensor_transpose_8bit')
    row = dict(schema='tnca-r41s2-source-check/1', commit=rec['commit'], files=len(have),
               files_sha256=hashlib.sha256(json.dumps(have, sort_keys=True).encode()).hexdigest(),
               registered_files_sha256=rec['files_sha256'], missing=missing[:20], changed=changed[:20],
               extra=extra[:20], profile=params.get('profile'), parameters=params['parameters'],
               status='pass' if not problems else 'fail', problems=problems)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(row, indent=1) + '\n')
    print(row['status'], problems)
    return 0 if not problems else 1


# ------------------------------------------------------------- fp8 summary ----
def count_cases(report):
    cases = report.get('cases') or report.get('rows') or []     # fp8_kernels_cases: cases; fp8_cases: rows
    if isinstance(cases, dict):
        cases = list(cases.values())
    status = [c.get('status') for c in cases if isinstance(c, dict)]
    return dict(cases=len(status), passed=status.count('pass'),
                not_passed=[c.get('case') or c.get('name') for c in cases
                            if isinstance(c, dict) and c.get('status') != 'pass'][:20])


def fp8_summary(out, groups, dumps, require_x8, image_label='run-1', require_dp=False):
    rows, problems = {}, []
    for spec in groups:
        name, path = spec.split('=', 1)
        p = Path(path)
        if not p.is_file():
            problems.append(f'{name}: no report')
            continue
        rep = json.loads(p.read_text())
        rows[name] = dict(report=str(p), sha256=file_sha(p), status=rep.get('status'), **count_cases(rep))
        if not rows[name]['cases'] or rows[name]['passed'] != rows[name]['cases']:
            problems.append(f"{name}: {rows[name]['passed']}/{rows[name]['cases']} pass")
    x8, dp = {}, {}
    for d in dumps:
        for s in sorted(Path(d).rglob('*.s')) if Path(d).is_dir() else []:
            assembly = s.read_text()
            n = len(re.findall(r'\b' + re.escape(X8_LOAD) + r'\b', assembly))
            if n:
                x8[str(s.relative_to(d))] = n
            n = len(re.findall(r'\b' + re.escape('m.mma.dp.async') + r'\b', assembly))
            if n:
                dp[str(s.relative_to(d))] = n
    if require_x8 and not x8:
        problems.append(f'no {X8_LOAD} in any dumped kernel: X8 not exercised')
    if require_dp and not dp:
        problems.append('no m.mma.dp.async in any dumped kernel: DP not exercised')
    record = dict(schema='tnca-r41s2-fp8-summary/1', groups=rows, x8_transposed_loads=x8,
                  x8_kernels=len(x8), x8_instructions=sum(x8.values()),
                  dp_kernels=len(dp), dp_instructions=sum(dp.values()),
                  result='pass' if not problems else 'fail', residual=problems or 'none',
                  conclusion=(f'every FP8 case passed on the {image_label} image' + (
                      f'; X8 transposed load in {len(x8)} kernels' if x8 else '')) if not problems
                  else 'FP8 board check failed: ' + '; '.join(problems),
                  next_step='compare cycles with the simulator reports' if not problems
                  else 'triage the failing cases from the preserved reports')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=1) + '\n')
    print(record['result'], problems)
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--image-source', action='store_true')
    sub = ap.add_subparsers(dest='cmd')
    sc = sub.add_parser('source-check')
    sc.add_argument('--root', type=Path, required=True)
    sc.add_argument('--out', type=Path, required=True)
    fs = sub.add_parser('fp8-summary')
    fs.add_argument('--out', type=Path, required=True)
    fs.add_argument('--group', action='append', default=[])
    fs.add_argument('--dump', action='append', default=[])
    fs.add_argument('--require-x8', action='store_true')
    fs.add_argument('--image-label', choices=('run-1', 'run-2'), default='run-1')
    fs.add_argument('--require-dp', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'source-check':
        return source_check(a.root, a.out)
    if a.cmd == 'fp8-summary':
        return fp8_summary(a.out, a.group, a.dump, a.require_x8, a.image_label, a.require_dp)
    if a.image_source:
        (ROOT / SOURCE).write_text(json.dumps(image_source(), indent=1) + '\n')
    if a.write:
        SPEC.mkdir(parents=True, exist_ok=True)
        for name in ORDER:
            (SPEC / f'{name}.json').write_text(json.dumps(manifest(name), indent=1) + '\n')
    if a.check:
        bad = [n for n in ORDER if json.loads((SPEC / f'{n}.json').read_text()) != manifest(n)]
        if bad:
            sys.exit(f'registered manifests differ from the generator: {bad}')
        print('stage2 board manifests: registered == generated')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
