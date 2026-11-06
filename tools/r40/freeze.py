#!/usr/bin/env python3
"""R40-00: freeze the plan, the subsets and the ablation matrix -- before any
R40 optimisation is measured.

    python3 tools/r40/freeze.py [--out spec/r40] [--check]

doc/r40_tasks_and_acceptance.md sec 3.2 item 3: every subset a performance
gate is judged on is chosen by a rule over the point definitions (family,
tier, shape, arithmetic intensity) and the frozen B39 records, and resolved
to a list of points with their definition hashes now, not after the
candidates are measured; a holdout is set aside that no strategy or
threshold may be tuned on. This writes

  spec/r40/suites/<name>.json   each subset: its rule, its points and hashes
  spec/r40/plan.json            the work packages, their dependencies, their
                                gates (command, evidence, kind), the
                                experiments (subset, knobs, timing,
                                threshold, baseline), the sample plan and the
                                formulas
  spec/r40/ablation_matrix.json every mechanism, its switch, its single-factor
                                pair, its leave-one-out and its interactions

`--check` regenerates in memory and fails if a written file differs: a
change to the suites, the thresholds or a point definition is an amendment
(sec 4, preamble) and has to be made visibly.

The rules read the registry, so this needs torch (the GB10 host's
torchenv); everything it writes is JSON the rest of the tooling reads
without it.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from archbench import registry     # noqa: E402

EXISTING = ('tiny', 'small', 'medium')
POINTWISE = ('add', 'silu', 'gelu_tanh', 'swiglu', 'copy', 'softmax', 'rmsnorm', 'layernorm',
             'argmax', 'cumsum', 'rope')
GEMM = ('matmul', 'linear', 'gemv', 'bmm', 'mlp')
STREAMING = ('add', 'silu', 'copy', 'transpose', 'softmax', 'rmsnorm')
ACTIVE_CORES = 3                     # the U280 target (doc sec 2.2)
HOLDOUT_SHARE = 0.3

# Board sampling (doc sec 3.3).
SAMPLE_PLAN = dict(
    board=dict(warmup=5, timed=30, blocks=3, order='interleaved A/B, randomised per block',
               long_task=dict(threshold_s=1.0, timed=10, blocks=3),
               report=['median', 'p95', 'paired bootstrap 95% interval']),
    simulation=dict(note='deterministic: one run is the measurement; RTL cycles are not seconds'),
    bootstrap=dict(resamples=2000, alpha=0.05, seed=20261010))

FORMULAS = dict(
    s_geo=('exp(mean_i log(T_A,i / T_B,i)) over the frozen subset, T the per-point median of the '
           'timed samples (R39\'s definition)'),
    s_geo_balanced_weighted=('exp(sum_i w_i log(T_A,i / T_B,i)), w balanced: every family the same '
                             'mass, uniform inside it -- reported beside S_geo, never instead of it'),
    E='the weighted harmonic mean of eta over the complete suite (archbench.report); a companion, '
      'never a speed',
    perf_pass='95% paired bootstrap lower bound of S_geo > 1 and the frozen threshold met',
    regress='a point slower by more than 2% needs a recorded cause and an automatic selection; '
            'more than 5% under the final default fails (doc sec 3.2 item 5)',
    loo='the final system with one mechanism off: at least 3% on its target subset, else '
        '"no significant contribution" (doc sec 3.2 item 6)')


def pt_entry(p):
    return dict(id=p.id, sha256=p.definition_sha256(), family=p.family, tier=p.tier, dtype=p.dtype)


def intensity(p):
    d = p.demand
    by = (d.read_bytes or 0) + (d.write_bytes or 0)
    return ((d.macs or 0) * 2 + (d.fp_ops or 0)) / by if by else 0.0


def kernels_of():
    """Point id -> number of kernels its B39 call launched, and the mean
    device cycles per launch (the frozen board records)."""
    out = {}
    for t in EXISTING:
        path = ROOT / 'spec' / 'r40' / 'b39' / f'{t}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            per = (r.get('time') or {}).get('per_launch') or []
            if per:
                out[r['point_id']] = (len(per), sum(x.get('cycles', 0) for x in per) / len(per))
    return out


def held_out(pid, salt='r40-holdout', share=HOLDOUT_SHARE):
    """A stable, rule-free choice: the point id's hash decides."""
    h = int(hashlib.sha256(f'{salt}:{pid}'.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def working_set(p):
    d = p.demand
    return (d.read_bytes or 0) + (d.write_bytes or 0)


def subsets():
    pts = registry.points()
    by_tier = {}
    for p in pts:
        by_tier.setdefault(p.tier, []).append(p)
    k = kernels_of()
    S = {}

    def add(name, rule, chosen):
        chosen = sorted(chosen, key=lambda p: p.id)
        S[name] = dict(schema='tnca-r40-subset/1', name=name, rule=rule,
                       points=[pt_entry(p) for p in chosen])

    existing = [p for t in EXISTING for p in by_tier.get(t, [])]
    large, hbm = by_tier.get('large', []), by_tier.get('hbm', [])
    add('existing', 'every point of tiny, small and medium', existing)
    add('large', 'every point of the original large tier, unshrunk', large)
    add('hbm-tier', 'every point of the hbm tier: archbench/hbm.py and the R40 decoder '
        '(archbench/models.py LLAMA["hbm"])', hbm)
    add('hbm', 'the HBM suite (R40-00): every existing point, the original large tier and the '
        'hbm tier; with the multi-step decoder workload of plan.json', existing + large + hbm)
    sdpa = [p for p in large if p.family in ('attention_prefill', 'attention_decode')]
    add('hbm-core', 'R40-09 acceptance 3: the hbm tier (streaming, GEMM, transpose, gather, norm, '
        'decoder) and the large SDPA points', hbm + sdpa)
    reps = [p for p in by_tier.get('medium', [])
            if (p.family, p.dtype) in (('matmul', 'fp16'), ('softmax', 'fp16'), ('add', 'fp16'))]
    add('R40-01.representatives', 'R40-01 acceptance 3: the medium fp16 matmul (GEMM), softmax '
        '(reduction) and add (pointwise), whose descriptor code and device events are shown', reps)
    add('R40-04.ge64mib', 'large and hbm points whose semantic bytes (read + write) are at least '
        '64 MiB', [p for p in large + hbm if working_set(p) >= 64 << 20])
    pool = large + hbm
    add('R40-04.calibration', 'half of large and hbm by the hash of the point id: the HBM memory '
        'model is calibrated on these (doc sec 3.4)',
        [p for p in pool if not held_out(p.id, 'r40-hbm-model', 0.5)])
    add('R40-04.model-holdout', 'the other half: the HBM model is judged on these and never fitted '
        'to them (doc sec 3.4: median error <= 10 %, p95 <= 20 %)',
        [p for p in pool if held_out(p.id, 'r40-hbm-model', 0.5)])
    add('R40-02.graphs3', 'existing points whose B39 call launched at least 3 kernels',
        [p for p in existing if k.get(p.id, (0, 0))[0] >= 3])
    add('R40-02.small-kernel-graphs',
        'R40-02.graphs3 points whose B39 kernels average at most 20000 device cycles',
        [p for p in existing if k.get(p.id, (0, 0))[0] >= 3 and k[p.id][1] <= 20000])
    add('R40-03.pointwise-reduction',
        f'medium and hbm points of the families {", ".join(POINTWISE)}',
        [p for p in by_tier.get('medium', []) + by_tier.get('hbm', []) if p.family in POINTWISE])
    add('R40-03.gemm-reuse', f'medium and hbm points of the families {", ".join(GEMM)}',
        [p for p in by_tier.get('medium', []) + by_tier.get('hbm', []) if p.family in GEMM])
    add('R40-04.streaming', f'hbm points of the families {", ".join(STREAMING)}',
        [p for p in hbm if p.family in STREAMING])
    big = by_tier.get('medium', []) + by_tier.get('large', []) + by_tier.get('hbm', [])
    add('R40-05.compute',
        f'medium, large and hbm points with at least 1e9 MACs -- 16x16 output tiles then give a '
        f'grid of at least 32 x {ACTIVE_CORES} programs -- and arithmetic intensity >= 32 FLOP/B',
        [p for p in big if (p.demand.macs or 0) >= 1e9 and intensity(p) >= 32])
    add('R40-06.reuse',
        'medium, large and hbm GEMM-family and attention points with arithmetic intensity >= 16 FLOP/B',
        [p for p in big if (p.family in GEMM or p.family.startswith('attention'))
         and intensity(p) >= 16])
    add('R40-07.long-k',
        'medium, large and hbm GEMM-family points with K >= 256',
        [p for p in big if p.family in GEMM and (p.params.get('k') or 0) >= 256])
    add('R40-07.compute-bound',
        'medium, large and hbm points with arithmetic intensity >= 64 FLOP/B',
        [p for p in big if intensity(p) >= 64])
    add('R40-08.long-prefill',
        'attention_prefill and llama_prefill points with a sequence of at least 512',
        [p for p in pts if p.family in ('attention_prefill', 'llama_prefill')
         and (p.params.get('s') or 0) >= 512])
    pool = [p for p in big]
    add('R40-09.holdout', f'{int(HOLDOUT_SHARE * 100)} % of medium, large and hbm, chosen by the '
        'hash of the point id; never used to choose a strategy or a threshold',
        [p for p in pool if held_out(p.id)])
    add('R40-09.tune', 'the rest of medium, large and hbm',
        [p for p in pool if not held_out(p.id)])
    for s in S.values():
        s['sha256'] = hashlib.sha256(json.dumps(s['points'], sort_keys=True).encode()).hexdigest()
    return S


def workloads():
    from archbench.models import LLAMA, MODEL_TOL, _llama_weight_bytes
    from dataclasses import asdict
    c, bsz, s, _p = LLAMA['hbm']
    pre = registry.points(tier='hbm', family='llama_prefill')[0]
    dec = registry.points(tier='hbm', family='llama_decode')[0]
    return dict(llama_generate=dict(
        harness='python -m archbench.generate --tier hbm --new 64',
        what='a decoder used the way a model is used: one prefill, then greedy decode steps, each '
             'fed the device\'s own token and cache (R40-09: >= 4 layers, >= 64 MiB of weights, '
             'prefill >= 512, >= 64 decode steps)',
        model=dict(asdict(c), dtype='fp16', batch=bsz,
                   weights='random: archbench.models.Llama(cfg, archbench.ops.gen(seed), "fp16")',
                   seed=0, weight_bytes=_llama_weight_bytes(c, 2) + 2 * c.dim * c.vocab),
        points=dict(prefill=dict(id=pre.id, sha256=pre.definition_sha256()),
                    decode=dict(id=dec.id, sha256=dec.definition_sha256())),
        prefill=s, decode_steps=64,
        reference=('eager PyTorch in binary32 on the CPU: every step against the reference given '
                   'the same inputs (the device\'s own tokens and cache) at tolerance '
                   f'{MODEL_TOL["fp16"]}, and the generated ids against a generation run wholly in '
                   'the reference, a differing id accepted only where the reference\'s two logits '
                   'nearly tie (archbench/generate.py)'),
        report=['TTFT', 'decode median and p95 ms/token', 'tokens/s', 'peak HBM',
                'input/output/weight/KV traffic']))


def gate(wp, gid, kind, command, required=True, note=''):
    return dict(id=f'{wp}.{gid}', kind=kind, command=command,
                evidence=f'sim/build/r40/{wp}.json', required=required, note=note)


def experiment(subset, vary, timing, threshold=None, baseline='B40-S', note=''):
    return dict(subset=f'spec/r40/suites/{subset}.json', vary=vary, timing=timing,
                threshold=threshold, baseline=baseline, note=note, explained=[])


def plan(S):
    W = []
    W.append(dict(id='R40-00', title='close R39, freeze the protocol', depends=[],
                  gates=[gate('R40-00', 'protocol', 'capability', 'make r40-protocol',
                              note='acceptance 1 and 3: every manipulation refused, identity is 1; '
                                   'acceptance.py passes nothing missing'),
                         gate('R40-00', 'b39-reproducible', 'capability', 'make r40-b39',
                              note='acceptance 2: B39 numerics, cycles and gates reproduce; time '
                                   'boundaries check out'),
                         gate('R40-00', 'frozen', 'capability', 'make r40-freeze-check',
                              note='plan, subsets, suites, API lock, B39 records and the HBM '
                                   'suite coverage as frozen')]))
    W.append(dict(id='R40-01', title='native PyTorch TNCA device', depends=['R40-00'],
                  gates=[gate('R40-01', 'device', 'capability', 'make r40-device'),
                         gate('R40-01', 'mixed-device', 'capability', 'make r40-device'),
                         gate('R40-01', 'device-time', 'capability', 'make r40-device',
                              note='a capability gate with a floor: S_geo >= 1.0 against B39, no '
                                   'unhandled regression over 5 % (doc R40-01 acceptance 3)')]))
    W.append(dict(id='R40-02', title='residency, bulk DMA, events, batched submission',
                  depends=['R40-01', 'R40-04'],
                  gates=[gate('R40-02', 'zero-transfer', 'capability', 'make r40-residency'),
                         gate('R40-02', 'resident', 'perf', 'make r40-residency'),
                         gate('R40-02', 'batched', 'perf', 'make r40-residency'),
                         gate('R40-02', 'stress', 'capability', 'make r40-residency')]))
    W.append(dict(id='R40-03', title='general tiles, layout, resource allocation', depends=['R40-01'],
                  gates=[gate('R40-03', 'tiles', 'capability', 'make r40-tiles'),
                         gate('R40-03', 'wide-blocks', 'perf', 'make r40-tiles'),
                         gate('R40-03', 'gemm-tile', 'perf', 'make r40-tiles'),
                         gate('R40-03', 'contract', 'capability', 'make r40-tiles')]))
    W.append(dict(id='R40-04', title='single-core HBM, scalable memory interface', depends=['R40-00'],
                  gates=[gate('R40-04', 'hbm', 'capability', 'make r40-hbm'),
                         gate('R40-04', 'bandwidth', 'perf', 'make r40-hbm'),
                         gate('R40-04', 'robustness', 'capability', 'make r40-hbm')]))
    W.append(dict(id='R40-05', title='multi-SLR physical cores in the common runtime',
                  depends=['R40-04'],
                  gates=[gate('R40-05', 'multicore', 'capability', 'make r40-multicore'),
                         gate('R40-05', 'scaling', 'perf', 'make r40-multicore'),
                         gate('R40-05', 'dispatch', 'capability', 'make r40-multicore')]))
    W.append(dict(id='R40-06', title='NUCA, several memory ports, locality', depends=['R40-05'],
                  gates=[gate('R40-06', 'nuca', 'capability', 'make r40-nuca'),
                         gate('R40-06', 'streaming-scaling', 'perf', 'make r40-nuca'),
                         gate('R40-06', 'locality', 'perf', 'make r40-nuca'),
                         gate('R40-06', 'semantics', 'capability', 'make r40-nuca')]))
    W.append(dict(id='R40-07', title='asynchronous pipelines, layout moves, matrix throughput',
                  depends=['R40-03', 'R40-06'],
                  gates=[gate('R40-07', 'pipeline', 'capability', 'make r40-pipeline'),
                         gate('R40-07', 'long-k', 'perf', 'make r40-pipeline'),
                         gate('R40-07', 'matrix', 'perf', 'make r40-pipeline'),
                         gate('R40-07', 'policy', 'capability', 'make r40-pipeline')]))
    W.append(dict(id='R40-08', title='standard Torch attention, reductions, resident KV',
                  depends=['R40-07', 'R40-02'],
                  gates=[gate('R40-08', 'attention', 'capability', 'make r40-attention'),
                         gate('R40-08', 'prefill', 'perf', 'make r40-attention'),
                         gate('R40-08', 'decode', 'capability', 'make r40-attention')]))
    W.append(dict(id='R40-09', title='budgeted automatic selection, applications, scaling',
                  depends=['R40-08'],
                  gates=[gate('R40-09', 'autotune', 'capability', 'make r40-autotune'),
                         gate('R40-09', 'holdout', 'perf', 'make r40-autotune'),
                         gate('R40-09', 'suites', 'capability', 'make r40-autotune'),
                         gate('R40-09', 'model', 'perf', 'make r40-autotune')]))
    W.append(dict(id='R40-10', title='final board sign-off', depends=['R40-09'],
                  gates=[gate('R40-10', 'acceptance', 'capability', 'make r40-acceptance')]))
    X = {
        'R40-01.device-time': dict(experiment(
            'existing', ['TNCA_TORCH_DEVICE', 'TNCA_RCTX_ROLES', 'platform.software_tree',
                         'platform.inductor', 'platform.device_id', 'platform.image'],
            'device-cycles-serial-grid', 1.0, baseline='B39',
            note='a capability gate: S_geo >= 1.0 and no unhandled >5% regression; A is B39 '
                 '(spec/r40/baseline.json, wrapped by tools/r40/b39.py), B the native device on '
                 'the same image with per-kernel staging'), legacy_a=True),
        'R40-02.resident': experiment('R40-02.graphs3', ['TNCA_TRANSFER_POLICY'], 'request-latency',
                                      1.5, note='per-kernel staging against residency'),
        'R40-02.batched': experiment('R40-02.small-kernel-graphs', ['TNCA_SUBMIT'],
                                     'resident-call-latency', 1.2),
        'R40-03.wide-blocks': experiment('R40-03.pointwise-reduction', ['TNCA_MAX_BLOCK'],
                                         'graph-makespan', 1.3),
        'R40-03.gemm-tile': experiment('R40-03.gemm-reuse', ['TNCA_GEMM_TILE'], 'graph-makespan',
                                       1.15),
        'R40-04.ingress': experiment('R40-04.streaming', ['TNCA_RTL_INGRESS'], 'graph-makespan',
                                     1.5, note='also >= 50% of the frozen bottleneck peak per direction'),
        'R40-05.three-cores': experiment('R40-05.compute', ['TNCA_ACTIVE_CORES'], 'graph-makespan',
                                         2.1, baseline='B40-N', note='1 -> 3 active cores; 2 cores >= 1.5'),
        'R40-06.streaming-cores': experiment('R40-04.streaming', ['TNCA_ACTIVE_CORES'],
                                             'graph-makespan', 1.8, baseline='B40-N'),
        'R40-06.ports': experiment('R40-04.streaming', ['TNCA_RTL_MEM_PORTS'], 'graph-makespan', 1.3),
        'R40-06.locality': experiment('R40-06.reuse', ['TNCA_LOCALITY'], 'graph-makespan', 1.1),
        'R40-06.dco': experiment('R40-06.reuse', ['TNCA_DCO'], 'graph-makespan', 1.1),
        'R40-07.pipeline': experiment('R40-07.long-k', ['TNCA_PIPELINE'], 'graph-makespan', 1.2),
        'R40-07.matrix': experiment('R40-07.compute-bound', ['TNCA_RTL_MMA'], 'graph-makespan', 1.2),
        'R40-08.prefill': experiment('R40-08.long-prefill', ['TNCA_SDPA'], 'graph-makespan', 1.5),
        'R40-09.holdout': experiment('R40-09.holdout', ['TNCA_TUNE'], 'graph-makespan', 1.1,
                                     note='regret p90 <= 15%'),
    }
    return dict(schema='tnca-r40-plan/1', signoff='R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM',
                document='doc/r40_tasks_and_acceptance.md', work_packages=W, experiments=X,
                subsets={n: dict(path=f'spec/r40/suites/{n}.json', sha256=s['sha256'],
                                 points=len(s['points'])) for n, s in S.items()},
                workloads=workloads(), sample_plan=SAMPLE_PLAN, formulas=FORMULAS,
                amendments=[],
                baselines=dict(
                    B39='spec/r40/baseline.json: R39\'s signed-off software, image and configuration',
                    **{'B40-S': 'native device, one core, the same HBM/NUCA system and software, '
                                'performance switches off',
                       'B40-N': 'one multi-core image; only the active core mask changes; slices, PCs '
                                'and cache capacity held',
                       'B40-F': 'the final candidate, every accepted switch on; leave-one-out from it'}),
                toolchain_lock='spec/r40/baseline.json:toolchain')


def ablation_matrix():
    M = [
        ('R40-01', 'native-device', 'runtime', 'TNCA_TORCH_DEVICE', 'cpu migration (A) / native per-kernel '
         'staging (B) / native resident (C): A-B to R40-01, B-C to R40-02'),
        ('R40-02', 'residency', 'runtime', 'TNCA_TRANSFER_POLICY', 'per-kernel / resident'),
        ('R40-02', 'submission', 'runtime', 'TNCA_SUBMIT', 'single / batched / replay'),
        ('R40-02', 'copy-overlap', 'runtime', 'TNCA_COPY_OVERLAP', '0 / 1'),
        ('R40-02', 'host-control', 'runtime', 'TNCA_HOST_CONTROL', 'mmio / queue'),
        ('R40-02', 'bulk-dma', 'runtime', 'TNCA_BULK_DMA', 'mmio words / XRT bulk'),
        ('R40-03', 'max-block', 'compile', 'TNCA_MAX_BLOCK', '16 / 32 / 64 / 128'),
        ('R40-03', 'multirow-reduction', 'compile', 'TNCA_MULTIROW', '0 / 1'),
        ('R40-03', 'gemm-tile', 'compile', 'TNCA_GEMM_TILE', '16x16 / larger'),
        ('R40-03', 'bank-allocation', 'compile', 'TNCA_BANK_ALLOC', '0 / 1'),
        ('R40-03', 'lifetime-reuse', 'compile', 'TNCA_LIFETIME', '0 / 1'),
        ('R40-03', 'spill-policy', 'compile', 'TNCA_SPILL', 'refuse / retile / spill'),
        ('R40-04', 'burst', 'rtl', 'TNCA_RTL_BURST', '0 / 1'),
        ('R40-04', 'outstanding', 'rtl', 'TNCA_RTL_OUTSTANDING', '1 / N'),
        ('R40-04', 'wide-ingress', 'rtl', 'TNCA_RTL_INGRESS', '8 B / 32 B per cycle'),
        ('R40-05', 'active-cores', 'runtime', 'TNCA_ACTIVE_CORES', '1 / 2 / 3'),
        ('R40-05', 'grid-assignment', 'runtime', 'TNCA_GRID_DISPATCH', 'static / dynamic'),
        ('R40-06', 'memory-ports', 'rtl', 'TNCA_RTL_MEM_PORTS', '1 / N'),
        ('R40-06', 'locality', 'runtime', 'TNCA_LOCALITY', 'naive / locality'),
        ('R40-06', 'dco', 'runtime', 'TNCA_DCO', 'off / on'),
        ('R40-06', 'cache-bypass', 'runtime', 'TNCA_CACHE_BYPASS', 'cache / bypass'),
        ('R40-06', 'home-mapping', 'runtime', 'TNCA_HOME_MAP', 'interleave / tensor home'),
        ('R40-06', 'dead-hint', 'runtime', 'TNCA_DEAD_HINT', '0 / 1'),
        ('R40-07', 'stages', 'compile', 'TNCA_PIPELINE', '1 / 2 / 3 stages'),
        ('R40-07', 'role-pipeline', 'compile', 'TNCA_RCTX_ROLES', 'off / gated (R39-12) / forced'),
        ('R40-07', 'transpose-pack', 'compile', 'TNCA_LAYOUT_MOVE', 'scratch repack / descriptor or pack'),
        ('R40-07', 'epilogue-fusion', 'compile', 'TNCA_EPILOGUE', 'materialised / fused'),
        ('R40-07', 'matrix', 'rtl', 'TNCA_RTL_MMA', 'v5 MTX_ABUF / overlapped or wider'),
        ('R40-08', 'sdpa', 'compile', 'TNCA_SDPA', 'decomposed / fused'),
        ('R40-08', 'kv-resident', 'runtime', 'TNCA_KV', 'copied / resident'),
        ('R40-08', 'reduction-fusion', 'compile', 'TNCA_REDUCTION_FUSION', '0 / 1'),
        ('R40-09', 'selection', 'compile', 'TNCA_TUNE', 'fixed / model / measured'),
    ]
    mech = [dict(wp=w, id=i, layer=l, switch=s, values=v, single_factor=True, leave_one_out=True)
            for w, i, l, s, v in M]
    inter = [
        dict(a='TNCA_ACTIVE_CORES', b='TNCA_RTL_MEM_PORTS', note='single/multi core x single/multi port'),
        dict(a='TNCA_LOCALITY', b='TNCA_DCO', note='naive/locality x DCO'),
        dict(a='TNCA_PIPELINE', b='TNCA_RCTX_ROLES', note='pipeline off/on x single/multi context'),
        dict(a='TNCA_PIPELINE', b='TNCA_RTL_MMA', note='pipeline x matrix'),
    ]
    return dict(schema='tnca-r40-ablation-matrix/1',
                rule=('every mechanism has its own switch and configuration hash, a single-factor '
                      'A/B and a leave-one-out from B40-F; an illegal combination is recorded '
                      'with its reason and a semantically equivalent control, never a made-up '
                      'number (doc sec 3.2)'),
                mechanisms=mech, interactions=inter)


def dump(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + '\n'


def frozen_checks(out, S):
    """Beyond the files this writes: the ArchBench suites, the API lock, the
    B39 records and the HBM suite's coverage record, each as frozen."""
    checks = []
    from archbench import suite as _suite
    for tier in registry.TIERS:
        f = _suite.path(tier)
        ok = f.exists() and json.loads(f.read_text())['sha256'] == _suite.build(tier)['sha256']
        checks.append(dict(name=f'archbench suite {tier}', ok=ok))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import api_lock
    lock = out / 'api_lock.json'
    bad = api_lock.compare(json.loads(lock.read_text()), api_lock.probe()) if lock.exists() else ['absent']
    checks.append(dict(name='api lock (torch 2.14.0, Triton 3.8.0, extension APIs)', ok=not bad,
                       why='; '.join(bad[:3])))
    base = json.loads((out / 'baseline.json').read_text())
    for tier, meta in base['records'].items():
        lines = [l for l in (ROOT / meta['path']).read_text().splitlines() if l.strip()]
        ok = hashlib.sha256('\n'.join(lines).encode()).hexdigest() == meta['sha256']
        checks.append(dict(name=f'B39 records {tier}', ok=ok))
    cov = out / 'hbm_coverage.json'
    if cov.exists():
        doc = json.loads(cov.read_text())
        have = {e['id']: e for e in doc['points']}
        want = [e for n in ('large', 'hbm-tier') for e in S[n]['points']]
        miss = [e['id'] for e in want if e['id'] not in have or have[e['id']]['sha256'] != e['sha256']]
        checks.append(dict(name='HBM suite coverage recorded for every large and hbm point',
                           ok=not miss, why=f'{len(miss)} missing or redefined' if miss else ''))
    else:
        checks.append(dict(name='HBM suite coverage recorded', ok=False, why='no spec/r40/hbm_coverage.json'))
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'spec' / 'r40')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--evidence', action='store_true', help='with --check: record gate R40-00.frozen')
    args = ap.parse_args(argv)
    S = subsets()
    files = {f'suites/{n}.json': dump(s) for n, s in S.items()}
    files['plan.json'] = dump(plan(S))
    files['ablation_matrix.json'] = dump(ablation_matrix())
    checks = []
    for rel, text in files.items():
        path = args.out / rel
        if args.check:
            same = path.exists() and path.read_text() == text
            checks.append(dict(name=f'spec/r40/{rel}', ok=same))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for n, s in S.items():
        print(f'  {n:<32s} {len(s["points"]):>3d} point(s)')
    if not args.check:
        return 0
    checks += frozen_checks(args.out, S)
    for c in checks:
        if not c['ok']:
            print(f"  FAIL  {c['name']}  {c.get('why', '')}")
    ok = all(c['ok'] for c in checks)
    print(f"r40-freeze-check  {'PASS' if ok else 'FAIL'} ({sum(c['ok'] for c in checks)}/{len(checks)})")
    if args.evidence:
        import evidence
        evidence.record('R40-00.frozen', 'capability-pass' if ok else 'fail', checks,
                        detail=f"{sum(c['ok'] for c in checks)}/{len(checks)} as frozen")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
