#!/usr/bin/env python3
"""R40 v3 audit of the R40-08 capability gates: attention and decode.

R40-08.attention (acceptance 1): the original attention/llama/GPT-2 graphs
pass on the delivery board; the new long-sequence, odd-length, all-masked-row,
GQA/MQA and 64-step cache-update cases match their references with zero CPU
numerical fallback. R40-08.decode (acceptance 3): full frozen generation in one
fixed KV bucket on the delivery board, no steady weight/KV retransfer, every
layer's launch completed and timed, one decode compilation, guard boundary
reported. R40-08.prefill (performance/traffic) is out of scope.

Identity rules are those of tools/r40/v3_r40_02_gates.py: the delivery
identity comes from contract_v3.delivery(plan); evidence on another product
tree is not-measured.

    R40_CONTRACT=v3 python3 tools/r40/v3_r40_08_gates.py [--dry-run]
        [--native DIR_OR_JSONL ...] [--extra-board JSONL ...] [--sim RUN_DIR ...]
        [--board QUEUE_DIR] [--board-snapshot SOURCE_ROOT] [--model-report report.json[.gz]]
        [--bucket-audit bucket_model_capability.json] [--clock-evidence qualified_clock.json]
"""
import argparse
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'tools/r40')]
from v3_r40_02_gates import (BACKEND, CLOCK_HZ, EVIDENCE, IMAGE, TREE, Missing, backend_args_ok,
                             contract_v3, default_sims, emit, file_sha, functional_queue, has, judge,
                             model_report, need, require, same_identity, sha, sim_identity, sim_run)

FAMILIES = {'attention_decode', 'attention_prefill', 'llama_decode', 'llama_prefill', 'gpt2_block'}
ONLINE_CASES = {
    # case: (heads, key heads, query length, key length, head dim, mask, causal)
    'decode_tail': (1, 1, 1, 17, 16, None, False),
    'prefill_causal_tail': (1, 1, 3, 33, 16, None, True),
    'bool_all_mask_row': (2, 2, 3, 19, 16, 'bool', False),
    'float_all_mask_row': (1, 1, 3, 17, 16, 'float', False),
    'strided_batch_heads': (2, 2, 3, 17, 16, None, False),
    'bf16_tail': (1, 1, 1, 17, 16, None, False),
    'gqa_tail': (4, 2, 3, 19, 16, None, False),
    'mqa_causal_tail': (4, 1, 3, 17, 16, None, True),
    'long_prefill_tail': (1, 1, 17, 129, 16, None, False),
    'decode_d32_tail': (1, 1, 1, 17, 32, None, False),
    'decode_d64_tail': (1, 1, 1, 17, 64, None, False),
    'bool_all_mask_d32': (2, 2, 3, 19, 32, 'bool', False),
    'gqa_d64_tail': (4, 2, 3, 19, 64, None, False),
    'mqa_d32_causal_tail': (4, 1, 3, 17, 32, None, True),
}
ATTENTION_SIM_STEPS = ('online-sdpa-14', 'sdpa-configuration')


def jsonl(path):
    raw = need(path).read_bytes()
    if Path(path).suffix == '.gz':
        raw = gzip.decompress(raw)
    return sha(raw), [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def board_record(r, where):
    p = r['platform']
    same_identity(p['software_tree'], f'{where}: {r["point_id"]}')
    require(p['image_sha256'] == IMAGE and p['clock_hz'] == CLOCK_HZ and p['backend'] == BACKEND and
            backend_args_ok(p['backend_args']) and p['knobs'].get('TNCA_TORCH_DEVICE') == 'native' and
            p['knobs'].get('TNCA_ACTIVE_CORES') == '3' and p['inductor'].get('tensor_descriptors') is True and
            r['evidence_level'] == 'measured-board',
            f'{where}: {r["point_id"]} not measured natively on the delivery board image')
    c = r['correctness']
    require(r['status'] == 'pass' and r['fallbacks'] == [] and not r['problems'] and
            0 <= c['max_rel_err'] <= c['tol'], f'{where}: {r["point_id"]} failed or fell back')
    require(r['host']['upload_bytes'] == 0 and r['host']['download_bytes'] == 0,
            f'{where}: {r["point_id"]} moved tensor bytes inside the measured call')


def native_files(paths):
    files = []
    for path in paths:
        path = need(path)
        files += sorted(path.glob('u280_*.jsonl*')) if path.is_dir() else [path]
    return files


def original_graphs(native, extra=()):
    """The 15 original (existing-suite) attention/llama/GPT-2 graphs on the delivery board."""
    suite = json.loads((ROOT / 'spec/r40/suites/existing.json').read_text())['points']
    expected = {p['id']: p['sha256'] for p in suite if p['family'] in FAMILIES}
    require(len(expected) == 15, 'frozen original attention inventory changed')
    records, sources = {}, {}
    for path in native_files(native):
        digest, rows = jsonl(path)
        sources[path.name] = digest
        for r in rows:
            require(r['point_id'] not in records, f'duplicate board record {r["point_id"]}')
            records[r['point_id']] = r
    for pid, digest in expected.items():
        if pid not in records:
            raise Missing(f'original graph {pid} has no board record')
        require(records[pid]['point_sha256'] == digest, f'{pid}: frozen definition changed')
        board_record(records[pid], 'native77')
    checks = [dict(name='original attention/llama/GPT-2 graphs (15/15, existing suite) pass on the delivery '
                        'board: native device, descriptors, zero fallback, zero host tensor bytes', ok=True,
                   points=sorted(expected), sources=sources,
                   max_rel_err=max(records[p]['correctness']['max_rel_err'] for p in expected))]
    for path in extra:     # larger frozen tiers whose attention-family points must also pass
        digest, rows = jsonl(path)
        mine = [r for r in rows if r['family'] in FAMILIES]
        require(mine, f'{path}: no attention-family records')
        for r in mine:
            board_record(r, Path(path).name)
        checks.append(dict(name=f'larger frozen attention-family points in {Path(path).name} pass on the '
                                 'delivery board', ok=True, points=sorted(r['point_id'] for r in mine),
                           raw_sha256=digest))
    return checks


def online_cases(logs, sim_dirs, used):
    text = logs['online-sdpa-14']
    has(text, 'r40-online-attention-tests PASS (14/14)', 'online-sdpa-14')
    run = next(Path(d) for d in sim_dirs if Path(d).name == used['online-sdpa-14']['run'])
    doc = json.loads(need(run / 'online-sdpa-14.json').read_text())
    require(doc['schema'] == 'tnca-online-sdpa-checkpoint/1' and doc['full_inventory'] is True and
            doc['measured_performance'] is False, 'online SDPA report incomplete')
    results = {r['case']: r for r in doc['results']}
    require(set(results) == set(ONLINE_CASES) and len(doc['results']) == 14, 'online SDPA cases missing')
    printed = [json.loads(m) for m in re.findall(r'^PASS (\{.*\})$', text, re.M)]
    require(printed == doc['results'], 'online SDPA report differs from its log')
    for case, (heads, _kv, length, _seq, width, mask, _causal) in ONLINE_CASES.items():
        r = results[case]
        require(r['status'] == 'pass' and r['head_dim'] == width and r['steady_tensor_transfers'] == 0,
                f'{case}: failed or transferred tensors')
        art = r['online_artifacts']
        require(len(art) == 1 and art[0]['descriptor_loads'] >= 2 * (width // 16),
                f'{case}: fused online descriptor kernel not executed')
        if mask is None:
            batch = 2 if case == 'strided_batch_heads' else 1
            require(r['kernels'] == 1 and r['tensor_allocations'] == [batch * heads * length * width * 2],
                    f'{case}: not one kernel with an output-only allocation')
    require(all(v[3] % 2 for v in ONLINE_CASES.values()), 'even key length in the odd-length inventory')
    require(re.search(r'^OK$', logs['sdpa-configuration'], re.M), 'SDPA mode switching unittest not OK')
    covered = dict(long_sequence=['long_prefill_tail'], odd_lengths=sorted(ONLINE_CASES),
                   all_masked_rows=['bool_all_mask_row', 'float_all_mask_row', 'bool_all_mask_d32'],
                   gqa=['gqa_tail', 'gqa_d64_tail'], mqa=['mqa_causal_tail', 'mqa_d32_causal_tail'],
                   causal=['prefill_causal_tail', 'mqa_causal_tail', 'mqa_d32_causal_tail'],
                   head_dims=[16, 32, 64], dtypes=['float16', 'bfloat16'])
    return [dict(name='new attention cases on delivery-tree RTL (14/14, standard fused SDPA vs FP32 reference '
                      'rtol/atol 2e-3): long sequence, odd lengths, all-masked rows, GQA/MQA, causal, '
                      'D16/32/64, FP16/BF16; actual descriptor loads; zero steady tensor transfers; unmasked '
                      'cases one kernel with output-only allocation', ok=True, coverage=covered,
                 step=used['online-sdpa-14']),
            dict(name='decomposed/fused SDPA configuration switching and guards', ok=True,
                 step=used['sdpa-configuration'])]


def decode64_sim(logs, used):
    text = logs['bucket-decode-64']
    rows = re.findall(r'^PASS position=(\d+) KV=(\d+)/80: (\d+) kernels, tensor transfers=0, '
                      r'max_rel_err=(\S+)$', text, re.M)
    require([int(p) for p, *_ in rows] == list(range(16, 80)) and
            all(int(kv) == int(p) + 1 and int(k) >= 3 for p, kv, k, _ in rows),
            'bucket decode did not complete 64 resident positions')
    has(text, 'AUDIT fixed bucket: one compiled graph, resident tensor position', 'bucket-decode-64')
    has(text, 'r40-decode-residency-tests PASS (64/64), KV 16->80, fixed cache addresses, weights unchanged',
        'bucket-decode-64')
    return dict(name='64-step resident KV cache update on delivery-tree RTL (frozen tiny LLaMA, one bucket): '
                     'every position matches the reference, zero tensor transfers, final KV state verified',
                ok=True, max_rel_err=max(float(e) for *_, e in rows), step=used['bucket-decode-64'])


def attention_checks(native, extra, sims, decode_inputs):
    checks = original_graphs(native, extra)
    states, used, logs = sim_run(sims, ATTENTION_SIM_STEPS)
    checks += online_cases(logs, sims, used)
    checks.append(sim_identity(states))
    # 64-step cache update: the frozen tiny-LLaMA bucket on delivery-tree RTL
    # (make r40-attention's decode phase) and/or the full frozen model's 64
    # decode calls in one bucket on the delivery board. At least one must be
    # complete; any present one must pass.
    cache, absent = [], []
    try:
        _s, dused, dlogs = sim_run(sims, ('bucket-decode-64',))
        cache.append(decode64_sim(dlogs, dused))
    except Missing as exc:
        absent.append(f'RTL 64-step bucket decode: {exc}')
    try:
        full = decode_checks(*decode_inputs)
        cache.append(dict(name='64-step cache update in the full frozen model on the delivery board (fixed '
                               'bucket, tokens equal the independent reference, every step within tolerance)',
                          ok=True, detail=full[0]))
    except Missing as exc:
        absent.append(f'board full-model 64-step generation: {exc}')
    if not cache:
        raise Missing('no complete 64-step cache-update evidence: ' + '; '.join(absent))
    cache[-1]['other_64_step_evidence_not_included'] = absent
    return checks + cache


def clock_evidence(path, expected_sha256):
    raw = need(path).read_bytes()
    require(sha(raw) == expected_sha256, 'operating clock evidence differs from the record the model bound')
    data = json.loads(raw)
    require(data['schema'] == 'tnca-qualified-operating-clock/1' and data['image_sha256'] == IMAGE and
            data['clock_hz'] == CLOCK_HZ and data['route_legal'] is True and data['cdc_review'] == 'passed' and
            data['clock_source'] == 'measured-board-counter' and
            data['physical_core_slrs'] == {'0': 0, '1': 1, '2': 2}, 'operating clock evidence incomplete')


def decode_checks(report_path, queue, snapshot, bucket_audit, clock):
    import residency_acceptance
    import bucket_model_acceptance
    state, commands = functional_queue(queue, snapshot)
    done = {c['name']: c for c in state['completed']}
    for stage in ('frozen-original-model', 'full-bucket-model-capability-audit'):
        if stage not in done:
            raise Missing(f'board stage {stage} not complete')
        require(done[stage]['exit_code'] == 0, f'{stage} exited {done[stage]["exit_code"]}')
    report, digest = model_report(report_path)
    ident = report['identity']
    same_identity(ident['software_tree'], 'model report')
    bucket_model_acceptance.bucket_policy(report)
    residency_acceptance.model(report, TREE, IMAGE, dict(backend_args=state['backend_args'], clock_hz=CLOCK_HZ))
    require(ident['clock_hz'] == CLOCK_HZ and ident['clock_policy'] == f'{CLOCK_HZ}Hz-undivided-conditional',
            'model clock identity changed')
    clock_evidence(clock, ident['operating_clock_evidence_sha256'])
    audit_raw = need(bucket_audit).read_bytes()
    audit = json.loads(audit_raw)
    require(audit['status'] == 'capability-pass' and audit['software_tree'] == TREE and
            audit['image_sha256'] == IMAGE and audit['input_sha256'] == digest['raw_sha256'] and
            audit['auditor_sha256'] == file_sha(ROOT / 'tools/r40/bucket_model_acceptance.py'),
            'on-board audit (image hash, qualified clock) did not pass this exact report')
    argv = commands['full-bucket-model-capability-audit']
    require(argv[argv.index('--xclbin') + 1] == state['backend_args']['xclbin'] and
            argv[argv.index('--image-sha256') + 1] == IMAGE, 'on-board audit used another image')
    sequences = {tuple(e['artifact'] for e in s['launches']) for s in report['steps'][1:]}
    require(len(sequences) == 1, 'decode steps executed different graphs')
    for s in report['steps']:
        require(all(e['device_timing'] and e['device_timing']['cycles'] > 0 for e in s['launches']) and
                all(e['device_timing'] for e in s['events']), 'a launch was not device-timed to completion')
    return [dict(name='full frozen generation (prefill512 + 64 decode) in one fixed KV bucket on the delivery '
                      'board: tokens equal the independent reference, every step within tolerance', ok=True,
                 tokens=len(report['tokens']), report=digest),
            dict(name='no steady weight/KV retransfer: weights uploaded once, fixed cache addresses, zero '
                      'tensor/staging transfers after initialization', ok=True,
                 initial_weight_upload_bytes=report['initial_weight_upload_bytes'],
                 cache_addresses=report['cache_addresses']),
            dict(name='timing covers every layer and dependency: identical complete decode graph each step, '
                      'every launch device-timed and drained before host visibility', ok=True,
                 decode_launches_per_step=len(next(iter(sequences)))),
            dict(name='exactly one decode compilation across all positions in the bucket; guard boundary '
                      'reported (capacity, next position; cross-bucket transition not measured)', ok=True,
                 decode_compilations=report['decode_compilations'], bucket_guard=report['bucket_guard']),
            dict(name='delivery product tree, qualified image and measured 90 MHz clock (the on-board audit '
                      'hashed the loaded xclbin)', ok=True, on_board_audit_sha256=sha(audit_raw))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true', help='print the records; write nothing')
    p.add_argument('--native', type=Path, nargs='+', default=[EVIDENCE / 'R40-08.attention/board_native'],
                   help='native77 u280_{tiny,small,medium}.jsonl[.gz] files or directories holding them')
    p.add_argument('--extra-board', type=Path, nargs='*',
                   default=[EVIDENCE / 'R40-08.attention/board_native/hbm14.jsonl.gz'],
                   help='further board jsonl[.gz] (HBM14, large24) whose attention-family points must pass')
    p.add_argument('--sim', type=Path, nargs='+', default=None, help='v3_sim_chains runs, oldest first')
    p.add_argument('--board', type=Path, default=EVIDENCE / 'R40-02.zero-transfer/board_queue')
    p.add_argument('--board-snapshot', type=Path, default=None)
    model = EVIDENCE / 'R40-08.decode/board_model'
    p.add_argument('--model-report', type=Path, default=model / 'model_report.json.gz')
    p.add_argument('--bucket-audit', type=Path, default=model / 'bucket_model_capability.json')
    p.add_argument('--clock-evidence', type=Path, default=model / 'qualified_clock.json')
    a = p.parse_args()
    contract_v3.verify()
    sims = a.sim or default_sims()
    decode_inputs = (a.model_report, a.board, a.board_snapshot or a.board / 'snapshot_harness',
                     a.bucket_audit, a.clock_evidence)
    results = {}
    status, checks, why = judge(attention_checks, a.native, a.extra_board, sims, decode_inputs)
    results['R40-08.attention'] = (status, checks, why or
        'acceptance 1: original graphs on the delivery board; new cases and 64-step cache on delivery-tree RTL '
        '(plus the full-model board generation when present); no performance or traffic claim')
    status, checks, why = judge(decode_checks, *decode_inputs)
    results['R40-08.decode'] = (status, checks, why or
        'acceptance 3: fixed-bucket full generation on the delivery board; functional, not performance')
    return emit(results, a.dry_run, __file__)


if __name__ == '__main__':
    raise SystemExit(main())
