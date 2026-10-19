#!/usr/bin/env python3
"""R40 v3 audit of the R40-03 capability gates: tiles and contract.

    R40_CONTRACT=v3 python3 tools/r40/v3_r40_03_gates.py [--dry-run]
        [--tiles-run DIR] [--layout-run DIR] [--contract-run DIR]

Acceptance (doc/r40_tasks_and_acceptance.md R40-03, unchanged by v3):

  1. `make r40-tiles`: cross-register tails (17/31/33/65/127 ...),
     transpose/stride, mask, int64 index, alias and the existing dtypes are
     correct; insufficient resources trigger a legal retile/spill, and the
     resource ledger agrees with what the machine code needs.   -> R40-03.tiles
  3. With the switches off the old profile's artifact contract holds; new
     capabilities are not decided by U280 PC/SLR or benchmark family names.
                                                              -> R40-03.contract
  (2, the frozen speed-ups, are the perf gates R40-03.wide-blocks/gemm-tile.)

Evidence: copies of tools/r40/v3_capgate_jobs.py runs (every recipe line of
`make r40-tiles`, one job each, fresh caches, on an immutable GB10 copy of
the delivery product tree; plus tools/r40/v3_layout_tests.py for the
transposed/strided layouts). This is RTL simulation of the delivered core
(rtl/ is part of the product tree) and offline compilation; it measures no
board time or performance. The source scan for board/benchmark dispatch runs
here on the repository, which must itself be the delivery tree.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/r40')]
import v3_capgate_audit as A  # noqa: E402

EVIDENCE = ROOT / 'evidence/r40/v3'
WIDTHS = (16, 32, 64, 128)
TILES_JOBS = ([f'tiles_w{w}' for w in WIDTHS] + [f'reduction_w{w}' for w in WIDTHS]
              + [f'fold_w{w}' for w in WIDTHS] + [f'reduction_w{w}_rows2' for w in (16, 32, 64)]
              + ['fold_w32_rows2_bias40'] + [f'gemm_n{c}_k16' for c in (16, 32, 64)]
              + [f'gemm_n{c}_k32_{d}' for c in (16, 32) for d in ('float16', 'bfloat16')]
              + ['gemm_n64_k32_window', 'gemm_n16_k64_window_float16',
                 'gemm_n16_k64_window_bfloat16', 'window_options', 'compile_switch',
                 'packed_window_options', 'bank_0', 'bank_1', 'contract'])
HARNESS = ['tools/r40/tiles_tests.py', 'tools/r40/reduction_tests.py', 'tools/r40/fold_tests.py',
           'tools/r40/gemm_tiles_tests.py', 'tools/r40/bank_tests.py',
           'tools/r40/window_options_tests.py', 'tools/r40/compile_switch_tests.py',
           'tools/r40/packed_window_options_tests.py', 'tools/r40/contract_tests.py',
           'tools/r38/harness.py', 'tools/r37/upstream_cases.py', 'tools/r37/upstream.py']
MAKE_FRAGMENTS = ['tools/r40/tiles_tests.py --width $width', 'tools/r40/reduction_tests.py --width $width',
                  'tools/r40/fold_tests.py --width $width', 'reduction_tests.py --width $width --rows 2',
                  'fold_tests.py --width 32 --rows 2 --index-bias 1099511627776',
                  'gemm_tiles_tests.py --columns $columns',
                  'gemm_tiles_tests.py --columns $columns --depth 32 --dtype $dtype',
                  'TNCA_WINDOW_REUSE=1 TNCA_HOST_TIMEOUT=172800',
                  'gemm_tiles_tests.py --columns 64 --depth 32',
                  'gemm_tiles_tests.py --columns 16 --depth 64 --dtype $dtype',
                  'window_options_tests.py', 'compile_switch_tests.py',
                  'packed_window_options_tests.py', 'bank_tests.py --enabled $mode',
                  'TNCA_BANK_ALLOC=0 TNCA_WINDOW_REUSE=0 TNCA_PACKED_WINDOW_UNPACK=0',
                  'contract_tests.py']
LAYOUT_CASES = ('transpose_pointwise_f32', 'transpose_pointwise_f16', 'transpose_plus_contiguous',
                'column_strided_view', 'int64_transposed', 'transpose_sum', 'transpose_amax',
                'cumsum_scan_retile', 'linear_transposed_weight', 'mm_transposed_a')
# Lowering sources whose decisions must depend only on shape/dtype/IR/target capability.
LOWERING = ['compiler/python/tnca_cc', 'python/tnca/triton_backend', 'python/tnca/torch_device',
            'python/tnca/inductor.py', 'python/tnca/aot.py', 'python/tnca/online_attention.py',
            'python/tnca/costmodel.py', 'python/tnca/autotune.py']
FORBIDDEN = re.compile(r'u280|u50|alveo|xclbin|\bslr\d*\b|\bhbm_?pc|\bpc\d+\b|archbench|tilebench'
                       r'|^(op|model)\.[a-z_0-9]+(/|$)|/(tiny|small|medium|large|hbm)/', re.I)


def tiles(args, tree):
    checks = []
    m, jobs, base = A.load_run(args.tiles_run, tree)
    checks += base
    checks.append(A.check('every recipe line of make r40-tiles is in the run (make -n text)',
                          all(f in m['make_n'].get('r40-tiles', '') for f in MAKE_FRAGMENTS),
                          fragments=MAKE_FRAGMENTS))
    # A recipe line whose first attempt failed only for a harness/host cause (600 s host
    # wait on the loaded shared host, reference absent from the copy, the bank ledger
    # harness) is taken from its rerun on the same immutable copy (v3_capgate_audit.SUPERSEDE).
    reruns, rm = {}, None
    if args.rerun_run and Path(args.rerun_run).exists():
        rm, reruns, rbase = A.load_run(args.rerun_run, tree)
        checks += [dict(c, name='rerun run: ' + c['name']) for c in rbase]
    jobs, superseded = A.merge_reruns(jobs, reruns, TILES_JOBS, label='make r40-tiles')
    checks += superseded
    replaced = {c['name'].split(': ', 1)[1].split(' ', 1)[0] for c in superseded if c['ok']}
    checks += A.job_checks(jobs, TILES_JOBS, label='make r40-tiles')
    users = {'tools/r40/bank_tests.py': ('bank_0', 'bank_1'), 'tools/r40/contract_tests.py': ('contract',)}
    runs = {h: (rm if rm is not None and users.get(h) and all(j in replaced for j in users[h]) else m)
            for h in HARNESS}
    checks.append(A.harness_same_runs(runs, HARNESS))
    text = {n: (jobs.get(n, (None, ''))[1] or '') for n in TILES_JOBS}
    tails = {w: sorted(int(t) for t in A.lines(text[f'tiles_w{w}'],
                                                rf'^PASS block {w}, tail (\d+), masked select/load/store$'))
             for w in WIDTHS}
    checks.append(A.check('tails 17/31/33/65/127 across registers, masked select/load/store, '
                          'widths 16/32/64/128', all(tails[w] == [17, 31, 33, 65, 127] for w in WIDTHS),
                          tails=tails))
    dtypes = {w: A.lines(text[f'tiles_w{w}'], rf'^PASS block {w}, (torch\.\w+)$') for w in WIDTHS}
    checks.append(A.check('existing dtypes fp16/bf16/i8/i16/i32 (fp32 graphs above)',
                          all(sorted(dtypes[w]) == ['torch.bfloat16', 'torch.float16', 'torch.int16',
                                                    'torch.int32', 'torch.int8'] for w in WIDTHS),
                          dtypes=dtypes))
    checks.append(A.check('int64 values beyond 32 bits at every width; int64 argmax indices + 2^40',
                          all(A.lines(text[f'tiles_w{w}'], rf'^PASS block {w}, signed int64 values beyond 32 bits$')
                              for w in WIDTHS)
                          and 'r40-fold-tests PASS (4/4)' in text['fold_w32_rows2_bias40']))
    checks.append(A.check('strided offset alias with in-place stores at every width',
                          all(A.lines(text[f'tiles_w{w}'], rf'^PASS block {w}, strided offset alias and in-place stores$')
                              for w in WIDTHS)))
    red = {n: len(A.lines(text[n], r'^PASS (sum|amax|rmsnorm|softmax|layernorm), width cap'))
           for n in text if n.startswith('reduction_')}
    checks.append(A.check('reductions sum/amax/RMSNorm/softmax/LayerNorm 20/20 per width, two-row 16/32/64',
                          all(v == 20 for v in red.values()) and len(red) == 7, counts=red))
    gemm_tiles = sorted({t for n in text if n.startswith('gemm_')
                         for t in A.lines(text[n], r'^PASS (?:mm|addmm), logical (16x\d+x\d+), shape')})
    checks.append(A.check('logical GEMM tiles over MMA fragments: 16x16/32/64x16, 16x16/32x32 '
                          '(fp16, bf16), 16x64x32, 16x16x64 (fp16, bf16), tails 17x65x33 and 33x31x65',
                          gemm_tiles == ['16x16x16', '16x16x32', '16x16x64', '16x32x16', '16x32x32',
                                         '16x64x16', '16x64x32'], tiles=gemm_tiles))
    # Resource pressure: spill and retile, with ledgers checked against the machine.
    spills = []
    for line in A.lines(text['tiles_w128'], r'^AUDIT (\{.*\})$'):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('dtype') == 'int64':
            spills.append(dict(plan=d.get('spill'), scratch=d['resources']['scratch_bytes']))
    checks.append(A.check('register pressure triggers a private-scratch vector spill (128-wide int64) '
                          'and the artifact reserves at least the planned scratch',
                          spills and all(s['plan'] and s['plan'].get('scratch_bytes', 0) > 0
                                         and s['scratch'] >= s['plan']['scratch_bytes'] for s in spills),
                          spills=spills))
    retiles = {n: A.lines(text[n], r'^AUDIT logical K=64, physical MMA K=32, retiles=\[\[64, 32\]\]')
               for n in ('gemm_n16_k64_window_float16', 'gemm_n16_k64_window_bfloat16')}
    checks.append(A.check('full VRF triggers a legal K64 -> 2 x K32 MMA retile (fp16 and bf16), '
                          'actual machine K encoding asserted by the test',
                          all(len(v) >= 4 for v in retiles.values()),
                          retile_audits={k: len(v) for k, v in retiles.items()}))
    windows = {n: len(A.lines(text[n], r'^AUDIT awaited window layout')) for n in
               ('gemm_n64_k32_window', 'gemm_n16_k64_window_float16', 'gemm_n16_k64_window_bfloat16')}
    checks.append(A.check('window reuse ledger equals the window definitions decoded from the executed '
                          'machine code, and every role reserves at least those registers',
                          all(v >= 4 for v in windows.values()), audits=windows))
    banks = {n: 'r40-bank-tests PASS (3/3)' in text[n] for n in ('bank_0', 'bank_1')}
    checks.append(A.check('bank allocation off/on: physical ledger reproduced from the verified '
                          'assembly of the executed artifact', all(banks.values()), banks=banks))
    # Transposed / strided layouts (v3 directed addition to the make target).
    lm, ljobs, lbase = A.load_run(args.layout_run, tree)
    checks += [dict(c, name='layout run: ' + c['name']) for c in lbase]
    checks.append(A.harness_same(lm, ['tools/r40/v3_layout_tests.py']))
    names = [f'layout_w{w}' for w in WIDTHS]
    checks += A.job_checks(ljobs, names, label='transpose/stride layouts')
    cases = {}
    for w in WIDTHS:
        # The per-case report (layout_w*.report.json); runs whose runner record overwrote a
        # report named like the job keep the per-case PASS/FAIL lines in the preserved log.
        doc = {}
        for name in (f'layout_w{w}.report.json', f'layout_w{w}.json'):
            path = Path(args.layout_run) / name
            if path.exists():
                doc = json.loads(path.read_text())
                if 'cases' in doc:
                    break
        if 'cases' in doc:
            cases[w] = {c['name']: (c.get('ok'), c.get('error')) for c in doc['cases']}
        else:
            log = ljobs.get(f'layout_w{w}', (None, ''))[1] or ''
            cases[w] = {}
            for verdict, name, rest in re.findall(rf'^(PASS|FAIL) (\w+), width {w}(.*)$', log, re.M):
                cases[w][name] = (verdict == 'PASS', None if verdict == 'PASS' else rest.lstrip(': ')[:300])
    groups = [
        ('transposed pointwise reads (fp32, fp16), transposed + contiguous operands, a column-strided '
         '2-D view', ('transpose_pointwise_f32', 'transpose_pointwise_f16', 'transpose_plus_contiguous',
                      'column_strided_view')),
        ('linear with its [n, k] weight (B transposed) and mm with a transposed A, wide logical tiles',
         ('linear_transposed_weight', 'mm_transposed_a')),
        ('cumsum scan kept at 16 lanes above a 16 cap (the bc5a narrow-block retile)', ('cumsum_scan_retile',)),
        ('reductions (sum, amax) over a transposed / non-innermost axis', ('transpose_sum', 'transpose_amax')),
        ('int64 beyond 32 bits read transposed (2-D tile under register/code pressure)', ('int64_transposed',)),
    ]
    for title, names in groups:
        got = {w: {n: cases[w].get(n, (None, 'not run')) for n in names} for w in WIDTHS}
        checks.append(A.check(f'transpose/stride layouts at widths 16/32/64/128: {title}',
                              all(v[0] is True for w in WIDTHS for v in got[w].values()),
                              results={w: {n: ('PASS' if v[0] else f'FAIL {v[1]}') for n, v in got[w].items()}
                                       for w in WIDTHS}))
    return checks, m


def contract(args, tree):
    checks = []
    m, jobs, base = A.load_run(args.contract_run, tree)
    checks += base

    rm = None
    if args.contract_rerun and Path(args.contract_rerun).exists():
        rm, reruns, rbase = A.load_run(args.contract_rerun, tree)
        checks += [dict(c, name='rerun run: ' + c['name']) for c in rbase]
        jobs, superseded = A.merge_reruns(jobs, reruns, ['contract'], label='contract')
        checks += superseded
    checks += A.job_checks(jobs, ['contract'], label='disabled-option legacy artifact contract')
    used = rm if (rm is not None and jobs.get('contract', (None,))[0] is not None
                  and jobs['contract'][0].get('log_sha256') in
                  {r['log_sha256'] for r in rm.get('results', [])}) else m
    checks.append(A.harness_same(used, ['tools/r40/contract_tests.py', 'tools/r38/harness.py',
                                        'tools/r37/upstream_cases.py', 'tools/r37/upstream.py']))
    rec, text = jobs.get('contract', (None, None))
    checks.append(A.check('all 447 frozen B39 artifacts (149 cases x O0/O1/O2) byte-identical, no DIFF',
                          bool(text) and text.count('r40-contract-tests PASS (447/447 byte-identical to B39)') == 1
                          and not re.search(r'^DIFF ', text, re.M)))
    checks.append(A.check('new switches off in the contract job (bank allocation, window reuse, packed unpack)',
                          rec is not None and rec['env'] == dict(TNCA_BANK_ALLOC='0', TNCA_WINDOW_REUSE='0',
                                                                 TNCA_PACKED_WINDOW_UNPACK='0')))
    ref_path = ROOT / 'evidence/r40/R40-03/contract_reference.json'
    ref = json.loads(ref_path.read_text())
    baseline = json.loads((ROOT / 'spec/r40/baseline.json').read_text())
    checks.append(A.check('reference is the independently rebuilt frozen B39 software commit, '
                          'complete inventory, no performance claim',
                          ref['schema'] == 'tnca-r40-artifact-contract-reference/1'
                          and ref['software_commit'] == baseline['software']['commit']
                          and ref['measured_performance'] is False and len(ref['artifacts']) == 447,
                          reference=str(ref_path.relative_to(ROOT)), sha256=A.sha_file(ref_path),
                          commit=ref['software_commit']))
    # Default caps unchanged: wide blocks / tiles / rows stay opt-in.
    src = (ROOT / 'python/tnca/inductor.py').read_text()
    tree_ast = ast.parse(src)
    stage1 = [n.value.value for n in ast.walk(tree_ast) if isinstance(n, ast.Assign)
              and any(getattr(t, 'id', None) == 'STAGE1_MAX_BLOCK' for t in n.targets)
              and isinstance(n.value, ast.Constant)]
    checks.append(A.check('default caps unchanged: block 16, GEMM tile 16,16,16, reduction rows 1',
                          stage1 == [16] and "os.environ.get('TNCA_GEMM_TILE', '16,16,16')" in src
                          and "os.environ.get('TNCA_REDUCTION_ROWS', '1')" in src))
    # No board placement / benchmark identity in lowering decisions.
    files = []
    for entry in LOWERING:
        p = ROOT / entry
        files += sorted(p.rglob('*.py')) if p.is_dir() else [p]
    hits, hashes = [], {}
    for f in files:
        source = f.read_text()
        hashes[str(f.relative_to(ROOT))] = A.sha_bytes(source.encode())
        module = ast.parse(source)
        docs = {id(n.body[0].value) for n in ast.walk(module)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(module):
            where = f'{f.relative_to(ROOT)}:{getattr(n, "lineno", "?")}'
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs \
                    and FORBIDDEN.search(n.value):
                hits.append([where, n.value[:80]])
            elif isinstance(n, ast.Name) and FORBIDDEN.search(n.id):
                hits.append([where, n.id])
            elif isinstance(n, ast.Attribute) and FORBIDDEN.search(n.attr):
                hits.append([where, n.attr])
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in n.names] + [getattr(n, 'module', None) or '']
                if any(x.split('.')[0] == 'archbench' for x in mods if x):
                    hits.append([where, 'imports archbench'])
    checks.append(A.check('lowering code (strings, names, imports; comments/docstrings excluded) never '
                          'names a U280/U50 PC or SLR, an xclbin, ArchBench/Tilebench or a frozen '
                          'point/family id: shared tiles are chosen by shape/dtype/IR/target capability',
                          not hits, files=len(files), hits=hits, file_sha256=hashes,
                          pattern=FORBIDDEN.pattern))
    return checks, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true', help='print the records; write nothing')
    p.add_argument('--tiles-run', type=Path, default=EVIDENCE / 'R40-03.tiles/run')
    p.add_argument('--layout-run', type=Path, default=EVIDENCE / 'R40-03.tiles/layout_run')
    p.add_argument('--rerun-run', type=Path, default=EVIDENCE / 'R40-03.tiles/rerun_run')
    p.add_argument('--contract-run', type=Path, default=EVIDENCE / 'R40-03.contract/run')
    p.add_argument('--contract-rerun', type=Path, default=EVIDENCE / 'R40-03.contract/rerun_run')
    args = p.parse_args()
    identity, lock = A.delivery()
    tree = identity['software_tree']
    here = A.local_tree()
    common = [A.check('auditor runs on the delivery product tree', here == tree, local=here, delivery=tree)]
    records = {}
    for gate, fn, what in (('R40-03.tiles', tiles, 'tails/transpose/stride/mask/int64/alias/dtypes, '
                            'register-block reductions, logical GEMM tiles, bank allocation, '
                            'legal spill and K retile with machine-checked ledgers'),
                           ('R40-03.contract', contract, 'disabled-option B39 artifact contract and '
                            'capability-only dispatch')):
        try:
            checks, manifest = fn(args, tree)
            checks = common + checks
            status = 'capability-pass' if all(c['ok'] for c in checks) else 'fail'
            detail = (f'{what}; RTL simulation of the delivered core and offline compilation on an '
                      f'immutable copy of {tree}; no board time, no performance claim')
            extra = dict(software_tree=tree, image='not-applicable: RTL simulation / compilation',
                         clock='not-applicable', measured_performance=False,
                         run_host=manifest.get('host'), run_root=manifest.get('root'),
                         run_started=manifest.get('started'), run_finished=manifest.get('finished'))
        except A.Missing as e:
            checks, status, detail, extra = common + [A.check(str(e), False)], 'not-measured', str(e), {}
        records[gate] = (status, checks, detail, extra)
    for gate, (status, checks, detail, extra) in records.items():
        bad = [c['name'] for c in checks if not c['ok']]
        print(f'{gate}: {status}  ({len(checks) - len(bad)}/{len(checks)} checks)'
              + (f'  failing: {bad}' if bad else ''))
        if args.dry_run:
            print(json.dumps(dict(status=status, detail=detail, checks=checks, **extra), indent=1,
                             default=str)[:20000])
        else:
            if os.environ.get('R40_CONTRACT') != 'v3':
                raise SystemExit('set R40_CONTRACT=v3 to record v3 gate evidence')
            import evidence
            evidence.record(gate, status, checks, detail=detail, **extra)
    return 0


if __name__ == '__main__':
    sys.exit(main())
