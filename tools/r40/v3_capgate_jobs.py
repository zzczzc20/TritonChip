#!/usr/bin/env python3
"""Run the R40-03/R40-07 capability test recipes on an immutable product copy.

    python3 tools/r40/v3_capgate_jobs.py --out DIR --suite tiles pipeline extra [--jobs 6]
        [--only NAME ...] [--list]

`make r40-tiles` and `make r40-pipeline-checks` are sequences of independent
commands. This runner executes exactly those recipe lines (loops expanded,
one job per command) concurrently, each with fresh TNCA/Triton/Inductor caches
so every artifact and simulator is rebuilt from the tree being judged, plus
the extra directed v3 checks (compiler RTL pipeline/protocol benches, layout
and adverse-memory pipeline matrices). It records the product software tree
before and after (it must be the delivery tree and must not change), the hash
of every harness/testbench/ISA file outside the hashed product tree, the
`make -n` text of both targets, and each job's command, environment, times,
exit status, log hash and pass marker. It measures no performance.
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIVERY_TREE = os.environ.get('R40_EXPECT_TREE', 'files:bc5a778ebc5f4afebf6c1a9fbf3b2d8717cb02e118134ebbb73fb5a927fc8183')
TORCH = os.environ.get('TORCH_PYTHON', str(Path.home() / 'tc_env/torchenv/bin/python'))
# The GB10 tritonenv has no torch; rctx_roles_tests and the contract harness
# import it. torchenv carries the locked torch 2.14.0 + Triton 3.8.0 pair.
TRITON = TORCH
LONG = '172800'


def job(name, group, cmd, marker, env=None, timeout=6 * 3600):
    return dict(name=name, group=group, cmd=cmd, marker=marker, env=dict(env or {}),
                timeout=timeout)


def jobs(out):
    rows = []
    # make r40-tiles, in recipe order.
    for w in (16, 32, 64, 128):
        rows.append(job(f'tiles_w{w}', 'tiles', f'{TORCH} tools/r40/tiles_tests.py --width {w}',
                        r'r40-tiles-tests PASS \(12/12\)'))
    for w in (16, 32, 64, 128):
        rows.append(job(f'reduction_w{w}', 'tiles',
                        f'{TORCH} tools/r40/reduction_tests.py --width {w}',
                        r'r40-reduction-tests PASS \(20/20\)'))
        rows.append(job(f'fold_w{w}', 'tiles', f'{TORCH} tools/r40/fold_tests.py --width {w}',
                        r'r40-fold-tests PASS \(4/4\)'))
    for w in (16, 32, 64):
        rows.append(job(f'reduction_w{w}_rows2', 'tiles',
                        f'{TORCH} tools/r40/reduction_tests.py --width {w} --rows 2',
                        r'r40-reduction-tests PASS \(20/20\)'))
    rows.append(job('fold_w32_rows2_bias40', 'tiles',
                    f'{TORCH} tools/r40/fold_tests.py --width 32 --rows 2 '
                    '--index-bias 1099511627776', r'r40-fold-tests PASS \(4/4\)'))
    for c in (16, 32, 64):
        rows.append(job(f'gemm_n{c}_k16', 'tiles',
                        f'{TORCH} tools/r40/gemm_tiles_tests.py --columns {c}',
                        r'r40-gemm-tiles-tests PASS \(4/4\)'))
    for c in (16, 32):
        for dt in ('float16', 'bfloat16'):
            rows.append(job(f'gemm_n{c}_k32_{dt}', 'tiles',
                            f'{TORCH} tools/r40/gemm_tiles_tests.py --columns {c} --depth 32 '
                            f'--dtype {dt}', r'r40-gemm-tiles-tests PASS \(4/4\)',
                            env=dict(TNCA_HOST_TIMEOUT=LONG)))
    rows.append(job('gemm_n64_k32_window', 'tiles',
                    f'{TORCH} tools/r40/gemm_tiles_tests.py --columns 64 --depth 32',
                    r'r40-gemm-tiles-tests PASS \(4/4\)',
                    env=dict(TNCA_WINDOW_REUSE='1', TNCA_HOST_TIMEOUT=LONG)))
    for dt in ('float16', 'bfloat16'):
        rows.append(job(f'gemm_n16_k64_window_{dt}', 'tiles',
                        f'{TORCH} tools/r40/gemm_tiles_tests.py --columns 16 --depth 64 '
                        f'--dtype {dt}', r'r40-gemm-tiles-tests PASS \(4/4\)',
                        env=dict(TNCA_WINDOW_REUSE='1', TNCA_HOST_TIMEOUT=LONG)))
    # Host-wait-only reruns (precedent: R40-03 gemm_bf16_n32_k32_600s/longhost): a job whose
    # first attempt hit the 600 s host wait on the shared, loaded build host runs again with
    # TNCA_HOST_TIMEOUT raised; the device cycle deadline is unchanged. Both logs are kept.
    for c in (16, 32, 64):
        rows.append(job(f'gemm_n{c}_k16_longhost', 'tiles-rerun',
                        f'{TORCH} tools/r40/gemm_tiles_tests.py --columns {c}',
                        r'r40-gemm-tiles-tests PASS \(4/4\)', env=dict(TNCA_HOST_TIMEOUT=LONG)))
    rows.append(job('window_options', 'tiles', f'{TRITON} tools/r40/window_options_tests.py',
                    r'PASS'))
    rows.append(job('compile_switch', 'tiles', f'{TRITON} tools/r40/compile_switch_tests.py',
                    r'PASS'))
    rows.append(job('packed_window_options', 'tiles',
                    'python3 tools/r40/packed_window_options_tests.py', r'\nOK\n'))
    for mode in (0, 1):
        rows.append(job(f'bank_{mode}', 'tiles',
                        f'{TORCH} tools/r40/bank_tests.py --enabled {mode} '
                        f'--out {out}/bank_{mode}.ledger.json', r'r40-bank-tests PASS \(3/3\)'))
    rows.append(job('contract', 'tiles', f'{TRITON} tools/r40/contract_tests.py',
                    r'r40-contract-tests PASS \(447/447 byte-identical to B39\)',
                    env=dict(TNCA_BANK_ALLOC='0', TNCA_WINDOW_REUSE='0',
                             TNCA_PACKED_WINDOW_UNPACK='0')))
    # make r40-pipeline-checks (r40-barrier-epoch first).
    rows.append(job('barrier_epoch', 'pipeline',
                    f'mkdir -p {out}/barrier && '
                    f'iverilog -g2005 -I rtl/common -s tb_barrier_epoch -o {out}/barrier/epoch.vvp '
                    'rtl/core/tnca_barrier.v tb/core/tb_barrier_epoch.v && '
                    f'vvp {out}/barrier/epoch.vvp && '
                    'iverilog -g2005 -I rtl/common -DEPOCH_EXHAUST -s tb_barrier_epoch '
                    f'-o {out}/barrier/exhaust.vvp rtl/core/tnca_barrier.v tb/core/tb_barrier_epoch.v && '
                    f'vvp {out}/barrier/exhaust.vvp',
                    r'PASS barrier generations 0\.\.65537[\s\S]*PASS barrier exhaustion faults'))
    rows.append(job('engine_overlap_parser', 'pipeline',
                    'python3 tools/r40/engine_overlap_tests.py', r'\nOK\n'))
    rows.append(job('stage_options', 'pipeline',
                    f'{TRITON} tools/r40/pipeline_stage_options_tests.py',
                    r'r40-pipeline-stage-options PASS \((\d+)/\1\)'))
    for s in (2, 3):
        rows.append(job(f'rctx_roles_s{s}', 'pipeline',
                        f'{TRITON} -c "from tnca import device; device.prepare(\'core-vrf-sync-v5\')" && '
                        f'{TRITON} tools/r39/rctx_roles_tests.py',
                        r'r39-rctx-roles-tests \[core-vrf-sync-v5\]  PASS',
                        env=dict(TNCA_RCTX_STAGES=str(s))))
        rows.append(job(f'native_pipeline_s{s}', 'pipeline',
                        f'{TORCH} tools/r40/native_pipeline_tests.py --stages {s} '
                        f'--trace-dir {out}/trace_native_s{s} --require-overlap',
                        r'r40-native-pipeline PASS \(4/4\)[\s\S]*AUDIT measured DMA/MMA overlap=[1-9]',
                        env=dict(TNCA_RCTX_STAGES=str(s))))
    # Extra directed v3 checks (harness outside the product tree).
    rows.append(job('compiler_rtl', 'extra',
                    'make --no-print-directory sim/build/tb_core.vvp sim/build/tb_core4.vvp '
                    'sim/build/tb_core_slow.vvp sim/build/tb_core_ooo.vvp '
                    'sim/build/tb_core4_slow.vvp sim/build/tb_core4_ooo.vvp && '
                    'python3 compiler/test/run_tests.py --rtl', r'compiler-rtl\s+PASS\s+\d+ tests\n',
                    env=dict(R1_STRICT='1')))
    # The C3 two-role pipeline suite (delayed / out-of-order completion, odd
    # trip counts, predicated-off async, faulting transfer propagation, three
    # stages, two consumers) and the protocol model, one line per test.
    rows.append(job('compiler_pipeline_protocol_v', 'extra',
                    'cd compiler/test && python3 -m unittest -v test_pipeline test_protocol',
                    r'\nOK\n'))
    for w in (16, 32, 64, 128):
        rows.append(job(f'layout_w{w}', 'extra',
                        f'{TORCH} tools/r40/v3_layout_tests.py --width {w} '
                        f'--json {out}/layout_w{w}.report.json', r'r40-v3-layout PASS'))
    for mem in ('fast', 'slow', 'hbm', 'hbm-adverse'):
        for s in (1, 2, 3):
            rows.append(job(f'pipe_{mem}_s{s}', 'extra',
                            f'{TORCH} tools/r40/v3_pipeline_matrix.py --memory {mem} --stages {s} '
                            f'--trace-dir {out}/trace_{mem}_s{s} --json {out}/pipe_{mem}_s{s}.report.json',
                            r'r40-v3-pipeline-matrix PASS'))
    # Fault drain: a faulting kernel goes FAILING -> DRAINING -> FAILED and an
    # independent task afterwards succeeds (R2-11 ERR suite on the delivery tree).
    rows.append(job('r2_fault_suite', 'extra',
                    f'{TORCH} tools/r2/fault_tests.py --json {out}/r2_fault.report.json',
                    r'r2-fault-tests\s+PASS'))
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree():
    sys.path.insert(0, str(ROOT))
    from archbench.provenance import software_tree
    return software_tree(ROOT, refresh=True)


def harness_hashes():
    out = {}
    for base in ('tools', 'tb', 'isa', 'compiler/test'):
        for p in sorted((ROOT / base).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'):
                out[str(p.relative_to(ROOT))] = sha(p)
    out['Makefile'] = sha(ROOT / 'Makefile')
    return out


def staged_matches():
    """python/tnca/data (what the device cache key and build read) must equal rtl/, tb/."""
    bad = []
    data = ROOT / 'python/tnca/data'
    for src, dst in (('rtl', 'rtl'), ('tb/common', 'tb/common'), ('tb/device', 'tb/device'),
                     ('isa', 'isa')):
        for p in sorted((ROOT / src).rglob('*')):
            if p.is_file():
                q = data / dst / p.relative_to(ROOT / src)
                if src == 'isa' and p.suffix != '.json':
                    continue
                if not q.is_file() or sha(p) != sha(q):
                    bad.append(str(p.relative_to(ROOT)))
    return bad


def import_origins(env):
    """Where the interpreter actually imports the product and Triton backend from:
    torchenv has an editable tnca/tnca_cc pointing at another checkout."""
    probe = ('import json,tnca,tnca_cc,triton,torch;'
             'from importlib.metadata import entry_points as e;'
             'b=[x.value for x in e(group="triton.backends")];'
             'print(json.dumps(dict(tnca=tnca.__file__,tnca_cc=tnca_cc.__file__,'
             'triton=triton.__version__,torch=torch.__version__,backends=b)))')
    r = subprocess.run([TORCH, '-c', probe], cwd='/', env=env, capture_output=True, text=True)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    ok = all(str(Path(info[k]).resolve()).startswith(str(ROOT) + '/') for k in ('tnca', 'tnca_cc'))
    return dict(info, ok=ok)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def run(row, out, base_env):
    cache = out / 'cache' / row['name']
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(base_env)
    env.update(TNCA_CACHE=str(cache), TRITON_CACHE_DIR=str(cache / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MAX_JOBS='1')
    env.update(row['env'])
    log = out / f"{row['name']}.log"
    start, t0 = now(), time.monotonic()
    with log.open('w') as fh:
        try:
            code = subprocess.run(['bash', '-c', row['cmd']], cwd=ROOT, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=row['timeout']).returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    text = log.read_text(errors='replace')
    marker = re.search(row['marker'], text) is not None
    rec = dict(name=row['name'], group=row['group'], cmd=row['cmd'], env=row['env'],
               start=start, end=now(), seconds=round(time.monotonic() - t0, 1), exit=code,
               marker=row['marker'], marker_found=marker, ok=(code == 0 and marker),
               log=log.name, log_sha256=sha(log))
    (out / f"{row['name']}.json").write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--suite', nargs='+', default=['tiles', 'pipeline', 'extra'])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--finalize', action='store_true',
                    help='close a stopped run: rebuild results from its per-job records, '
                         'recompute the product tree, and mark the jobs that never ran')
    args = ap.parse_args()
    out = args.out.resolve()
    if args.finalize:
        mpath = next(out.glob('manifest_*.json'))
        manifest = json.loads(mpath.read_text())
        if manifest.get('status') != 'running':
            raise SystemExit(f'{mpath} is not a stopped run')
        done = []
        for name in manifest['jobs_planned']:
            rec = out / f'{name}.json'
            if rec.exists():
                r = json.loads(rec.read_text())
                if r.get('log_sha256') == sha(out / r['log']):
                    done.append(r)
        after = tree()
        manifest.update(finished=now(), software_tree_after=after, results=done,
                        jobs_not_run=[n for n in manifest['jobs_planned']
                                      if n not in {r['name'] for r in done}],
                        status='partial' if after == manifest['software_tree_before'] else 'tree-changed',
                        finalized_after_stop=True,
                        all_ok=all(r['ok'] for r in done) and after == manifest['software_tree_before'])
        mpath.write_text(json.dumps(manifest, indent=1) + '\n')
        print(f"finalized {mpath}: {len(done)} job(s) recorded, not run: {manifest['jobs_not_run']}")
        return 0
    rows = [r for r in jobs(out) if r['group'] in args.suite
            and (not args.only or r['name'] in args.only)]
    if args.list:
        for r in rows:
            print(r['name'], r['group'], r['cmd'])
        return 0
    out.mkdir(parents=True, exist_ok=True)
    before = tree()
    if before != DELIVERY_TREE:
        raise SystemExit(f'product tree {before} is not the delivery tree')
    stale = staged_matches()
    if stale:
        raise SystemExit(f'staged device sources differ from the tree: {stale[:5]}')
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'R40_'))}
    base_env['PYTHONPATH'] = os.pathsep.join(str(ROOT / d) for d in
                                             ('python', 'compiler/python', 'tools/asm'))
    origins = import_origins(dict(base_env, CUDA_VISIBLE_DEVICES=''))
    if not origins['ok']:
        raise SystemExit(f'product imports do not resolve to {ROOT}: {origins}')
    make_n = {t: subprocess.run(['make', '-n', t, f'TORCH_PYTHON={TORCH}',
                                 f'TRITON_PYTHON={TRITON}'], cwd=ROOT, capture_output=True,
                                text=True).stdout
              for t in ('r40-tiles', 'r40-pipeline-checks')}
    manifest = dict(schema='tnca-r40-v3-capgate-run/1', root=str(ROOT), out=str(out),
                    host=os.uname().nodename, started=now(), software_tree_before=before,
                    torch_python=TORCH, triton_python=TRITON, import_origins=origins,
                    make_n=make_n,
                    concurrency=args.jobs, jobs_planned=[r['name'] for r in rows],
                    harness_hashes=harness_hashes(), status='running')
    mpath = out / f"manifest_{'_'.join(args.suite)}{'_only' if args.only else ''}.json"
    mpath.write_text(json.dumps(manifest, indent=1) + '\n')
    results = []
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run, r, out, base_env): r for r in rows}
        for fut in cf.as_completed(futures):
            rec = fut.result()
            results.append(rec)
            print(f"{'PASS' if rec['ok'] else 'FAIL'} {rec['name']} exit={rec['exit']} "
                  f"{rec['seconds']}s", flush=True)
    after = tree()
    manifest.update(finished=now(), software_tree_after=after,
                    results=sorted(results, key=lambda r: [x['name'] for x in rows].index(r['name'])),
                    status='complete' if after == before else 'tree-changed',
                    all_ok=all(r['ok'] for r in results) and after == before)
    mpath.write_text(json.dumps(manifest, indent=1) + '\n')
    print(f"v3-capgate-jobs {'PASS' if manifest['all_ok'] else 'FAIL'} "
          f"({sum(r['ok'] for r in results)}/{len(results)})", flush=True)
    return 0 if manifest['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
