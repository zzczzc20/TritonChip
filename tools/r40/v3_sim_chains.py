#!/usr/bin/env python3
"""Run the R40-02.stress / R40-08.attention simulation checks on an immutable tree.

Pure RTL-simulation evidence for capability gates (v3 change 2): every check
runs from one source directory whose product tree must equal --expect-tree
before and after, with the staged RTL/data byte-identical to the tree's own
sources. No FPGA device is opened. Each chain runs its steps sequentially in
fresh caches; chains run in parallel. Every step keeps its raw log, exit code,
wall time and log hash in state.json. This tool never issues a gate verdict;
tools/r40/v3_r0208_gates.py audits the result.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]

# (chain, step, argv-after-python, extra env). Mirrors make r40-residency
# (stress/zero-transfer items) and make r40-attention's online/decode phases.
PLAN = {
    'fast': [
        ('child-exec', ['tools/r40/child_exec_tests.py'], {}),
        ('io-concurrency', ['tools/r40/io_concurrency_tests.py'], {}),
        ('lifetime', ['tools/r40/lifetime_tests.py'], {}),
        ('allocation-pressure', ['tools/r40/allocation_pressure_tests.py'], {'TNCA_BACKEND': 'sim-file'}),
        ('residency-controls', ['tools/r40/residency_tests.py'], {}),
        ('device-copy', ['tools/r40/device_copy_tests.py'], {}),
        ('frozen-attention-tiny', ['tools/r40/frozen_residency_tests.py', '--point',
                                   'op.attention_decode/tiny/fp16-b1-h2-l16-d16'], {}),
        ('sdpa-configuration', ['tools/r40/sdpa_configuration_tests.py'], {}),
    ],
    'batch': [
        ('batch-1000', ['tools/r40/batch_tests.py', '--calls', '1000'], {}),
    ],
    'replay': [
        ('replay-1000', ['tools/r40/replay_tests.py', '--replays', '1000'], {'TNCA_BACKEND': 'sim-file'}),
        ('recovery', ['tools/r40/recovery_tests.py'], {'TNCA_BACKEND': 'sim-file'}),
    ],
    'multicore-recovery': [
        ('recovery-multicore-dma', ['tools/r40/recovery_tests.py', '--multicore-dma'], {}),
    ],
    'online': [
        ('online-sdpa-14', ['tools/r40/online_attention_tests.py', '--out', '{out}/online-sdpa-14.json'],
         {'TNCA_PROFILE': 'core-vrf-sync-v5/hbm'}),
    ],
    'decode64': [
        ('bucket-decode-64', ['tools/r40/decode_residency_tests.py', '--steps', '64', '--bucket'],
         {'TNCA_PROFILE': 'core-vrf-sync-v5/hbm'}),
    ],
}
PROFILES = ('core-vrf-sync-v5', 'core-vrf-sync-v5/hbm', 'core-sim')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tree_digest(base):
    return sha(''.join(f'{p.relative_to(base)}:{sha(p.read_bytes())}\n'
                       for p in sorted(base.rglob('*')) if p.is_file()).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--expect-tree', required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--chain', action='append', choices=sorted(PLAN))
    ap.add_argument('--step', action='append', help='run only these steps of the selected chains')
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        ap.error('--out must be new; preserve prior attempts')
    out.mkdir(parents=True)
    sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'tools/r40')]
    from archbench.provenance import software_tree
    chains = a.chain or sorted(PLAN)
    if a.step:
        known = {n for c in chains for n, _, _ in PLAN[c]}
        if set(a.step) - known:
            ap.error(f'unknown steps {sorted(set(a.step) - known)}')
        chains = [c for c in chains if any(n in a.step for n, _, _ in PLAN[c])]
    lock = threading.Lock()
    state = dict(schema='tnca-r40-v3-sim-chains/1', status='preflight', source=str(ROOT),
                 expect_tree=a.expect_tree, python=a.python, chains={c: [] for c in chains},
                 plan={c: [dict(name=n, argv=v, env=e) for n, v, e in PLAN[c] if not a.step or n in a.step]
                       for c in chains},
                 runner_sha256=sha(Path(__file__).read_bytes()),
                 host=os.uname().nodename, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def save():
        state['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = out / 'state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(out / 'state.json')

    try:
        tree = software_tree(refresh=True)
        state['software_tree'] = tree
        assert tree == a.expect_tree, f'product tree {tree} != {a.expect_tree}'
        # Staged copies are excluded from the product digest; prove they are
        # byte-identical to the tree's own RTL/testbench/data sources.
        staged = ROOT / 'python/tnca/data'
        pairs = {'rtl': (ROOT / 'rtl', staged / 'rtl'), 'tb/device': (ROOT / 'tb/device', staged / 'tb/device'),
                 'tb/common': (ROOT / 'tb/common', staged / 'tb/common')}
        state['staged_identity'] = {k: dict(source=tree_digest(s), staged=tree_digest(d))
                                    for k, (s, d) in pairs.items()}
        assert all(v['source'] == v['staged'] for v in state['staged_identity'].values()), 'stale staged RTL'
        files = sorted({argv[0] for c in chains for _, argv, _ in PLAN[c]} |
                       {'tools/r40/frozen_residency_tests.py', 'tools/r40/bucket_decode.py',
                        'tools/r40/v3_sim_chains.py'})
        state['harness_sha256'] = {f: sha((ROOT / f).read_bytes()) for f in files}
        base = {k: v for k, v in os.environ.items() if not k.startswith('TNCA_')}
        base.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MAX_JOBS='1',
                    PYTHONFAULTHANDLER='1', TNCA_HOST_TIMEOUT='172800',
                    PYTHONPATH=os.pathsep.join(str(ROOT / d) for d in ('', 'python', 'compiler/python', 'tools/asm')))
        state['base_env'] = {k: base[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                  'MAX_JOBS', 'TNCA_HOST_TIMEOUT', 'PYTHONPATH')}
        state['status'] = 'preparing'
        save()
        prep = [a.python, '-c', 'import sys; from tnca import device\n'
                'for p in sys.argv[1:]: device.prepare(p)', *PROFILES]
        with (out / 'prepare.log').open('x') as log:
            code = subprocess.run(prep, cwd=ROOT, env=base, stdout=log, stderr=subprocess.STDOUT).returncode
        state['prepare'] = dict(argv=prep, exit_code=code)
        assert code == 0, 'device preparation failed'
        state['status'] = 'running'
        save()

        def run_chain(chain):
            cache = out / 'cache' / chain
            for name, argv, extra in PLAN[chain]:
                if a.step and name not in a.step:
                    continue
                env = dict(base, TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                           TRITON_CACHE_DIR=str(cache / 'triton'), **extra)
                cmd = [a.python, '-u', *[x.format(out=out) for x in argv]]
                rec = dict(name=name, argv=cmd, env=extra,
                           started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                with lock:
                    state['chains'][chain].append(rec)
                    save()
                t0 = time.monotonic()
                with (out / f'{name}.log').open('x') as log:
                    code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT).returncode
                with lock:
                    rec.update(exit_code=code, wall_s=round(time.monotonic() - t0, 1),
                               log=f'{name}.log', log_sha256=sha((out / f'{name}.log').read_bytes()),
                               ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    save()

        workers = [threading.Thread(target=run_chain, args=(c,)) for c in chains]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        after = software_tree(refresh=True)
        state['software_tree_after'] = after
        assert after == a.expect_tree, 'product tree changed during the run'
        failed = [r['name'] for c in chains for r in state['chains'][c] if r.get('exit_code') != 0]
        state['status'] = 'complete-review-required' if not failed else 'complete-with-failures'
        state['failed'] = failed
        save()
        return 0 if not failed else 1
    except Exception as exc:
        import traceback
        state.update(status='failed-preserved', error=traceback.format_exc())
        save()
        raise SystemExit(f'v3 sim chains failed: {exc}')


if __name__ == '__main__':
    raise SystemExit(main())
