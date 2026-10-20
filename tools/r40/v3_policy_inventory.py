#!/usr/bin/env python3
"""R40-07.policy: compile-only inventory of the pipeline decision on tiny+small+medium.

    v3_policy_inventory.py --source-root SRC --software-tree files:<sha> --out DIR
        [--jobs 4] [--configs off pipe2 gated2] [--only POINT ...]

For every point of spec/r40/suites/existing.json (the complete tiny, small
and medium suites, 77 points; frozen definition hashes checked) and each
configuration, one fresh process with its own Triton/Inductor/TNCA caches
compiles the ArchBench point with torch.compile(fullgraph=True) on `tnca`
tensors over a host-memory stand-in session (TNCA_LAUNCH=skip; every device
session is forbidden; the stand-in follows tools/r40/v3_compile_preflight.py)
and records, per launched kernel: the machine-code (text segment) hash, roles
per program, and the pipeline decision the compiler wrote in the provenance
(rctx_roles_ring, or rctx_roles_refused / rctx_roles_fallback with the
reason), with the launch grids (parallelism).

Configurations (B40-S compile knobs otherwise; TNCA_BACKEND=fpga-u280-multicore
as the board campaign compiles):
  off     TNCA_PIPELINE=1                       (B40-S)
  pipe2   TNCA_PIPELINE=2                       (v3 R40-07 candidate; roles default 0)
  gated2  TNCA_PIPELINE=2, TNCA_RCTX_ROLES=1    (R39-12 automatic policy: ring_estimate)
  gated3  TNCA_PIPELINE=3, TNCA_RCTX_ROLES=1
  forced1 / forced2  TNCA_RCTX_ROLES=2 (legal split forced) at 1 / 2 stages

A kernel whose executable payload (every code/data segment, entry points, ABI,
execution map and resources) is identical between a configuration and `off`
cannot change device time on the same image; only the changed ones need a
board A/B. Not measured here: numerics, execution, time.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve()
BASE = dict(TNCA_MAX_BLOCK='16', TNCA_GEMM_TILE='16,16,16', TNCA_REDUCTION_ROWS='1',
            TNCA_BANK_ALLOC='0', TNCA_WINDOW_REUSE='0', TNCA_PACKED_WINDOW_UNPACK='0',
            TNCA_SDPA='decomposed', TNCA_TUNE='0', TNCA_TRANSFER_POLICY='per-kernel',
            TNCA_ACTIVE_CORES='1', TNCA_RTL_MEM_PORTS='1', TNCA_RTL_OUTSTANDING='1',
            TNCA_SUBMIT='single')
CONFIGS = {'off': dict(TNCA_PIPELINE='1'),
           'pipe2': dict(TNCA_PIPELINE='2'),
           'gated2': dict(TNCA_PIPELINE='2', TNCA_RCTX_ROLES='1'),
           'gated3': dict(TNCA_PIPELINE='3', TNCA_RCTX_ROLES='1'),
           # Roles forced legal (the mechanism tests' setting), for sizing a
           # stage A/B that holds role specialization constant.
           'forced1': dict(TNCA_PIPELINE='1', TNCA_RCTX_ROLES='2'),
           'forced2': dict(TNCA_PIPELINE='2', TNCA_RCTX_ROLES='2')}
HARNESS = dict(CUDA_VISIBLE_DEVICES='', TNCA_LAUNCH='skip', TNCA_PROFILE='core-vrf-sync-v5',
               TNCA_OPT_LEVEL='2', TNCA_TRITON='1', TNCA_BACKEND='fpga-u280-multicore',
               TNCA_TORCH_DEVICE='native', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


class HostSession:
    """Host bytes in place of a device Session (allocation, upload, download);
    anything else the native path asks raises (as v3_compile_preflight.py)."""

    class Heap:
        def __init__(self, base, size):
            self.base, self.limit, self.next, self.free_list = base, base + size, base, []

    class Buffer:
        def __init__(self, addr, size, name, sid, epoch):
            self.addr, self.size, self.name = addr, size, name
            self.session_id, self.epoch, self.alive, self.reserved = sid, epoch, True, False

    def __init__(self, profile, size=64 << 30):
        self.profile, self.id, self.epoch = profile, 'host-stub', 0
        self._lock = threading.RLock()
        self._heap = self.Heap(0x1000_0000, size)
        self.unfinished, self.latest_event, self.closed, self.poisoned = [], None, False, False
        self.store = {}

    def allocate(self, nbytes, *, name=None):
        with self._lock:
            addr = (self._heap.next + 63) // 64 * 64
            if addr + nbytes > self._heap.limit:
                raise MemoryError('host stub heap exhausted')
            self._heap.next = addr + max(int(nbytes), 1)
            return self.Buffer(addr, int(nbytes), name, self.id, self.epoch)

    def retire(self, buf):
        buf.alive = False
        self.store.pop(buf.addr, None)

    def upload(self, buf, data, offset=0):
        b = self.store.setdefault(buf.addr, bytearray(buf.size))
        b[offset:offset + len(data)] = data

    def download(self, buf, nbytes=None, offset=0):
        n = buf.size - offset if nbytes is None else nbytes
        return bytes(self.store.get(buf.addr, b'')[offset:offset + n]).ljust(n, b'\0')

    def drain(self, timeout=None):
        pass

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError(f'policy inventory: the native path asked the stand-in for {name!r}')


def child(a):
    root = Path(a.source_root).resolve()
    out = Path(a.run_dir)
    res = dict(point=a.point, config=a.config, status='incomplete',
               knobs={k: os.environ.get(k) for k in sorted(set(BASE) | {'TNCA_PIPELINE',
                                                                        'TNCA_RCTX_ROLES'})})
    t0 = time.monotonic()
    try:
        sys.path[:0] = [str(root), str(root / 'python'), str(root / 'compiler/python'),
                        str(root / 'tools/asm')]
        import torch
        import triton  # noqa: F401  (before tnca: the backend's driver module is discovered by Triton)
        import tnca
        import tnca.inductor
        import tnca_cc
        import tnca.torch_device as tdev
        from tnca.triton_backend import driver
        from tnca import artifact
        from archbench import registry
        from archbench.provenance import software_tree
        for name, mod in (('tnca', tnca), ('tnca_cc', tnca_cc), ('registry', registry)):
            if not Path(mod.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f'{name} imported from {mod.__file__}, outside {root}')
        if software_tree(root, refresh=True) != a.software_tree:
            raise RuntimeError('software tree mismatch')
        point = registry.get(a.point)
        if point.definition_sha256() != a.point_sha256:
            raise RuntimeError('point definition differs from the frozen suite')

        def forbidden(*args, **kwargs):
            raise RuntimeError('policy inventory forbids every device session')
        driver._Sessions.get = classmethod(forbidden)
        captured = []
        original_call = driver.TNCALauncher.__call__

        def call(self, *args, **kwargs):
            assert os.environ['TNCA_LAUNCH'] == 'skip'
            captured.append((self.metadata.name, tuple(args[:3]), str(args[4])))
            return original_call(self, *args, **kwargs)
        driver.TNCALauncher.__call__ = call
        tnca.inductor.configure()
        stub = HostSession(os.environ['TNCA_PROFILE'])
        tdev.MEMORY.session = lambda index=0: stub
        tdev.register()
        torch._dynamo.reset()
        m, inputs = point.make(0)
        m = m.eval().to('tnca')
        inputs = tuple(x.to('tnca') if isinstance(x, torch.Tensor) else x for x in inputs)
        compiled = torch.compile(m, fullgraph=True)
        with torch.no_grad():
            compiled(*inputs)                 # cold: compile (and literal copies)
            captured.clear()
            with driver.launch_log() as log:
                compiled(*inputs)             # the steady call's launches
        kernels = {}
        for (name, grid, path), entry in zip(captured, log):
            rec = kernels.get(path)
            if rec is None:
                manifest, segments = artifact.load(path)
                prov = manifest['optional_metadata']['provenance']
                p2r = manifest['execution'].get('program_to_role', {})
                # Everything that decides execution: every segment, the entry
                # points, parameters, target, execution map, resources and semantics -- not identity or
                # provenance, which name the options and differ by construction.
                payload = json.dumps(dict(
                    segments={k: hashlib.sha256(v).hexdigest() for k, v in sorted(segments.items())},
                    **{k: manifest.get(k) for k in ('target', 'code', 'parameters', 'execution', 'resources', 'semantics')}),
                    sort_keys=True, default=str)
                rec = kernels[path] = dict(
                    kernel=name, text_sha256=hashlib.sha256(segments['text']).hexdigest(),
                    payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    artifact_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    roles_per_program=len(p2r.get('0', [])) if p2r else None,
                    programs_per_launch=len(p2r) if p2r else None,
                    ring=prov.get('rctx_roles_ring'), refused=prov.get('rctx_roles_refused'),
                    fallback=prov.get('rctx_roles_fallback'), pairs=prov.get('rctx_roles_pairs'),
                    rctx_spmd=prov.get('rctx_spmd'), grids=[])
            rec['grids'].append(list(grid))
        res.update(status='pass', launches=len(log), kernels=list(kernels.values()))
    except Exception as e:                                   # noqa: BLE001
        res.update(status='fail', error=f'{type(e).__name__}: {e}'[:3000],
                   traceback=traceback.format_exc()[-6000:])
    res['seconds'] = round(time.monotonic() - t0, 1)
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(f"INVENTORY {res['status']} {a.point} [{a.config}] {res['seconds']}s", flush=True)
    return 0 if res['status'] == 'pass' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--software-tree', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--configs', nargs='+', default=['off', 'pipe2', 'gated2'])
    p.add_argument('--only', nargs='*')
    p.add_argument('--suite', default='spec/r40/suites/existing.json')
    p.add_argument('--timeout', type=float, default=4 * 3600)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--point')
    p.add_argument('--point-sha256')
    p.add_argument('--config')
    p.add_argument('--run-dir')
    a = p.parse_args()
    if a.child:
        return child(a)
    root = a.source_root.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root)]
    from archbench.provenance import software_tree
    if software_tree(root, refresh=True) != a.software_tree:
        raise SystemExit('software tree mismatch; refusing to run')
    suite_raw = (root / a.suite).read_bytes()
    suite = json.loads(suite_raw)
    points = [pt for pt in suite['points'] if not a.only or pt['id'] in a.only]
    report = dict(schema='tnca-r40-v3-policy-inventory/1', software_tree=a.software_tree,
                  source_root=str(root), host=os.uname().nodename,
                  harness_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                  suite=a.suite,
                  suite_sha256=hashlib.sha256(suite_raw).hexdigest(), suite_points=len(suite['points']),
                  base=BASE, configs={c: CONFIGS[c] for c in a.configs}, harness_env=HARNESS,
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs=[])
    lock = threading.Lock()

    def save():
        with lock:
            tmp = a.out / 'report.tmp'
            tmp.write_text(json.dumps(report, indent=1, default=str) + '\n')
            tmp.replace(a.out / 'report.json')

    def launch(pt, config):
        run_dir = a.out / 'runs' / (pt['id'].replace('/', '__') + '__' + config)
        run_dir.mkdir(parents=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_', 'R40_'))}
        env.update(HARNESS)
        env.update(BASE)
        env.update(CONFIGS[config])
        env.update(TRITON_CACHE_DIR=str(run_dir / 'triton'), TRITON_HOME=str(run_dir / 'home'),
                   TORCHINDUCTOR_CACHE_DIR=str(run_dir / 'inductor'), TNCA_CACHE=str(run_dir / 'tnca'),
                   TMPDIR=str(run_dir / 'tmp'), TNCA_DUMP_DIR=str(run_dir / 'dump'),
                   PYTHONPATH=os.pathsep.join(str(root / d) for d in
                                              ('', 'python', 'compiler/python', 'tools/asm')))
        (run_dir / 'tmp').mkdir()
        cmd = [sys.executable, str(HERE), '--child', '--source-root', str(root),
               '--software-tree', a.software_tree, '--point', pt['id'],
               '--point-sha256', pt['sha256'], '--config', config, '--run-dir', str(run_dir)]
        with (run_dir / 'child.log').open('w') as fh:
            try:
                code = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        try:
            res = json.loads((run_dir / 'result.json').read_text())
        except (OSError, json.JSONDecodeError):
            res = dict(point=pt['id'], config=config, status='fail', error='no result')
        res['exit'] = code
        for k in res.get('kernels', []):
            k.pop('artifact_path', None)
        with lock:
            report['runs'].append(res)
        save()
        print(f"{res['status']} {pt['id']} [{config}] exit={code}", flush=True)
        # Bound disk use: keep the result and log, drop caches and dumps.
        subprocess.run(['rm', '-rf', str(run_dir / 'triton'), str(run_dir / 'inductor'),
                        str(run_dir / 'tnca'), str(run_dir / 'tmp'), str(run_dir / 'dump'),
                        str(run_dir / 'home')])

    save()
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(lambda args: launch(*args), [(pt, c) for pt in points for c in a.configs]))
    # Summary: per point, kernels whose machine code differs from `off`.
    by = {}
    for r in report['runs']:
        by.setdefault(r['point'], {})[r['config']] = r
    summary = {}
    for pid, rows in sorted(by.items()):
        off = rows.get('off')
        entry = {}
        for c, r in rows.items():
            if r.get('status') != 'pass':
                entry[c] = dict(status=r.get('status'), error=r.get('error'))
                continue
            texts = sorted(k['payload_sha256'] for k in r['kernels'])
            base = sorted(k['payload_sha256'] for k in off['kernels']) \
                if off and off.get('status') == 'pass' else None
            entry[c] = dict(status='pass', kernels=len(r['kernels']),
                            rings=sum(bool(k['ring']) for k in r['kernels']),
                            refused=sorted({k['refused'] for k in r['kernels'] if k['refused']}),
                            fallback=sorted({k['fallback'] for k in r['kernels'] if k['fallback']}),
                            code_identical_to_off=(texts == base) if base is not None else None)
        summary[pid] = entry
    report.update(summary=summary, finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  status='complete' if all(r.get('status') == 'pass' for r in report['runs'])
                  else 'incomplete')
    save()
    print(f"policy-inventory {report['status']} runs={len(report['runs'])}", flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
