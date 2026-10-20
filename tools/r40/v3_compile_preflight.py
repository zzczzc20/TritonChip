#!/usr/bin/env python3
"""R40 v3: compile-only preflight of every board A/B compile configuration.

Before a board campaign on a fixed image, prove offline -- no board, no
simulator, every device session forbidden -- that each compile configuration
the campaign uses lowers for each frozen point. Harness only: it is outside
the hashed product tree and changes nothing in it.

    python3 v3_compile_preflight.py --source-root SRC --software-tree files:<sha> --out DIR

Each (point, configuration) runs in a fresh process (no knob state can leak
through configure()-time registration) with its own Triton, Inductor and
temporary directories (no artifact compiled under another configuration can
be served from a cache and mask a failure). Per run: torch.compile
(fullgraph, as the native campaign compiles) of the ArchBench point on CPU
tensors with TNCA_LAUNCH=skip, so every kernel of the graph is lowered by the
TNCA Triton backend and sealed, and none is launched. The approach is
tools/r40/bucket_compile_coverage.py's.

Not measured: numerics, native execution, performance.

--device cpu (default, the prior art's path): CPU tensors. Inductor generates
for device "cpu" with the TNCA heuristics; the TNCA_SDPA=fused template only
engages on `tnca` tensors (tnca.online_attention.eligible_inputs), so this
path cannot exercise it -- each record counts the fused SDPA graph nodes
(sdpa_online_nodes).

--device native-stub (supplementary): TNCA_TORCH_DEVICE=native, the module
and inputs on the `tnca` PyTorch device as the campaign runs them, Inductor
generating for device "tnca". Storage is a host-memory stand-in for the
runtime Session (HostSession below: allocation, upload, download only); no
device, simulator or board is opened, and anything else the native path asks
of a session raises, so the stand-in cannot quietly answer for a device.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve()

# "B40-S compile": the campaign's control-side compile knobs.
BASE = dict(TNCA_MAX_BLOCK='16', TNCA_GEMM_TILE='16,16,16', TNCA_REDUCTION_ROWS='1',
            TNCA_BANK_ALLOC='0', TNCA_WINDOW_REUSE='0', TNCA_SDPA='decomposed',
            TNCA_PIPELINE='1', TNCA_TUNE='0')
# B40-S runtime knobs (tools/r40/board_ablation_collect.py baseline_knobs). No
# compiler module reads them; set so the environment is the control side's.
RUNTIME = dict(TNCA_TRANSFER_POLICY='per-kernel', TNCA_ACTIVE_CORES='1',
               TNCA_RTL_MEM_PORTS='1', TNCA_RTL_OUTSTANDING='1', TNCA_SUBMIT='single')
# Compile-only harness environment. TNCA_BACKEND names the campaign's backend
# so the compile options resolved from it (physical_chunks, via
# tnca.backend.physical_grid) are the campaign's; no backend is ever opened.
HARNESS = dict(CUDA_VISIBLE_DEVICES='', TNCA_LAUNCH='skip',
               TNCA_PROFILE='core-vrf-sync-v5', TNCA_OPT_LEVEL='2', TNCA_TRITON='1',
               TNCA_KERNEL_DUMP='1', TNCA_BACKEND='fpga-u280-multicore',
               PYTHONDONTWRITEBYTECODE='1')
KNOB_KEYS = sorted(set(BASE) | {'TNCA_PACKED_WINDOW_UNPACK'})
DEVICE_ENV = {'cpu': 'cpu', 'native-stub': 'native'}


class HostSession:
    """Host bytes in place of a device Session, for compile-only native runs.

    Only what tnca.torch_device.DeviceMemory and the profile properties use:
    a heap (64-byte aligned bump allocation, runtime.DEFAULT_ALIGN), upload,
    download, retire, drain. Every other attribute raises."""

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
        self.store, self.allocations, self.uploaded = {}, 0, 0

    def allocate(self, nbytes, *, name=None):
        with self._lock:
            addr = (self._heap.next + 63) // 64 * 64
            if addr + nbytes > self._heap.limit:
                raise MemoryError('host stub heap exhausted')
            self._heap.next = addr + max(int(nbytes), 1)
            self.allocations += 1
            return self.Buffer(addr, int(nbytes), name, self.id, self.epoch)

    def retire(self, buf):
        buf.alive = False
        self.store.pop(buf.addr, None)

    def upload(self, buf, data, offset=0):
        b = self.store.setdefault(buf.addr, bytearray(buf.size))
        b[offset:offset + len(data)] = data
        self.uploaded += len(data)

    def download(self, buf, nbytes=None, offset=0):
        n = buf.size - offset if nbytes is None else nbytes
        b = self.store.get(buf.addr, b'')
        return bytes(b[offset:offset + n]).ljust(n, b'\0')

    def drain(self, timeout=None):
        pass

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError(f'compile preflight: the native path asked the stand-in session for '
                           f'{name!r}, which only a device has')

SUBSETS = ('R40-03.pointwise-reduction', 'R40-03.gemm-reuse', 'R40-07.long-k',
           'R40-08.long-prefill', 'R40-02.graphs3', 'R40-02.small-kernel-graphs',
           'R40-04.streaming', 'R40-05.compute')


def cfg(label, **over):
    knobs = dict(BASE)
    knobs.update(over)
    return dict(label=label, knobs=knobs)


# Items of the request, in order. `fallback`: run for the whole subset only
# when the primary candidate fails on any point.
ITEMS = [
    dict(item=1, subset='R40-03.pointwise-reduction', experiment='R40-03.wide-blocks',
         configs=[cfg('base'), cfg('max_block=64', TNCA_MAX_BLOCK='64')],
         fallback=dict(after='max_block=64', config=cfg('max_block=32', TNCA_MAX_BLOCK='32'))),
    dict(item=2, subset='R40-03.gemm-reuse', experiment='R40-03.gemm-tile',
         configs=[cfg('base'), cfg('gemm_tile=16,64,16', TNCA_GEMM_TILE='16,64,16')],
         fallback=dict(after='gemm_tile=16,64,16',
                       config=cfg('gemm_tile=16,32,16', TNCA_GEMM_TILE='16,32,16'))),
    dict(item=3, subset='R40-07.long-k', experiment='R40-07.pipeline',
         configs=[cfg('base'), cfg('pipeline=2', TNCA_PIPELINE='2')]),
    # As requested, and (supplementary) with the prerequisite
    # tnca.online_attention.enable() demands, to see past configure().
    dict(item=4, subset='R40-08.long-prefill', experiment='R40-08.prefill',
         configs=[cfg('base'), cfg('sdpa=fused', TNCA_SDPA='fused'),
                  cfg('sdpa=fused+packed_window_unpack=1', TNCA_SDPA='fused',
                      TNCA_PACKED_WINDOW_UNPACK='1')]),
    dict(item=5, subset='R40-02.graphs3', experiment='R40-02.*', configs=[cfg('base')]),
    dict(item=5, subset='R40-02.small-kernel-graphs', experiment='R40-02.batched',
         configs=[cfg('base')]),
    dict(item=5, subset='R40-04.streaming', experiment='R40-04.*', configs=[cfg('base')]),
    dict(item=5, subset='R40-05.compute', experiment='R40-05.three-cores', configs=[cfg('base')]),
]


def knob_key(knobs):
    return '|'.join(f'{k}={v}' for k, v in sorted(knobs.items()))


def heavy(point_id):
    """Model points of the large and hbm tiers: one at a time."""
    tier = point_id.split('/')[1]
    return point_id.startswith('model.') and tier in ('large', 'hbm')


# ---------------------------------------------------------------- child ----
def child(a):
    root = Path(a.source_root).resolve()
    out = Path(a.run_dir)
    res = dict(point=a.point, point_sha256=a.point_sha256, config=a.config,
               knobs={k: os.environ.get(k) for k in KNOB_KEYS},
               harness_env={k: os.environ.get(k) for k in sorted(HARNESS) + sorted(RUNTIME)},
               status='incomplete', pid=os.getpid())
    times = res['wall_s'] = {}
    t_start = time.monotonic()

    def save():
        tmp = out / 'result.tmp'
        tmp.write_text(json.dumps(res, indent=1, default=str) + '\n')
        tmp.replace(out / 'result.json')

    save()
    try:
        sys.path[:0] = [str(root), str(root / 'python'), str(root / 'compiler/python'),
                        str(root / 'tools/asm')]
        # The knobs must be in the environment before configure() reads them.
        for k, v in json.loads(a.knobs).items():
            assert os.environ.get(k) == v, f'{k}: environment {os.environ.get(k)!r}, wanted {v!r}'
        import torch
        import triton
        import tnca
        import tnca.inductor
        import tnca_cc
        from tnca import online_attention
        from tnca.triton_backend import driver
        from tnca.artifact import load, read_header
        from archbench import registry
        from archbench.provenance import software_tree
        from archbench.platforms.tnca import _wrapper_launches
        from torch._inductor.utils import run_and_get_code
        mods = dict(tnca=tnca.__file__, tnca_cc=tnca_cc.__file__, driver=driver.__file__,
                    archbench=registry.__file__)
        for name, f in mods.items():
            if not Path(f).resolve().is_relative_to(root):
                raise RuntimeError(f'{name} imported from {f}, outside the source root {root}')
        res['modules'] = mods
        tree = software_tree(root, refresh=True)
        if tree != a.software_tree:
            raise RuntimeError(f'software tree {tree} != {a.software_tree}')
        res['software_tree'] = tree
        res['toolchain'] = dict(torch=str(torch.__version__), triton=str(triton.__version__),
                                python=sys.version.split()[0])
        point = registry.get(a.point)
        if point.definition_sha256() != a.point_sha256:
            raise RuntimeError(f'{a.point}: definition {point.definition_sha256()} != frozen {a.point_sha256}')

        def forbidden(*args, **kwargs):
            raise RuntimeError('compile preflight forbids every device session')
        driver._Sessions.get = classmethod(forbidden)
        # Profile-only properties, as bucket_compile_coverage.py declares them;
        # production reads them after opening a native heap.
        from tnca import device
        from tnca.adapters import r1
        target, _info = r1.target_for(device.base_profile(os.environ['TNCA_PROFILE']))
        declared = dict(max_shared_mem=0,
                        multiprocessor_count=int(getattr(target, 'num_cores', 1) or 1),
                        max_num_regs=int(target.phys_vregs), warpSize=1,
                        sm_clock_rate=0, mem_clock_rate=0, mem_bus_width=64,
                        max_threads_per_block=int(getattr(target, 'vlen', 16) or 16))
        native = a.device == 'native-stub'
        res['device'] = a.device
        if not native:
            driver.TNCAUtils.get_device_properties = lambda self, index: dict(declared)
            res['compile_target_properties'] = declared

        launchers, captured = [], []
        original_init, original_call = driver.TNCALauncher.__init__, driver.TNCALauncher.__call__

        def init(self, src, metadata):
            original_init(self, src, metadata)
            info = dict(name=getattr(metadata, 'name', None))
            try:
                sig = getattr(src, 'signature', {}) or {}
                names = list(sig) if isinstance(sig, dict) else []
                consts = {}
                for k, v in (getattr(src, 'constants', {}) or {}).items():
                    if isinstance(k, tuple) and len(k) == 1 and isinstance(k[0], int) and k[0] < len(names):
                        k = names[k[0]]
                    if isinstance(v, (bool, int, float, str)):
                        consts[str(k)] = v
                info['constexprs'] = consts
                text = getattr(getattr(src, 'fn', None), 'src', None) or ''
                info['source_constexprs'] = {
                    m.group(1): int(m.group(2)) for m in
                    re.finditer(r'\b(\w*BLOCK\w*)\s*:\s*tl\.constexpr\s*=\s*(\d+)', text)}
            except Exception as e:                           # noqa: BLE001
                info['introspection_error'] = repr(e)
            launchers.append(info)

        def call(self, *args, **kwargs):
            assert os.environ['TNCA_LAUNCH'] == 'skip'
            captured.append(Path(args[4] if len(args) > 4 else kwargs['function']))
            return original_call(self, *args, **kwargs)
        driver.TNCALauncher.__init__ = init
        driver.TNCALauncher.__call__ = call

        times['setup'] = round(time.monotonic() - t_start, 2)
        stage = 'configure'
        res['stage'] = stage
        save()
        inductor_config = tnca.inductor.configure()
        res['inductor_config'] = inductor_config
        if native:
            import tnca.torch_device as tdev
            stub = HostSession(os.environ['TNCA_PROFILE'])

            def session(index=0):
                if index != 0:
                    raise RuntimeError(f'tnca:{index}: one device')
                return stub
            tdev.MEMORY.session = session
            tdev.register()           # archbench order: configure(), then register()
            res['inductor_config'] = dict(inductor_config, device='tnca')
            res['compile_target_properties'] = driver.TNCAUtils.get_device_properties(None, 0)
        from torch._inductor.decomposition import decompositions
        res['sdpa_fused_decomposition_installed'] = bool(online_attention._bindings) and all(
            decompositions.get(op) is fused for op, (_, fused) in online_attention._bindings.items())

        res['stage'] = stage = 'make'
        save()
        t = time.monotonic()
        torch._dynamo.reset()
        m, inputs = point.make(0)
        m = m.eval()
        if native:
            m = m.to('tnca')
            inputs = tuple(x.to('tnca') if isinstance(x, torch.Tensor) else x for x in inputs)
        res['parameter_bytes'] = sum(p.numel() * p.element_size() for p in m.parameters())
        res['input_bytes'] = sum(x.numel() * x.element_size() for x in inputs
                                 if isinstance(x, torch.Tensor))
        times['make'] = round(time.monotonic() - t, 2)

        res['stage'] = stage = 'compile'
        save()
        compiled = torch.compile(m, fullgraph=True)
        t = time.monotonic()
        with torch.no_grad(), driver.launch_log() as log:
            _out, codes = run_and_get_code(compiled, *inputs)
        times['compile'] = round(time.monotonic() - t, 2)
        cold = None
        if native:
            # Native compilation can launch kernels on real device literals
            # (FakeTensor constant propagation; archbench's `cold_setup`). As
            # archbench does, the graph's own launches are a steady call's.
            cold, cold_captured = list(log), list(captured)
            captured.clear()
            _out = None
            t = time.monotonic()
            with torch.no_grad(), driver.launch_log() as log:
                _out = compiled(*inputs)
            times['steady_call'] = round(time.monotonic() - t, 2)
            from collections import Counter
            res['cold_call'] = dict(launches=len(cold), kernels=dict(Counter(e['kernel'] for e in cold)))

        res['stage'] = stage = 'audit'
        for i, src in enumerate(codes):
            (out / f'wrapper_{i}.py').write_text(src)
        problems = []
        fallbacks = tnca.inductor.host_fallbacks(codes)
        wrapper_kernels = tnca.inductor.triton_kernels(codes)
        expected = _wrapper_launches(codes)
        if fallbacks:
            problems.append(f'host fallback: {fallbacks}')
        if not log:
            problems.append('no kernel was launched (skipped) at all')
        if len(log) != len(captured):
            problems.append(f'{len(log)} logged launches, {len(captured)} captured')
        if expected is not None and expected != len(log):
            problems.append(f'{len(log)} launches for {expected} in the wrapper'
                            + (' (steady call)' if native else ''))
        if not all(e.get('skipped') is True and e['device_cycles'] is None for e in log):
            problems.append('a launch was not skipped')
        if driver._Sessions.open:
            problems.append('a device session is open')
        artifacts, options, kernels = {}, set(), {}
        for e, path in zip(log, captured):
            if str(path) in artifacts:
                continue
            raw = path.read_bytes()
            load(path)                    # the sealed artifact, every hash checked
            man, _at = read_header(path)
            ident = man.get('identity', {})
            artifacts[str(path)] = hashlib.sha256(raw).hexdigest()
            options.add(json.dumps(ident.get('options', {}), sort_keys=True))
            kernels[ident.get('kernel_name') or e['kernel']] = dict(
                artifact=artifacts[str(path)], bytes=len(raw), constexprs=ident.get('constexprs'),
                roles=len(man.get('code', {}).get('entry_points', {})))
        dump = Path(os.environ['TNCA_DUMP_DIR'])
        dumped = sorted(p.name for p in dump.iterdir() if p.is_dir()) if dump.is_dir() else []
        # Block / tile actually compiled: every *BLOCK* constexpr each kernel got.
        # Only the graph's kernels (native cold-call literal copies excluded).
        graph_names = {e['kernel'] for e in log}
        launchers = [info for info in launchers if info.get('name') in graph_names]
        blocks = {}
        for info in launchers:
            for k, v in {**info.get('constexprs', {}), **info.get('source_constexprs', {})}.items():
                if 'BLOCK' in k and isinstance(v, int) and not isinstance(v, bool):
                    blocks[k] = max(blocks.get(k, 0), v)
        tiles = sorted({(c.get('BLOCK_M'), c.get('BLOCK_N'), c.get('BLOCK_K'))
                        for info in launchers
                        for c in [{**info.get('constexprs', {}), **info.get('source_constexprs', {})}]
                        if c.get('BLOCK_M') and c.get('BLOCK_N') and c.get('BLOCK_K')})
        # The fused template's graph nodes, as the wrapper's source-node
        # comments name them (its kernels are named triton_tem_fused_...).
        sdpa_nodes = sum(src.count('target=torch.ops.tnca.sdpa_online_v1') for src in codes)
        res.update(
            launches=len(log), expected_launches=expected,
            kernels_compiled=len(dumped), kernels_launched=len(kernels),
            graph_launchers=len(launchers), wrapper_triton_kernels=len(wrapper_kernels),
            artifacts=len(set(artifacts.values())), artifact_options=[json.loads(o) for o in sorted(options)],
            max_block_constexprs=blocks, gemm_tiles=[list(t) for t in tiles],
            sdpa_online_nodes=sdpa_nodes, host_fallbacks=fallbacks,
            max_launch_bytes=max((e['upload_bytes'] for e in log), default=0),
            kernels=kernels, launcher_constexprs=launchers, problems=problems)
        if native:
            res['native_memory'] = dict(peak_bytes=tdev.MEMORY.peak, live_bytes=tdev.MEMORY.live,
                                        allocations=stub.allocations, uploaded_bytes=stub.uploaded)
        if software_tree(root, refresh=True) != a.software_tree:
            problems.append('software tree changed during the run')
        res['status'] = 'pass' if not problems else 'fail'
        res['stage'] = 'done'
    except Exception as e:                                   # noqa: BLE001
        res.update(status='fail', error=f'{type(e).__name__}: {e}'[:4000],
                   traceback=traceback.format_exc())
        dump = Path(os.environ.get('TNCA_DUMP_DIR', '/nonexistent'))
        failed = []
        if dump.is_dir():
            for meta in dump.glob('*/*.json'):
                try:
                    md = json.loads(meta.read_text())
                    if isinstance(md, dict) and md.get('failed'):
                        failed.append(dict(kernel=meta.parent.name, failed=md['failed']))
                except Exception:                            # noqa: BLE001
                    pass
            res['kernels_compiled'] = sum(1 for p in dump.iterdir() if p.is_dir())
        res['failed_kernel_dumps'] = failed
    times['total'] = round(time.monotonic() - t_start, 2)
    res['peak_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    save()
    print(f'PREFLIGHT {res["status"]} {a.point} [{a.config}] {times}', flush=True)
    return 0 if res['status'] == 'pass' else 1


# --------------------------------------------------------------- parent ----
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--software-tree', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--timeout', type=float, default=3 * 3600)
    p.add_argument('--only', nargs='*', help='debug: limit to these point ids')
    p.add_argument('--device', choices=sorted(DEVICE_ENV), default='cpu')
    p.add_argument('--torch-extensions-dir', type=Path,
                   help='native-stub: private build dir of the tnca device extension')
    p.add_argument('--items', type=Path,
                   help='JSON list replacing ITEMS: [{"item", "subset", "experiment", '
                        '"configs": [{"label", "knobs": {overrides of BASE}}]}]; subsets are frozen v2 ones')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--point')
    p.add_argument('--point-sha256')
    p.add_argument('--config')
    p.add_argument('--knobs')
    p.add_argument('--run-dir')
    a = p.parse_args()
    if a.child:
        return child(a)

    root = a.source_root.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root), str(root / 'python'), str(root / 'compiler/python'),
                    str(root / 'tools/asm'), str(root / 'tools/r40')]
    from archbench.provenance import software_tree
    import contract_v2
    tree = software_tree(root, refresh=True)
    if tree != a.software_tree:
        raise SystemExit(f'software tree {tree} != {a.software_tree}; refusing to run')
    contract_v2.verify(root)
    plan = json.loads((root / 'spec/r40/v2/plan.json').read_text())

    global ITEMS
    if a.items:
        ITEMS = [dict(it, configs=[cfg(c['label'], **c.get('knobs', {})) for c in it['configs']])
                 for it in json.loads(a.items.read_text())]
    subsets = {}
    for name in sorted({it['subset'] for it in ITEMS}):
        path = root / f'spec/r40/suites/{name}.json'
        raw = path.read_bytes()
        doc = json.loads(raw)
        frozen = plan['subsets'][name]
        assert doc['sha256'] == frozen['sha256'] and len(doc['points']) == frozen['points'], name
        subsets[name] = dict(path=str(path.relative_to(root)), sha256=doc['sha256'],
                             file_sha256=hashlib.sha256(raw).hexdigest(), points=doc['points'])

    report = dict(schema='tnca-r40-v3-compile-preflight/1', status='running',
                  software_tree=tree, contract_lock_sha256=contract_v2.LOCK_SHA256,
                  source_root=str(root), host=os.uname().nodename,
                  harness=str(HERE), harness_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                  device=a.device, harness_env=dict(HARNESS, TNCA_TORCH_DEVICE=DEVICE_ENV[a.device], **RUNTIME),
                  base_knobs=BASE,
                  method=('compile-only: fresh process and fresh Triton/Inductor/tmp dirs per '
                          '(point, config); torch.compile(fullgraph=True), TNCA_LAUNCH=skip, '
                          'device sessions forbidden; ' + ('CPU tensors' if a.device == 'cpu' else
                          'tnca-device tensors over a host-memory stand-in session') +
                          '; pass = every kernel lowered and sealed, artifacts load, no host '
                          'fallback, launch count equals the wrapper\'s'),
                  not_measured=['numerics', 'native execution', 'performance']
                               + (['tnca-device (native) code generation'] if a.device == 'cpu' else []),
                  subsets={k: {kk: vv for kk, vv in v.items() if kk != 'points'} | dict(n=len(v['points']))
                           for k, v in subsets.items()},
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs={}, items=[])
    lock = threading.Lock()

    def save():
        with lock:
            tmp = a.out / 'report.tmp'
            tmp.write_text(json.dumps(report, indent=1, default=str) + '\n')
            tmp.replace(a.out / 'report.json')

    runs = report['runs']           # key -> run record
    queue = []                      # keys not yet started
    shas = {}

    def add(point, config):
        if a.only and point['id'] not in a.only:
            return None
        shas[point['id']] = point['sha256']
        key = point['id'] + ' @ ' + knob_key(config['knobs'])
        if key not in runs:
            runs[key] = dict(point=point['id'], config=config['label'], knobs=config['knobs'],
                             status='queued', heavy=heavy(point['id']))
            queue.append(key)
        return key

    for it in ITEMS:
        it['runs'] = {c['label']: [add(pt, c) for pt in subsets[it['subset']]['points']]
                      for c in it['configs']}
    save()

    cond = threading.Condition()
    state = dict(running=0, heavy=0)

    def launch(key):
        r = runs[key]
        run_dir = a.out / 'runs' / (r['point'].replace('/', '__') + '__' + re.sub(r'[^\w.=,+-]', '_', r['config']))
        run_dir.mkdir(parents=True, exist_ok=False)
        for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump'):
            (run_dir / d).mkdir()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
        env.update(HARNESS)
        env['TNCA_TORCH_DEVICE'] = DEVICE_ENV[a.device]
        if a.torch_extensions_dir:
            env['TORCH_EXTENSIONS_DIR'] = str(a.torch_extensions_dir)
        env.update(RUNTIME)
        env.update(r['knobs'])
        env.update(PYTHONPATH=os.pathsep.join(str(x) for x in (root, root / 'python',
                                                                root / 'compiler/python', root / 'tools/asm')),
                   TRITON_CACHE_DIR=str(run_dir / 'triton_cache'), TRITON_HOME=str(run_dir / 'triton_home'),
                   TORCHINDUCTOR_CACHE_DIR=str(run_dir / 'inductor_cache'), TMPDIR=str(run_dir / 'tmp'),
                   TNCA_DUMP_DIR=str(run_dir / 'dump'))
        cmd = [sys.executable, '-u', str(HERE), '--child', '--source-root', str(root),
               '--software-tree', tree, '--device', a.device,
               '--point', r['point'], '--point-sha256', shas[r['point']],
               '--config', r['config'], '--knobs', json.dumps(r['knobs']), '--run-dir', str(run_dir)]
        r.update(status='running', run_dir=str(run_dir), started=time.strftime('%H:%M:%S'))
        save()
        t = time.monotonic()
        with open(run_dir / 'child.log', 'w') as fh:
            try:
                proc = subprocess.run(cmd, cwd=run_dir, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=a.timeout)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        wall = round(time.monotonic() - t, 1)
        try:
            res = json.loads((run_dir / 'result.json').read_text())
        except Exception:                                    # noqa: BLE001
            res = dict(status='fail', error='child left no result')
        if code == 'timeout':
            res.update(status='fail', error=f'timeout after {a.timeout}s (stage {res.get("stage")})')
        elif res.get('status') == 'incomplete' or (code != 0 and res.get('status') == 'pass'):
            tail = (run_dir / 'child.log').read_text(errors='replace')[-3000:]
            res.update(status='fail', error=res.get('error') or f'child exit {code}: {tail}')
        r.update(status=res['status'], exit=code, process_wall_s=wall, result=res,
                 finished=time.strftime('%H:%M:%S'))
        # Caches duplicate the artifacts the result already hashes.
        for d in ('triton_cache', 'inductor_cache', 'triton_home'):
            subprocess.run(['rm', '-rf', str(run_dir / d)])
        print(f'{time.strftime("%H:%M:%S")} {res["status"]:4s} {wall:8.1f}s {r["point"]} [{r["config"]}]'
              + (f'\n      {str(res.get("error"))[:400]}' if res['status'] != 'pass' else ''), flush=True)
        save()

    def fallbacks():
        """Queue each item's fallback once its primary candidate has an answer."""
        for it in ITEMS:
            fb = it.get('fallback')
            if not fb or it.get('fallback_decided'):
                continue
            keys = [k for k in it['runs'][fb['after']] if k]
            if keys and all(runs[k]['status'] in ('pass', 'fail') for k in keys):
                it['fallback_decided'] = True
                if any(runs[k]['status'] == 'fail' for k in keys):
                    it['configs'].append(fb['config'])
                    it['runs'][fb['config']['label']] = [
                        add(pt, fb['config']) for pt in subsets[it['subset']]['points']]

    def worker(key):
        try:
            launch(key)
        except Exception:                                    # noqa: BLE001
            runs[key].update(status='fail', result=dict(status='fail', error=traceback.format_exc()))
        finally:
            with cond:
                state['running'] -= 1
                state['heavy'] -= runs[key]['heavy']
                fallbacks()
                cond.notify_all()

    threads = []
    with cond:
        while True:
            fallbacks()
            pending = [k for k in queue if runs[k]['status'] == 'queued']
            if not pending and state['running'] == 0:
                break
            pick = None
            if state['running'] < a.jobs:
                # Heavy first while the heavy slot is free; else the next light run.
                pick = next((k for k in pending if runs[k]['heavy']), None) if state['heavy'] == 0 else None
                pick = pick or next((k for k in pending if not runs[k]['heavy']), None)
            if pick is None:
                cond.wait(30)
                continue
            runs[pick]['status'] = 'starting'
            state['running'] += 1
            state['heavy'] += runs[pick]['heavy']
            th = threading.Thread(target=worker, args=(pick,), daemon=True)
            th.start()
            threads.append(th)
    for th in threads:
        th.join()

    # Summary per item and configuration.
    for it in ITEMS:
        for c in it['configs']:
            keys = [k for k in it['runs'][c['label']] if k]
            fails = [dict(point=runs[k]['point'], error=str(runs[k]['result'].get('error')
                                                            or runs[k]['result'].get('problems'))[:600])
                     for k in keys if runs[k]['status'] != 'pass']
            report['items'].append(dict(
                item=it['item'], subset=it['subset'], experiment=it['experiment'], config=c['label'],
                knobs=c['knobs'], compiled=len(keys) - len(fails), total=len(keys),
                summary=f'{c["label"]} compiles {len(keys) - len(fails)}/{len(keys)}', failures=fails))
    report['software_tree_after'] = software_tree(root, refresh=True)
    report['status'] = 'complete' if report['software_tree_after'] == tree else 'tree-changed'
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()
    for s in report['items']:
        print(f'item {s["item"]} {s["subset"]:28s} {s["summary"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
