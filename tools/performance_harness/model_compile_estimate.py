#!/usr/bin/env python3
"""Compile-only TNCA estimates for the official Qwen3.5 language branch.

    python3 model_compile_estimate.py --config C --inventory I --sources S --out DIR
        [--graphs prefill decode] [--prefill 128] [--decode-slots 16]

The standard path, without a device: the pinned official Transformers model,
built on meta tensors from the official text configuration, is placed on the
`tnca` PyTorch device (TNCA_TORCH_DEVICE=native) whose storage is a
host-side stand-in for the runtime Session -- the runtime's own first-fit
allocator over the device address map, and no data -- and each graph is
compiled with torch.compile(backend="inductor", fullgraph=True), every kernel
lowered by the TNCA Triton backend and sealed, none launched (TNCA_LAUNCH=skip),
as tools/r40/v3_compile_preflight.py --device native-stub does. No board,
simulator or device session is opened; no checkpoint data is read.

Graphs (each in a fresh process with private Triton/Inductor/tmp directories):

  prefill  B=1, `--prefill` tokens, official StaticCache of prefill+decode
           slots written in place, logits for the last token (generation's
           logits_to_keep=1)
  decode   B=1, one token at position `--prefill` in the same fixed bucket;
           a second call at the next position must not recompile
  prefill-nocache
           B=1, no cache, all logits: the graph language_meta_analysis.py
           traced, for an allocator-vs-ATen-liveness cross-check

Both cached graphs pass the public mask-mapping argument (full-attention
mask built in-graph from position_ids over the fixed slot bucket; linear
attention None), as language_meta_analysis.py does, instead of value-dependent
mask discovery.

Measured per graph: Dynamo/AOT/Inductor/TNCA compile wall times; peak RSS of
the compiling process (and its children); Dynamo and post-decomposition FX
graph sizes; Inductor buffers and kernels; every compiled TNCA artifact
(code bytes, resources, IR stage sizes, compile seconds); the launch sequence
and argument blocks of a steady call; host fallbacks in the generated wrapper
and the ATen operators the tnca dispatcher refuses at run time; the device
allocator trace (live bytes and first-fit address extent) of weights, states,
inputs, graph constants and the steady call's buffers; an LRU replay of the
driver's code-region policy over the launch sequence.

Blockers are findings, recorded and then measured past where possible:

  * a kernel TNCA cannot lower is recorded with its diagnostic; the graph
    goes on and its launches of that kernel are logged as failed (not run),
    so one compile inventories every failing kernel and the allocation plan
    stays complete; its code bytes are absent from the working set;
  * an ATen operator the tnca dispatcher refuses at run time is recorded;
    a shape-only stand-in (outputs allocated by meta shape inference, no
    arithmetic) is then registered for exactly that operator and the steady
    call retried, so the allocation and launch trace finishes;
  * an Inductor lowering failure aborts the graph. `--shims` names
    harness-only workarounds (SHIMS) to measure beyond such blockers; every
    record carries the shims it used.

Nothing here is numerical execution, a performance measurement or a Stage 2
capacity PASS.

    python3 model_compile_estimate.py --summarize LABEL=RUN/GRAPH/result.json ... \
        --estimates-out model_estimates.json

combines finished runs: per-stage compile seconds, IR sizes, code working set
and LRU code-region replay, and the heap composition compared with the 4 GiB
B40 window and 6/8 GiB Stage 2 windows under C_usable >= 1.25 H_peak.
"""
import argparse
from collections import Counter, OrderedDict
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]

# Compile-only harness environment (v3_compile_preflight.py HARNESS, native).
HARNESS = dict(CUDA_VISIBLE_DEVICES='', TNCA_LAUNCH='skip', TNCA_PROFILE='core-vrf-sync-v5',
               TNCA_OPT_LEVEL='2', TNCA_TRITON='1', TNCA_KERNEL_DUMP='1',
               TNCA_BACKEND='fpga-u280-multicore', TNCA_TORCH_DEVICE='native',
               TNCA_TENSOR_DESCRIPTORS='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_JOBS='4')
GIB = 1 << 30
# Harness-only workarounds, each OFF unless named with --shims. They exist to
# measure past a blocker the unmodified standard path hits; a run that uses
# one says so in every record, and none is a product fix or numerically
# qualified.
SHIMS = {
    'decompose-k-off': (
        'torch._inductor.config.triton.num_decompose_k_splits = 0. For skinny GEMMs (K >= 64*max(M,N), '
        'e.g. decode in_proj_a/in_proj_b 1x2048x16) Inductor drops its mm template in favour of '
        'decompose-K, for which the tnca device offers no configuration: NoValidChoicesError.'),
    'log1p-kahan': (
        'libdevice.log1p supplied by the harness as u = 1 + x; where(u == 1, x, log(u) * (x / (u - 1))). '
        'TNCA libdevice deliberately omits log1p; Inductor emits it for softplus (linear-attention decay). '
        'Compile-size stand-in only; its accuracy on TNCA is not qualified.'),
    'scan-no-split': (
        'torch._inductor.ir.Scan.num_splits forced to 1 on the tnca device. Inductor splits the '
        'linear-attention cumsum into a TritonSplitScanKernel whose R0_BLOCK is raised to '
        'min_split_scan_rblock=256 after the device heuristic, which TNCA refuses (register block '
        'width 256; VLEN multiples up to 128).'),
    'int32-positions': (
        'position_ids passed as int32 and the in-graph mask slot index built as int32. The public '
        'interface passes int64 positions; RoPE converts them to float in-graph, a 64-bit tensor '
        'operation (arith.sitofp) TNCA refuses.'),
    'delta-solve-export-branch': (
        'official torch_chunk_gated_delta_rule forward-substitution branch (the code it runs when '
        'is_torchdynamo_exporting() is true) instead of torch.linalg.solve_triangular, which Inductor '
        'leaves as a host ATen call that the tnca device refuses.'),
}
LOG1P_SHIM = '''import triton
import triton.language as tl


@triton.jit
def log1p(x):
    # Goldberg's form: exact where 1 + x rounds to 1, log(u) * x / (u - 1) elsewhere.
    u = 1.0 + x
    return tl.where(u == 1.0, x, tl.log(u) * (x / (u - 1.0)))
'''
WINDOWS = {'B40 4 GiB': 4 * GIB, 'Stage 2 6 GiB': 6 * GIB, 'Stage 2 8 GiB': 8 * GIB}
MARGIN = 1.25


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha256(Path(path).read_bytes())


class StandInSession:
    """Host stand-in for the runtime Session in a compile-only run.

    Allocation uses the runtime's own allocator class over the device's
    address map (heap above the code region, the reserved ranges skipped), so
    alignment, first-fit reuse and fragmentation are the runtime's. No byte
    is held except what is explicitly uploaded (small cache/input tensors).
    Everything else a session can be asked raises."""

    class Buffer:
        __slots__ = ('addr', 'size', 'reserved', 'name', 'session_id', 'epoch', 'alive', 'leases')

        def __init__(self, addr, size, reserved, name, sid, epoch):
            self.addr, self.size, self.reserved, self.name = addr, size, reserved, name
            self.session_id, self.epoch, self.alive, self.leases = sid, epoch, True, 0

    def __init__(self, profile, heap_base, heap_bytes):
        from tnca.runtime import _Allocator, DEVICE_RESERVED_RANGES
        self.profile, self.id, self.epoch = profile, 'compile-stand-in', 0
        self._lock = threading.RLock()
        self._heap = _Allocator(heap_base, heap_base + heap_bytes, reserved=DEVICE_RESERVED_RANGES)
        self.unfinished, self.latest_event, self.closed, self.poisoned = [], None, False, False
        self.store, self.uploaded = {}, 0
        self.phase = 'setup'
        self.events = []                 # (phase, +/-reserved, name)
        self.extent = {}                 # phase -> max address extent above heap base
        self.live = 0
        self.live_peak = {}              # phase -> max reserved live bytes

    def _note(self, delta, name):
        self.live += delta
        self.events.append((self.phase, delta, name))
        top = self._heap.next - self._heap.base
        self.extent[self.phase] = max(self.extent.get(self.phase, 0), top)
        self.live_peak[self.phase] = max(self.live_peak.get(self.phase, 0), self.live)

    def allocate(self, nbytes, *, name=None):
        with self._lock:
            addr, reserved = self._heap.alloc(int(nbytes))
            buf = self.Buffer(addr, int(nbytes), reserved, name, self.id, self.epoch)
            self._note(reserved, name)
            return buf

    def retire(self, buf):
        with self._lock:
            if not buf.alive:
                return False
            buf.alive = False
            self.store.pop(buf.addr, None)
            self._heap.release(buf.addr, buf.reserved)
            self._note(-buf.reserved, buf.name)
            return True

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

    def wait(self, event):
        pass

    def reset_phase(self, phase):
        self.phase = phase
        self.extent[phase] = self._heap.next - self._heap.base
        self.live_peak[phase] = self.live

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError(f'compile-only run: the native path asked the stand-in session for '
                           f'{name!r}, which only a device has')


def simulate_code_store(sequence, sizes, capacity, passes=2):
    """Replay the driver's code-region policy (tnca.triton_backend.driver._load):
    first-fit 64-byte aligned placement in [0, capacity); on a miss that does
    not fit, unload least-recently-used modules one at a time until it does."""
    from tnca.runtime import _Allocator
    from tnca.errors import BufferError
    alloc = _Allocator(0, capacity)
    loaded = OrderedDict()
    out = []
    for _ in range(passes):
        loads = evictions = upload = 0
        for key in sequence:
            if key in loaded:
                loaded.move_to_end(key)
                continue
            try:
                placed = alloc.alloc(sizes[key], align=64)
            except BufferError:
                placed = None
                for old in list(loaded):
                    addr, size = loaded.pop(old)
                    alloc.release(addr, size)
                    evictions += 1
                    try:
                        placed = alloc.alloc(sizes[key], align=64)
                        break
                    except BufferError:
                        continue
                if placed is None:
                    raise ValueError(f'kernel of {sizes[key]} B cannot be placed in {capacity} B')
            loaded[key] = placed
            loads += 1
            upload += sizes[key]
        out.append(dict(launches=len(sequence), loads=loads, evictions=evictions, code_upload_bytes=upload))
    return out


# --------------------------------------------------------------------- child --
def child(a):
    out = Path(a.run_dir)
    res = dict(schema='tnca-qwen-compile-estimate-run/1', graph=a.graph, status='incomplete',
               pid=os.getpid(), host=os.uname().nodename, stage='start',
               harness_env={k: os.environ.get(k) for k in sorted(HARNESS)},
               tnca_env={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
    wall = res['wall_seconds'] = {}
    t0 = time.monotonic()

    def save():
        tmp = out / 'result.tmp'
        tmp.write_text(json.dumps(res, indent=1, default=str) + '\n')
        tmp.replace(out / 'result.json')

    def stage(name):
        res['stage'] = name
        save()
        return time.monotonic()

    save()
    stub = None
    try:
        t = stage('imports')
        sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm')]
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        import triton
        import transformers
        import tnca
        import tnca.inductor
        import tnca.aot
        import tnca.dump
        import tnca.runtime as rt
        from tnca import artifact, abi
        from tnca.triton_backend import driver
        import tnca_cc
        from archbench.provenance import software_tree
        from torch._inductor.runtime import triton_heuristics as H
        from torch._inductor.graph import GraphLowering
        import torch._inductor.compile_fx as cfx
        for name, f in dict(tnca=tnca.__file__, tnca_cc=tnca_cc.__file__, driver=driver.__file__).items():
            if not Path(f).resolve().is_relative_to(ROOT):
                raise RuntimeError(f'{name} imported from {f}, outside the source root {ROOT}')
        sources = json.loads(Path(a.sources).read_text())
        audited = {}
        for name, entry in sources['files'].items():
            module_name = name.removeprefix('src/').removesuffix('.py').replace('/', '.')
            module = importlib.import_module(module_name)
            if digest(module.__file__) != entry['sha256']:
                raise ValueError(f'official source identity mismatch: {name}')
            audited[name] = entry['sha256']
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        if 'delta-solve-export-branch' in (a.shims or ''):
            modeling.is_torchdynamo_exporting = lambda: True
        from transformers.cache_utils import StaticCache, LinearAttentionCacheLayerMixin, StaticLayer
        tree = software_tree(ROOT, refresh=True)
        res.update(software_tree=tree, toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__),
                                                      transformers=transformers.__version__,
                                                      python=sys.version.split()[0]),
                   official_source_revision=sources['revision'], official_source_sha256=audited,
                   harness_sha256=digest(HERE))
        wall['imports'] = time.monotonic() - t

        # ---- the tnca device over the stand-in session ------------------------
        t = stage('device')
        def forbidden(*args, **kwargs):
            raise RuntimeError('compile-only run forbids every device session')
        driver._Sessions.get = classmethod(forbidden)
        inductor_config = tnca.inductor.configure()
        import tnca.torch_device as tdev
        stub = StandInSession(os.environ['TNCA_PROFILE'], a.heap_base, a.stand_in_heap_bytes)
        def session(index=0):
            if index != 0:
                raise RuntimeError(f'tnca:{index}: one device')
            return stub
        tdev.MEMORY.session = session
        tdev.register()
        res['inductor_config'] = dict(inductor_config, device='tnca')
        props = driver.TNCAUtils.get_device_properties(None, 0)
        props = props if isinstance(props, dict) else vars(props)
        res['device_properties'] = {k: v for k, v in props.items() if isinstance(v, (int, str, float, bool))}
        res['stand_in'] = dict(heap_base=a.heap_base, heap_bytes=a.stand_in_heap_bytes,
                               reserved_ranges=[list(r) for r in rt.DEVICE_RESERVED_RANGES],
                               allocator='tnca.runtime._Allocator (first fit, 64-byte, merging)',
                               launch_reserve_bytes=tdev.LAUNCH_RESERVE)
        wall['device_register'] = time.monotonic() - t
        shims = [x for x in (a.shims or '').split(',') if x]
        res['shims'] = {name: SHIMS[name] for name in shims}
        if 'decompose-k-off' in shims:
            import torch._inductor.config as iconfig
            iconfig.triton.num_decompose_k_splits = 0
        if 'scan-no-split' in shims:
            import torch._inductor.ir as iir
            upstream_splits = iir.Scan.num_splits.__func__
            def num_splits(cls, *args, **kwargs):
                hint, splits = upstream_splits(cls, *args, **kwargs)
                device = kwargs.get('device', args[0] if args else None)
                return (hint, 1) if getattr(device, 'type', None) in ('tnca', 'cpu') else (hint, splits)
            iir.Scan.num_splits = classmethod(num_splits)
        if 'log1p-kahan' in shims:
            import importlib.util as importlib_util
            import tnca.triton_backend.libdevice as libdevice
            if hasattr(libdevice, 'log1p'):
                raise RuntimeError('libdevice now provides log1p; drop the shim')
            shim_path = out / 'harness_log1p_shim.py'
            shim_path.write_text(LOG1P_SHIM)
            spec = importlib_util.spec_from_file_location('harness_log1p_shim', shim_path)
            shim_module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(shim_module)
            libdevice.log1p = shim_module.log1p
            res['log1p_shim_sha256'] = digest(shim_path)

        # ---- official model on meta, then device storage (no data) ------------
        t = stage('model')
        config_doc = json.loads(Path(a.config).read_text())['text_config']
        subset = None
        if a.layers:
            subset = [int(i) for i in a.layers.split(',')]
            types = [config_doc['layer_types'][i] for i in subset]
            config_doc = dict(config_doc, num_hidden_layers=len(subset), layer_types=types)
        config = Qwen3_5TextConfig(**config_doc)
        config._attn_implementation = 'eager'
        with torch.device('meta'):
            model = modeling.Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        inventory = json.loads(Path(a.inventory).read_text())
        selected = inventory['selected_tensors']
        matched = {}
        dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32}
        for name, parameter in model.named_parameters():
            source_name = name.replace('model.', 'model.language_model.', 1)
            if subset is not None:
                m = re.match(r'model\.language_model\.layers\.(\d+)\.(.*)', source_name)
                if m:
                    source_name = f'model.language_model.layers.{subset[int(m.group(1))]}.{m.group(2)}'
            entry = selected[source_name]
            if list(parameter.shape) != entry['shape']:
                raise ValueError(f'checkpoint shape mismatch: {name}')
            parameter.data = parameter.data.to(dtype=dtypes[entry['dtype']])
            matched[source_name] = parameter.numel() * parameter.element_size()
        if subset is None and (set(matched) != set(selected) or
                               sum(matched.values()) != inventory['selected_distinct_bytes']):
            raise ValueError('official language parameters do not exactly cover the audited inventory')
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise ValueError('official tied embedding alias was lost on meta')
        wall['model_build_meta'] = time.monotonic() - t

        t = stage('weights')
        stub.reset_phase('weights')
        params, buffers = {}, {}
        keep_meta = list(model.parameters())       # ids stay unique while replacing
        def place(tensor):
            return torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device='tnca')
        for module in model.modules():
            for key, p in list(module._parameters.items()):
                if p is None:
                    continue
                if id(p) not in params:
                    params[id(p)] = torch.nn.Parameter(place(p), requires_grad=False)
                module._parameters[key] = params[id(p)]
        stub.reset_phase('buffers')
        for module in model.modules():
            for key, b in list(module._buffers.items()):
                if b is not None:
                    module._buffers[key] = place(b)
                    buffers[f'{type(module).__name__}.{key}'] = b.numel() * b.element_size()
        del keep_meta
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise ValueError('official tied embedding alias was lost on the device')
        storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
        res['weights'] = dict(parameters=len(matched), distinct_storages=len(storages),
                              distinct_bytes=sum(storages.values()),
                              inventory_selected_distinct_bytes=inventory['selected_distinct_bytes'],
                              tied_embedding_single_storage=True,
                              allocator_reserved_bytes=sum(d for ph, d, _ in stub.events if ph == 'weights'),
                              extent_after_weights=stub.extent['weights'],
                              non_persistent_buffers=buffers,
                              buffers_reserved_bytes=sum(d for ph, d, _ in stub.events if ph == 'buffers'),
                              layer_subset=subset, layer_types=list(config.layer_types))
        wall['weights_place'] = time.monotonic() - t
        save()

        # ---- cache and inputs -------------------------------------------------
        t = stage('cache')
        stub.reset_phase('cache')
        L = a.prefill if a.graph.startswith('prefill') else 1
        slots = a.prefill + a.decode_slots
        bf16 = torch.bfloat16
        cache = None
        if a.graph in ('prefill', 'decode'):
            cache = StaticCache(config=config, max_cache_len=slots)
            conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 + \
                config.linear_value_head_dim * config.linear_num_value_heads
            K = config.linear_conv_kernel_dim
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    layer.lazy_initialization(
                        conv_states=torch.empty((1, conv_dim, K), dtype=bf16, device='tnca'),
                        recurrent_states=torch.empty((1, config.linear_num_value_heads, config.linear_key_head_dim,
                                                      config.linear_value_head_dim), dtype=torch.float32,
                                                     device='tnca'),
                        conv_kernel_size=K)
                elif isinstance(layer, StaticLayer):
                    kv = torch.empty((1, config.num_key_value_heads, 1, config.head_dim), dtype=bf16, device='tnca')
                    layer.lazy_initialization(kv, kv)
                else:
                    raise ValueError(f'unexpected cache layer {type(layer).__name__}')
            state = Counter()
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    state['conv'] += layer.conv_states[0].untyped_storage().nbytes()
                    state['recurrent'] += layer.recurrent_states[0].untyped_storage().nbytes()
                    if layer.recurrent_states[0].dtype != torch.float32:
                        raise ValueError('recurrent state is not FP32')
                else:
                    state['kv'] += layer.keys.untyped_storage().nbytes() + layer.values.untyped_storage().nbytes()
                    state['cumulative_length'] += layer.cumulative_length.untyped_storage().nbytes()
            res['cache'] = dict(cls='transformers.cache_utils.StaticCache', slots=slots,
                                state_bytes=dict(state), state_total=sum(state.values()),
                                allocator_reserved_bytes=sum(d for ph, d, _ in stub.events if ph == 'cache'),
                                decode_position=a.prefill if a.graph == 'decode' else None)
        def prepare_call(position):
            """The cache state a call at `position` starts from, set through the
            public cache API outside the graph: a fresh prompt (Cache.reset())
            for prefill, a filled fixed bucket for decode. Without it, the
            Python side effects of a completed prefill (has_previous_state)
            would make the next call a different, continuation graph."""
            if cache is None:
                return
            if a.graph == 'prefill':
                cache.reset()
                return
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    layer.has_previous_state[0] = True
                else:
                    layer.cumulative_length.fill_(position)

        stub.reset_phase('inputs')
        start = a.prefill if a.graph == 'decode' else 0
        position_dtype = torch.int32 if 'int32-positions' in (a.shims or '') else torch.int64
        input_ids = torch.zeros((1, L), dtype=torch.int64).to('tnca')
        position_ids = torch.arange(start, start + L, dtype=position_dtype)[None].to('tnca')
        res['inputs'] = dict(tokens=L, start_position=start,
                             allocator_reserved_bytes=sum(d for ph, d, _ in stub.events if ph == 'inputs'))
        wall['cache_and_inputs'] = time.monotonic() - t

        if a.graph == 'prefill-nocache':
            def step(input_ids, position_ids):
                mask = torch.full((L, L), float('-inf'), dtype=bf16, device=input_ids.device).triu(1)
                masks = {'full_attention': mask[None, None, :, :], 'linear_attention': None}
                return model(input_ids=input_ids, position_ids=position_ids, attention_mask=masks,
                             use_cache=False).logits
            res['adapter'] = 'public mask mapping, unpadded causal, no cache, all logits (language_meta_analysis.py)'
        else:
            def step(input_ids, position_ids):
                # Public mask-mapping argument over the fixed slot bucket: a
                # query at position p sees slots 0..p.
                allowed = torch.arange(slots, dtype=position_ids.dtype,
                                       device=input_ids.device)[None, :] <= position_ids[0][:, None]
                mask = torch.full((L, slots), float('-inf'), dtype=bf16, device=input_ids.device)
                mask = mask.masked_fill(allowed, 0.0)
                masks = {'full_attention': mask[None, None, :, :], 'linear_attention': None}
                return model(input_ids=input_ids, position_ids=position_ids, attention_mask=masks,
                             past_key_values=cache, use_cache=True, logits_to_keep=1).logits
            res['adapter'] = ('public mask mapping built in-graph from position_ids over the fixed StaticCache '
                              'slot bucket; linear-attention mask None; logits_to_keep=1')

        # ---- instrumentation ----------------------------------------------------
        art_dir = out / 'artifacts'
        art_dir.mkdir(exist_ok=True)
        kernels = {}          # artifact sha -> record
        attempts = []         # every _precompile_config
        lowerings = []        # every TNCA lowering call
        launches = []
        dynamo_graphs, lowering_graphs, wrappers = [], [], []

        orig_precompile = H.CachingAutotuner._precompile_config
        def precompile_config(self, cfg, **kw):
            started = time.monotonic()
            name = self.inductor_meta.get('kernel_name') or getattr(self.fn, '__name__', '?')
            row = dict(kernel=name, config=dict(cfg.kwargs), num_warps=cfg.num_warps)
            try:
                result = orig_precompile(self, cfg, **kw)
            except Exception as e:                                  # noqa: BLE001
                row.update(status='error', error=f'{type(e).__name__}: {str(e)[:600]}',
                           seconds=time.monotonic() - started)
                attempts.append(row)
                raise
            row['seconds'] = time.monotonic() - started
            blob = (getattr(getattr(result, 'kernel', None), 'asm', None) or {}).get('tnca')
            if blob:
                h = sha256(blob)
                path = art_dir / f'{h}.tnca'
                if not path.exists():
                    path.write_bytes(blob)
                man, segments = artifact.load(path)
                row.update(status='compiled', artifact=h)
                rec = kernels.setdefault(h, dict(
                    kernel=name, artifact_bytes=len(blob), code_bytes=len(segments.get('text') or b''),
                    segments={k: len(v) for k, v in segments.items() if isinstance(v, (bytes, bytearray))},
                    resources=man.get('resources'), parameters=len(man['parameters']['signature']),
                    program_to_role=len(man.get('execution', {}).get('program_to_role', {'0': ['role0']})),
                    options=man.get('identity', {}).get('options'), config=dict(cfg.kwargs),
                    compile_seconds=row['seconds'], compiled_for=[]))
                rec['compiled_for'].append(name)
            else:
                row.update(status='no-tnca-blob')
            attempts.append(row)
            return result
        H.CachingAutotuner._precompile_config = precompile_config

        # A kernel TNCA cannot lower is recorded and the graph goes on: the
        # wrapper still runs (its launches of that kernel are logged as
        # failed, not run), so one compile reports every failing kernel and
        # the allocation plan is still complete.
        failed_kernels = {}
        orig_precompile_all = H.CachingAutotuner.precompile
        def precompile_all(self, *args, **kwargs):
            try:
                return orig_precompile_all(self, *args, **kwargs)
            except Exception as e:                                  # noqa: BLE001
                name = self.inductor_meta.get('kernel_name') or getattr(self.fn, '__name__', '?')
                diag = re.search(r'diagnostic: (.*)', str(e))
                failed_kernels[name] = dict(kernel=name, error=f'{type(e).__name__}: {str(e)[:1500]}',
                                            diagnostic=diag.group(1).strip() if diag else None,
                                            configs=[dict(c.kwargs) for c in (self.configs or [])])
                self._harness_failed = failed_kernels[name]['error']
        H.CachingAutotuner.precompile = precompile_all
        orig_autotuner_run = H.CachingAutotuner.run
        def autotuner_run(self, *args, **kwargs):
            if getattr(self, '_harness_failed', None):
                name = self.inductor_meta.get('kernel_name') or '?'
                launches.append(dict(kernel=name, artifact=None, failed=True, grid=None, maps=0, phase=stub.phase))
                return None
            return orig_autotuner_run(self, *args, **kwargs)
        H.CachingAutotuner.run = autotuner_run

        orig_lower = tnca.aot._lower_ttir
        def lower(module, ctx, kernel_name, params, target, prov, options, stages, opt_level):
            started = time.monotonic()
            try:
                return orig_lower(module, ctx, kernel_name, params, target, prov, options, stages, opt_level)
            finally:
                sizes = {s: len(v.encode() if isinstance(v, str) else v) for s, v in stages.stages.items()
                         if isinstance(v, (str, bytes, bytearray))}
                lowerings.append(dict(kernel=kernel_name, seconds=time.monotonic() - started, stage_bytes=sizes))
        tnca.aot._lower_ttir = lower

        write_dump = tnca.dump.Dump.write
        def distinct_dump(stages, directory, **keywords):
            key = sha256(json.dumps(stages.stages, sort_keys=True, default=str).encode())
            return write_dump(stages, Path(directory) / key[:16], **keywords)
        tnca.dump.Dump.write = distinct_dump

        orig_compile_fx = cfx.compile_fx
        def compile_fx(model_, example_inputs_, *args, **kwargs):
            started = time.monotonic()
            row = dict(nodes=len(model_.graph.nodes),
                       ops=dict(Counter(n.op for n in model_.graph.nodes)),
                       python_bytes=len(model_.code.encode()))
            (out / f'dynamo_graph_{len(dynamo_graphs)}.py').write_text(model_.code)
            dynamo_graphs.append(row)
            try:
                return orig_compile_fx(model_, example_inputs_, *args, **kwargs)
            finally:
                row['backend_seconds'] = time.monotonic() - started
        cfx.compile_fx = compile_fx

        orig_run, orig_codegen, orig_ctm = GraphLowering.run, GraphLowering.codegen, GraphLowering.compile_to_module
        def run(self, *args):
            started = time.monotonic()
            try:
                return orig_run(self, *args)
            finally:
                g = self.module.graph
                code = self.module.code
                (out / f'aten_graph_{len(lowering_graphs)}.py').write_text(code)
                lowering_graphs.append(dict(
                    aten_nodes=len(g.nodes), aten_python_bytes=len(code.encode()),
                    aten_ops=dict(Counter(str(n.target) for n in g.nodes if n.op == 'call_function')),
                    lowering_seconds=time.monotonic() - started))
        def codegen(self, *args, **kwargs):
            started = time.monotonic()
            try:
                return orig_codegen(self, *args, **kwargs)
            finally:
                if lowering_graphs:
                    sched = getattr(self, 'scheduler', None)
                    lowering_graphs[-1].update(
                        inductor_buffers=len(self.buffers), inductor_operations=len(self.operations),
                        scheduler_nodes=len(sched.nodes) if sched is not None else None,
                        codegen_seconds=time.monotonic() - started)
        def compile_to_module(self, *args, **kwargs):
            started = time.monotonic()
            try:
                return orig_ctm(self, *args, **kwargs)
            finally:
                if lowering_graphs:
                    lowering_graphs[-1]['compile_to_module_seconds'] = time.monotonic() - started
        GraphLowering.run, GraphLowering.codegen, GraphLowering.compile_to_module = run, codegen, compile_to_module
        GraphLowering.save_output_code = staticmethod(lambda code: wrappers.append(code))

        manifests = {}
        orig_call = driver.TNCALauncher.__call__
        def call(self, *args, **kwargs):
            assert os.environ['TNCA_LAUNCH'] == 'skip'
            path = Path(args[4] if len(args) > 4 else kwargs['function'])
            key = str(path)
            if key not in manifests:
                raw = path.read_bytes()
                man, segments = artifact.load(path)
                manifests[key] = (sha256(raw), man, len(segments.get('text') or b''))
            h, man, code = manifests[key]
            maps = sum(1 for v in args[9:] if type(v).__name__ == 'TensorDescriptor')
            launches.append(dict(kernel=self.metadata.name, artifact=h, grid=list(args[:3]), maps=maps,
                                 phase=stub.phase))
            return orig_call(self, *args, **kwargs)
        driver.TNCALauncher.__call__ = call

        # ---- compile (and the cold call, launches skipped) ----------------------
        t = stage('compile')
        stub.reset_phase('compile')
        import torch._dynamo.utils as dutils
        torch._dynamo.reset()
        compiled = torch.compile(step, backend='inductor', fullgraph=True, dynamic=False)
        refused, stand_ins = [], []
        cold_error = None
        compile_started = time.monotonic()
        prepare_call(start)
        try:
            with torch.no_grad():
                result = compiled(input_ids, position_ids)
            del result
        except Exception as e:                                       # noqa: BLE001
            cold_error = f'{type(e).__name__}: {str(e)[:2000]}'
            res['cold_call_traceback'] = traceback.format_exc()[-6000:]
        wall['compile_and_cold_call'] = time.monotonic() - compile_started
        try:
            res['dynamo_time_spent'] = dutils.calculate_time_spent()
        except Exception as e:                                       # noqa: BLE001
            res['dynamo_time_spent'] = repr(e)
        try:
            headers, values = dutils.compile_times(repr='csv', aggregate=True)
            res['dynamo_compile_times'] = dict(zip(headers, values))
        except Exception as e:                                       # noqa: BLE001
            res['dynamo_compile_times'] = repr(e)
        unique_graphs = dutils.counters['stats']['unique_graphs']
        res['cold_call_error'] = cold_error
        gc.collect()
        res['wrappers'] = len(wrappers)
        for i, src in enumerate(wrappers):
            (out / f'wrapper_{i}.py').write_text(src)
        save()
        if not wrappers:
            raise RuntimeError(f'compilation produced no wrapper: {cold_error}')

        # ---- steady call: refused operators, then shape-only stand-ins ----------
        t = stage('steady')
        lib = torch.library.Library('aten', 'IMPL')

        def stand_in(op_name):
            packet = getattr(torch.ops.aten, op_name.split('.')[0])
            overload = getattr(packet, op_name.split('.')[1] if '.' in op_name else 'default')
            def shape_only(*args, **kwargs):
                def meta(v):
                    return v.to('meta') if isinstance(v, torch.Tensor) else v
                from torch.utils._pytree import tree_map
                result = overload(*tree_map(meta, args), **tree_map(meta, kwargs))
                def realize(v):
                    if isinstance(v, torch.Tensor):
                        return torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device='tnca')
                    return v
                return tree_map(realize, result)
            lib.impl(op_name, shape_only, 'PrivateUse1')

        steady_ok = False
        steady_error = None
        for attempt in range(8):
            launches.clear()
            gc.collect()
            prepare_call(start)
            stub.reset_phase(f'steady-attempt-{attempt}')
            baseline = stub.live
            try:
                with torch.no_grad():
                    result = compiled(input_ids, position_ids)
                steady_ok = True
            except NotImplementedError as e:
                message = str(e)
                m = re.search(r"Could not run '(?:aten::)?([\w.]+)'", message)
                op = m.group(1) if m else None
                refused.append(dict(attempt=attempt, op=op, message=message[:800],
                                    launches_before_refusal=len(launches)))
                del e
                if op is None or op in stand_ins:
                    break
                stand_in(op)
                stand_ins.append(op)
                continue
            except Exception as e:                                   # noqa: BLE001
                steady_error = f'{type(e).__name__}: {str(e)[:2000]}'
                res['steady_traceback'] = traceback.format_exc()[-6000:]
                del e
            break
        gc.collect()
        res['refused_operators'] = refused
        res['shape_only_stand_ins'] = stand_ins
        res['steady_call_complete'] = steady_ok
        res['steady_error'] = steady_error
        res['unique_graphs_after_steady'] = dutils.counters['stats']['unique_graphs']
        res['recompiled_on_steady'] = res['unique_graphs_after_steady'] != unique_graphs
        phase = stub.phase
        steady_launches = list(launches)
        res['steady'] = dict(phase=phase, baseline_live_bytes=baseline,
                             peak_live_bytes=stub.live_peak.get(phase), extent_peak_bytes=stub.extent.get(phase),
                             output_bytes=sum(x.untyped_storage().nbytes() for x in
                                              ([result] if steady_ok and isinstance(result, torch.Tensor) else [])))
        if steady_ok:
            del result
        gc.collect()
        res['steady']['live_after_bytes'] = stub.live
        if a.graph == 'decode' and steady_ok:
            before = dutils.counters['stats']['unique_graphs']
            stub.reset_phase('next-position')
            launches.clear()
            next_pos = torch.tensor([[a.prefill + 1]], dtype=position_dtype).to('tnca')
            prepare_call(a.prefill + 1)
            with torch.no_grad():
                result = compiled(input_ids, next_pos)
            del result
            res['next_position'] = dict(position=a.prefill + 1, unique_graphs_before=before,
                                        unique_graphs_after=dutils.counters['stats']['unique_graphs'],
                                        recompiled=dutils.counters['stats']['unique_graphs'] != before,
                                        launches=len(launches),
                                        same_launch_sequence=[l['artifact'] for l in launches] ==
                                        [l['artifact'] for l in steady_launches])
        wall['steady_calls'] = time.monotonic() - t
        res['unique_graphs'] = unique_graphs
        save()

        # ---- audit ----------------------------------------------------------------
        t = stage('audit')
        fallbacks = tnca.inductor.host_fallbacks(wrappers)
        triton_names = tnca.inductor.triton_kernels(wrappers)
        sources_bytes = {}
        for src in wrappers:
            for m in re.finditer(r"(\w+) = async_compile\.triton\('(\w+)', '''(.*?)'''", src, re.S):
                sources_bytes[m.group(2)] = len(m.group(3).encode())
        failed_launches = [l for l in steady_launches if l.get('failed')]
        steady_launches = [l for l in steady_launches if not l.get('failed')]
        res['failed_kernels'] = sorted(failed_kernels.values(), key=lambda r: r['kernel'])
        res['failed_kernel_launches'] = dict(Counter(l['kernel'] for l in failed_launches))
        launched = Counter(l['artifact'] for l in steady_launches)
        missing = [h for h in launched if h not in kernels]
        for l in steady_launches:
            if l['artifact'] not in kernels:
                # Not seen at precompile (e.g. a runtime helper): take it from the launch.
                for key, (h, man, code) in manifests.items():
                    if h == l['artifact']:
                        kernels[h] = dict(kernel=l['kernel'], code_bytes=code, artifact_bytes=Path(key).stat().st_size,
                                          resources=man.get('resources'),
                                          parameters=len(man['parameters']['signature']),
                                          program_to_role=len(man.get('execution', {}).get('program_to_role',
                                                                                            {'0': ['role0']})),
                                          compiled_for=[l['kernel']], source='launch-only')
        absent = sorted(h for h in launched if h not in kernels)
        if absent:
            raise RuntimeError(f'launched artifacts without a compile record: {absent}')
        unique = {h: kernels[h] for h in launched}
        aligned = {h: (k['code_bytes'] + 63) // 64 * 64 for h, k in kernels.items()}
        code_ws = sum(aligned[h] for h in launched)
        sequence = [l['artifact'] for l in steady_launches]
        lru = simulate_code_store(sequence, {h: kernels[h]['code_bytes'] for h in launched},
                                  tnca.inductor.CODE_STORE_BYTES) if sequence else []

        def arg_block(l):
            k = kernels[l['artifact']]
            scratch = (k.get('resources') or {}).get('scratch_bytes', 0)
            scratch = scratch if isinstance(scratch, int) and scratch > 0 else 0
            scratch *= k.get('program_to_role', 1)
            slots_b = abi.block_size(k['parameters']) + (8 if scratch else 0)
            map_start = (slots_b + 255) // 256 * 256
            scratch_start = map_start + 256 * l['maps']
            size = max(scratch_start + scratch, slots_b, 8)
            stride = (size + 255) // 256 * 256
            return size, stride * a.physical_cores, scratch
        blocks = [arg_block(l) for l in steady_launches]
        per_launch = [b[1] for b in blocks]
        top = sorted(per_launch, reverse=True)[:rt.MAX_UNFINISHED]
        res['launch_scratch'] = dict(
            physical_cores=a.physical_cores, max_unfinished=rt.MAX_UNFINISHED,
            max_argument_block_bytes=max((b[0] for b in blocks), default=0),
            max_physical_argument_table_bytes=max(per_launch, default=0),
            max_kernel_scratch_bytes=max((b[2] for b in blocks), default=0),
            single_submit_peak_bytes=max(per_launch, default=0),
            batched_worst_case_bytes=sum(top),
            formula='tnca.runtime.Session.submit: block + 256 per tensor map + scratch x roles, '
                    '256-aligned stride x physical cores; batched = MAX_UNFINISHED largest')
        stage_totals = Counter()
        for row in lowerings:
            for s, n in row['stage_bytes'].items():
                stage_totals[s] += n
        res.update(
            kernels_compiled=len(kernels), compile_attempts=len(attempts),
            compile_attempt_errors=[r for r in attempts if r.get('status') == 'error'],
            block_fallbacks=list(tnca.inductor._BLOCK_FALLBACKS),
            tnca_lowerings=len(lowerings), tnca_lowering_seconds=sum(r['seconds'] for r in lowerings),
            kernel_compile_seconds=sum(r['seconds'] for r in attempts),
            ir=dict(dynamo_graphs=dynamo_graphs, aten_graphs=lowering_graphs,
                    wrapper_bytes=[len(w.encode()) for w in wrappers],
                    wrapper_triton_kernels=len(triton_names),
                    triton_source_bytes_total=sum(sources_bytes.values()),
                    triton_source_bytes_max=max(sources_bytes.values(), default=0),
                    tnca_stage_bytes_total=dict(stage_totals),
                    tnca_stage_bytes_max={s: max((r['stage_bytes'].get(s, 0) for r in lowerings), default=0)
                                          for s in stage_totals}),
            host_fallbacks=fallbacks,
            launches=len(steady_launches), distinct_launched_artifacts=len(launched),
            launched_without_precompile_record=missing,
            code=dict(code_store_bytes=tnca.inductor.CODE_STORE_BYTES,
                      wide_code_limit=tnca.inductor.WIDE_CODE_LIMIT,
                      unique_launched_code_bytes=sum(k['code_bytes'] for k in unique.values()),
                      unique_launched_code_bytes_64_aligned=code_ws,
                      working_set_over_code_store=code_ws / tnca.inductor.CODE_STORE_BYTES,
                      all_compiled_code_bytes=sum(k['code_bytes'] for k in kernels.values()),
                      largest_kernel_code_bytes=max((k['code_bytes'] for k in unique.values()), default=0),
                      kernels_over_wide_limit=sorted(k['kernel'] for k in unique.values()
                                                     if k['code_bytes'] > tnca.inductor.WIDE_CODE_LIMIT),
                      lru_replay=lru,
                      lru_policy='driver._load: first-fit 64-byte placement; unload LRU modules until it fits'),
            kernel_table=[dict(artifact=h, launches=launched.get(h, 0), **{k: v for k, v in rec.items()
                                                                          if k not in ('options',)})
                          for h, rec in sorted(kernels.items(), key=lambda kv: -kv[1]['code_bytes'])],
            kernel_options=sorted({json.dumps(k.get('options'), sort_keys=True) for k in kernels.values()}),
            launch_sequence=[dict(kernel=l['kernel'], artifact=l['artifact'][:16], grid=l['grid']) for l in
                             steady_launches])
        # Device-memory composition of the steady call, from the allocator trace.
        by_phase = Counter()
        for ph, d, _ in stub.events:
            by_phase[ph] += d
        res['allocator'] = dict(net_reserved_by_phase=dict(by_phase), extent_by_phase=stub.extent,
                                live_peak_by_phase=stub.live_peak, uploaded_bytes=stub.uploaded,
                                events=len(stub.events))
        if software_tree(ROOT, refresh=True) != tree:
            raise RuntimeError('software tree changed during the run')
        if res['recompiled_on_steady']:
            raise RuntimeError('the steady call recompiled: it measured a different graph')
        res['status'] = 'compiled' + ('' if steady_ok and not refused and not fallbacks
                                      and not failed_kernels else '-with-findings')
        wall['audit'] = time.monotonic() - t
    except Exception as e:                                                    # noqa: BLE001
        res.update(status='failed', error=f'{type(e).__name__}: {str(e)[:4000]}', traceback=traceback.format_exc())
    wall['total'] = time.monotonic() - t0
    res['peak_rss_kib'] = dict(self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save()
    print(f'RUN {res["status"]} {a.graph} {json.dumps({k: round(v, 1) for k, v in wall.items()})}', flush=True)
    os._exit(0 if res['status'].startswith('compiled') else 1)


# ----------------------------------------------------------------- summary --
def compile_stages(r):
    """Wall seconds per compile stage from Dynamo's cumulative phase timers
    and the harness's per-kernel hooks (nested timers are differenced)."""
    t = r.get('dynamo_time_spent') or {}
    if not isinstance(t, dict) or 'entire_frame_compile' not in t:
        return None
    kernels = r.get('kernel_compile_seconds') or 0.0
    tnca = r.get('tnca_lowering_seconds') or 0.0
    precompile = t.get('async_compile.precompile', kernels)
    return dict(
        dynamo_trace_and_guards=t['entire_frame_compile'] - t.get('backend_compile', 0),
        aot_autograd_decompositions=t.get('backend_compile', 0) - t.get('inductor_compile', 0),
        inductor_lowering_and_scheduling=t.get('inductor_compile', 0) - t.get('code_gen', 0),
        inductor_codegen_excluding_kernels=t.get('code_gen', 0) - precompile,
        triton_frontend_ttir=max(kernels - tnca, 0.0),
        tnca_backend_lowering=tnca,
        frame_compile_total=t['entire_frame_compile'],
        process_total=r['wall_seconds'].get('total'),
        device_extension_register=r['wall_seconds'].get('device_register'),
        compile_and_cold_call=r['wall_seconds'].get('compile_and_cold_call'))


def memory_plan(r):
    """Device heap composition of one graph's steady call, from the allocator trace."""
    st = r.get('steady') or {}
    scratch = r.get('launch_scratch') or {}
    if not st.get('peak_live_bytes'):
        return None
    w = r['weights']
    persistent = st['baseline_live_bytes']
    components = dict(
        weights_distinct=w['distinct_bytes'],
        weights_allocator_reserved=w['allocator_reserved_bytes'],
        non_persistent_buffers=w['buffers_reserved_bytes'],
        states=(r.get('cache') or {}).get('allocator_reserved_bytes', 0),
        inputs=r['inputs']['allocator_reserved_bytes'])
    components['graph_held_other'] = persistent - sum(v for k, v in components.items()
                                                      if k not in ('weights_distinct',))
    components['steady_transient_peak'] = st['peak_live_bytes'] - persistent
    components['fragmentation_alignment_reserved_skip'] = st['extent_peak_bytes'] - st['peak_live_bytes']
    launch = max(scratch.get('batched_worst_case_bytes', 0), r['stand_in']['launch_reserve_bytes'])
    heap_peak = st['extent_peak_bytes'] + launch
    return dict(components=components, persistent_live=persistent, steady_peak_live=st['peak_live_bytes'],
                steady_extent_peak=st['extent_peak_bytes'],
                launch_argument_blocks=dict(single_submit=scratch.get('single_submit_peak_bytes'),
                                            batched_worst_case=scratch.get('batched_worst_case_bytes'),
                                            launch_reserve=r['stand_in']['launch_reserve_bytes'],
                                            counted=launch),
                heap_peak=heap_peak, code_region=r['code']['code_store_bytes'],
                h_peak_including_code_region=heap_peak + r['code']['code_store_bytes'])


def blocker_keys(r):
    """(kind, normalised detail, kernel or op, launches per call) for every
    blocker a run recorded."""
    out = []
    if r.get('status') == 'failed' and r.get('error'):
        e = r['error']
        m = re.search(r'(NoValidChoicesError)', e)
        target = re.search(r'target: ([\w.]+)', e)
        detail = (f'Inductor lowering: {m.group(1)} for {target.group(1) if target else "?"} '
                  '(no tnca template choice)') if m else e.splitlines()[0][:200]
        out.append(('lowering-abort', detail, target.group(1) if target else None, None))
    launches = r.get('failed_kernel_launches') or {}
    for f in r.get('failed_kernels') or []:
        attr = re.search(r"has no attribute '(\w+)'", f['error'])
        detail = (f'libdevice.{attr.group(1)} absent from TNCA libdevice' if attr else
                  f.get('diagnostic') or f['error'].splitlines()[0][:200])
        out.append(('kernel-lowering', detail, f['kernel'], launches.get(f['kernel'])))
    for fb in r.get('host_fallbacks') or []:
        out.append(('host-fallback', f'wrapper leaves {fb} on the host', fb, None))
    for x in r.get('refused_operators') or []:
        out.append(('refused-at-run', f'tnca dispatcher refuses aten::{x["op"]}', x['op'], None))
    return out


def summarize(a):
    runs = {}
    for spec in a.summarize:
        label, path = spec.split('=', 1)
        runs[label] = json.loads(Path(path).read_text())
    from tnca.runtime import DEVICE_RESERVED_RANGES
    reserved = sum(hi - lo for lo, hi in DEVICE_RESERVED_RANGES)
    est = dict(schema='tnca-qwen-model-estimates/1', item='S1-04.compile_code_IR_memory_estimates',
               boundary=('Estimates from compile-only standard-path builds of the official model with meta/'
                         'uninitialised weights: no numerical execution, no device or simulator, no '
                         'performance claim, and not a Stage 2 capacity PASS.'),
               runs={})
    for label, r in runs.items():
        code = r.get('code') or {}
        est['runs'][label] = dict(
            graph=r['graph'], status=r['status'], stage=r.get('stage'), shims=sorted(r.get('shims') or {}),
            error=(r.get('error') or '')[:600] or None,
            layers=(r.get('weights') or {}).get('layer_types'),
            kernels_compiled=r.get('kernels_compiled'), launches=r.get('launches'),
            distinct_launched_artifacts=r.get('distinct_launched_artifacts'),
            failed_kernels=[dict(kernel=f['kernel'], diagnostic=f.get('diagnostic') or f['error'][:300])
                            for f in r.get('failed_kernels') or []],
            failed_kernel_launches=r.get('failed_kernel_launches'),
            host_fallbacks=r.get('host_fallbacks'),
            refused_operators=[dict(op=x['op'], launches_before_refusal=x['launches_before_refusal'])
                               for x in r.get('refused_operators') or []],
            shape_only_stand_ins=r.get('shape_only_stand_ins'),
            next_position=r.get('next_position'),
            compile_seconds=compile_stages(r),
            peak_rss_bytes=dict(compiler_process=(r.get('peak_rss_kib') or {}).get('self', 0) * 1024,
                                compiler_children=(r.get('peak_rss_kib') or {}).get('children', 0) * 1024,
                                sampled_process_tree=(r.get('watchdog') or {}).get('peak_tree_rss')),
            ir=r.get('ir'),
            code={k: code.get(k) for k in ('unique_launched_code_bytes', 'unique_launched_code_bytes_64_aligned',
                                           'working_set_over_code_store', 'all_compiled_code_bytes',
                                           'largest_kernel_code_bytes', 'kernels_over_wide_limit',
                                           'lru_replay', 'code_store_bytes')} if code else None,
            memory=memory_plan(r) if r.get('steady') else None,
            kernels=[dict(kernel=k['kernel'], artifact=k['artifact'][:16], code_bytes=k['code_bytes'],
                          launches=k.get('launches', 0), compile_seconds=round(k.get('compile_seconds') or 0, 3),
                          scratch_bytes=(k.get('resources') or {}).get('scratch_bytes'))
                     for k in r.get('kernel_table') or []],
            weights=r.get('weights'), cache=r.get('cache'),
            identity=dict(software_tree=r.get('software_tree'), harness_sha256=r.get('harness_sha256'),
                          toolchain=r.get('toolchain'), official_source_revision=r.get('official_source_revision'),
                          tnca_env=r.get('tnca_env')))
    # Every blocker, where it was seen, and which shimmed runs of the same
    # graph no longer show it.
    blockers = {}
    for label, r in runs.items():
        for kind, detail, where, count in blocker_keys(r):
            b = blockers.setdefault(detail, dict(kind=kind, detail=detail, seen_in={}, graphs=set()))
            b['seen_in'][label] = dict(where=where, launches_per_call=count,
                                       shims=sorted(r.get('shims') or {}))
            b['graphs'].add(r['graph'])
    for b in blockers.values():
        # Measured past: a completed run of the same graph whose shims include
        # every shim of a run that showed the blocker, and that no longer does.
        # (A kernel that fails stops at its first diagnostic, so the absence of
        # a blocker in a run with fewer shims means nothing.)
        b['measured_past_in'] = sorted(
            label for label, r in runs.items()
            if label not in b['seen_in'] and r['status'] != 'failed' and r.get('stage') == 'audit'
            and any(runs[seen]['graph'] == r['graph'] and
                    set(runs[seen].get('shims') or {}) <= set(r.get('shims') or {})
                    for seen in b['seen_in']))
        b['graphs'] = sorted(b['graphs'])
    est['blockers'] = list(blockers.values())
    est['measurement_scope'] = dict(
        measured=['compile wall seconds per stage on the compiling host', 'compiler process peak RSS',
                  'Dynamo, ATen and Inductor IR sizes', 'TNCA IR stage bytes per lowered kernel',
                  'code bytes and resources of every sealed TNCA artifact',
                  'launch sequence of one steady call', 'first-fit device allocator trace of weights, '
                  'states, inputs and every Inductor buffer of one steady call'],
        modelled=['code-region residency: LRU replay of the driver policy over the launch sequence',
                  'launch argument blocks: tnca.runtime.Session.submit formula over artifact manifests',
                  'usable window capacity: window bytes minus the device-reserved range'],
        not_measured=['numerics and device execution', 'code bytes of kernels TNCA failed to lower',
                      'device implementation and workspace of host-fallback operators (shape-only '
                      'stand-ins allocate their outputs only)',
                      'buffer lifetime extension under asynchronous submission (TNCA_SUBMIT=batched)',
                      'host staging, reset epochs and multi-session effects'])

    # Session view: prefill then decode share weights and states; transient
    # buffers of the two graphs are never live together.
    est['windows'] = {}
    for label, r in est['runs'].items():
        m = r.get('memory')
        if not m:
            continue
        rows = {}
        for name, size in WINDOWS.items():
            usable = size - reserved
            need = MARGIN * m['h_peak_including_code_region']
            rows[name] = dict(c_usable=usable, required_c_usable=need,
                              fits_without_margin=m['h_peak_including_code_region'] <= usable,
                              meets_1_25_margin=usable >= need,
                              headroom_vs_margin=usable - need)
        est['windows'][label] = rows
    # A generation session: one prefill, then decode steps through the fixed
    # bucket, in one device session (shared weights, states and code region).
    est['sessions'] = {}
    for pair in a.session or []:
        pre_label, dec_label = pair.split(',')
        pre, dec = runs[pre_label], runs[dec_label]
        sizes = {}
        for r in (pre, dec):
            for k in r['kernel_table']:
                sizes[k['artifact'][:16]] = k['code_bytes']
        pre_seq = [l['artifact'] for l in pre['launch_sequence']]
        dec_seq = [l['artifact'] for l in dec['launch_sequence']]
        steps = a.decode_slots
        union = set(pre_seq) | set(dec_seq)
        capacity = pre['code']['code_store_bytes']
        replay = simulate_code_store(pre_seq + dec_seq * steps, sizes, capacity, passes=1)[0]
        pm, dm = memory_plan(pre), memory_plan(dec)
        h_peak = max(pm['h_peak_including_code_region'], dm['h_peak_including_code_region'])
        est['sessions'][f'{pre_label}+{dec_label}'] = dict(
            decode_steps=steps,
            distinct_kernels_union=len(union),
            unique_code_bytes_union=sum(sizes[h] for h in union),
            unique_code_bytes_union_64_aligned=sum((sizes[h] + 63) // 64 * 64 for h in union),
            code_store_bytes=capacity,
            union_over_code_store=sum((sizes[h] + 63) // 64 * 64 for h in union) / capacity,
            launches=len(pre_seq) + steps * len(dec_seq),
            lru_replay_prefill_then_decode=replay,
            steady_decode_step_code_upload_bytes=dec['code']['lru_replay'][-1]['code_upload_bytes'],
            steady_decode_step_code_loads=dec['code']['lru_replay'][-1]['loads'],
            prefill_call_code_upload_bytes=pre['code']['lru_replay'][-1]['code_upload_bytes'],
            h_peak=h_peak, h_peak_from=pre_label if h_peak == pm['h_peak_including_code_region'] else dec_label,
            windows={name: dict(c_usable=size - reserved, required_c_usable=MARGIN * h_peak,
                                fits_without_margin=h_peak <= size - reserved,
                                meets_1_25_margin=size - reserved >= MARGIN * h_peak,
                                headroom_vs_margin=size - reserved - MARGIN * h_peak,
                                max_h_peak_for_margin=(size - reserved) / MARGIN)
                     for name, size in WINDOWS.items()})
    # Allocator plan versus the earlier straight-line ATen storage liveness
    # of the same no-cache, all-logits prefill (language_meta_analysis.py).
    est['aten_liveness_crosscheck'] = {}
    for spec in a.aten_crosscheck or []:
        label, path = spec.split('=', 1)
        aten = json.loads(Path(path).read_text())
        m = memory_plan(runs[label])
        est['aten_liveness_crosscheck'][label] = dict(
            aten_report=dict(path=path, sha256=digest(path), prefill=aten.get('prefill'),
                             aten_nodes=aten.get('aten_nodes'),
                             activation_storage_peak_bytes=aten.get('activation_storage_peak_bytes')),
            inductor_allocator_transient_peak_bytes=m['components']['steady_transient_peak'],
            equal=m['components']['steady_transient_peak'] == aten.get('activation_storage_peak_bytes'))
    est['margin_rule'] = 'C_usable >= 1.25 * H_peak (doc/r41_tasks_and_acceptance.md Stage 2 D)'
    est['window_capacity_rule'] = (f'C_usable = window bytes - {reserved} B device-reserved range inside the heap; '
                                   'H_peak = steady first-fit heap extent + max(launch argument blocks, '
                                   'LAUNCH_RESERVE) + 128 KiB code region')
    Path(a.estimates_out).write_text(json.dumps(est, indent=1, default=str) + '\n')
    print(json.dumps({k: dict(status=v['status'], code=(v.get('code') or {}).get('unique_launched_code_bytes'),
                              heap=(v.get('memory') or {}).get('h_peak_including_code_region'))
                      for k, v in est['runs'].items()}, indent=1))
    return 0


# -------------------------------------------------------------------- parent --
def tree_rss(root_pid):
    """Resident bytes of a process and all its descendants, from /proc."""
    parents = {}
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:
                stat = Path(f'/proc/{d}/stat').read_text()
                parents[int(d)] = int(stat.rsplit(')', 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
    pids, frontier = {root_pid}, [root_pid]
    while frontier:
        p = frontier.pop()
        for c, pp in parents.items():
            if pp == p and c not in pids:
                pids.add(c)
                frontier.append(c)
    total = 0
    for p in pids:
        try:
            for line in Path(f'/proc/{p}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
    return total, sorted(pids)


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def run_child(a, graph, run_dir, tree):
    for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump'):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
    env.update(HARNESS)
    env.update(PYTHONPATH=os.pathsep.join(str(x) for x in (ROOT, ROOT / 'python', ROOT / 'compiler/python',
                                                           ROOT / 'tools/asm')),
               TRITON_CACHE_DIR=str(run_dir / 'triton_cache'), TRITON_HOME=str(run_dir / 'triton_home'),
               TORCHINDUCTOR_CACHE_DIR=str(run_dir / 'inductor_cache'), TMPDIR=str(run_dir / 'tmp'),
               TNCA_DUMP_DIR=str(run_dir / 'dump'))
    if a.torch_extensions_dir:
        env['TORCH_EXTENSIONS_DIR'] = str(a.torch_extensions_dir)
    cmd = [sys.executable, '-u', str(HERE), '--child', '--graph', graph, '--run-dir', str(run_dir),
           '--config', str(a.config), '--inventory', str(a.inventory), '--sources', str(a.sources),
           '--prefill', str(a.prefill), '--decode-slots', str(a.decode_slots),
           '--physical-cores', str(a.physical_cores), '--heap-base', str(a.heap_base),
           '--stand-in-heap-bytes', str(a.stand_in_heap_bytes)]
    if a.layers:
        cmd += ['--layers', a.layers]
    if a.shims:
        cmd += ['--shims', a.shims]
    watch = dict(peak_tree_rss=0, min_available=mem_available(), killed=None, samples=0)
    started = time.monotonic()
    with open(run_dir / 'child.log', 'w') as fh:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=fh, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            rss, pids = tree_rss(proc.pid)
            avail = mem_available()
            watch['peak_tree_rss'] = max(watch['peak_tree_rss'], rss)
            watch['min_available'] = min(watch['min_available'], avail)
            watch['samples'] += 1
            reason = None
            if rss > a.rss_limit_gib * GIB:
                reason = f'process-tree RSS {rss} B over {a.rss_limit_gib} GiB'
            elif avail < a.min_available_gib * GIB:
                reason = f'host MemAvailable {avail} B under {a.min_available_gib} GiB'
            elif time.monotonic() - started > a.timeout:
                reason = f'timeout after {a.timeout} s'
            if reason:
                watch['killed'] = reason
                for p in reversed(pids):          # by PID, children first
                    try:
                        os.kill(p, signal.SIGKILL)
                    except OSError:
                        pass
                proc.wait()
                break
            time.sleep(2)
    watch['process_wall_seconds'] = time.monotonic() - started
    watch['exit'] = proc.returncode
    try:
        res = json.loads((run_dir / 'result.json').read_text())
    except Exception:                                                         # noqa: BLE001
        res = dict(status='failed', error='child left no result')
    if watch['killed']:
        res.update(status='failed', error=f'harness stopped the child: {watch["killed"]} (stage {res.get("stage")})')
    res['watchdog'] = watch
    for d in ('triton_cache', 'inductor_cache', 'triton_home'):
        subprocess.run(['rm', '-rf', str(run_dir / d)])
    (run_dir / 'result.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--summarize', nargs='+', metavar='LABEL=RESULT_JSON',
                   help='combine finished run results into an estimates JSON (no compilation)')
    p.add_argument('--estimates-out', type=Path)
    p.add_argument('--aten-crosscheck', action='append', metavar='LABEL=ATEN_REPORT_JSON',
                   help='with --summarize: compare a run with a language_meta_analysis.py report')
    p.add_argument('--session', action='append', metavar='PREFILL_LABEL,DECODE_LABEL',
                   help='with --summarize: one prefill then --decode-slots decode steps in one session')
    p.add_argument('--out', type=Path)
    p.add_argument('--graphs', nargs='+', default=['prefill', 'decode'],
                   choices=['prefill', 'decode', 'prefill-nocache'])
    p.add_argument('--prefill', type=int, default=128)
    p.add_argument('--decode-slots', type=int, default=16)
    p.add_argument('--layers', help='debug/extrapolation only: comma-separated official layer indices')
    p.add_argument('--shims', default='', help='comma-separated harness workarounds: ' + ', '.join(SHIMS))
    p.add_argument('--physical-cores', type=int, default=3, help='B40: three cores')
    p.add_argument('--heap-base', type=int, default=0x20000, help='heap above the 128 KiB code region')
    p.add_argument('--stand-in-heap-bytes', type=int, default=16 * GIB,
                   help='stand-in address space; large so the peak is measured rather than refused')
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--rss-limit-gib', type=float, default=20)
    p.add_argument('--min-available-gib', type=float, default=41)
    p.add_argument('--timeout', type=float, default=4 * 3600)
    p.add_argument('--torch-extensions-dir', type=Path)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--graph', help=argparse.SUPPRESS)
    p.add_argument('--run-dir', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        return child(a)
    if a.summarize:
        sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
        return summarize(a)
    if not (a.config and a.inventory and a.sources and a.out):
        p.error('--config, --inventory, --sources and --out are required to compile')
    if a.prefill <= 0 or a.decode_slots <= 0:
        p.error('positive prefill and decode slots required')
    unknown = [n for n in a.shims.split(',') if n and n not in SHIMS]
    if unknown:
        p.error(f'unknown shims {unknown}')
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    for name in ('config', 'inventory', 'sources', 'torch_extensions_dir'):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    sys.path[:0] = [str(ROOT)]
    from archbench.provenance import software_tree
    tree = software_tree(ROOT, refresh=True)
    report = dict(schema='tnca-qwen-compile-estimate/1', status='running', software_tree=tree,
                  harness_sha256=digest(HERE), host=os.uname().nodename,
                  inputs={n: dict(path=str(getattr(a, n)), sha256=digest(getattr(a, n)))
                          for n in ('config', 'inventory', 'sources')},
                  geometry=dict(batch=1, prefill=a.prefill, decode_slots=a.decode_slots,
                                kv_slots=a.prefill + a.decode_slots, layers=a.layers or 'all'),
                  shims={n: SHIMS[n] for n in a.shims.split(',') if n},
                  harness_env=HARNESS, limits=dict(rss_limit_gib=a.rss_limit_gib,
                                                   min_available_gib=a.min_available_gib,
                                                   cpu_affinity=sorted(os.sched_getaffinity(0))),
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs={})
    lock = threading.Lock()

    def save():
        with lock:
            (a.out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    save()

    def work(graph):
        res = run_child(a, graph, a.out / graph, tree)
        with lock:
            report['runs'][graph] = res
        print(f'{time.strftime("%H:%M:%S")} {graph}: {res["status"]} '
              f'{res.get("error", "")[:300]}', flush=True)
        save()
    threads = []
    for graph in a.graphs:
        while sum(t.is_alive() for t in threads) >= a.jobs:
            time.sleep(1)
        th = threading.Thread(target=work, args=(graph,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    report['software_tree_after'] = software_tree(ROOT, refresh=True)
    report['status'] = 'complete' if report['software_tree_after'] == tree else 'tree-changed'
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
