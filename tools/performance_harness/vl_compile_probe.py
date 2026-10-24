#!/usr/bin/env python3
"""Compile-only TNCA probe of the official Qwen3-VL text and vision branches.

    python3 vl_compile_probe.py --config C --header H --out DIR
        [--graphs text-prefill text-decode text-prefill-deepstack vision vision-api]
        [--text-layers N] [--vision-depth N] [--grid T,H,W] [--prefill 128]
        [--shims int32-positions,...] [--backend tnca|cpu-inductor] [--attn eager|sdpa]
        [--capture-dynamic]

The standard path without a device, as tools/performance_harness/
model_compile_estimate.py runs it for the Qwen3.5 language branch: the pinned
official Transformers model is built on meta tensors from the official
`config.json`, every parameter is checked against the checkpoint's safetensors
header (names, shapes, dtypes; no tensor byte is read), and the model is placed
on the `tnca` PyTorch device (TNCA_TORCH_DEVICE=native) over the same host
stand-in Session (model_compile_estimate.StandInSession: the runtime's own
first-fit allocator over the device map, no data). Each graph is compiled with
torch.compile(backend="inductor", fullgraph=True, dynamic=False); every kernel
is lowered by the TNCA Triton backend and sealed, none is launched
(TNCA_LAUNCH=skip). Each graph runs in a fresh process with private caches
under the same RSS / host-memory watchdog.

Graphs
  text-prefill   language model, B=1, --prefill tokens, 3-row (T/H/W) mrope
                 position ids [3, 1, L], causal 4-D mask built in-graph (a
                 prepared 4-D mask is returned as-is by create_causal_mask),
                 no cache, logits of the last token
  text-decode    one token at position --prefill in an official StaticCache of
                 --prefill + --decode-slots slots, mask over the slot bucket
  text-prefill-deepstack
                 text-prefill with the DeepStack inputs the multimodal model
                 passes (visual_pos_masks [1, L] bool, one [Nv, hidden] feature
                 per deepstack layer); Nv = merged tokens of --grid
  mm-prefill     the multimodal prefill as Qwen3VLModel.forward runs it, with the
                 vision outputs passed through the public `mm_encoder_outputs`
                 (the tower is its own graph): placeholder mask, masked_scatter
                 of the image features, DeepStack, text; 3-D M-RoPE positions
                 from get_rope_index on the host
  vision         the vision tower for one image of --grid patches, with the
                 data-dependent grid tensors precomputed through the public
                 vision_utils API (cu_seqlens, position_ids, interp_indices,
                 interp_weights passed as kwargs), so the graph has static shapes
  vision-api     the vision tower called as the model calls it,
                 visual(pixel_values, grid_thw)

Blockers are recorded, then measured past where possible, exactly as in
model_compile_estimate.py: a kernel TNCA cannot lower is recorded with its
diagnostic and skipped; an ATen operator the tnca dispatcher refuses at run
time gets a shape-only stand-in and the steady call is retried; a Dynamo or
Inductor failure ends the graph with its traceback. `--shims` takes the
harness-only workarounds of model_compile_estimate.SHIMS that apply here
(decompose-k-off, log1p-kahan, scan-no-split, int32-positions: position ids
and the vision rotary positions passed as int32), plus its own `erf-standin`.
`--candidate-fixes` installs a generic fix module (CANDIDATE_FIXES) after
tnca.inductor.configure(), to measure it before it is hooked into configure().

`--backend cpu-inductor` runs the same graph on CPU tensors with stock Inductor
(C++ backend, no TNCA import, random weights, reduced sizes allowed) to tell a
model-API limitation (fails on any device) from a TNCA gap.

Nothing here is numerical execution or a performance measurement.
"""
import argparse
from collections import Counter
import gc
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
sys.path.insert(0, str(HERE.parent))
import model_compile_estimate as mce  # noqa: E402  (helpers only; never modified here)

GIB = mce.GIB
GRAPHS = ('text-prefill', 'text-decode', 'text-prefill-deepstack', 'mm-prefill', 'vision', 'vision-api')
APPLICABLE_SHIMS = ('decompose-k-off', 'log1p-kahan', 'scan-no-split', 'int32-positions', 'erf-standin')
# Harness-only, like model_compile_estimate.SHIMS: a compile-size stand-in so a
# graph is measured past a missing libdevice function. Not a product fix and
# not numerically qualified.
LOCAL_SHIMS = {
    'erf-standin': (
        'libdevice.erf supplied by the harness as Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7 in exact '
        'arithmetic). TNCA libdevice deliberately omits erf; Inductor emits it for exact GELU '
        '(nn.GELU(approximate="none")). Compile-size stand-in only; its accuracy on TNCA is not qualified.'),
}
ERF_SHIM = '''import triton
import triton.language as tl


@triton.jit
def erf(x):
    a = tl.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    y = 1.0 - p * tl.exp(-a * a)
    return tl.where(x < 0, -y, y)
'''
# Candidate generic fixes in their own modules, applied after
# tnca.inductor.configure() so a probe can show what they change before they
# are hooked into configure().
CANDIDATE_FIXES = {
}


def _apply_shims(shims, out, res):
    res['shims'] = {n: LOCAL_SHIMS.get(n) or mce.SHIMS[n] for n in shims}
    if 'erf-standin' in shims:
        import importlib.util as iu
        import tnca.triton_backend.libdevice as libdevice
        if not hasattr(libdevice, 'erf'):
            path = out / 'harness_erf_shim.py'
            path.write_text(ERF_SHIM)
            spec = iu.spec_from_file_location('harness_erf_shim', path)
            m = iu.module_from_spec(spec)
            spec.loader.exec_module(m)
            libdevice.erf = m.erf
        else:
            res['shims']['erf-standin'] = 'not applied: libdevice provides erf'
    if 'decompose-k-off' in shims:
        import torch._inductor.config as iconfig
        iconfig.triton.num_decompose_k_splits = 0
    if 'scan-no-split' in shims:
        import torch._inductor.ir as iir
        upstream = iir.Scan.num_splits.__func__

        def num_splits(cls, *args, **kwargs):
            hint, splits = upstream(cls, *args, **kwargs)
            device = kwargs.get('device', args[0] if args else None)
            return (hint, 1) if getattr(device, 'type', None) in ('tnca', 'cpu') else (hint, splits)
        iir.Scan.num_splits = classmethod(num_splits)
    if 'log1p-kahan' in shims:
        import importlib.util as iu
        import tnca.triton_backend.libdevice as libdevice
        if not hasattr(libdevice, 'log1p'):
            path = out / 'harness_log1p_shim.py'
            path.write_text(mce.LOG1P_SHIM)
            spec = iu.spec_from_file_location('harness_log1p_shim', path)
            m = iu.module_from_spec(spec)
            spec.loader.exec_module(m)
            libdevice.log1p = m.log1p
        else:
            res['shims']['log1p-kahan'] = 'not applied: libdevice provides log1p'


# --------------------------------------------------------------------- child --
def child(a):
    out = Path(a.run_dir)
    res = dict(schema='tnca-qwen3vl-compile-probe-run/1', graph=a.graph, backend=a.backend, status='incomplete',
               pid=os.getpid(), host=os.uname().nodename, stage='start',
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

    tnca_backend = a.backend == 'tnca'
    shims = [x for x in (a.shims or '').split(',') if x]
    save()
    stub = None
    try:
        t = stage('imports')
        sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm')]
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
        import transformers
        from transformers.models.qwen3_vl import modeling_qwen3_vl as modeling
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers import vision_utils
        from transformers.cache_utils import StaticCache
        res['toolchain'] = dict(torch=str(torch.__version__), transformers=transformers.__version__,
                                modeling_sha256=mce.digest(modeling.__file__),
                                vision_utils_sha256=mce.digest(vision_utils.__file__),
                                python=sys.version.split()[0])
        res['harness_sha256'] = mce.digest(HERE)
        if a.capture_dynamic:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        res['dynamo_capture_dynamic'] = bool(a.capture_dynamic)
        device = 'cpu'
        if tnca_backend:
            import triton
            import tnca
            import tnca.inductor
            import tnca.aot
            import tnca.dump
            import tnca.runtime as rt
            from tnca import artifact
            from tnca.triton_backend import driver
            from torch._inductor.runtime import triton_heuristics as H
            from torch._inductor.graph import GraphLowering
            import torch._inductor.compile_fx as cfx
            if not Path(tnca.__file__).resolve().is_relative_to(ROOT):
                raise RuntimeError(f'tnca imported from {tnca.__file__}, outside {ROOT}')
            from archbench.provenance import software_tree
            tree = software_tree(ROOT, refresh=True)
            res['software_tree'] = tree
            res['toolchain']['triton'] = str(triton.__version__)

            def forbidden(*args, **kwargs):
                raise RuntimeError('compile-only run forbids every device session')
            driver._Sessions.get = classmethod(forbidden)
            res['inductor_config'] = tnca.inductor.configure()
            import tnca.torch_device as tdev
            stub = mce.StandInSession(os.environ['TNCA_PROFILE'], a.heap_base, a.stand_in_heap_bytes)

            def session(index=0):
                if index != 0:
                    raise RuntimeError(f'tnca:{index}: one device')
                return stub
            tdev.MEMORY.session = session
            tdev.register()
            device = 'tnca'
            fixes = [x for x in (a.candidate_fixes or '').split(',') if x]
            for name in fixes:
                import importlib
                importlib.import_module(CANDIDATE_FIXES[name]).install()
            res['candidate_fixes'] = {n: CANDIDATE_FIXES[n] for n in fixes}
            _apply_shims(shims, out, res)
        elif shims:
            raise ValueError('shims apply to the tnca backend only')
        wall['imports_and_device'] = time.monotonic() - t

        # ---- official model on meta, checked against the checkpoint header -------
        t = stage('model')
        doc = json.loads(Path(a.config).read_text())
        if a.text_layers:
            doc['text_config'] = dict(doc['text_config'], num_hidden_layers=a.text_layers)
        if a.vision_depth:
            vc = doc['vision_config']
            keep = [i for i in vc['deepstack_visual_indexes'] if i < a.vision_depth]
            if not keep:
                keep = [a.vision_depth - 1]
            doc['vision_config'] = dict(vc, depth=a.vision_depth, deepstack_visual_indexes=keep)
        config = Qwen3VLConfig(**doc)
        for c in (config, config.text_config, config.vision_config):
            c._attn_implementation = a.attn
        res['config'] = dict(effective_attn=dict(text=config.text_config._attn_implementation,
                                                 vision=config.vision_config._attn_implementation),
                             text_layers=config.text_config.num_hidden_layers,
                             vision_depth=config.vision_config.depth,
                             deepstack_visual_indexes=list(config.vision_config.deepstack_visual_indexes),
                             text_rope_parameters=config.text_config.rope_parameters,
                             attn_implementation=a.attn)
        dtype = torch.bfloat16
        if tnca_backend:
            with torch.device('meta'):
                model = modeling.Qwen3VLForConditionalGeneration(config).to(dtype=dtype).eval()
        else:
            torch.manual_seed(0)
            model = modeling.Qwen3VLForConditionalGeneration(config).to(dtype=dtype).eval()
        res['config']['model_attn'] = dict(text=model.model.language_model.config._attn_implementation,
                                           vision=model.model.visual.config._attn_implementation)
        header = json.loads(Path(a.header).read_text())
        header.pop('__metadata__', None)
        dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32}
        checked, mismatched = 0, []
        for name, p in model.named_parameters():
            entry = header.get(name)
            if entry is None:
                if a.text_layers or a.vision_depth:
                    continue
                raise ValueError(f'parameter {name} absent from the checkpoint header')
            if list(p.shape) != entry['shape'] or dtypes[entry['dtype']] != p.dtype:
                mismatched.append(name)
            checked += 1
        if mismatched:
            raise ValueError(f'checkpoint shape/dtype mismatch: {mismatched[:5]}')
        full = not (a.text_layers or a.vision_depth)
        if full and checked != len(header):
            raise ValueError(f'{checked} parameters vs {len(header)} checkpoint tensors')
        res['weights_checked'] = dict(parameters=checked, header_tensors=len(header), full_model=full,
                                      tied_embedding=model.lm_head.weight is
                                      model.model.language_model.embed_tokens.weight)
        if tnca_backend:
            stub.reset_phase('weights')
            params = {}
            for module in model.modules():
                for key, p in list(module._parameters.items()):
                    if p is not None:
                        if id(p) not in params:
                            params[id(p)] = torch.nn.Parameter(
                                torch.empty_strided(p.shape, p.stride(), dtype=p.dtype, device='tnca'),
                                requires_grad=False)
                        module._parameters[key] = params[id(p)]
                for key, b in list(module._buffers.items()):
                    if b is not None:
                        module._buffers[key] = torch.empty_strided(b.shape, b.stride(), dtype=b.dtype,
                                                                   device='tnca')
            res['weights_placed_bytes'] = stub.live
        wall['model'] = time.monotonic() - t

        # ---- inputs and the graph --------------------------------------------------
        t = stage('inputs')
        if stub is not None:
            stub.reset_phase('inputs')
        pos_dtype = torch.int32 if 'int32-positions' in shims else torch.int64
        tc, vc = config.text_config, config.vision_config
        gt, gh, gw = (int(x) for x in a.grid.split(','))
        merge = vc.spatial_merge_size
        patches = gt * gh * gw
        visual_tokens = patches // (merge * merge)
        lm, visual = model.model.language_model, model.model.visual
        res['geometry'] = dict(grid_thw=[gt, gh, gw], patches=patches, visual_tokens=visual_tokens,
                               prefill=a.prefill, decode_slots=a.decode_slots)

        def dev(x):
            return x.to(device)

        cache = None
        prepare = lambda: None                                          # noqa: E731
        if a.graph.startswith('text'):
            L = 1 if a.graph == 'text-decode' else a.prefill
            start = a.prefill if a.graph == 'text-decode' else 0
            input_ids = dev(torch.zeros((1, L), dtype=torch.int64))
            pos = torch.arange(start, start + L, dtype=pos_dtype)
            # Text tokens: T = H = W = position (get_rope_index for text).
            position_ids = dev(pos[None, None, :].expand(3, 1, L).contiguous())
            extra = []
            if a.graph == 'text-prefill-deepstack':
                if visual_tokens > L:
                    raise ValueError(f'{visual_tokens} visual tokens exceed the {L}-token prefill')
                mask = torch.zeros((1, L), dtype=torch.bool)
                mask[0, 1:1 + visual_tokens] = True
                extra = [dev(mask)] + [dev(torch.zeros((visual_tokens, tc.hidden_size), dtype=dtype))
                                       for _ in vc.deepstack_visual_indexes]
            if a.graph == 'text-decode':
                slots = a.prefill + a.decode_slots
                cache = StaticCache(config=tc, max_cache_len=slots)
                for layer in cache.layers:
                    kv = torch.empty((1, tc.num_key_value_heads, 1, tc.head_dim), dtype=dtype, device=device)
                    layer.lazy_initialization(kv, kv)

                def prepare():
                    for layer in cache.layers:
                        layer.cumulative_length.fill_(a.prefill)

                def step(input_ids, position_ids):
                    allowed = torch.arange(slots, dtype=position_ids.dtype,
                                           device=input_ids.device)[None, :] <= position_ids[0, 0][:, None]
                    m = torch.full((L, slots), float('-inf'), dtype=dtype, device=input_ids.device)
                    m = m.masked_fill(allowed, 0.0)[None, None]
                    h = lm(input_ids=input_ids, position_ids=position_ids, attention_mask=m,
                           past_key_values=cache, use_cache=True).last_hidden_state
                    return model.lm_head(h[:, -1:])
                res['adapter'] = ('official StaticCache, 4-D mask over the slot bucket built in-graph from the '
                                  'T row of the position ids, logits of the token')
            else:
                def step(input_ids, position_ids, *deepstack):
                    m = torch.full((L, L), float('-inf'), dtype=dtype, device=input_ids.device).triu(1)
                    kw = {}
                    if deepstack:
                        kw = dict(visual_pos_masks=deepstack[0], deepstack_visual_embeds=list(deepstack[1:]))
                    h = lm(input_ids=input_ids, position_ids=position_ids, attention_mask=m[None, None],
                           use_cache=False, **kw).last_hidden_state
                    return model.lm_head(h[:, -1:])
                res['adapter'] = ('no cache, causal 4-D mask built in-graph, logits of the last token'
                                  + (', DeepStack visual_pos_masks + features' if extra else ''))
            args = [input_ids, position_ids] + extra
        elif a.graph == 'mm-prefill':
            L = a.prefill
            if visual_tokens + 2 > L:
                raise ValueError(f'{visual_tokens} visual tokens do not fit the {L}-token prefill')
            ids = torch.zeros((1, L), dtype=torch.int64)
            ids[0, 1:1 + visual_tokens] = config.image_token_id
            token_types = torch.zeros((1, L), dtype=torch.int32)
            token_types[0, 1:1 + visual_tokens] = 1
            grid_cpu = torch.tensor([[gt, gh, gw]], dtype=torch.int64)
            # Host orchestration, as generation does it: 3-D M-RoPE positions
            # from the official get_rope_index on host tensors (python loops,
            # .tolist(); no weights).
            pos3, _deltas = model.model.get_rope_index(ids, mm_token_type_ids=token_types,
                                                       image_grid_thw=grid_cpu)
            res['mrope_positions'] = dict(shape=list(pos3.shape), max=int(pos3.max()))
            enc_args = [dev(torch.zeros((visual_tokens, tc.hidden_size), dtype=dtype))
                        for _ in range(1 + len(vc.deepstack_visual_indexes))]
            Output = modeling.BaseModelOutputWithDeepstackFeatures

            def step(input_ids, position_ids, image_embeds, *deepstack):
                m = torch.full((L, L), float('-inf'), dtype=dtype, device=input_ids.device).triu(1)
                enc = Output(pooler_output=(image_embeds,), deepstack_features=[(f,) for f in deepstack])
                h = model.model(input_ids=input_ids, position_ids=position_ids, attention_mask=m[None, None],
                                mm_encoder_outputs={'image': enc}, use_cache=False).last_hidden_state
                return model.lm_head(h[:, -1:])
            args = [dev(ids), dev(pos3.to(pos_dtype))] + enc_args
            res['adapter'] = ('multimodal prefill: Qwen3VLModel.forward with the public mm_encoder_outputs '
                              '(vision tower outputs computed by a separate graph), 3-D positions from '
                              'get_rope_index on the host, causal 4-D mask in-graph, logits of the last token')
        else:
            in_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
            pixel_values = dev(torch.zeros((patches, in_dim), dtype=torch.float32))
            grid_cpu = torch.tensor([[gt, gh, gw]], dtype=torch.int64)
            # The grid is host integer metadata (the processor's), and so are
            # the packed-attention offsets derived from it: they stay on the
            # CPU, where the official attention's `lengths.tolist()` reads them
            # without a device readback (tnca.inductor.configure: CPU tensors
            # on the native device are host data, compiled for the host).
            grid_thw = grid_cpu if a.graph == 'vision' else dev(grid_cpu)
            if a.graph == 'vision':
                kw = {}
                ii, iw = vision_utils.get_vision_interpolation_indices_and_weights(
                    grid_cpu, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
                    align_corners=visual.interpolation_align_corners, spatial_merge_size=merge)
                vpos = vision_utils.get_vision_position_ids(grid_cpu, merge).to(pos_dtype)
                cu = vision_utils.get_vision_cu_seqlens(grid_cpu)
                pre = [cu, dev(vpos), dev(ii), dev(iw)]
                res['precomputed'] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in
                                      zip(('cu_seqlens', 'position_ids', 'interp_indices', 'interp_weights'), pre)}

                def step(pixel_values, grid_thw, cu_seqlens, position_ids, interp_indices, interp_weights):
                    o = visual(pixel_values, grid_thw, cu_seqlens=cu_seqlens, position_ids=position_ids,
                               interp_indices=interp_indices, interp_weights=interp_weights)
                    return (o.pooler_output, *o.deepstack_features)
                args = [pixel_values, grid_thw] + pre
                res['adapter'] = ('vision tower with cu_seqlens, rotary position ids and interpolation '
                                  'indices/weights precomputed by transformers.vision_utils (the public '
                                  'precompute API) and passed as kwargs; grid_thw and cu_seqlens (integer '
                                  'host metadata) on the CPU, the rest on tnca')
                res['host_metadata_inputs'] = ['grid_thw', 'cu_seqlens']
            else:
                def step(pixel_values, grid_thw):
                    o = visual(pixel_values, grid_thw)
                    return (o.pooler_output, *o.deepstack_features)
                args = [pixel_values, grid_thw]
                res['adapter'] = 'vision tower as the model calls it: visual(pixel_values, grid_thw)'
        res['inputs'] = [dict(shape=list(x.shape), dtype=str(x.dtype)) for x in args]
        wall['inputs'] = time.monotonic() - t

        # ---- instrumentation (tnca) -------------------------------------------------
        kernels, failed_kernels, launches, wrappers, aten_graphs = {}, {}, [], [], []
        if tnca_backend:
            art_dir = out / 'artifacts'
            art_dir.mkdir(exist_ok=True)
            orig_precompile = H.CachingAutotuner._precompile_config

            def precompile_config(self, cfg, **kw):
                result = orig_precompile(self, cfg, **kw)
                blob = (getattr(getattr(result, 'kernel', None), 'asm', None) or {}).get('tnca')
                if blob:
                    h = mce.sha256(blob)
                    path = art_dir / f'{h}.tnca'
                    if not path.exists():
                        path.write_bytes(blob)
                    _man, segments = artifact.load(path)
                    kernels.setdefault(h, dict(kernel=self.inductor_meta.get('kernel_name'),
                                               code_bytes=len(segments.get('text') or b''),
                                               config=dict(cfg.kwargs)))
                return result
            H.CachingAutotuner._precompile_config = precompile_config
            orig_precompile_all = H.CachingAutotuner.precompile

            def precompile_all(self, *args_, **kwargs_):
                try:
                    return orig_precompile_all(self, *args_, **kwargs_)
                except Exception as e:                              # noqa: BLE001
                    name = self.inductor_meta.get('kernel_name') or getattr(self.fn, '__name__', '?')
                    diag = re.search(r'diagnostic: (.*)', str(e))
                    failed_kernels[name] = dict(kernel=name, error=f'{type(e).__name__}: {str(e)[:1500]}',
                                                diagnostic=diag.group(1).strip() if diag else None,
                                                configs=[dict(c.kwargs) for c in (self.configs or [])])
                    self._harness_failed = failed_kernels[name]['error']
            H.CachingAutotuner.precompile = precompile_all
            orig_autotuner_run = H.CachingAutotuner.run

            def autotuner_run(self, *args_, **kwargs_):
                if getattr(self, '_harness_failed', None):
                    launches.append(dict(kernel=self.inductor_meta.get('kernel_name'), failed=True))
                    return None
                return orig_autotuner_run(self, *args_, **kwargs_)
            H.CachingAutotuner.run = autotuner_run
            orig_call = driver.TNCALauncher.__call__

            def call(self, *args_, **kwargs_):
                assert os.environ['TNCA_LAUNCH'] == 'skip'
                launches.append(dict(kernel=self.metadata.name, grid=list(args_[:3])))
                return orig_call(self, *args_, **kwargs_)
            driver.TNCALauncher.__call__ = call
            write_dump = tnca.dump.Dump.write

            def distinct_dump(stages, directory, **keywords):
                key = mce.sha256(json.dumps(stages.stages, sort_keys=True, default=str).encode())
                return write_dump(stages, Path(directory) / key[:16], **keywords)
            tnca.dump.Dump.write = distinct_dump
            GraphLowering.save_output_code = staticmethod(lambda code: wrappers.append(code))
        from torch._inductor.graph import GraphLowering as GL
        orig_run = GL.run

        def run(self, *args_):
            try:
                return orig_run(self, *args_)
            finally:
                code = self.module.code
                (out / f'aten_graph_{len(aten_graphs)}.py').write_text(code)
                aten_graphs.append(dict(nodes=len(self.module.graph.nodes), ops=dict(Counter(
                    str(n.target) for n in self.module.graph.nodes if n.op == 'call_function'))))
        GL.run = run

        # ---- compile and cold call -----------------------------------------------
        t = stage('compile')
        if stub is not None:
            stub.reset_phase('compile')
        import torch._dynamo.utils as dutils
        torch._dynamo.reset()
        compiled = torch.compile(step, backend='inductor', fullgraph=True, dynamic=False)
        prepare()
        cold_error = None
        try:
            with torch.no_grad():
                result = compiled(*args)
            del result
        except Exception as e:                                       # noqa: BLE001
            cold_error = f'{type(e).__name__}: {str(e)[:3000]}'
            res['cold_call_traceback'] = traceback.format_exc()[-8000:]
        wall['compile_and_cold_call'] = time.monotonic() - t
        res['cold_call_error'] = cold_error
        res['aten_graphs'] = aten_graphs
        res['wrappers'] = len(wrappers)
        for i, src in enumerate(wrappers):
            (out / f'wrapper_{i}.py').write_text(src)
        unique_graphs = dutils.counters['stats']['unique_graphs']
        save()
        if not tnca_backend:
            res['status'] = 'compiled' if cold_error is None else 'failed'
            if cold_error:
                res['error'] = cold_error
            raise SystemExit
        if not wrappers:
            raise RuntimeError(f'compilation produced no wrapper: {cold_error}')

        # ---- steady call: refused operators, then shape-only stand-ins ------------
        t = stage('steady')
        lib = torch.library.Library('aten', 'IMPL')

        def stand_in(op_name):
            packet = getattr(torch.ops.aten, op_name.split('.')[0])
            overload = getattr(packet, op_name.split('.')[1] if '.' in op_name else 'default')

            def shape_only(*args_, **kwargs_):
                from torch.utils._pytree import tree_map
                meta = lambda v: v.to('meta') if isinstance(v, torch.Tensor) else v      # noqa: E731
                r = overload(*tree_map(meta, args_), **tree_map(meta, kwargs_))
                return tree_map(lambda v: torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device='tnca')
                                if isinstance(v, torch.Tensor) else v, r)
            lib.impl(op_name, shape_only, 'PrivateUse1')
        refused, stand_ins, steady_ok, steady_error = [], [], False, None
        for attempt in range(8):
            launches.clear()
            prepare()
            stub.reset_phase(f'steady-{attempt}')
            try:
                with torch.no_grad():
                    result = compiled(*args)
                steady_ok = True
                del result
            except NotImplementedError as e:
                m = re.search(r"Could not run '(?:aten::)?([\w.]+)'", str(e))
                op = m.group(1) if m else None
                refused.append(dict(attempt=attempt, op=op, message=str(e)[:800]))
                if op is None or op in stand_ins:
                    break
                stand_in(op)
                stand_ins.append(op)
                continue
            except Exception as e:                                   # noqa: BLE001
                steady_error = f'{type(e).__name__}: {str(e)[:2000]}'
                res['steady_traceback'] = traceback.format_exc()[-6000:]
            break
        res.update(refused_operators=refused, shape_only_stand_ins=stand_ins, steady_call_complete=steady_ok,
                   steady_error=steady_error,
                   recompiled_on_steady=dutils.counters['stats']['unique_graphs'] != unique_graphs)
        wall['steady'] = time.monotonic() - t

        # ---- audit ---------------------------------------------------------------
        fallbacks = tnca.inductor.host_fallbacks(wrappers)
        ok_launches = [x for x in launches if not x.get('failed')]
        res.update(
            failed_kernels=sorted(failed_kernels.values(), key=lambda r: r['kernel']),
            failed_kernel_launches=dict(Counter(x['kernel'] for x in launches if x.get('failed'))),
            kernels_compiled=len(kernels), distinct_kernels_launched=len({x['kernel'] for x in ok_launches}),
            launches=len(ok_launches), host_fallbacks=fallbacks,
            host_metadata_kernels=tnca.inductor.host_metadata_kernels(wrappers),
            scalar_reads=tnca.inductor.scalar_reads(wrappers),
            triton_kernels_in_wrappers=len(tnca.inductor.triton_kernels(wrappers)),
            block_fallbacks=list(tnca.inductor._BLOCK_FALLBACKS),
            unique_compiled_code_bytes=sum(k['code_bytes'] for k in kernels.values()),
            largest_kernel=max(kernels.values(), key=lambda k: k['code_bytes'], default=None),
            allocator=dict(live_peak_by_phase=stub.live_peak, extent_by_phase=stub.extent))
        if software_tree(ROOT, refresh=True) != tree:
            raise RuntimeError('software tree changed during the run')
        clean = steady_ok and not refused and not fallbacks and not failed_kernels and not cold_error
        res['status'] = 'compiled' if clean else 'compiled-with-findings'
    except SystemExit:
        pass
    except Exception as e:                                                    # noqa: BLE001
        res.update(status='failed', error=f'{type(e).__name__}: {str(e)[:4000]}', traceback=traceback.format_exc())
    wall['total'] = time.monotonic() - t0
    res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save()
    print(f'RUN {res["status"]} {a.graph} {json.dumps({k: round(v, 1) for k, v in wall.items()})}', flush=True)
    os._exit(0 if res['status'].startswith('compiled') else 1)


# -------------------------------------------------------------------- parent --
def run_child(a, graph, run_dir):
    for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump'):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
    if a.backend == 'tnca':
        env.update(mce.HARNESS)
    else:
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), MAX_JOBS=str(a.threads),
               PYTHONPATH=os.pathsep.join(str(x) for x in (ROOT, ROOT / 'python', ROOT / 'compiler/python',
                                                           ROOT / 'tools/asm')),
               TRITON_CACHE_DIR=str(run_dir / 'triton_cache'), TRITON_HOME=str(run_dir / 'triton_home'),
               TORCHINDUCTOR_CACHE_DIR=str(run_dir / 'inductor_cache'), TMPDIR=str(run_dir / 'tmp'),
               TNCA_DUMP_DIR=str(run_dir / 'dump'))
    cmd = [sys.executable, '-u', str(HERE), '--child', '--graph', graph, '--run-dir', str(run_dir),
           '--config', str(a.config), '--header', str(a.header), '--backend', a.backend, '--attn', a.attn,
           '--grid', a.grid, '--prefill', str(a.prefill), '--decode-slots', str(a.decode_slots),
           '--heap-base', str(a.heap_base), '--stand-in-heap-bytes', str(a.stand_in_heap_bytes),
           '--threads', str(a.threads)]
    for flag in ('text_layers', 'vision_depth'):
        if getattr(a, flag):
            cmd += ['--' + flag.replace('_', '-'), str(getattr(a, flag))]
    if a.shims:
        cmd += ['--shims', a.shims]
    if a.capture_dynamic:
        cmd += ['--capture-dynamic']
    if a.candidate_fixes:
        cmd += ['--candidate-fixes', a.candidate_fixes]
    watch = dict(peak_tree_rss=0, min_available=mce.mem_available(), killed=None)
    started = time.monotonic()
    with open(run_dir / 'child.log', 'w') as fh:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=fh, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            rss, pids = mce.tree_rss(proc.pid)
            avail = mce.mem_available()
            watch['peak_tree_rss'] = max(watch['peak_tree_rss'], rss)
            watch['min_available'] = min(watch['min_available'], avail)
            reason = None
            if rss > a.rss_limit_gib * GIB:
                reason = f'process-tree RSS {rss} B over {a.rss_limit_gib} GiB'
            elif avail < a.min_available_gib * GIB:
                reason = f'host MemAvailable {avail} B under {a.min_available_gib} GiB'
            elif time.monotonic() - started > a.timeout:
                reason = f'timeout after {a.timeout} s'
            if reason:
                watch['killed'] = reason
                for p in reversed(pids):
                    try:
                        os.kill(p, signal.SIGKILL)
                    except OSError:
                        pass
                proc.wait()
                break
            time.sleep(2)
    watch.update(process_wall_seconds=time.monotonic() - started, exit=proc.returncode)
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


def summary_line(res):
    keys = ('status', 'stage', 'kernels_compiled', 'launches', 'host_fallbacks', 'scalar_reads',
            'shape_only_stand_ins')
    row = {k: res.get(k) for k in keys if res.get(k) not in (None, [], {})}
    row['failed'] = [(f['kernel'], f.get('diagnostic') or f['error'][:160]) for f in res.get('failed_kernels', [])]
    if res.get('cold_call_error'):
        row['cold_call_error'] = res['cold_call_error'][:400]
    if res.get('error'):
        row['error'] = res['error'][:400]
    return json.dumps(row, default=str)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True, help='official Qwen3-VL config.json')
    p.add_argument('--header', type=Path, required=True, help='the checkpoint safetensors header (JSON)')
    p.add_argument('--out', type=Path)
    p.add_argument('--graphs', nargs='+', default=['text-prefill', 'text-decode', 'vision'], choices=GRAPHS)
    p.add_argument('--backend', default='tnca', choices=['tnca', 'cpu-inductor'])
    p.add_argument('--attn', default='eager', choices=['eager', 'sdpa'])
    p.add_argument('--text-layers', type=int, help='debug: first N text layers')
    p.add_argument('--vision-depth', type=int, help='debug: first N vision blocks (deepstack indexes kept below N)')
    p.add_argument('--grid', default='1,16,16', help='image grid T,H,W in patches')
    p.add_argument('--prefill', type=int, default=128)
    p.add_argument('--decode-slots', type=int, default=16)
    p.add_argument('--shims', default='', help='harness workarounds: ' + ', '.join(APPLICABLE_SHIMS))
    p.add_argument('--candidate-fixes', default='',
                   help='generic fix modules to install after configure(): ' + ', '.join(CANDIDATE_FIXES))
    p.add_argument('--capture-dynamic', action='store_true',
                   help='Dynamo capture_scalar_outputs + capture_dynamic_output_shape_ops')
    p.add_argument('--heap-base', type=int, default=0x20000)
    p.add_argument('--stand-in-heap-bytes', type=int, default=16 * GIB)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--rss-limit-gib', type=float, default=8)
    p.add_argument('--min-available-gib', type=float, default=40)
    p.add_argument('--timeout', type=float, default=3 * 3600)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--graph', help=argparse.SUPPRESS)
    p.add_argument('--run-dir', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        return child(a)
    if not a.out:
        p.error('--out is required')
    unknown = [n for n in a.shims.split(',') if n and n not in APPLICABLE_SHIMS]
    if unknown:
        p.error(f'unknown or inapplicable shims {unknown}')
    unknown = [n for n in a.candidate_fixes.split(',') if n and n not in CANDIDATE_FIXES]
    if unknown:
        p.error(f'unknown candidate fixes {unknown}')
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    a.config, a.header = a.config.resolve(), a.header.resolve()
    report = dict(schema='tnca-qwen3vl-compile-probe/1', harness_sha256=mce.digest(HERE), host=os.uname().nodename,
                  inputs={n: dict(path=str(getattr(a, n)), sha256=mce.digest(getattr(a, n)))
                          for n in ('config', 'header')},
                  args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()
                        if k not in ('child', 'graph', 'run_dir')},
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs={})
    lock = threading.Lock()

    def save():
        with lock:
            (a.out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    save()

    def work(graph):
        res = run_child(a, graph, a.out / graph)
        with lock:
            report['runs'][graph] = res
        print(f'{time.strftime("%H:%M:%S")} {graph}: {summary_line(res)}', flush=True)
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
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
