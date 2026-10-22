"""Shared runner for transformers checkpoints in the fine-grained FP8 format (quantization_config.quant_method
"fp8": float8_e4m3fn weights with per-128x128-block `weight_scale_inv`, dynamic per-token activations), used by
qwen3vl_run.py and qwen35_run.py.  The model code is the official transformers code, loaded with
`<Class>.from_pretrained`; nothing here is specific to a model or a layer name.

FP8 handling, keyed only on the checkpoint's quantization format:
  * loading: transformers' own FineGrainedFP8 quantizer.  `--weights fp8` keeps FP8Linear (W8A8);
    `--weights bf16` sets the documented loading attribute `dequantize=True` (the quantizer's Fp8Dequantize).
    transformers flips dequantize to True by itself on any host without CUDA/XPU (validate_environment); the
    runner pins the requested value so the result does not depend on the host (pinned_fp8_quantizer).
  * compute: FP8Linear.forward -> fp8_linear -> finegrained_fp8_linear -> <kernel>.matmul_2d.  `--fp8-kernel`
    picks <kernel>:  official = transformers loads kernels-community/finegrained-fp8 itself (the Hub Triton
    kernel; it has to run on the stage's device);  reference = tools/models/fp8_blockwise_ref.py (independent
    CPU/torch-op reference of the same math, `--ref-dot exact|fp32`, `--ref-scale ieee|recip`).

Stages (each one module, placed on cpu (eager, or --cpu-compile inductor), tnca (always
torch.compile(fullgraph=True)) or cuda (eager; for the official kernel on a GPU)):
  vision   the vision tower on pixel values, with the grid-dependent index tensors precomputed on the host by
           transformers' own vision_utils helpers (they exist for exactly this: data-dependent ops before compile)
  prefill  the whole conditional-generation model on the prompt, image features passed through the official
           `mm_encoder_outputs` input, writing a StaticCache of a fixed bucket length
  decode   one token through the same model and the same StaticCache (fixed bucket: one compiled graph for every
           step); greedy argmax on the host
Host side: processor (tokenizer + image processor), 3-D RoPE positions (the model's own
_prepare_position_ids_for_generation), argmax, stopping.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fp8_blockwise_ref as ref  # noqa: E402
from flux2_klein_run import (LiveBytes, Meter, capture_graphs, consumed_bytes, digest, flop_mapping,  # noqa: E402
                             graph_dtypes, op_inventory, prepare_for_compile, require_available, rss_watchdog,
                             save_tensors, stats, tensor_key)

GIB = 2 ** 30
WINDOWS_GIB = (4, 6, 8)              # 6 and 8 GiB are the targets, 4 GiB (B40 board) is the reference column
USABLE_FACTOR = 1.25                 # R41 rule: usable capacity >= 1.25 x peak


@dataclass
class ModelSpec:
    name: str
    repo: str
    revision: str
    cls: str                           # transformers class (architectures[0] of the checkpoint)
    prompt: str
    image: bool = True
    chat_kwargs: dict = field(default_factory=dict)
    new_tokens: int = 32
    bucket: int = 256


# ---- environment ----------------------------------------------------------------------------------
def snapshot(spec):
    from huggingface_hub import constants
    path = Path(constants.HF_HUB_CACHE) / ("models--" + spec.repo.replace("/", "--")) / "snapshots" / spec.revision
    if not path.is_dir():
        raise SystemExit(f"{spec.repo}@{spec.revision} is not in {constants.HF_HUB_CACHE} (tools/models/fetch_qwen_fp8.py)")
    return path


def loadable_snapshot(spec):
    """The snapshot directory from_pretrained can read.  transformers 5.20 (utils/hub.py
    get_checkpoint_shard_files) reads index["metadata"] unconditionally; an index with only "weight_map" (as in
    lovedheart/Qwen3.5-4B-FP8) raises KeyError.  For such a checkpoint a view directory is built: every snapshot
    file symlinked unchanged, except the index, written with the same weight_map plus
    metadata.total_size (sum of the shards' tensor bytes).  Weights and configs are the cached files."""
    snap = snapshot(spec)
    idx = snap / "model.safetensors.index.json"
    if not idx.exists() or "metadata" in json.load(open(idx)):
        return snap, None
    view = Path(os.environ.get("QWEN_FP8_VIEW_DIR", HERE.parent / "snapview")) / f"{spec.repo.replace('/', '--')}--{spec.revision}"
    view.mkdir(parents=True, exist_ok=True)
    index = json.load(open(idx))
    total = 0
    for f in sorted(set(index["weight_map"].values())):
        for meta in safetensors_header(snap / f).values():
            total += meta["data_offsets"][1] - meta["data_offsets"][0]
    for f in snap.iterdir():
        link = view / f.name
        if f.name != idx.name and not link.exists():
            link.symlink_to(f.resolve())
    new = dict(metadata=dict(total_size=total), weight_map=index["weight_map"])
    (view / idx.name).write_text(json.dumps(new, indent=1))
    return view, dict(reason="index has no 'metadata' (transformers 5.20 get_checkpoint_shard_files reads it)",
                      view=str(view), added_metadata=new["metadata"],
                      original_index_sha256=hashlib.sha256(idx.read_bytes()).hexdigest())


def versions():
    import accelerate
    import huggingface_hub
    import safetensors
    import tokenizers
    import transformers
    v = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
             accelerate=accelerate.__version__, safetensors=safetensors.__version__, tokenizers=tokenizers.__version__,
             huggingface_hub=huggingface_hub.__version__, numpy=np.__version__, machine=platform.machine(),
             cpu_capability=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads())
    try:
        import torchvision
        v["torchvision"] = torchvision.__version__
    except ImportError:
        v["torchvision"] = None
    try:
        import kernels
        v["kernels"] = kernels.__version__
    except ImportError:
        v["kernels"] = None
    return v


_TNCA = False


def use_tnca(tree=None):
    """Register the tnca device from a TritonChip tree (python/, compiler/python, tools/asm, the TLM backend),
    before anything imports triton (torchenv's triton discovers a tnca backend; torch caches a failed probe)."""
    global _TNCA
    if not _TNCA:
        add_tree_to_path(tree)
        import tnca.torch_device as device
        if os.environ.get("TNCA_BACKEND", "").startswith("systemc-tlm"):
            import tlm_backend
            tlm_backend.register()
        device.register()
        _TNCA = True
    return torch.device("tnca")


def add_tree_to_path(tree=None):
    tree = Path(tree or os.environ.get("TNCA_TREE") or ROOT)
    paths = [tree / "python", tree / "python/_staged", tree / "compiler/python", tree / "tools/asm",
             tree / "tools/performance_harness/systemc"]
    sys.path[:0] = [str(p) for p in paths if p.is_dir() and str(p) not in sys.path]
    return tree


# ---- FP8 format, loading, kernel route ------------------------------------------------------------------
class UnsupportedQuantization(RuntimeError):
    pass


def fp8_format(config):
    """The checkpoint's quantization format as transformers reads it; None for an unquantized checkpoint.
    Only the fine-grained FP8 format (quant_method 'fp8') is accepted."""
    qc = getattr(config, "quantization_config", None)
    if qc is None and hasattr(config, "get_text_config"):
        qc = getattr(config.get_text_config(decoder=True), "quantization_config", None)
    if qc is None:
        return None
    qd = dict(qc) if isinstance(qc, dict) else qc.to_dict()
    if str(qd.get("quant_method")) != "fp8":
        raise UnsupportedQuantization(f"quant_method {qd.get('quant_method')!r} is not the fine-grained FP8 format")
    skip = qd.get("modules_to_not_convert") or qd.get("ignored_layers") or []
    return dict(quant_method="fp8", fmt=qd.get("fmt", "e4m3"), activation_scheme=qd.get("activation_scheme", "dynamic"),
                weight_block_size=qd.get("weight_block_size"), scale_fmt=qd.get("scale_fmt", "float"),
                modules_not_converted=len(skip))


@contextlib.contextmanager
def pinned_fp8_quantizer(record):
    """Run transformers' FineGrainedFP8 quantizer unchanged except for one thing: the `dequantize` value the
    caller asked for survives validate_environment, which otherwise sets dequantize=True whenever
    torch.cuda.is_available() and is_torch_xpu_available() are both False (pre-quantized checkpoint).
    The swap is in the quantizer registry, under the format key 'fp8'; restored on exit."""
    from transformers.quantizers import auto as qauto
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    class PinnedFineGrainedFP8HfQuantizer(FineGrainedFP8HfQuantizer):
        def validate_environment(self, *args, **kwargs):
            want = self.quantization_config.dequantize
            super().validate_environment(*args, **kwargs)
            record.update(requested_dequantize=bool(want),
                          library_validate_environment_dequantize=bool(self.quantization_config.dequantize),
                          torch_cuda_is_available=torch.cuda.is_available())
            self.quantization_config.dequantize = want

    keys = [k for k, v in qauto.AUTO_QUANTIZER_MAPPING.items() if v is FineGrainedFP8HfQuantizer]
    old = {k: qauto.AUTO_QUANTIZER_MAPPING[k] for k in keys}
    for k in keys:
        qauto.AUTO_QUANTIZER_MAPPING[k] = PinnedFineGrainedFP8HfQuantizer
    record["registry_keys"] = [str(k) for k in keys]
    try:
        yield
    finally:
        qauto.AUTO_QUANTIZER_MAPPING.update(old)


def load_model(spec, weights="fp8", dtype=torch.bfloat16):
    """Official <Class>.from_pretrained on the pinned snapshot.  weights: fp8 (FP8Linear, W8A8) or bf16
    (transformers' own dequantize=True path).  Returns (model, report)."""
    import transformers
    from transformers import AutoConfig
    snap, view_note = loadable_snapshot(spec)
    cfg = AutoConfig.from_pretrained(snap)
    fmt = fp8_format(cfg)
    if fmt is None:
        raise UnsupportedQuantization("checkpoint has no quantization_config")
    qc = dict(cfg.quantization_config) if isinstance(cfg.quantization_config, dict) else cfg.quantization_config.to_dict()
    qc["dequantize"] = weights == "bf16"
    cfg.quantization_config = qc
    record = {}
    t0 = time.perf_counter()
    with pinned_fp8_quantizer(record):
        model, info = getattr(transformers, spec.cls).from_pretrained(snap, config=cfg, dtype=dtype,
                                                                       output_loading_info=True)
    model.eval()
    report = dict(format=fmt, weights=weights, quantizer=record, load_seconds=round(time.perf_counter() - t0, 3),
                  snapshot_view=view_note,
                  loading_info={k: sorted(map(str, v)) if isinstance(v, (list, set, tuple)) else str(v) for k, v in info.items()},
                  attn_implementation=getattr(model.config, "_attn_implementation", None))
    fp8 = fp8_modules(model)
    shells = fp8_modules(model, quantized_only=False)
    report["fp8_linear_modules"] = len(fp8)
    report["fp8_linear_shells_with_unquantized_weight"] = {n: str(m.weight.dtype) for n, m in shells.items() if n not in fp8}
    if fp8:
        first = next(iter(fp8.values()))
        report["fp8_linear"] = dict(weight_dtype=str(first.weight.dtype), scale_dtype=str(first.weight_scale_inv.dtype),
                                    block_size=list(first.block_size) if first.block_size else None,
                                    activation_scheme=first.activation_scheme)
    out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    report["tied_embeddings"] = bool(out_emb is not None and out_emb.weight is in_emb.weight)
    return model, report


def fp8_modules(model, quantized_only=True):
    """FP8Linear modules transformers built from the quantization config.  quantized_only: those whose weight
    really is float8 (an FP8Linear can end up holding a tied BF16 weight, e.g. an lm_head tied to an
    unconverted embedding; its forward is then a plain F.linear)."""
    from transformers.integrations.finegrained_fp8 import FP8Linear
    return {n: m for n, m in model.named_modules()
            if isinstance(m, FP8Linear) and (not quantized_only or m.weight.element_size() == 1)}


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def checkpoint_files(snap):
    idx = snap / "model.safetensors.index.json"
    if idx.exists():
        return sorted({snap / f for f in json.load(open(idx))["weight_map"].values()})
    return sorted(snap.glob("*.safetensors"))


def verify_loaded_weights(spec, model, weights):
    """Every float8 tensor of the checkpoint against the loaded model: fp8 -> the FP8Linear holds the same bytes
    and the same scales (exactly, after transformers' dtype handling); bf16 -> the dequantized nn.Linear weight
    equals fp8_blockwise_ref.dequant_weight of the checkpoint tensors bit for bit.  Also lists float8 tensors the
    model does not load (e.g. a multi-token-prediction head the class has no module for)."""
    from safetensors import safe_open
    snap = snapshot(spec)
    params = dict(model.named_parameters())
    rows, unloaded, bad = {}, [], []
    fmt = fp8_format(model.config) or {}
    block = fmt.get("weight_block_size") or [128, 128]
    for f in checkpoint_files(snap):
        h = safetensors_header(f)
        with safe_open(str(f), "pt") as st:
            for key, meta in h.items():
                if meta["dtype"] != "F8_E4M3":
                    continue
                name = key[: -len(".weight")]
                if f"{name}.weight" not in params:
                    unloaded.append(key)
                    continue
                q = st.get_tensor(key)
                s = st.get_tensor(name + ".weight_scale_inv")
                p = params[f"{name}.weight"]
                if weights == "fp8":
                    ok_w = p.dtype == q.dtype and torch.equal(p.detach().view(torch.uint8), q.view(torch.uint8))
                    ps = params[f"{name}.weight_scale_inv"]
                    ok_s = torch.equal(ps.detach().float(), s.float())
                    rows[name] = dict(weight_bytes_equal=bool(ok_w), scale_equal=bool(ok_s), scale_ckpt_dtype=str(s.dtype),
                                      scale_loaded_dtype=str(ps.dtype))
                    if not (ok_w and ok_s):
                        bad.append(name)
                else:
                    want = ref.dequant_weight(q, s, block, p.dtype)
                    ok = torch.equal(p.detach(), want)
                    rows[name] = dict(dequant_bitwise_equal_reference=bool(ok), loaded_dtype=str(p.dtype))
                    if not ok:
                        bad.append(name)
    return dict(weights=weights, checked=len(rows), mismatches=bad, float8_tensors_not_loaded=unloaded,
                all_equal=not bad, layers=rows)


def install_fp8_route(kind, *, dot="exact", scale_div="ieee", fma=False):
    """Select what FP8Linear computes with.  Keyed on nothing but the FP8Linear class transformers builds from
    the quantization format."""
    from transformers.integrations import finegrained_fp8 as tf_fp8
    if kind == "reference":
        kernel = ref.ReferenceKernel(dot=dot, fma=fma, scale_div=scale_div)
        ref.install_into_transformers(kernel)
        return kernel.describe()
    if kind == "official":
        tf_fp8._FINEGRAINED_FP8 = None          # transformers loads the Hub kernel itself on the first FP8 call
        return dict(kind="official", loader="transformers.integrations.finegrained_fp8.load_finegrained_fp8_kernel",
                    repo="kernels-community/finegrained-fp8")
    raise ValueError(kind)


def fp8_route_in_use():
    from transformers.integrations import finegrained_fp8 as tf_fp8
    b = tf_fp8._FINEGRAINED_FP8
    if b is None:
        return None
    owner = getattr(b.matmul, "__self__", None)
    if isinstance(owner, ref.ReferenceKernel):
        return owner.describe()
    return dict(kind="official", matmul=f"{getattr(b.matmul, '__module__', '?')}.{getattr(b.matmul, '__name__', '?')}")


# ---- inputs (host) -------------------------------------------------------------------------------------
def make_image():
    """The fixed test image: 256x256 (the processor's min_pixels, a multiple of patch 16 x merge 2, so no
    resize happens: only rescale and normalize), three primitives on white."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((16, 24, 112, 120), fill=(220, 30, 30))                    # red circle
    d.rectangle((144, 32, 232, 120), fill=(30, 60, 200))                 # blue square
    d.polygon([(128, 140), (72, 236), (184, 236)], fill=(40, 160, 60))   # green triangle
    return img


def build_inputs(spec, snap, prompt, image):
    """Official processor (AutoProcessor) and its chat template."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(snap)
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
    inputs = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt", **spec.chat_kwargs)
    inputs = {k: v for k, v in inputs.items()}
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                    **spec.chat_kwargs)
    return proc, inputs, text


def prefill_position_ids(model, inputs):
    """The 4 x B x T position ids generate() builds (text row + 3-D M-RoPE rows): the model's own
    _prepare_position_ids_for_generation, called exactly as generate() calls it."""
    kw = {k: v for k, v in inputs.items() if k != "pixel_values"}
    return model._prepare_position_ids_for_generation(inputs["input_ids"], dict(kw))


def vision_precompute(visual, grid_thw):
    """The grid-dependent tensors the vision tower would compute with host-side ops (.tolist, repeat_interleave,
    loops), computed here with the same transformers helpers and the module's own attributes; the tower pops them
    from its kwargs (transformers.vision_utils: 'can be precomputed before torch.compile')."""
    from transformers import vision_utils as vu
    ii, iw = vu.get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=visual.config.spatial_merge_size)
    pos = vu.get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    cu = vu.get_vision_cu_seqlens(grid_thw)
    return dict(interp_indices=ii, interp_weights=iw, position_ids=pos, cu_seqlens=cu)


# ---- stages ---------------------------------------------------------------------------------------------
class VisionStage(nn.Module):
    """get_image_features' device part: the vision tower on the pixel values (precomputed grid tensors)."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, grid_thw, interp_indices, interp_weights, position_ids, cu_seqlens):
        out = self.visual(pixel_values.to(self.visual.dtype), grid_thw=grid_thw, interp_indices=interp_indices,
                          interp_weights=interp_weights, position_ids=position_ids, cu_seqlens=cu_seqlens,
                          return_dict=True)
        deep = getattr(out, "deepstack_features", None) or []
        return (out.pooler_output, *deep)


def vision_output_class(model):
    """The output class the model family's get_image_features returns (with deepstack features, if any)."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    return getattr(sys.modules[type(model).__module__], "BaseModelOutputWithDeepstackFeatures", BaseModelOutputWithPooling)


def encoder_outputs(vision, cls):
    """Image features in the form get_image_features returns them (one image: per-image tuples), for the
    model's documented `mm_encoder_outputs` input."""
    if vision is None:
        return None
    embeds, deep = vision[0], list(vision[1:])
    kw = dict(pooler_output=(embeds,))
    if deep:
        kw["deepstack_features"] = [(d,) for d in deep]
    return {"image": cls(**kw)}


class PrefillStage(nn.Module):
    """The whole model on the prompt (image features already computed), filling the static cache."""

    def __init__(self, model, all_logits=True, hidden_states=True):
        super().__init__()
        self.model, self.all_logits, self.hidden_states = model, all_logits, hidden_states
        self.vision_cls = vision_output_class(model)

    def forward(self, input_ids, position_ids, cache, vision=None):
        out = self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True,
                         mm_encoder_outputs=encoder_outputs(vision, self.vision_cls),
                         logits_to_keep=0 if self.all_logits else 1,
                         output_hidden_states=self.hidden_states)
        hs = tuple(out.hidden_states) if self.hidden_states else ()
        return (out.logits, *hs)


class DecodeStage(nn.Module):
    """One token through the model and the same static cache (fixed bucket)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, cache):
        out = self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
        return out.logits[:, -1, :]


def make_cache(model, bucket, device, dtype=torch.bfloat16):
    """transformers' StaticCache (hybrid models get their linear-attention layers from the config); attention
    layers allocated up front on the stage device, linear-attention states on first use."""
    from transformers import StaticCache
    cfg = model.config.get_text_config(decoder=True)
    cache = StaticCache(config=model.config, max_cache_len=bucket)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cache.early_initialization(batch_size=1, num_heads=cfg.num_key_value_heads, head_dim=head_dim, dtype=dtype,
                               device=device)
    return cache


def cache_state(cache):
    """Valid contents of every cache layer (attention: the filled positions; linear attention: conv and
    recurrent states) plus their sizes, as plain tensors on the host."""
    sd, meta = {}, {}
    for i, layer in enumerate(cache.layers):
        if torch.is_tensor(getattr(layer, "keys", None)) and getattr(layer, "is_initialized", False):
            n = int(layer.cumulative_length)
            sd[f"{i}.keys"] = layer.keys[:, :, :n].detach().cpu().clone()
            sd[f"{i}.values"] = layer.values[:, :, :n].detach().cpu().clone()
            meta[f"{i}.length"] = n
        for kind in ("conv_states", "recurrent_states"):
            for j, t in (getattr(layer, kind, None) or {}).items():
                if torch.is_tensor(t):
                    sd[f"{i}.{kind}.{j}"] = t.detach().cpu().clone()
    return sd, meta


def cache_bytes(cache):
    total, seen = 0, set()
    for layer in cache.layers:
        for t in [getattr(layer, "keys", None), getattr(layer, "values", None),
                  *(getattr(layer, "conv_states", None) or {}).values(),
                  *(getattr(layer, "recurrent_states", None) or {}).values()]:
            if torch.is_tensor(t) and id(t) not in seen:
                seen.add(id(t))
                total += t.numel() * t.element_size()
    return total


def load_cache_state(cache, sd):
    """Put a saved state into a (fresh) cache on its device: what a teacher-forced decode starts from."""
    for key, t in sd.items():
        parts = key.split(".")
        layer = cache.layers[int(parts[0])]
        if parts[1] in ("keys", "values"):
            n = t.shape[2]
            dst = getattr(layer, parts[1])
            dst.zero_()
            dst[:, :, :n].copy_(t.to(dst.device, dst.dtype))
            layer.cumulative_length.fill_(n)
        else:
            kind, j = parts[1], int(parts[2])
            dev = layer.device or cache_device(cache)
            if kind == "conv_states":
                if not layer.is_conv_states_initialized[j]:
                    layer.lazy_initialization(conv_states=t.to(dev), state_idx=j, conv_kernel_size=t.shape[-1])
                layer.conv_states[j].copy_(t.to(dev))
            else:
                if not layer.is_recurrent_states_initialized[j]:
                    layer.lazy_initialization(recurrent_states=t.to(dev), state_idx=j)
                layer.recurrent_states[j].copy_(t.to(dev))
            layer.has_previous_state[j] = True


def cache_device(cache):
    for layer in cache.layers:
        if torch.is_tensor(getattr(layer, "keys", None)):
            return layer.keys.device
    return torch.device("cpu")


# ---- placement ------------------------------------------------------------------------------------------
def to_device_except(module, dev, skip=None):
    """module.to(dev) for every submodule except the subtree `skip` (a language-model stage does not need the
    vision tower on its device).  Tied parameters stay tied (Module._apply keeps the Parameter objects)."""
    skip_ids = {id(m) for m in skip.modules()} if skip is not None else set()
    for m in module.modules():
        if id(m) not in skip_ids:
            m._apply(lambda t: t.to(dev), recurse=False)
    return module


def place(stage, device, compile_mode, tree=None, skip=None):
    """tnca: torch.compile(fullgraph=True) always; cpu: eager unless --cpu-compile inductor; cuda: eager."""
    if device == "tnca" or compile_mode == "inductor":
        prepare_for_compile(stage)
    dev = use_tnca(tree) if device == "tnca" else torch.device(device)
    to_device_except(stage, dev, skip)
    if device == "tnca" or (device == "cpu" and compile_mode == "inductor"):
        return torch.compile(stage, fullgraph=True), dev
    return stage, dev


def vision_tower(model):
    return getattr(getattr(model, "model", model), "visual", None)


def device_memory(dev):
    if dev.type == "tnca":
        return dict(device_peak=int(torch.tnca.max_memory_allocated()), device_live=int(torch.tnca.memory_allocated()))
    if dev.type == "cuda":
        return dict(device_peak=int(torch.cuda.max_memory_allocated()), device_live=int(torch.cuda.memory_allocated()))
    return {}


def reset_device_peak(dev):
    if dev.type == "tnca" and hasattr(torch.tnca, "reset_peak_memory_stats"):
        torch.tnca.reset_peak_memory_stats()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ---- run ------------------------------------------------------------------------------------------------
def set_threads(n):
    torch.set_num_threads(n)


def generation_eos(model):
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()


def topk(logits, k=16):
    v, i = logits.float().topk(k, dim=-1)
    return v, i


def cmd_run(spec, a):
    if "tnca" in (a.vision_device, a.prefill_device, a.decode_device):
        use_tnca(a.tnca_tree)
    set_threads(a.threads)
    if a.emulate_precision_casts:
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True
    rss_watchdog(int(a.rss_limit_gib * GIB))
    avail = require_available(int(a.min_available_gib * GIB))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snap = snapshot(spec)
    meter = Meter()
    route = install_fp8_route(a.fp8_kernel, dot=a.ref_dot, scale_div=a.ref_scale, fma=a.ref_fma)
    report = dict(schema="tnca-qwen-fp8-run/1", model=spec.name, repo=spec.repo, revision=spec.revision,
                  versions=versions(), mem_available_at_start=avail,
                  settings=dict(prompt=a.prompt, image=not a.no_image, new_tokens=a.new_tokens, bucket=a.bucket,
                                weights=a.weights, fp8_kernel=route, devices=dict(vision=a.vision_device,
                                prefill=a.prefill_device, decode=a.decode_device), cpu_compile=a.cpu_compile,
                                emulate_precision_casts=a.emulate_precision_casts, threads=a.threads,
                                perturb=a.perturb),
                  checks={})
    refd = None
    if a.teacher_force:
        from safetensors.torch import load_file
        refd = Path(a.teacher_force)
        rt = load_file(str(refd / "tensors.safetensors"))
        report["teacher_forced_from"] = str(refd)
    t_all = time.perf_counter()

    with meter("load_model", weights=a.weights) as row:
        model, load_report = load_model(spec, a.weights)
        row["weight_bytes"] = module_bytes(model)
    report["load"] = load_report
    if a.verify_weights:
        with meter("verify_weights"):
            vw = verify_loaded_weights(spec, model, a.weights)
        json.dump(vw, open(out / "verify_weights.json", "w"), indent=1)
        report["checks"]["loaded_weights_match_checkpoint"] = vw["all_equal"]
        report["float8_tensors_not_loaded"] = vw["float8_tensors_not_loaded"]
    if a.perturb:
        perturb_hook(model, a.perturb)

    # host: processor, positions
    image = None if a.no_image else make_image()
    if image is not None:
        image.save(out / "input.png")
        report["input_png_sha256"] = hashlib.sha256((out / "input.png").read_bytes()).hexdigest()
    proc, inputs, prompt_text = build_inputs(spec, snap, a.prompt, image)
    report["prompt_text"] = prompt_text
    pos = prefill_position_ids(model, inputs)
    tensors = {k: v for k, v in inputs.items()}
    tensors["position_ids_prefill"] = pos
    if refd is not None:
        report["checks"]["inputs_equal_reference"] = all(torch.equal(v, rt[k]) for k, v in tensors.items() if k in rt)
    T = int(inputs["input_ids"].shape[1])
    report["prompt_tokens"] = T
    if T + a.new_tokens > a.bucket:
        raise SystemExit(f"bucket {a.bucket} < prompt {T} + new tokens {a.new_tokens}")

    # 1. vision
    vision = None
    if "pixel_values" in inputs:
        visual = model.model.visual
        pre = vision_precompute(visual, inputs["image_grid_thw"])
        stage, dev = place(VisionStage(visual), a.vision_device, a.cpu_compile, a.tnca_tree)
        args = [inputs["pixel_values"], inputs["image_grid_thw"], pre["interp_indices"], pre["interp_weights"],
                pre["position_ids"], pre["cu_seqlens"]]
        reset_device_peak(dev)
        with torch.no_grad(), meter("vision", device=str(dev)) as row:
            vision = tuple(t.cpu() for t in stage(*[t.to(dev) for t in args]))
            row.update(device_memory(dev))
        if a.check_official and a.vision_device == "cpu":
            with torch.no_grad():
                off = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            want = (torch.cat(off.pooler_output), *[torch.cat(f) for f in (getattr(off, "deepstack_features", None) or [])])
            report["checks"]["vision_bitwise_equal_get_image_features"] = all(torch.equal(x, y) for x, y in zip(vision, want))
        tensors["vision_embeds"] = vision[0]
        for i, d in enumerate(vision[1:]):
            tensors[f"vision_deepstack_{i}"] = d
        if refd is not None:
            vision = (rt["vision_embeds"], *[rt[f"vision_deepstack_{i}"] for i in range(len(vision) - 1)])
        if dev.type != "cpu" or a.vision_device != a.prefill_device:
            del stage
        gc.collect()

    # 2. prefill
    stage, dev = place(PrefillStage(model, all_logits=True, hidden_states=True), a.prefill_device, a.cpu_compile,
                       a.tnca_tree, skip=vision_tower(model))
    cache = make_cache(model, a.bucket, dev)
    reset_device_peak(dev)
    with torch.no_grad(), meter("prefill", device=str(dev), tokens=T) as row:
        res = stage(inputs["input_ids"].to(dev), pos.to(dev), cache,
                    None if vision is None else tuple(t.to(dev) for t in vision))
        logits, hidden = res[0].cpu(), [h.cpu() for h in res[1:]]
        row.update(device_memory(dev), cache_bytes=cache_bytes(cache))
    del res
    report["hidden_states_count"] = len(hidden)
    full_logits = {"prefill_logits": logits[0]}
    tensors["prefill_logits_last"] = logits[0, -1:]
    for i, h in enumerate(hidden):
        tensors[f"prefill_hidden_{i:02d}"] = h[0]
    prefill_state, prefill_meta = cache_state(cache)
    report["cache_after_prefill"] = dict(prefill_meta, bytes_allocated=cache_bytes(cache))
    save_tensors(out / "cache_prefill.safetensors", prefill_state)
    if a.check_official and a.prefill_device == "cpu" and a.cpu_compile == "eager":
        with torch.no_grad(), meter("prefill_official_check"):
            off = model(**inputs, position_ids=pos, use_cache=False, output_hidden_states=True)
        report["checks"]["prefill_logits_bitwise_equal_model_forward"] = bool(torch.equal(off.logits, logits))
        report["checks"]["prefill_logits_rel_l2_vs_model_forward"] = rel_l2(logits, off.logits)
        report["checks"]["prefill_hidden_bitwise_equal_model_forward"] = [bool(torch.equal(x[0], y.cpu()[0])) for x, y in
                                                                           zip(hidden, off.hidden_states)]
        del off

    # 3. decode (greedy on the host; fixed bucket: one graph for all steps)
    dstage, ddev = place(DecodeStage(model), a.decode_device, a.cpu_compile, a.tnca_tree, skip=vision_tower(model))
    if ddev != dev:
        dcache = make_cache(model, a.bucket, ddev)
        load_cache_state(dcache, prefill_state)
        cache = dcache
    if refd is not None:                                   # teacher forcing: the reference's prefill cache
        from safetensors.torch import load_file
        cache = make_cache(model, a.bucket, ddev)
        load_cache_state(cache, load_file(str(refd / "cache_prefill.safetensors")))
    eos = generation_eos(model)
    ref_tokens = rt["generated_tokens"].tolist() if refd is not None else None
    tokens = [int(logits[0, -1].float().argmax())]
    feed = ref_tokens if ref_tokens is not None else tokens
    step_logits, p = [], pos[..., -1:]
    reset_device_peak(ddev)
    n_steps = a.new_tokens - 1
    for i in range(n_steps):
        if ref_tokens is None and tokens[-1] in eos:
            break
        if ref_tokens is not None and i >= len(ref_tokens) - 1:
            break
        p = p + 1
        tok = torch.tensor([[feed[i]]], dtype=torch.long)
        with torch.no_grad(), meter(f"decode_step_{i}", device=str(ddev)) as row:
            lg = dstage(tok.to(ddev), p.to(ddev), cache).cpu()
            row.update(device_memory(ddev))
        step_logits.append(lg[0])
        tokens.append(int(lg[0].float().argmax()))
    report["decode_steps"] = len(step_logits)
    gen = torch.tensor(tokens, dtype=torch.long)
    tensors["generated_tokens"] = gen
    if step_logits:
        sl = torch.stack(step_logits)
        full_logits["decode_logits"] = sl
        v, ix = topk(sl)
        tensors["decode_logits_top16_values"], tensors["decode_logits_top16_indices"] = v, ix
        tensors["decode_logits_step0"] = sl[0]
    report["generated_text"] = proc.batch_decode(gen[None], skip_special_tokens=True)[0]
    report["generated_tokens"] = tokens
    report["stopped_at_eos"] = bool(tokens and tokens[-1] in eos)
    report["fp8_route_in_use"] = fp8_route_in_use()

    # official generate() on the same model and route, for the token sequence (and logits) equality
    if a.check_official and "cpu" == a.prefill_device == a.decode_device and a.cpu_compile == "eager" and refd is None:
        with torch.no_grad(), meter("generate_official_static_cache"):
            g = official_generate(model, inputs, a.new_tokens, cache=make_cache(model, a.bucket, torch.device("cpu")))
        report["checks"]["tokens_equal_generate_static_cache"] = g["tokens"] == tokens
        report["checks"]["logits_bitwise_equal_generate_static_cache"] = logits_equal(g, logits, step_logits)
        with torch.no_grad(), meter("generate_official_dynamic_cache"):
            g2 = official_generate(model, inputs, a.new_tokens, cache=None)
        report["checks"]["tokens_equal_generate_dynamic_cache"] = g2["tokens"] == tokens
        report["generate_dynamic_cache_tokens"] = g2["tokens"]
        report["generate_dynamic_cache_text"] = proc.batch_decode(torch.tensor(g2["tokens"])[None], skip_special_tokens=True)[0]
        report["checks"]["logits_vs_generate_dynamic_cache"] = logits_close(g2, logits, step_logits)
    report["wall_seconds_total"] = round(time.perf_counter() - t_all, 3)
    report["stages"] = meter.stages
    report["tensors"] = {k: stats(v) for k, v in tensors.items()}
    report["full_logits"] = {k: stats(v) for k, v in full_logits.items()}
    save_tensors(out / "tensors.safetensors", tensors)
    save_tensors(out / "logits.safetensors", full_logits)
    json.dump(report, open(out / "run.json", "w"), indent=1)
    print(json.dumps(dict(text=report["generated_text"], checks=report["checks"], total=report["wall_seconds_total"]),
                     indent=1))


def official_generate(model, inputs, new_tokens, cache=None):
    kw = dict(do_sample=False, max_new_tokens=new_tokens, return_dict_in_generate=True, output_logits=True)
    if cache is not None:
        kw["past_key_values"] = cache
    g = model.generate(**inputs, **kw)
    T = inputs["input_ids"].shape[1]
    return dict(tokens=g.sequences[0, T:].tolist(), logits=[x[0] for x in g.logits])


def logits_equal(g, prefill_logits, step_logits):
    ours = [prefill_logits[0, -1]] + list(step_logits)
    n = min(len(ours), len(g["logits"]))
    return [bool(torch.equal(x.float(), y.float())) for x, y in zip(ours[:n], g["logits"][:n])]


def logits_close(g, prefill_logits, step_logits):
    ours = [prefill_logits[0, -1]] + list(step_logits)
    n = min(len(ours), len(g["logits"]))
    return [rel_l2(x, y) for x, y in zip(ours[:n], g["logits"][:n])]


def rel_l2(x, y):
    x, y = x.double(), y.double()
    return float((x - y).norm() / y.norm().clamp_min(1e-300))


def cosine(x, y):
    return float(torch.nn.functional.cosine_similarity(x.double().flatten(), y.double().flatten(), dim=0))


def module_bytes(module, include=None):
    seen, total = set(), 0
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        if include is not None and not include(name):
            continue
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def perturb_hook(model, spec_str):
    """--perturb NAME:KIND  (KIND = noise<pct> | zero): corrupt one FP8Linear's output (sensitivity study)."""
    name, kind = spec_str.split(":")
    mod = dict(model.named_modules())[name]
    g = torch.Generator().manual_seed(0)

    def hook(m, inp, out):
        if kind == "zero":
            return torch.zeros_like(out)
        pct = float(kind.removeprefix("noise")) / 100
        noise = torch.randn(out.shape, generator=g).to(out.device)
        return (out.float() * (1 + pct * noise.float())).to(out.dtype)
    mod.register_forward_hook(hook)


# ---- official generate on its own (cpu or cuda, either kernel route) --------------------------------------
def cmd_official(spec, a):
    set_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * GIB))
    require_available(int(a.min_available_gib * GIB))
    if a.device == "cuda" and a.tnca_tree:
        os.environ.setdefault("TNCA_TRITON", "0")
        add_tree_to_path(a.tnca_tree)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    route = install_fp8_route(a.fp8_kernel, dot=a.ref_dot, scale_div=a.ref_scale, fma=a.ref_fma)
    model, load_report = load_model(spec, a.weights)
    model = model.to(a.device)
    image = None if a.no_image else make_image()
    proc, inputs, _ = build_inputs(spec, snapshot(spec), a.prompt, image)
    inputs = {k: v.to(a.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        cache = make_cache(model, a.bucket, torch.device(a.device)) if a.static_cache else None
        g = official_generate(model, inputs, a.new_tokens, cache=cache)
        fw = model(**inputs, output_hidden_states=True, use_cache=False)
    secs = time.perf_counter() - t0
    tensors = dict(generated_tokens=torch.tensor(g["tokens"]), prefill_logits_last=fw.logits[0, -1:].cpu())
    for i, h in enumerate(fw.hidden_states):
        tensors[f"prefill_hidden_{i:02d}"] = h[0].cpu()
    if len(g["logits"]) > 1:
        sl = torch.stack([x.cpu() for x in g["logits"][1:]])
        tensors["decode_logits_top16_values"], tensors["decode_logits_top16_indices"] = topk(sl)
        save_tensors(out / "logits.safetensors", dict(prefill_logits=fw.logits[0].cpu(), decode_logits=sl))
    report = dict(schema="tnca-qwen-fp8-run/1", mode="official_generate", model=spec.name, versions=versions(),
                  device=a.device, settings=dict(weights=a.weights, fp8_kernel=route, static_cache=a.static_cache,
                                                 new_tokens=a.new_tokens),
                  load=load_report, generated_tokens=g["tokens"],
                  generated_text=proc.batch_decode(torch.tensor(g["tokens"])[None], skip_special_tokens=True)[0],
                  fp8_route_in_use=fp8_route_in_use(), seconds=round(secs, 3),
                  device_name=torch.cuda.get_device_name(0) if a.device == "cuda" else platform.machine(),
                  tensors={k: stats(v) for k, v in tensors.items()})
    if a.device == "cuda":
        report["cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    save_tensors(out / "tensors.safetensors", tensors)
    json.dump(report, open(out / "run.json", "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("generated_text", "seconds", "fp8_route_in_use")}, indent=1))


# ---- compare --------------------------------------------------------------------------------------------
def cmd_compare(spec, a):
    from safetensors.torch import load_file
    A, B = Path(a.a), Path(a.b)
    ta, tb = load_file(str(A / "tensors.safetensors")), load_file(str(B / "tensors.safetensors"))
    rows = {}
    for k in sorted(set(ta) & set(tb)):
        x, y = ta[k], tb[k]
        if x.shape != y.shape:
            rows[k] = dict(shape_a=list(x.shape), shape_b=list(y.shape))
            continue
        if not torch.is_floating_point(x):
            rows[k] = dict(equal=bool(torch.equal(x, y)))
            continue
        rows[k] = dict(bitwise_equal=bool(torch.equal(x, y)), rel_l2=rel_l2(x, y), cosine=cosine(x, y),
                       max_abs=float((x.double() - y.double()).abs().max()))
    res = dict(a=str(A), b=str(B), tensors=rows)
    hid = sorted(k for k in rows if k.startswith("prefill_hidden_") and "rel_l2" in rows[k])
    res["prefill_hidden_rel_l2"] = [rows[k]["rel_l2"] for k in hid]
    if "generated_tokens" in ta and "generated_tokens" in tb:
        ga, gb = ta["generated_tokens"].tolist(), tb["generated_tokens"].tolist()
        n = 0
        while n < min(len(ga), len(gb)) and ga[n] == gb[n]:
            n += 1
        res["tokens"] = dict(a=ga, b=gb, equal=ga == gb, common_prefix=n)
    la, lb = A / "logits.safetensors", B / "logits.safetensors"
    if la.exists() and lb.exists():
        fa, fb = load_file(str(la)), load_file(str(lb))
        if "prefill_logits" in fa and "prefill_logits" in fb and fa["prefill_logits"].shape == fb["prefill_logits"].shape:
            x, y = fa["prefill_logits"], fb["prefill_logits"]
            res["prefill_logits"] = dict(rel_l2=rel_l2(x, y), cosine=cosine(x, y), bitwise_equal=bool(torch.equal(x, y)),
                                         argmax_agree=float((x.float().argmax(-1) == y.float().argmax(-1)).double().mean()))
        if "decode_logits" in fa and "decode_logits" in fb:
            n = min(len(fa["decode_logits"]), len(fb["decode_logits"]))
            x, y = fa["decode_logits"][:n], fb["decode_logits"][:n]
            res["decode_logits_per_step"] = [dict(rel_l2=rel_l2(u, v), cosine=cosine(u, v), argmax_agree=bool(
                int(u.float().argmax()) == int(v.float().argmax()))) for u, v in zip(x, y)]
            res["decode_logits_max_rel_l2"] = max((r["rel_l2"] for r in res["decode_logits_per_step"]), default=None)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    summary = {k: v for k, v in res.items() if k not in ("tensors", "decode_logits_per_step")}
    print(json.dumps(summary, indent=1))


# ---- layer check ----------------------------------------------------------------------------------------
def cmd_layer_check(spec, a):
    """Every FP8Linear alone, on the input it sees in the reference prefill (teacher-forced from REF's vision
    features), run on the target (cpu eager / cpu inductor / tnca compiled / cuda) with the selected kernel route,
    against the exact W8A8 value (float64, fp8_blockwise_ref.exact_w8a8) and against the cpu reference output.
    Also per layer: the scale-rounding variant (recip vs ieee) and the dequantize-to-BF16 alternative."""
    import torch.nn.functional as F
    from safetensors.torch import load_file
    if a.device == "tnca":
        use_tnca(a.tnca_tree)
    if a.device == "cuda":
        os.environ.setdefault("TNCA_TRITON", "0")
        add_tree_to_path(a.tnca_tree)
    set_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * GIB))
    rt = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
    model, _ = load_model(spec, "fp8")
    install_fp8_route("reference", dot="exact", scale_div="ieee")
    captured = {}
    mods = fp8_modules(model)
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: captured.__setitem__(n, (inp[0].detach().clone(),
                                                                                         out.detach().clone())))
             for n, m in mods.items()]
    vision = None
    if "vision_embeds" in rt:
        n_deep = sum(1 for k in rt if k.startswith("vision_deepstack_"))
        vision = (rt["vision_embeds"], *[rt[f"vision_deepstack_{i}"] for i in range(n_deep)])
    with torch.no_grad():
        PrefillStage(model, all_logits=False, hidden_states=False)(rt["input_ids"], rt["position_ids_prefill"],
                                                                     make_cache(model, a.bucket, torch.device("cpu")), vision)
    for h in hooks:
        h.remove()
    route = install_fp8_route(a.fp8_kernel, dot=a.ref_dot, scale_div=a.ref_scale, fma=a.ref_fma)
    dev = torch.device(a.device) if a.device != "tnca" else torch.device("tnca")
    compile_it = a.device == "tnca" or a.cpu_compile == "inductor"
    rows = {}
    for name, m in mods.items():
        x, y_cpu = captured[name]
        W, S, blk = m.weight, m.weight_scale_inv, list(m.block_size)
        with torch.no_grad():
            exact = ref.exact_w8a8(x, W, S, blk)
            mod = copy.deepcopy(m).to(dev)
            fn = torch.compile(mod, fullgraph=True) if compile_it else mod
            y_dev = fn(x.to(dev)).cpu()
            y_recip = ref.w8a8_block_matmul(x, W, S, blk, scale_div="recip")
            exact_recip = ref.exact_w8a8(x, W, S, blk, scale_div="recip")
            y_deq = F.linear(x, ref.dequant_weight(W, S, blk, x.dtype))
            unq = (x.double().reshape(-1, x.shape[-1]) @ ref.dequant_weight(W, S, blk, torch.float64).t()).reshape(exact.shape)
        rows[name] = dict(shape=[int(np.prod(x.shape[:-1])), m.in_features, m.out_features],
                          bitwise_equal_cpu_reference=bool(torch.equal(y_dev, y_cpu)), rel_l2_vs_cpu=rel_l2(y_dev, y_cpu),
                          rel_l2_vs_exact=rel_l2(y_dev, exact), cpu_rel_l2_vs_exact=rel_l2(y_cpu, exact),
                          bf16_rounding_of_exact=rel_l2(exact.to(torch.bfloat16), exact),
                          recip_scale_rel_l2_vs_ieee_output=rel_l2(y_recip, y_cpu),
                          recip_scale_frac_outputs_equal=float((y_recip == y_cpu).double().mean()),
                          exact_recip_vs_exact_ieee=rel_l2(exact_recip, exact),
                          dequant_bf16_rel_l2_vs_exact_w8a8=rel_l2(y_deq, exact),
                          w8a8_rel_l2_vs_unquantized_activation=rel_l2(y_cpu, unq),
                          dequant_bf16_rel_l2_vs_unquantized_activation=rel_l2(y_deq, unq),
                          input_absmax=float(x.float().abs().max()))
        del mod, fn
    worst = max(rows.values(), key=lambda r: r["rel_l2_vs_exact"])
    agg = lambda k, f=max: f(r[k] for r in rows.values())  # noqa: E731
    result = dict(model=spec.name, device=a.device, compiled=compile_it, fp8_kernel=route, layers=len(rows),
                  all_bitwise_equal_cpu_reference=all(r["bitwise_equal_cpu_reference"] for r in rows.values()),
                  max_rel_l2_vs_cpu=agg("rel_l2_vs_cpu"), max_rel_l2_vs_exact=worst["rel_l2_vs_exact"],
                  max_cpu_rel_l2_vs_exact=agg("cpu_rel_l2_vs_exact"),
                  max_bf16_rounding_of_exact=agg("bf16_rounding_of_exact"),
                  recip_scale=dict(max_rel_l2_vs_ieee_output=agg("recip_scale_rel_l2_vs_ieee_output"),
                                   min_frac_outputs_equal=agg("recip_scale_frac_outputs_equal", min),
                                   max_exact_recip_vs_exact_ieee=agg("exact_recip_vs_exact_ieee")),
                  dequant_bf16=dict(max_rel_l2_vs_exact_w8a8=agg("dequant_bf16_rel_l2_vs_exact_w8a8"),
                                    min_rel_l2_vs_exact_w8a8=agg("dequant_bf16_rel_l2_vs_exact_w8a8", min),
                                    max_rel_l2_vs_unquantized_activation=agg("dequant_bf16_rel_l2_vs_unquantized_activation")),
                  w8a8_max_rel_l2_vs_unquantized_activation=agg("w8a8_rel_l2_vs_unquantized_activation"),
                  criterion="rel_l2_vs_exact of every layer within ~1.1x bf16_rounding_of_exact (a correct device GEMM "
                            "only adds output rounding); bitwise vs the cpu reference only if the device rounds alike",
                  rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


# ---- plan: capture, ops, FLOPs, memory ---------------------------------------------------------------------
def fp8_flop_hooks(model, counter):
    def hook(m, inp, out):
        x = inp[0]
        counter["fp8_gemm_flops"] = counter.get("fp8_gemm_flops", 0) + 2 * (x.numel() // x.shape[-1]) * m.in_features * m.out_features
    return [m.register_forward_hook(hook) for m in fp8_modules(model).values()]


def stage_measure(label, stage, inputs, weight_filter, model, cache_bytes_=0, capture=True, before_capture=None):
    """Weights the stage reads, inputs, eager activation peak, eager FLOPs (+ FP8 GEMM share), and the
    fullgraph capture (ops/dtypes, graph FLOPs, unfused graph activation peak, weights the graph consumes)."""
    from torch.utils.flop_counter import FlopCounterMode
    row = dict(stage=label)
    in_bytes = sum(t.numel() * t.element_size() for t in inputs if torch.is_tensor(t))
    in_bytes += sum(t.numel() * t.element_size() for x in inputs if isinstance(x, tuple) for t in x if torch.is_tensor(t))
    row.update(input_bytes=in_bytes, module_weight_bytes=module_bytes(model, weight_filter), kv_state_bytes=cache_bytes_)
    counter = {}
    hooks = fp8_flop_hooks(model, counter)
    fcm = FlopCounterMode(display=False, custom_mapping=flop_mapping())
    live = LiveBytes()
    with torch.no_grad(), fcm, live:
        out = stage(*inputs)
    for h in hooks:
        h.remove()
    outs = out if isinstance(out, tuple) else (out,)
    row.update(eager_flops=int(fcm.get_total_flops()), fp8_gemm_flops=int(counter.get("fp8_gemm_flops", 0)),
               eager_flops_by_op={str(k).replace("aten.", ""): int(v) for k, v in fcm.get_flop_counts().get("Global", {}).items()},
               eager_activation_peak_bytes=live.peak, output_bytes=sum(t.numel() * t.element_size() for t in outs))
    del out, outs
    if capture:
        if before_capture is not None:
            before_capture()
        try:
            graphs, runs = capture_graphs(stage, inputs)
            consumed = set().union(*(r["consumed"] for r in runs)) if runs else set()
            row.update(fullgraph="ok", graphs=len(graphs), graph_nodes=sum(len(g.graph.nodes) for g in graphs),
                       graph_flops=sum(r["flops"] for r in runs),
                       graph_activation_peak_bytes=max((r["activation_peak"] for r in runs), default=0),
                       graph_weight_bytes_read=consumed_bytes(model, consumed), ops=op_inventory(graphs),
                       dtypes=graph_dtypes(graphs))
        except Exception as e:  # noqa: BLE001 - a graph break is a finding
            row.update(fullgraph=f"FAILED: {type(e).__name__}: {str(e)[:3000]}")
    w = row.get("graph_weight_bytes_read") or row["module_weight_bytes"]
    act = row["eager_activation_peak_bytes"]
    peak = w + cache_bytes_ + in_bytes + act
    row["peak_bytes"] = peak
    row["peak_bytes_graph_bound"] = w + cache_bytes_ + in_bytes + max(act, row.get("graph_activation_peak_bytes", 0))
    row["windows"] = {f"{g}GiB": dict(allowed_peak_gib=round(g / USABLE_FACTOR, 3), fits=peak * USABLE_FACTOR <= g * GIB,
                                       fits_graph_bound=row["peak_bytes_graph_bound"] * USABLE_FACTOR <= g * GIB)
                      for g in WINDOWS_GIB}
    print(f"[plan] {label}: weights {w / GIB:.3f} GiB, kv/state {cache_bytes_ / GIB:.3f}, act {act / GIB:.3f} "
          f"(graph {row.get('graph_activation_peak_bytes', 0) / GIB:.3f}), peak {peak / GIB:.3f} GiB, "
          f"flops {row['eager_flops'] / 1e9:.1f} G (fp8 {row['fp8_gemm_flops'] / 1e9:.1f} G), "
          f"fullgraph {row.get('fullgraph', '-')[:80]}", flush=True)
    return row


def cmd_plan(spec, a):
    set_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * GIB))
    route = install_fp8_route("reference", dot=a.ref_dot, scale_div=a.ref_scale)
    model, load_report = load_model(spec, a.weights)
    snap = snapshot(spec)
    image = None if a.no_image else make_image()
    proc, inputs, _ = build_inputs(spec, snap, a.prompt, image)
    pos = prefill_position_ids(model, inputs)
    T = int(inputs["input_ids"].shape[1])
    report = dict(schema="tnca-qwen-fp8-plan/1", model=spec.name, versions=versions(), weights=a.weights, fp8_kernel=route,
                  prompt_tokens=T, bucket=a.bucket, load=load_report, stages=[],
                  note="ops: ATen graph after dynamo fullgraph capture and Inductor's decomposition table, executed "
                       "node by node on cpu. eager_*: op-by-op eager run. Activation bytes: storages created during the "
                       "stage (weights, inputs and the cache excluded). peak = weights read + kv/state + inputs + "
                       "eager activations; peak_graph_bound uses the larger of the eager and unfused-graph activations.")
    lm_filter = lambda n: not n.startswith("model.visual.")  # noqa: E731
    vision = None
    if "pixel_values" in inputs:
        visual = model.model.visual
        pre = vision_precompute(visual, inputs["image_grid_thw"])
        args = (inputs["pixel_values"], inputs["image_grid_thw"], pre["interp_indices"], pre["interp_weights"],
                pre["position_ids"], pre["cu_seqlens"])
        row = stage_measure("vision", VisionStage(visual), args, lambda n: n.startswith("model.visual."), model,
                            capture=not a.no_capture)
        row["image_tokens"] = int(inputs["image_grid_thw"].prod() // visual.spatial_merge_size ** 2)
        row["patches"] = int(inputs["pixel_values"].shape[0])
        report["stages"].append(row)
        with torch.no_grad():
            vision = VisionStage(visual)(*args)
    # prefill: warm the cache once (linear-attention states are allocated on first use), then measure on a reset cache
    stage = PrefillStage(model, all_logits=False, hidden_states=False)
    cache = make_cache(model, a.bucket, torch.device("cpu"))
    with torch.no_grad():
        stage(inputs["input_ids"], pos, cache, vision)
    cb = cache_bytes(cache)
    cache.reset()
    row = stage_measure("prefill", stage, (inputs["input_ids"], pos, cache, vision), lm_filter, model, cb,
                        capture=not a.no_capture, before_capture=cache.reset)
    row["tokens"] = T
    report["stages"].append(row)
    # decode: one step with a cache holding the prompt (bucket fixed); the capture runs on a copy of that state
    cache.reset()
    with torch.no_grad():
        stage(inputs["input_ids"], pos, cache, vision)
    state, _ = cache_state(cache)
    dstage = DecodeStage(model)
    tok = torch.tensor([[int(0)]])
    p = pos[..., -1:] + 1
    dcache = make_cache(model, a.bucket, torch.device("cpu"))
    load_cache_state(dcache, state)
    row = stage_measure("decode_step", dstage, (tok, p, dcache), lm_filter, model, cache_bytes(dcache),
                        capture=not a.no_capture, before_capture=lambda: load_cache_state(dcache, state))
    row["context"] = T
    report["stages"].append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)


# ---- the official quantizer's own behaviour (evidence) -----------------------------------------------------
def cmd_probe_quantizer(spec, a):
    """What unmodified transformers does with this checkpoint: (1) on this host, (2) on a host where
    torch.cuda.is_available() is False (simulated by patching that function and the XPU check during the load),
    (3) an FP8Linear forward on cpu tensors with transformers' own kernel loading.  Only for the record: the
    runner never patches anything."""
    import logging
    import traceback
    import transformers
    from transformers import AutoConfig
    from transformers.integrations import finegrained_fp8 as tf_fp8
    from transformers.quantizers import quantizer_finegrained_fp8 as qmod
    set_threads(a.threads)
    os.environ.setdefault("TNCA_TRITON", "0")
    add_tree_to_path(a.tnca_tree)                # triton (imported by the Hub kernel) needs the tnca package
    snap, view_note = loadable_snapshot(spec)
    cls = getattr(transformers, spec.cls)
    res = dict(model=spec.name, versions=versions(), snapshot_view=view_note)
    if view_note:
        try:
            cls.from_pretrained(snapshot(spec), dtype=torch.bfloat16)
            res["from_pretrained_on_original_snapshot"] = "loaded"
        except Exception as e:  # noqa: BLE001
            res["from_pretrained_on_original_snapshot"] = f"{type(e).__name__}: {e}"

    class Grab(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    def load(tag):
        grab = Grab()
        lg = logging.getLogger("transformers.quantizers.quantizer_finegrained_fp8")
        lg.addHandler(grab)
        try:
            m = cls.from_pretrained(snap, dtype=torch.bfloat16)
        finally:
            lg.removeHandler(grab)
        lin = [n for n, x in m.named_modules() if type(x).__name__ == "FP8Linear"]
        qc = m.config.quantization_config
        deq = qc.dequantize if hasattr(qc, "dequantize") else dict(qc).get("dequantize")
        sample = next(n for n, x in m.named_modules() if n.endswith("mlp.down_proj"))
        w = dict(m.named_parameters())[sample + ".weight"]
        return m, dict(fp8_linear_modules=len(lin), config_dequantize=deq, warnings=grab.msgs,
                       sample=sample, sample_weight_dtype=str(w.dtype), sample_module=type(dict(m.named_modules())[sample]).__name__)

    m1, res["this_host"] = load("host")
    res["this_host"]["torch_cuda_is_available"] = torch.cuda.is_available()
    # FP8Linear forward on cpu with transformers' own kernel loading (no route installed)
    tf_fp8._FINEGRAINED_FP8 = None
    lin = next(x for x in m1.modules() if type(x).__name__ == "FP8Linear")
    try:
        with torch.no_grad():
            lin(torch.randn(3, lin.in_features, dtype=torch.bfloat16))
        res["this_host"]["fp8linear_forward_cpu"] = "ran"
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        res["this_host"]["fp8linear_forward_cpu"] = dict(error=f"{type(e).__name__}: {str(e)[:500]}",
                                                         frames=[f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in tb[-4:]])
    del m1
    gc.collect()
    # simulated host without CUDA/XPU
    orig_cuda, orig_xpu = torch.cuda.is_available, qmod.is_torch_xpu_available
    torch.cuda.is_available = lambda: False
    qmod.is_torch_xpu_available = lambda: False
    try:
        m2, res["no_cuda_host_simulated"] = load("nocuda")
    finally:
        torch.cuda.is_available, qmod.is_torch_xpu_available = orig_cuda, orig_xpu
    # the dequantized weights it loads == fp8_blockwise_ref.dequant_weight of the checkpoint
    vw = verify_loaded_weights(spec, m2, "bf16")
    res["no_cuda_host_simulated"]["dequantized_weights_bitwise_equal_reference"] = vw["all_equal"]
    res["no_cuda_host_simulated"]["dequantized_layers_checked"] = vw["checked"]
    res["quantizer_source"] = dict(file=qmod.__file__, rule="validate_environment: if not dequantize and not "
                                   "torch.cuda.is_available() and not is_torch_xpu_available(): pre-quantized -> warning and "
                                   "quantization_config.dequantize = True; not pre-quantized -> RuntimeError")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])


# ---- CLI ------------------------------------------------------------------------------------------------
def main(spec: ModelSpec, doc: str):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=True):
        p.add_argument("--prompt", default=spec.prompt)
        p.add_argument("--no-image", action="store_true", default=not spec.image)
        p.add_argument("--new-tokens", type=int, default=spec.new_tokens)
        p.add_argument("--bucket", type=int, default=spec.bucket, help="static cache length (decode bucket)")
        p.add_argument("--weights", choices=("fp8", "bf16"), default="fp8",
                       help="fp8: FP8Linear (W8A8); bf16: transformers' dequantize=True")
        p.add_argument("--fp8-kernel", choices=("reference", "official"), default="reference")
        p.add_argument("--ref-dot", choices=ref.DOT_MODES, default="exact")
        p.add_argument("--ref-scale", choices=ref.SCALE_DIVS, default="ieee")
        p.add_argument("--ref-fma", action="store_true")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--rss-limit-gib", type=float, default=38.0)
        p.add_argument("--min-available-gib", type=float, default=40.0)
        p.add_argument("--tnca-tree", help="TritonChip tree whose python/ provides the tnca device (default: this repo)")
        if out:
            p.add_argument("--out", required=True)

    p = sub.add_parser("run", help="staged run (vision, prefill, decode) with per-stage time and memory")
    common(p)
    for s in ("vision", "prefill", "decode"):
        p.add_argument(f"--{s}-device", choices=("cpu", "tnca", "cuda"), default="cpu")
    p.add_argument("--cpu-compile", choices=("eager", "inductor"), default="eager")
    p.add_argument("--emulate-precision-casts", action="store_true")
    p.add_argument("--check-official", action="store_true",
                   help="cpu: stages bitwise vs get_image_features / model forward, tokens vs generate()")
    p.add_argument("--verify-weights", action="store_true", help="loaded weights vs checkpoint, bit for bit")
    p.add_argument("--teacher-force", metavar="REF_DIR",
                   help="each stage takes REF's inputs: vision features, prefill cache, the reference tokens")
    p.add_argument("--perturb", help="NAME:noise<pct>|zero on one FP8Linear output (sensitivity)")
    p = sub.add_parser("official", help="unmodified model.generate() on cpu or cuda")
    common(p)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--static-cache", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    p = sub.add_parser("layer-check", help="every FP8Linear alone on its reference input vs exact W8A8")
    common(p)
    p.add_argument("--teacher-force", required=True, metavar="REF_DIR")
    p.add_argument("--device", choices=("cpu", "tnca", "cuda"), default="cpu")
    p.add_argument("--cpu-compile", choices=("eager", "inductor"), default="eager")
    p = sub.add_parser("plan", help="per-stage fullgraph capture, op/dtype list, FLOPs, memory vs windows")
    common(p)
    p.add_argument("--no-capture", action="store_true")
    p = sub.add_parser("probe-quantizer", help="what unmodified transformers does with the checkpoint per host type")
    common(p)
    a = ap.parse_args()
    dict(run=cmd_run, official=cmd_official, compare=cmd_compare, plan=cmd_plan,
         **{"layer-check": cmd_layer_check, "probe-quantizer": cmd_probe_quantizer})[a.cmd](spec, a)
