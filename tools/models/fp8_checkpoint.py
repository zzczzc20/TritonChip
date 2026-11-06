"""Generic loader for safetensors checkpoints that carry `_quantization_metadata` (static FP8 W8A8).

The format, as read from the file and nothing else:

    __metadata__["_quantization_metadata"] = '{"format_version": "1.0",
                                              "layers": {"<layer>": {"format": "float8_e4m3fn"}, ...}}'
    <layer>.weight        float8 tensor [out, in]          (the format names the float8 dtype)
    <layer>.weight_scale  float32, per tensor (0-d) or per output row
    <layer>.input_scale   float32, per tensor (0-d)        optional: absent = weight-only
    <layer>.bias          optional, high precision

Nothing here knows any model, module or layer name.  The quantized layers come
from the metadata; the mapping from checkpoint keys to model modules comes from
the model's own official key conversion (for diffusers: the class's
`checkpoint_mapping_fn` in `SINGLE_FILE_LOADABLE_CLASSES`, the same function
`from_single_file` uses).  The conversion is run once on a probe copy of the
checkpoint in which every quantized weight is replaced by a zero-copy int64
"row tag" view (magic | source id | row).  Whatever the converter does -- renames,
fused-QKV splits, concatenations, prefix stripping -- the tags record which
source rows each converted tensor holds, and the float8 rows and their scales
follow them.  A converter that changes tag values (arithmetic, dtype casts) or
mixes rows of differently scaled sources is refused, never guessed around.

Every converted float8 `X.weight` must land on a `torch.nn.Linear` named `X`;
it is replaced by `FP8Linear`, which keeps the float8 weight, its scale and the
static input scale, and in forward quantizes the input with the stored scale and
calls `torch._scaled_mm(..., out_dtype=<input dtype>, bias=...)`.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field

import torch
from torch import nn

FORMAT_VERSIONS = {"1.0"}
FLOAT8_FORMATS = {  # format string in the metadata -> storage dtype
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}
SCALE_KEYS = ("weight_scale", "input_scale")
# probe tags: MAGIC in bits 48..62, source index in bits 24..47, row in bits 0..23
_TAG_MAGIC, _TAG_MAGIC_SHIFT, _TAG_SHIFT = 0x5A17, 48, 24
_TAG_FIELD = (1 << _TAG_SHIFT) - 1


class UnsupportedQuantizedCheckpoint(ValueError):
    pass


# ---- the format -----------------------------------------------------------------
def read_quantization_metadata(path):
    """{layer: format} from a safetensors header; {} when the file has no such metadata."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    return parse_quantization_metadata(meta.get("_quantization_metadata"))


def parse_quantization_metadata(raw):
    if raw is None:
        return {}
    q = json.loads(raw) if isinstance(raw, str) else raw
    version = str(q.get("format_version"))
    if version not in FORMAT_VERSIONS:
        raise UnsupportedQuantizedCheckpoint(f"_quantization_metadata format_version {version!r} not supported")
    layers = {}
    for name, spec in q["layers"].items():
        fmt = spec.get("format")
        if fmt not in FLOAT8_FORMATS:
            raise UnsupportedQuantizedCheckpoint(f"layer {name}: quantization format {fmt!r} not supported")
        layers[name] = fmt
    return layers


def float8_max(dtype):
    return float(torch.finfo(dtype).max)


def quantize_static(x, scale, dtype):
    """x / scale in fp32, saturated to the float8 range, rounded to nearest-even float8."""
    fmax = float8_max(dtype)
    return (x.to(torch.float32) / scale).clamp(-fmax, fmax).to(dtype)


# ---- the module -------------------------------------------------------------------
class FP8Linear(nn.Module):
    """y = (q(x) * input_scale) @ (W8 * weight_scale)^T + bias, float8 operands, fp32 accumulate.

    mode "scaled_mm": torch._scaled_mm on the float8 operands (per-tensor scales).
    mode "emulate":   the same arithmetic spelled in fp32 ops (dequantize, matmul); for
                      devices without _scaled_mm and as a numerical cross-check.
    Without an input scale the layer is weight-only: W8 * weight_scale in the input dtype.
    """

    def __init__(self, in_features, out_features, *, bias, weight_dtype, scale_shape=(),
                 has_input_scale=True, mode="scaled_mm", device=None, bias_dtype=torch.bfloat16):
        super().__init__()
        if mode not in ("scaled_mm", "emulate"):
            raise ValueError(mode)
        self.in_features, self.out_features, self.mode = in_features, out_features, mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=weight_dtype, device=device),
                                   requires_grad=False)
        self.register_buffer("weight_scale", torch.empty(scale_shape, dtype=torch.float32, device=device))
        if has_input_scale:
            self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))
        else:
            self.input_scale = None
        self.bias = (nn.Parameter(torch.empty(out_features, dtype=bias_dtype, device=device), requires_grad=False)
                     if bias else None)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"weight={self.weight.dtype}, static_input={self.input_scale is not None}, mode={self.mode}")

    def dequantized_weight(self, dtype=torch.float32):
        ws = self.weight_scale.reshape(-1, 1) if self.weight_scale.dim() else self.weight_scale
        return (self.weight.to(torch.float32) * ws).to(dtype)

    def forward(self, x):
        out_dtype = x.dtype
        bias = None if self.bias is None else self.bias.to(out_dtype)
        if self.input_scale is None:                       # weight-only
            return nn.functional.linear(x, self.dequantized_weight(out_dtype), bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        xq = quantize_static(x2, self.input_scale, self.weight.dtype)
        if self.mode == "scaled_mm" and self.weight_scale.dim() == 0:
            y = torch._scaled_mm(xq, self.weight.t(), scale_a=self.input_scale, scale_b=self.weight_scale,
                                 bias=bias, out_dtype=out_dtype)
        else:
            y = (xq.to(torch.float32) * self.input_scale) @ self.dequantized_weight(torch.float32).t()
            if bias is not None:
                y = y + bias.to(torch.float32)
            y = y.to(out_dtype)
        return y.reshape(*lead, self.out_features)


# ---- key conversion with provenance ------------------------------------------------
@dataclass
class QuantizedTensor:
    """One converted float8 weight: its rows of one source layer, with that layer's scales."""
    source: str
    rows: torch.Tensor                 # int64 row indices into the source weight
    weight: torch.Tensor               # float8 [len(rows), in]
    weight_scale: torch.Tensor
    input_scale: torch.Tensor | None
    fmt: str


@dataclass
class ConvertedCheckpoint:
    state_dict: dict                   # converted keys -> tensors (quantized weights excluded)
    quantized: dict = field(default_factory=dict)   # converted "X.weight" -> QuantizedTensor


def split_quantized(state_dict, layers):
    """Pop each quantized layer's weight and scales out of `state_dict` (in place)."""
    sources = {}
    for layer, fmt in sorted(layers.items()):
        wkey = f"{layer}.weight"
        if wkey not in state_dict:
            raise UnsupportedQuantizedCheckpoint(f"metadata names {layer} but {wkey} is not in the checkpoint")
        w = state_dict.pop(wkey)
        if w.dtype != FLOAT8_FORMATS[fmt] or w.dim() != 2:
            raise UnsupportedQuantizedCheckpoint(f"{wkey}: {w.dtype} {tuple(w.shape)} does not match format {fmt}")
        scales = {s: state_dict.pop(f"{layer}.{s}", None) for s in SCALE_KEYS}
        if scales["weight_scale"] is None:
            raise UnsupportedQuantizedCheckpoint(f"{layer}: no weight_scale")
        ws = scales["weight_scale"].to(torch.float32)
        if ws.numel() not in (1, w.shape[0]):
            raise UnsupportedQuantizedCheckpoint(f"{layer}: weight_scale {tuple(ws.shape)} is neither per tensor nor per row")
        ins = scales["input_scale"]
        if ins is not None and ins.numel() != 1:
            raise UnsupportedQuantizedCheckpoint(f"{layer}: input_scale {tuple(ins.shape)} is not per tensor")
        sources[layer] = dict(weight=w, fmt=fmt,
                              weight_scale=ws.reshape(()) if ws.numel() == 1 else ws.reshape(-1),
                              input_scale=None if ins is None else ins.to(torch.float32).reshape(()))
    leftover = [k for k in state_dict if k.rsplit(".", 1)[-1] in SCALE_KEYS
                and k.rsplit(".", 1)[0] not in layers and state_dict[k].dtype == torch.float32]
    stray8 = [k for k, v in state_dict.items() if v.dtype in FLOAT8_FORMATS.values()]
    if stray8:
        raise UnsupportedQuantizedCheckpoint(f"float8 tensors not described by the metadata: {stray8[:5]}")
    if leftover:
        raise UnsupportedQuantizedCheckpoint(f"scale tensors of layers not in the metadata: {leftover[:5]}")
    return sources


def _tag(index, weight):
    if weight.shape[0] > _TAG_FIELD or index > _TAG_FIELD:
        raise UnsupportedQuantizedCheckpoint("too many rows or layers for the probe tag")
    rows = torch.arange(weight.shape[0], dtype=torch.int64) | (index << _TAG_SHIFT) | (_TAG_MAGIC << _TAG_MAGIC_SHIFT)
    return rows.view(-1, 1).expand(weight.shape[0], weight.shape[1])   # zero-copy, same shape


def _is_tagged(value):
    return (value.dtype == torch.int64 and value.numel() > 0
            and bool(((value.reshape(-1)[:1] >> _TAG_MAGIC_SHIFT) == _TAG_MAGIC).all()))


def convert_with_provenance(state_dict, layers, convert_fn=None):
    """Run the official key conversion once, tracking the float8 weights through it.

    convert_fn(dict) -> dict is the model's own conversion (None: keys are already the
    model's).  Returns a ConvertedCheckpoint; the input dict is consumed.
    """
    sources = split_quantized(state_dict, layers)
    names = sorted(sources)
    probe = dict(state_dict)
    for i, name in enumerate(names):
        probe[f"{name}.weight"] = _tag(i, sources[name]["weight"])
    converted = convert_fn(probe) if convert_fn is not None else probe
    out = ConvertedCheckpoint(state_dict={})
    covered = {name: torch.zeros(sources[name]["weight"].shape[0], dtype=torch.int64) for name in names}
    for key, value in converted.items():
        if not _is_tagged(value):
            if torch.is_floating_point(value) and value.numel() and abs(float(value.reshape(-1)[0])) >= 1e18:
                raise UnsupportedQuantizedCheckpoint(f"{key}: converter cast a quantized weight's probe")
            out.state_dict[key] = value
            continue
        if value.dim() != 2:
            raise UnsupportedQuantizedCheckpoint(f"{key}: converter reshaped a quantized weight")
        first = value[:, 0]
        if not bool(((first >> _TAG_MAGIC_SHIFT) == _TAG_MAGIC).all()) or not bool((value == first.view(-1, 1)).all()):
            raise UnsupportedQuantizedCheckpoint(f"{key}: converter mixed a quantized weight with other data")
        src = torch.unique((first >> _TAG_SHIFT) & _TAG_FIELD)
        if src.numel() != 1 or not 0 <= int(src) < len(names):
            raise UnsupportedQuantizedCheckpoint(f"{key}: rows of {src.numel()} quantized sources")
        name = names[int(src)]
        s = sources[name]
        rows = first & _TAG_FIELD
        if value.shape[1] != s["weight"].shape[1] or int(rows.max()) >= s["weight"].shape[0]:
            raise UnsupportedQuantizedCheckpoint(f"{key}: tag shape/range does not match {name}")
        if not key.endswith(".weight"):
            raise UnsupportedQuantizedCheckpoint(f"{key}: quantized weight converted to a non-weight key")
        covered[name].index_add_(0, rows, torch.ones_like(rows))
        w = s["weight"]
        r0 = int(rows[0])
        if torch.equal(rows, torch.arange(r0, r0 + rows.numel())):
            w8 = w[r0:r0 + rows.numel()]
        else:
            w8 = w.view(torch.uint8).index_select(0, rows).view(w.dtype)
        ws = s["weight_scale"] if s["weight_scale"].dim() == 0 else s["weight_scale"].index_select(0, rows)
        out.quantized[key] = QuantizedTensor(source=name, rows=rows, weight=w8.contiguous(), weight_scale=ws,
                                             input_scale=s["input_scale"], fmt=s["fmt"])
    for name, cov in covered.items():
        if not bool((cov == 1).all()):
            raise UnsupportedQuantizedCheckpoint(f"{name}: converter did not place every row exactly once")
    return out


# ---- building the model -------------------------------------------------------------
def _cast(t, dtype):
    return t.to(dtype) if dtype is not None and torch.is_floating_point(t) and t.dtype != dtype else t


def install_fp8_linears(model, converted, *, torch_dtype=torch.bfloat16, mode="scaled_mm",
                        keep_in_fp32_modules=()):
    """Replace each nn.Linear that receives a float8 weight by FP8Linear, then load everything.

    `model` may live on the meta device (built under accelerate.init_empty_weights).
    Missing or unexpected keys are errors.
    """
    full = {}
    for key, value in converted.state_dict.items():
        keep32 = any(m in key.split(".") for m in keep_in_fp32_modules)
        full[key] = _cast(value, torch.float32 if keep32 else torch_dtype)
    for key, q in converted.quantized.items():
        path = key[: -len(".weight")]
        parent_name, _, child = path.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child, None)
        if not isinstance(old, nn.Linear):
            raise UnsupportedQuantizedCheckpoint(f"{path}: float8 weight for a {type(old).__name__}, not nn.Linear")
        if (old.out_features, old.in_features) != tuple(q.weight.shape):
            raise UnsupportedQuantizedCheckpoint(f"{path}: weight {tuple(q.weight.shape)} vs Linear "
                                                 f"{(old.out_features, old.in_features)}")
        bias_key = f"{path}.bias"
        new = FP8Linear(old.in_features, old.out_features, bias=old.bias is not None,
                        weight_dtype=q.weight.dtype, scale_shape=tuple(q.weight_scale.shape),
                        has_input_scale=q.input_scale is not None, mode=mode, device="meta",
                        bias_dtype=full[bias_key].dtype if bias_key in full else torch_dtype)
        setattr(parent, child, new)
        full[key] = q.weight
        full[f"{path}.weight_scale"] = q.weight_scale
        if q.input_scale is not None:
            full[f"{path}.input_scale"] = q.input_scale
    result = model.load_state_dict(full, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise UnsupportedQuantizedCheckpoint(f"missing={result.missing_keys[:8]} unexpected={result.unexpected_keys[:8]}")
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta:
        raise UnsupportedQuantizedCheckpoint(f"parameters left uninitialised: {meta[:8]}")
    return model


def fp8_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, FP8Linear)}


def set_mode(model, mode):
    for m in fp8_modules(model).values():
        m.mode = mode


# ---- diffusers entry point ------------------------------------------------------------------
def diffusers_mapping_fn(model_cls):
    """The class's own single-file key conversion, exactly as `from_single_file` selects it."""
    from diffusers.loaders.single_file_model import (SINGLE_FILE_LOADABLE_CLASSES,
                                                     _get_single_file_loadable_mapping_class)
    name = _get_single_file_loadable_mapping_class(model_cls)
    if name is None:
        raise UnsupportedQuantizedCheckpoint(f"{model_cls.__name__} has no diffusers single-file mapping")
    return SINGLE_FILE_LOADABLE_CLASSES[name]["checkpoint_mapping_fn"]


def load_diffusers_single_file(model_cls, path, *, config, subfolder=None, torch_dtype=torch.bfloat16,
                               mode="scaled_mm", device="cpu", **config_kwargs):
    """`model_cls.from_single_file` for checkpoints with `_quantization_metadata`.

    config / subfolder: where the diffusers config lives (repo id or directory), as for from_single_file.
    Returns (model, report).
    """
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_model import _should_convert_state_dict_to_diffusers
    from safetensors.torch import load_file

    layers = read_quantization_metadata(path)
    if not layers:
        raise UnsupportedQuantizedCheckpoint(f"{path}: no _quantization_metadata; use from_single_file")
    cfg = model_cls.load_config(config, subfolder=subfolder, **config_kwargs)
    with init_empty_weights():
        model = model_cls.from_config(cfg)
    checkpoint = load_file(str(path))
    mapping_fn = diffusers_mapping_fn(model_cls)
    raw_keys = sorted(checkpoint)

    def convert(sd):
        if not _should_convert_state_dict_to_diffusers(model.state_dict(), sd):
            return sd
        return mapping_fn(config=cfg, checkpoint=sd)

    converted = convert_with_provenance(checkpoint, layers, convert)
    keep32 = list(getattr(model_cls, "_keep_in_fp32_modules", None) or [])
    install_fp8_linears(model, converted, torch_dtype=torch_dtype, mode=mode, keep_in_fp32_modules=keep32)
    model.eval()
    model.to(device)            # device only: a dtype .to() would convert the float8 weights
    report = dict(checkpoint_keys=len(raw_keys), quantized_layers=len(layers),
                  formats=sorted(set(layers.values())),
                  fp8_linears=len(converted.quantized),
                  mapping={k: dict(source=q.source, rows=[int(q.rows[0]), int(q.rows[-1]) + 1],
                                   contiguous=bool(torch.equal(q.rows, torch.arange(int(q.rows[0]), int(q.rows[0]) + q.rows.numel()))),
                                   weight_scale=q.weight_scale.tolist() if q.weight_scale.dim() == 0 else "per-row",
                                   input_scale=None if q.input_scale is None else q.input_scale.item())
                           for k, q in sorted(converted.quantized.items())},
                  model_dtype=str(getattr(model, "dtype", None)))
    return model, report


@contextlib.contextmanager
def no_init():
    """Skip parameter initialisation while building a model to be overwritten."""
    from accelerate import init_empty_weights
    with init_empty_weights():
        yield
