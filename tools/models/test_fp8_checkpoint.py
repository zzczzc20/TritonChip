"""Tests for tools/models/fp8_checkpoint.py on toy models with the `_quantization_metadata` format.

    python tools/models/test_fp8_checkpoint.py        (or pytest)

The loader is model-agnostic; these toys exercise it three ways:
  * a plain nn.Module whose checkpoint keys are already the module's (no converter);
  * the same module behind an arbitrary converter that splits, renames and permutes rows;
  * a tiny diffusers Flux2Transformer2DModel through diffusers' own single-file conversion,
    compared with diffusers' official from_single_file on the dequantized checkpoint.
The original-layout key names in the last test are test data (a fixture for the
converter's input format), not something the loader reads.
"""
import json
import os
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_checkpoint as fc  # noqa: E402

F8 = torch.float8_e4m3fn


def quantize_layer(w, *, per_row=False, input_scale=4.0 / 448):
    amax = w.abs().amax(dim=1) if per_row else w.abs().max()
    ws = (amax / 448.0).float()
    w8 = (w.float() / (ws.view(-1, 1) if per_row else ws)).clamp(-448, 448).to(F8)
    out = {"weight": w8, "weight_scale": ws}
    if input_scale is not None:
        out["input_scale"] = torch.tensor(input_scale, dtype=torch.float32)
    return out


def write_checkpoint(path, sd, layers, fmt="float8_e4m3fn"):
    from safetensors.torch import save_file
    meta = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0", "layers": {name: {"format": fmt} for name in layers}})}
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path), metadata=meta)


def load_checkpoint(path):
    from safetensors.torch import load_file
    return load_file(str(path)), fc.read_quantization_metadata(path)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 96, bias=True), nn.GELU(), nn.Linear(96, 64, bias=False))
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 32, bias=True)
        self.keep = nn.Linear(32, 16)                       # stays high precision

    def forward(self, x):
        return self.keep(self.head(self.norm(self.enc(x) + x)))


def toy_checkpoint(quantized=("enc.0", "head"), **qkw):
    torch.manual_seed(0)
    ref = Toy()
    sd = {}
    for k, v in ref.state_dict().items():
        layer, _, p = k.rpartition(".")
        if layer in quantized and p == "weight":
            for s, t in quantize_layer(v, **qkw).items():
                sd[f"{layer}.{s}"] = t
        else:
            sd[k] = v.to(torch.bfloat16)
    return sd


def fake_quant_reference(model_sd, quantized, input_scales):
    """fp32 Toy with dequantized weights and the same static input quantization."""
    ref = Toy()
    ref.load_state_dict(model_sd)
    for layer in quantized:
        s = input_scales.get(layer)
        if s is not None:
            ref.get_submodule(layer).register_forward_pre_hook(
                lambda m, a, s=s: (fc.quantize_static(a[0], s, F8).float() * s,))
    return ref


def build(sd, layers, convert_fn=None, dtype=torch.float32, mode="scaled_mm", cls=Toy):
    with fc.no_init():
        model = cls()
    conv = fc.convert_with_provenance(dict(sd), layers, convert_fn)
    return fc.install_fp8_linears(model, conv, torch_dtype=dtype, mode=mode), conv


def test_metadata_and_identity_mapping():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "toy.safetensors")
        write_checkpoint(p, toy_checkpoint(), ["enc.0", "head"])
        sd, layers = load_checkpoint(p)
    assert layers == {"enc.0": "float8_e4m3fn", "head": "float8_e4m3fn"}
    model, conv = build(sd, layers)
    mods = fc.fp8_modules(model)
    assert sorted(mods) == ["enc.0", "head"], mods
    assert isinstance(model.enc[2], nn.Linear) and isinstance(model.keep, nn.Linear)
    assert model.enc[0].weight.dtype == F8 and model.enc[0].bias.dtype == torch.float32
    for name, m in mods.items():
        assert torch.equal(m.weight.view(torch.uint8), sd[f"{name}.weight"].view(torch.uint8))
        assert m.weight_scale.item() == sd[f"{name}.weight_scale"].item()
        assert m.input_scale.item() == sd[f"{name}.input_scale"].item()
    # forward: scaled_mm vs fp32 emulation vs an fp32 model with dequantized weights + fake-quant inputs
    x = torch.randn(3, 5, 64)
    y_mm = model(x)
    fc.set_mode(model, "emulate")
    y_em = model(x)
    dq = {k: v.float() for k, v in sd.items() if not k.endswith(("_scale",)) and v.dtype != F8}
    for name in mods:
        dq[f"{name}.weight"] = mods[name].dequantized_weight()
    ref = fake_quant_reference(dq, mods, {n: m.input_scale for n, m in mods.items()})
    y_ref = ref(x)
    assert torch.allclose(y_em, y_ref, rtol=1e-5, atol=1e-5), (y_em - y_ref).abs().max()
    assert torch.allclose(y_mm, y_ref, rtol=1e-4, atol=1e-4), (y_mm - y_ref).abs().max()
    # bf16 activations: scaled_mm emits bf16, matches the fp32 path to bf16 precision
    model_bf, _ = build(sd, layers, dtype=torch.bfloat16)
    y_bf = model_bf(x.bfloat16())
    assert y_bf.dtype == torch.bfloat16
    assert (y_bf.float() - y_ref).abs().max() < 0.05 * y_ref.abs().max()


def test_per_row_and_weight_only():
    sd = toy_checkpoint(per_row=True, input_scale=None)
    model, _ = build(sd, {"enc.0": "float8_e4m3fn", "head": "float8_e4m3fn"})
    m = model.head
    assert m.weight_scale.shape == (32,) and m.input_scale is None
    x = torch.randn(4, 64)
    w = sd["head.weight"].float() * sd["head.weight_scale"].view(-1, 1)
    assert torch.allclose(m(x), nn.functional.linear(x, w, sd["head.bias"].float()), atol=1e-5)


class Fused(nn.Module):
    """Target layout for the toy converter: q, k, v from one fused source; an up-projection whose
    halves the converter swaps; a renamed output layer."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(32, 16, bias=False), nn.Linear(32, 16, bias=False), nn.Linear(32, 16, bias=False)
        self.up = nn.Linear(32, 48, bias=True)
        self.out = nn.Linear(48, 32, bias=False)


def toy_converter(sd):
    """Not a real model's converter: split, swap halves (a row permutation), rename, strip a prefix."""
    out = {}
    for k in list(sd):
        v = sd.pop(k)
        k = k.removeprefix("model.")
        if k.startswith("qkv."):
            for name, part in zip("qkv", torch.chunk(v, 3, dim=0)):
                out[f"{name}.{k[4:]}"] = part
        elif k.startswith("fc1."):
            a, b = v.chunk(2, dim=0)
            out[f"up.{k[4:]}"] = torch.cat([b, a], dim=0)
        else:
            out[k.replace("fc2.", "out.")] = v
    return out


def fused_checkpoint():
    torch.manual_seed(1)
    raw = {"model.qkv.weight": torch.randn(48, 32) * 0.1, "model.fc1.weight": torch.randn(48, 32) * 0.1,
           "model.fc1.bias": torch.randn(48) * 0.1, "model.fc2.weight": torch.randn(32, 48) * 0.1}
    sd = {}
    for k, v in raw.items():
        layer, _, p = k.rpartition(".")
        if p == "weight":
            for s, t in quantize_layer(v).items():
                sd[f"{layer}.{s}"] = t
        else:
            sd[k] = v
    return sd, {"model.qkv": "float8_e4m3fn", "model.fc1": "float8_e4m3fn", "model.fc2": "float8_e4m3fn"}


def test_converter_provenance():
    sd, layers = fused_checkpoint()
    model, conv = build(sd, layers, toy_converter, cls=Fused)
    assert sorted(fc.fp8_modules(model)) == ["k", "out", "q", "up", "v"]
    w = sd["model.qkv.weight"].view(torch.uint8)
    for i, name in enumerate("qkv"):
        assert torch.equal(getattr(model, name).weight.view(torch.uint8), w[16 * i:16 * (i + 1)])
        assert conv.quantized[f"{name}.weight"].source == "model.qkv"
    up = sd["model.fc1.weight"].view(torch.uint8)
    assert torch.equal(model.up.weight.view(torch.uint8), torch.cat([up[24:], up[:24]]))
    assert torch.equal(model.up.bias, torch.cat([sd["model.fc1.bias"][24:], sd["model.fc1.bias"][:24]]))
    assert model.q.input_scale.item() == sd["model.qkv.input_scale"].item()


def expect_refusal(fn, needle):
    try:
        fn()
    except fc.UnsupportedQuantizedCheckpoint as e:
        assert needle in str(e), e
        return
    raise AssertionError(f"no refusal ({needle})")


def test_refusals():
    sd, layers = fused_checkpoint()
    expect_refusal(lambda: fc.parse_quantization_metadata('{"format_version": "2.0", "layers": {}}'), "format_version")
    expect_refusal(lambda: fc.parse_quantization_metadata('{"format_version": "1.0", "layers": {"a": {"format": "nvfp4"}}}'),
                   "nvfp4")
    bad = dict(sd)
    del bad["model.fc2.weight_scale"]
    expect_refusal(lambda: fc.convert_with_provenance(bad, layers, toy_converter), "no weight_scale")

    def casting(d):
        return {k: v.float() * 1 for k, v in toy_converter(d).items()}
    expect_refusal(lambda: fc.convert_with_provenance(dict(sd), layers, casting), "cast")

    def merging(d):
        out = toy_converter(d)
        out["q.weight"] = torch.cat([out["q.weight"][:8], out["up.weight"][:8]])    # rows of two scaled sources
        return out
    expect_refusal(lambda: fc.convert_with_provenance(dict(sd), layers, merging), "sources")

    def dropping(d):
        out = toy_converter(d)
        out["v.weight"] = out["v.weight"][:8]
        return out
    expect_refusal(lambda: fc.convert_with_provenance(dict(sd), layers, dropping), "every row")
    no_meta = {k: v for k, v in sd.items() if not k.startswith("model.fc2")}
    no_meta["model.fc2.weight"] = sd["model.fc2.weight"]
    expect_refusal(lambda: fc.convert_with_provenance(no_meta, {"model.qkv": "float8_e4m3fn", "model.fc1": "float8_e4m3fn"},
                                                      toy_converter), "not described")


# ---- diffusers: a tiny Flux2Transformer2DModel through the official single-file conversion ----
TINY = dict(patch_size=1, in_channels=16, num_layers=1, num_single_layers=2, attention_head_dim=32,
            num_attention_heads=2, joint_attention_dim=48, timestep_guidance_channels=32, mlp_ratio=3.0,
            axes_dims_rope=(8, 8, 8, 8), rope_theta=2000, eps=1e-6, guidance_embeds=False)


def tiny_original_checkpoint(cfg):
    """Fixture: random weights in the original (non-diffusers) single-file layout the converter reads."""
    g = torch.Generator().manual_seed(2)
    D, hd, M = cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["attention_head_dim"], int(cfg["num_attention_heads"] * cfg["attention_head_dim"] * cfg["mlp_ratio"])
    shapes = {"img_in.weight": (D, cfg["in_channels"]), "txt_in.weight": (D, cfg["joint_attention_dim"]),
              "time_in.in_layer.weight": (D, cfg["timestep_guidance_channels"]), "time_in.out_layer.weight": (D, D),
              "double_stream_modulation_img.lin.weight": (6 * D, D), "double_stream_modulation_txt.lin.weight": (6 * D, D),
              "single_stream_modulation.lin.weight": (3 * D, D), "final_layer.adaLN_modulation.1.weight": (2 * D, D),
              "final_layer.linear.weight": (cfg["in_channels"], D)}
    for i in range(cfg["num_layers"]):
        for s in ("img", "txt"):
            p = f"double_blocks.{i}.{s}"
            shapes.update({f"{p}_attn.qkv.weight": (3 * D, D), f"{p}_attn.proj.weight": (D, D),
                           f"{p}_attn.norm.query_norm.scale": (hd,), f"{p}_attn.norm.key_norm.scale": (hd,),
                           f"{p}_mlp.0.weight": (2 * M, D), f"{p}_mlp.2.weight": (D, M)})
    for i in range(cfg["num_single_layers"]):
        p = f"single_blocks.{i}"
        shapes.update({f"{p}.linear1.weight": (3 * D + 2 * M, D), f"{p}.linear2.weight": (D, D + M),
                       f"{p}.norm.query_norm.scale": (hd,), f"{p}.norm.key_norm.scale": (hd,)})
    return {k: (torch.randn(s, generator=g) * (0.5 / s[-1] ** 0.5 if len(s) == 2 else 0.1) + (1.0 if len(s) == 1 else 0.0))
            for k, s in shapes.items()}


def test_diffusers_official_conversion():
    try:
        from diffusers import Flux2Transformer2DModel
    except ImportError:
        print("skip: diffusers not installed")
        return
    raw = tiny_original_checkpoint(TINY)
    # the format: every 2-D block weight quantized, as in the published checkpoint's metadata
    qlayers = sorted(k[:-len(".weight")] for k, v in raw.items() if v.dim() == 2 and k.startswith(("double_blocks", "single_blocks")))
    q_sd, deq = {}, {}
    for k, v in raw.items():
        layer = k[:-len(".weight")]
        if layer in qlayers:
            q = quantize_layer(v, input_scale=6.0 / 448)
            q_sd.update({f"{layer}.{s}": t for s, t in q.items()})
            deq[k] = q["weight"].float() * q["weight_scale"]
        else:
            q_sd[k] = v.to(torch.bfloat16)
            deq[k] = v.to(torch.bfloat16).float()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "transformer"))
        cfg = dict(TINY, axes_dims_rope=list(TINY["axes_dims_rope"]), _class_name="Flux2Transformer2DModel")
        json.dump(cfg, open(os.path.join(d, "transformer", "config.json"), "w"))
        p = os.path.join(d, "tiny-fp8.safetensors")
        write_checkpoint(p, q_sd, qlayers)
        # 1) diffusers' own from_single_file cannot take the format (fused-QKV split of a 0-d scale)
        try:
            Flux2Transformer2DModel.from_single_file(p, config=d, subfolder="transformer", torch_dtype=torch.float32)
            official_fp8 = "loaded"
        except Exception as e:  # noqa: BLE001
            official_fp8 = f"{type(e).__name__}: {e}"
        assert "chunk" in official_fp8, official_fp8
        # 2) the generic loader, through the same conversion function
        model, report = fc.load_diffusers_single_file(Flux2Transformer2DModel, p, config=d, subfolder="transformer",
                                                      torch_dtype=torch.float32, mode="emulate")
        # 3) official from_single_file on the dequantized checkpoint (fp32)
        ref = Flux2Transformer2DModel.from_single_file(deq, config=d, subfolder="transformer", torch_dtype=torch.float32)
    mods = fc.fp8_modules(model)
    assert report["fp8_linears"] == len(mods) and len(mods) > len(qlayers)  # fused QKV became three Linears
    for name, m in mods.items():
        assert torch.equal(m.dequantized_weight(), ref.get_submodule(name).weight), name
        ref.get_submodule(name).register_forward_pre_hook(
            lambda mod, a, s=m.input_scale: (fc.quantize_static(a[0], s, F8).float() * s,))
    sd_ref = ref.state_dict()
    for k, v in model.state_dict().items():
        if k in sd_ref and not k.endswith("_scale") and v.dtype != F8:
            assert torch.equal(v, sd_ref[k]), k
    torch.manual_seed(3)
    B, L, T = 1, 16, 8
    inputs = dict(hidden_states=torch.randn(B, L, TINY["in_channels"]),
                  encoder_hidden_states=torch.randn(B, T, TINY["joint_attention_dim"]),
                  timestep=torch.tensor([0.7]), guidance=None,
                  img_ids=torch.cartesian_prod(torch.arange(1), torch.arange(4), torch.arange(4), torch.arange(1)).unsqueeze(0),
                  txt_ids=torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(T)).unsqueeze(0),
                  return_dict=False)
    with torch.no_grad():
        y = model(**inputs)[0]
        y_ref = ref(**inputs)[0]
        fc.set_mode(model, "scaled_mm")
        y_mm = model(**inputs)[0]
    err_em = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
    err_mm = ((y_mm - y_ref).abs().max() / y_ref.abs().max()).item()
    print(f"tiny flux2: {len(mods)} FP8Linear, emulate rel err {err_em:.2e}, scaled_mm rel err {err_mm:.2e}")
    assert err_em < 1e-5 and err_mm < 1e-2, (err_em, err_mm)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
