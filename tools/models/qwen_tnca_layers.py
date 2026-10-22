#!/usr/bin/env python3
"""Qwen FP8 checkpoints on tnca, layer by layer (R41 FP8 validation).

A whole-model run on the SystemC/TLM model is impractical in wall time, so the FP8 path is validated in pieces,
each on real checkpoint weights and on real activations captured from the CPU W8A8 reference run (REF_DIR is
`qwen*_run.py run --out` of the CPU reference, e.g. ~/qwen-work/out/vl/ref_fp8):

    linears   chosen FP8Linear modules alone, compiled with torch.compile(fullgraph=True), through the official
              Hub-kernel route (transformers' FP8Linear -> kernels-community/finegrained-fp8 matmul_2d), at
              M = prefill length and M = 1 (the last prefill row), against
                - the CPU reference (fp8_blockwise_ref, dot=exact, IEEE activation scale),
                - bf16 rounding of the exact W8A8 value (float64, no rounding after quantization),
                - fp8_block_reference.block_dynamic, the re-derivation in the order tnca executes.
    layer     one whole decoder layer (official module code: norms, attention or linear attention, MLP) compiled
              with fullgraph=True on tnca, FP8 linears through the same route, teacher-forced with the CPU
              reference's input hidden state; against the same layer on CPU with the reference route.
    vision    one vision block (BF16) on its captured CPU input.

Nothing here names a model beyond the spec module chosen with --model; layer/module selection is by regex.
"""
import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1] / "tools/performance_harness"))
import torch  # noqa: E402

import fp8_blockwise_ref as ref  # noqa: E402
import qwen_fp8_lib as L  # noqa: E402


def spec_of(name):
    if name == "vl":
        from qwen3vl_run import SPEC
    else:
        from qwen35_run import SPEC
    return SPEC


def tmap(v, f):
    if isinstance(v, torch.Tensor):
        return f(v)
    if isinstance(v, (tuple, list)):
        return type(v)(tmap(x, f) for x in v)
    if isinstance(v, dict):
        return {k: tmap(x, f) for k, x in v.items()}
    return v


def first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def cmp(got, want):
    got, want = got.float(), want.float()
    d = (got - want).abs()
    r = dict(bitwise=bool(torch.equal(got, want)), mismatches=int((got != want).sum()), elements=got.numel(),
             rel_l2=L.rel_l2(got, want), max_abs=float(d.max()) if d.numel() else 0.0,
             cosine=L.cosine(got, want))
    return r


def bf16_ulps(got, want):
    """max |difference| in bf16 units in the last place (same-sign values)."""
    g = got.to(torch.bfloat16).view(torch.int16).int()
    w = want.to(torch.bfloat16).view(torch.int16).int()
    return int((g - w).abs().max()) if g.numel() else 0


def setup(a):
    L.set_threads(a.threads)
    L.rss_watchdog(int(a.rss_limit_gib * L.GIB))
    if a.device == "tnca":
        dev = L.use_tnca(a.tnca_tree)
        import torch._dynamo as dynamo
        dynamo.config.automatic_dynamic_shapes = False
        dynamo.config.cache_size_limit = 64
        return dev
    return torch.device("cpu")


def reference_inputs(model, rt):
    vision = None
    if "vision_embeds" in rt:
        n_deep = sum(1 for k in rt if k.startswith("vision_deepstack_"))
        vision = (rt["vision_embeds"], *[rt[f"vision_deepstack_{i}"] for i in range(n_deep)])
    return vision


def text_layers(model):
    return model.model.language_model.layers


# ---- linears --------------------------------------------------------------------------------------------
def cmd_linears(a):
    from safetensors.torch import load_file
    import fp8_block_reference as tref
    spec = spec_of(a.model)
    dev = setup(a)
    rt = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
    model, _ = L.load_model(spec, "fp8")
    L.install_fp8_route("reference", dot="exact", scale_div="ieee")
    mods = {n: m for n, m in L.fp8_modules(model).items() if any(re.search(p, n) for p in a.modules)}
    captured = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: captured.__setitem__(n, (inp[0].detach().clone(),
                                                                                         out.detach().clone())))
             for n, m in mods.items()]
    with torch.no_grad():
        L.PrefillStage(model, all_logits=False, hidden_states=False)(
            rt["input_ids"], rt["position_ids_prefill"], L.make_cache(model, a.bucket, torch.device("cpu")),
            reference_inputs(model, rt))
    for h in hooks:
        h.remove()
    route = L.install_fp8_route(a.fp8_kernel, dot="exact", scale_div="ieee")
    rows = []
    for name, m in mods.items():
        x_full, y_hook = captured[name]
        W, S, blk = m.weight, m.weight_scale_inv, list(m.block_size)
        mod = copy.deepcopy(m).to(dev)
        fn = torch.compile(mod, fullgraph=True) if (a.device == "tnca" or a.compile) else mod
        for mode in a.modes:
            x = x_full if mode == "prefill" else x_full[:, -1:, :].contiguous()
            row = dict(module=name, mode=mode, M=int(x.numel() // x.shape[-1]), K=m.in_features, N=m.out_features)
            try:
                t0 = time.time()
                with torch.no_grad():
                    y = fn(x.to(dev)).cpu()
                row["seconds"] = round(time.time() - t0, 1)
                y_ref = ref.w8a8_block_matmul(x, W, S, blk, dot="exact", scale_div="ieee")
                exact = ref.exact_w8a8(x, W, S, blk)
                y_ord = tref.block_dynamic(x, W, S, blk, torch.bfloat16, ieee=True)
                row.update(
                    cpu_reference_equals_hook=bool(torch.equal(y_ref, y_hook)) if mode == "prefill" else None,
                    vs_cpu_reference=cmp(y, y_ref), max_ulp_vs_cpu_reference=bf16_ulps(y, y_ref),
                    vs_bf16_of_exact=cmp(y, exact.to(torch.bfloat16)),
                    vs_tnca_order_rederivation=cmp(y, y_ord),
                    rel_l2_vs_exact=L.rel_l2(y, exact), bf16_rounding_of_exact=L.rel_l2(exact.to(torch.bfloat16), exact),
                    cpu_reference_rel_l2_vs_exact=L.rel_l2(y_ref, exact),
                    cpu_reference_vs_bf16_of_exact=cmp(y_ref, exact.to(torch.bfloat16)))
                row["pass"] = (row["vs_tnca_order_rederivation"]["bitwise"]
                               and row["rel_l2_vs_exact"] <= 1.1 * row["bf16_rounding_of_exact"])
            except Exception as e:  # noqa: BLE001
                row.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:], **{"pass": False})
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("module", "mode", "M", "K", "N", "seconds", "pass")}
                             | {"rel_l2_vs_cpu": row.get("vs_cpu_reference", {}).get("rel_l2"),
                                "bitwise_cpu": row.get("vs_cpu_reference", {}).get("bitwise"),
                                "bitwise_order": row.get("vs_tnca_order_rederivation", {}).get("bitwise"),
                                "error": row.get("error")}), flush=True)
            write(a, dict(model=spec.name, device=a.device, route=route, knobs=knobs(), rows=rows))
        del mod, fn
    res = dict(model=spec.name, device=a.device, route=route, knobs=knobs(), rows=rows,
               passed=sum(r["pass"] for r in rows), total=len(rows))
    write(a, res)


# ---- one decoder layer ----------------------------------------------------------------------------------
def capture_layer(model, rt, idx, a):
    """kwargs and output of decoder layer `idx` in a cache-free CPU prefill with the reference route."""
    layer = text_layers(model)[idx]
    got = {}

    def pre(mod, args, kwargs):
        got["args"], got["kwargs"] = tmap(args, lambda t: t.detach().clone()), tmap(kwargs, lambda t: t.detach().clone())

    def post(mod, args, kwargs, out):
        got["out"] = first(out).detach().clone()
    h1 = layer.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = layer.register_forward_hook(post, with_kwargs=True)
    vision = reference_inputs(model, rt)
    with torch.no_grad():
        model(input_ids=rt["input_ids"], position_ids=rt["position_ids_prefill"], use_cache=False,
              mm_encoder_outputs=L.encoder_outputs(vision, L.vision_output_class(model)), logits_to_keep=1)
    h1.remove()
    h2.remove()
    return layer, got


def run_layer(layer, args, kwargs, dev, compiled):
    mod = copy.deepcopy(layer).to(dev)
    fn = torch.compile(mod, fullgraph=True) if compiled else mod
    with torch.no_grad():
        out = first(fn(*tmap(args, lambda t: t.to(dev)), **tmap(kwargs, lambda t: t.to(dev))))
    return out.cpu()


def cmd_layer(a):
    from safetensors.torch import load_file
    spec = spec_of(a.model)
    dev = setup(a)
    rt = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
    model, _ = L.load_model(spec, "fp8")
    L.prepare_for_compile(model)
    L.install_fp8_route("reference", dot="exact", scale_div="ieee")
    idx = a.index
    cfg = model.config.get_text_config()
    kind = (getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers)[idx]
    layer, got = capture_layer(model, rt, idx, a)
    key = lambda i: f"prefill_hidden_{i:02d}"  # noqa: E731
    h_in = rt[key(idx)]
    hs = got["kwargs"].get("hidden_states", got["args"][0] if got["args"] else None)
    res = dict(model=spec.name, layer=idx, layer_type=kind, device=a.device, knobs=knobs(),
               tokens=int(h_in.shape[1]), kwargs={k: (list(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor)
                                                    else (type(v).__name__ if not isinstance(v, (int, float, bool, str))
                                                          else v) for k, v in got["kwargs"].items()},
               cache_free_input_vs_reference_input=cmp(hs, h_in))
    if "hidden_states" in got["kwargs"]:
        got["kwargs"]["hidden_states"] = h_in
    else:
        got["args"] = (h_in, *got["args"][1:])
    # CPU reference layer (dot exact, IEEE scale) and its rounding floor (dot fp32 / CUDA's reciprocal scale)
    y_ref = run_layer(layer, got["args"], got["kwargs"], torch.device("cpu"), False)
    res["cpu_reference_vs_next_reference_hidden"] = cmp(y_ref, rt[key(idx + 1)]) if key(idx + 1) in rt else None
    floor = {}
    for tag, kw in (("dot_fp32", dict(dot="fp32")), ("scale_recip", dict(scale_div="recip"))):
        L.install_fp8_route("reference", **kw)
        floor[tag] = cmp(run_layer(layer, got["args"], got["kwargs"], torch.device("cpu"), False), y_ref)
    res["cpu_rounding_floor_vs_reference"] = floor
    res["route"] = L.install_fp8_route(a.fp8_kernel, dot="exact", scale_div="ieee")
    write(a, res)
    t0 = time.time()
    try:
        y = run_layer(layer, got["args"], got["kwargs"], dev, a.device == "tnca" or a.compile)
        res["seconds"] = round(time.time() - t0, 1)
        res["device_vs_cpu_reference"] = cmp(y, y_ref)
        res["finite"] = bool(torch.isfinite(y.float()).all())
        worst_floor = max(v["rel_l2"] for v in floor.values())
        res["pass"] = res["finite"] and res["device_vs_cpu_reference"]["rel_l2"] <= max(2 * worst_floor, 1e-2)
        res["criterion"] = "finite and rel_l2 vs the CPU W8A8 layer <= max(2x the CPU rounding floor, 1e-2)"
    except Exception as e:  # noqa: BLE001
        res.update(seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-6000:], **{"pass": False})
    write(a, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("traceback", "knobs")}, indent=1, default=str))


# ---- one vision block -----------------------------------------------------------------------------------
def cmd_vision(a):
    from safetensors.torch import load_file
    spec = spec_of(a.model)
    dev = setup(a)
    if a.device == "tnca":
        import torch._dynamo as dynamo
        dynamo.config.capture_scalar_outputs = True
    rt = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
    model, _ = L.load_model(spec, "fp8")
    visual = L.vision_tower(model)
    block = visual.blocks[a.index]
    got = {}
    h1 = block.register_forward_pre_hook(lambda m, args, kw: got.update(
        args=tmap(args, lambda t: t.detach().clone()), kwargs=tmap(kw, lambda t: t.detach().clone())), with_kwargs=True)
    h2 = block.register_forward_hook(lambda m, args, kw, out: got.update(out=first(out).detach().clone()),
                                     with_kwargs=True)
    with torch.no_grad():
        model.model.get_image_features(rt["pixel_values"], rt["image_grid_thw"])
    h1.remove()
    h2.remove()
    res = dict(model=spec.name, vision_block=a.index, device=a.device, knobs=knobs(),
               kwargs={k: (list(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                       for k, v in got["kwargs"].items()})
    t0 = time.time()
    for fullgraph in (True, False):
        try:
            mod = copy.deepcopy(block).to(dev)
            fn = torch.compile(mod, fullgraph=fullgraph) if a.device == "tnca" else mod
            with torch.no_grad():
                y = first(fn(*tmap(got["args"], lambda t: t.to(dev)), **tmap(got["kwargs"], lambda t: t.to(dev)))).cpu()
            res.update(fullgraph=fullgraph, seconds=round(time.time() - t0, 1), device_vs_cpu=cmp(y, got["out"]),
                       finite=bool(torch.isfinite(y.float()).all()))
            res["pass"] = res["finite"] and res["device_vs_cpu"]["rel_l2"] <= 1e-2
            break
        except Exception as e:  # noqa: BLE001
            res[f"error_fullgraph_{fullgraph}"] = f"{type(e).__name__}: {e}"[:3000]
            res["pass"] = False
            if a.device != "tnca":
                break
    write(a, res)
    print(json.dumps({k: v for k, v in res.items() if k != "knobs"}, indent=1, default=str))


def knobs():
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("TNCA_")}


def write(a, res):
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("linears", "layer", "vision"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=("vl", "q35"), required=True)
        p.add_argument("--teacher-force", required=True, metavar="REF_DIR")
        p.add_argument("--device", choices=("cpu", "tnca"), default="tnca")
        p.add_argument("--compile", action="store_true", help="also compile on cpu")
        p.add_argument("--fp8-kernel", choices=("official", "reference"), default="official")
        p.add_argument("--tnca-tree")
        p.add_argument("--bucket", type=int, default=256)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--rss-limit-gib", type=float, default=22.0)
        p.add_argument("--out", required=True)
        if name == "linears":
            p.add_argument("--modules", nargs="+", required=True, help="regexes over FP8Linear module names")
            p.add_argument("--modes", nargs="+", choices=("prefill", "decode"), default=["decode", "prefill"])
        else:
            p.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    dict(linears=cmd_linears, layer=cmd_layer, vision=cmd_vision)[a.cmd](a)


if __name__ == "__main__":
    main()
