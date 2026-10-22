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
    decode    two consecutive decoder layers with their KV cache / linear-attention state on the device: a short
              text prompt is prefilled, then --steps decode steps run with a fixed bucket (one compiled decode
              graph, recompiles refused after the first decode step), each step teacher-forced with the CPU
              reference's input hidden state of the first layer; every step's outputs and the two layers' cache
              contents are compared against the whole-model CPU W8A8 run, and the host<->device transfers made
              inside each step are counted (runtime launch log + every byte read or written by the host).
    tower     the whole vision tower (all blocks, merger, deepstack mergers) on the reference image.

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
    hs = got["kwargs"].get("hidden_states", got["args"][0] if got["args"] else None)
    h_in = rt[key(idx)].reshape(hs.shape)       # the reference stores [tokens, hidden]
    res = dict(model=spec.name, layer=idx, layer_type=kind, device=a.device, knobs=knobs(),
               tokens=int(h_in.shape[-2]), kwargs={k: (list(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor)
                                                    else (type(v).__name__ if not isinstance(v, (int, float, bool, str))
                                                          else v) for k, v in got["kwargs"].items()},
               cache_free_input_vs_reference_input=cmp(hs, h_in))
    if "hidden_states" in got["kwargs"]:
        got["kwargs"]["hidden_states"] = h_in
    else:
        got["args"] = (h_in, *got["args"][1:])
    # CPU reference layer (dot exact, IEEE scale) and its rounding floor (dot fp32 / CUDA's reciprocal scale)
    y_ref = run_layer(layer, got["args"], got["kwargs"], torch.device("cpu"), False)
    res["cpu_reference_vs_next_reference_hidden"] = cmp(y_ref, rt[key(idx + 1)].reshape(y_ref.shape)) if key(idx + 1) in rt else None
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


# ---- two layers, decode with the cache on the device ----------------------------------------------------
class Pair(torch.nn.Module):
    """Two consecutive decoder layers, chained; the cache is an input (its tensors live on the device)."""

    def __init__(self, l0, l1, cache_key):
        super().__init__()
        self.l0, self.l1, self.cache_key = l0, l1, cache_key

    def forward(self, h, kw0, kw1, cache):
        h0 = first(self.l0(h, **kw0, **{self.cache_key: cache}))
        h1 = first(self.l1(h0, **kw1, **{self.cache_key: cache}))
        return h0, h1


def layer_state(cache, idx):
    sd, meta = L.cache_state(cache)
    return {k: v for k, v in sd.items() if int(k.split(".")[0]) in idx}, meta


class TransferMeter:
    """Bytes the host moves to or from tnca memory (tnca.torch_device.MEMORY: to_host, from_host, fill_, .item())
    plus the runtime launch log (per-launch staging uploads/downloads) while open."""

    def __init__(self):
        import tnca.torch_device as td
        self.mem = td.MEMORY
        self.stats = dict(host_read_calls=0, host_read_bytes=0, host_write_calls=0, host_write_bytes=0)
        self._r, self._w = self.mem.read, self.mem.write

        self.sites = {}

        def site(kind, n):
            import traceback as tb
            frames = [f for f in tb.extract_stack()[:-2] if "/tnca/torch_device/" not in f.filename][-4:]
            key = kind + " <- " + " < ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in reversed(frames))
            self.sites[key] = self.sites.get(key, 0) + int(n)

        def r(addr, n):
            self.stats["host_read_calls"] += 1
            self.stats["host_read_bytes"] += int(n)
            site("read", n)
            return self._r(addr, n)

        def w(addr, data):
            self.stats["host_write_calls"] += 1
            self.stats["host_write_bytes"] += len(data)
            site("write", len(data))
            return self._w(addr, data)
        self.mem.read, self.mem.write = r, w

    def __enter__(self):
        from tnca.triton_backend.driver import launch_log
        self._ctx = launch_log()
        self.log = self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        del self.mem.read, self.mem.write     # back to the class methods
        return False

    def summary(self):
        lg = self.log
        cyc = [e.get("device_cycles") for e in lg]
        return dict(self.stats, host_sites=self.sites, launches=len(lg),
                    staging_upload_bytes=sum(e.get("upload_bytes", 0) for e in lg),
                    staging_download_bytes=sum(e.get("download_bytes", 0) for e in lg),
                    resident_bytes=sum(e.get("resident_bytes", 0) for e in lg),
                    skipped_launches=sum(bool(e.get("skipped")) for e in lg),
                    device_cycles=sum(c for c in cyc if c) if all(c is not None for c in cyc) else None)


def cmd_decode(a):
    spec = spec_of(a.model)
    dev = setup(a)                         # registers tnca before anything imports triton (torch._dynamo does)
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from transformers.cache_utils import Cache
    cpu = torch.device("cpu")
    model, _ = L.load_model(spec, "fp8")
    L.prepare_for_compile(model)
    L.install_fp8_route("reference", dot="exact", scale_div="ieee")
    snap, _ = L.loadable_snapshot(spec)
    _proc, inputs, text = L.build_inputs(spec, snap, a.prompt, None)
    pos = L.prefill_position_ids(model, inputs)
    T = int(inputs["input_ids"].shape[1])
    if T + a.steps + 1 > a.bucket:
        raise SystemExit(f"bucket {a.bucket} < prompt {T} + {a.steps + 1}")
    idx = (a.index, a.index + 1)
    layers = text_layers(model)
    cfg = model.config.get_text_config()
    kinds = [(getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers)[i] for i in idx]
    res = dict(model=spec.name, layers=list(idx), layer_types=kinds, prompt_text=text, prompt_tokens=T,
               bucket=a.bucket, steps=a.steps, device=a.device, knobs=knobs())

    # 1. the CPU W8A8 reference: the whole model, static cache, prefill + steps; the two layers' inputs, kwargs,
    #    outputs and cache contents at every call
    cur, cache_key = {}, {}

    def pre(i):
        def f(mod, args, kwargs):
            kw = {}
            for k, v in kwargs.items():
                if isinstance(v, Cache):
                    cache_key[i] = k
                else:
                    kw[k] = tmap(v, lambda t: t.detach().clone())
            h = kw.pop("hidden_states") if "hidden_states" in kw else args[0].detach().clone()
            cur[i] = dict(h=h, kw=kw)
        return f

    def post(i):
        def f(mod, args, kwargs, out):
            cur[i]["out"] = first(out).detach().clone()
        return f
    hooks = [layers[i].register_forward_pre_hook(pre(i), with_kwargs=True) for i in idx] + \
            [layers[i].register_forward_hook(post(i), with_kwargs=True) for i in idx]
    cache = L.make_cache(model, a.bucket, cpu)
    calls, tokens = [], []

    def call(ids, p):
        cur.clear()
        with torch.no_grad():
            lg = model(input_ids=ids, position_ids=p, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        calls.append(dict(layers={i: dict(cur[i]) for i in idx}, state=layer_state(cache, idx)[0]))
        tokens.append(int(lg[0, -1].float().argmax()))
    t0 = time.time()
    call(inputs["input_ids"], pos)
    p = pos[..., -1:]
    for _ in range(a.steps):
        p = p + 1
        call(torch.tensor([[tokens[-1]]], dtype=torch.long), p)
    for h in hooks:
        h.remove()
    res["cpu_reference_seconds"] = round(time.time() - t0, 1)
    res["tokens"] = tokens
    assert cache_key[idx[0]] == cache_key[idx[1]], cache_key
    res["state_shapes"] = {k: [list(v.shape), str(v.dtype)] for k, v in calls[-1]["state"].items()}
    res["kwargs"] = {k: (list(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                     for k, v in calls[-1]["layers"][idx[0]]["kw"].items()}
    del cache

    # 2. the pair on a device with its own cache, teacher-forced with the CPU input of the first layer
    def run_pair(device, compiled, route, meter=False):
        L.install_fp8_route(route[0], **route[1])
        pair = Pair(copy.deepcopy(layers[idx[0]]), copy.deepcopy(layers[idx[1]]), cache_key[idx[0]]).to(device)
        fn = torch.compile(pair, fullgraph=True) if compiled else pair
        c = L.make_cache(model, a.bucket, device)
        for key, t in calls[0]["state"].items():        # linear-attention states: allocated up front, zero
            parts = key.split(".")
            if parts[1] in ("conv_states", "recurrent_states"):
                lay, j = c.layers[int(parts[0])], int(parts[2])
                if parts[1] == "conv_states":
                    lay.lazy_initialization(conv_states=t.to(device), state_idx=j, conv_kernel_size=t.shape[-1])
                else:
                    lay.lazy_initialization(recurrent_states=t.to(device), state_idx=j)
        rows = []
        dynamo.config.error_on_recompile = False
        for s, cl in enumerate(calls):
            l0, l1 = cl["layers"][idx[0]], cl["layers"][idx[1]]
            h = l0["h"].to(device)
            kw0, kw1 = (tmap(x["kw"], lambda t: t.to(device)) for x in (l0, l1))
            row = dict(step=s, kind="prefill" if s == 0 else "decode")
            if compiled and s >= 2:
                dynamo.config.error_on_recompile = True    # the decode graph of step 1 serves every later step
            g0 = counters["stats"]["unique_graphs"]
            t1 = time.time()
            try:
                if meter:
                    with TransferMeter() as tm, torch.no_grad():
                        y0, y1 = fn(h, kw0, kw1, c)
                        tm.mem.synchronize()
                    row["transfers_inside_step"] = tm.summary()
                else:
                    with torch.no_grad():
                        y0, y1 = fn(h, kw0, kw1, c)
            except Exception as e:  # noqa: BLE001
                row.update(error=f"{type(e).__name__}: {e}"[:3000], traceback=traceback.format_exc()[-5000:])
                rows.append(row)
                break
            finally:
                dynamo.config.error_on_recompile = False
            row["seconds"] = round(time.time() - t1, 1)
            row["new_graphs"] = counters["stats"]["unique_graphs"] - g0
            st = layer_state(c, idx)[0]
            row.update(y0=y0.cpu(), y1=y1.cpu(), state=st,
                       state_dtypes={k: str(v.dtype) for k, v in st.items()})
            rows.append(row)
        del fn, pair, c
        return rows

    def compare(rows, ref_rows=None):
        out = []
        for s, row in enumerate(rows):
            cl = calls[s]
            want = dict(y0=cl["layers"][idx[0]]["out"], y1=cl["layers"][idx[1]]["out"], **cl["state"])
            r = {k: v for k, v in row.items() if k not in ("y0", "y1", "state", "traceback")}
            if "y0" in row:
                got = dict(y0=row["y0"], y1=row["y1"], **row["state"])
                r["vs_cpu_reference"] = {k: cmp(got[k], want[k]) | dict(dtype_equal=got[k].dtype == want[k].dtype,
                                                                        finite=bool(torch.isfinite(got[k].float()).all()))
                                         for k in want}
            out.append(r)
        return out
    route_ref = ("reference", dict(dot="exact", scale_div="ieee"))
    t0 = time.time()
    res["cpu_pair_reference"] = compare(run_pair(cpu, False, route_ref))
    res["cpu_pair_reproduces_whole_model"] = all(all(v["bitwise"] for v in r["vs_cpu_reference"].values())
                                                 for r in res["cpu_pair_reference"])
    res["floor_scale_recip"] = compare(run_pair(cpu, False, ("reference", dict(dot="exact", scale_div="recip"))))
    res["floor_dot_fp32"] = compare(run_pair(cpu, False, ("reference", dict(dot="fp32", scale_div="ieee"))))
    # the same pair compiled by Inductor for the CPU (reference route): fused pointwise ops computed in fp32 and
    # rounded once, as any Inductor backend does, instead of eager's bf16 rounding after every op
    # (tnca.inductor.configure() leaves Triton templates as the only GEMM choice; the host needs ATen's)
    import torch._inductor.config as icfg
    with icfg.patch(max_autotune_gemm=False, max_autotune_gemm_backends="ATEN,TRITON,CPP",
                    max_autotune_conv_backends="ATEN,TRITON", autotune_fallback_to_aten=True):
        res["floor_cpu_inductor"] = compare(run_pair(cpu, True, route_ref))
    res["cpu_seconds"] = round(time.time() - t0, 1)
    write(a, res)
    t0 = time.time()
    res["route"] = L.install_fp8_route(a.fp8_kernel, dot="exact", scale_div="ieee")
    dev_rows = run_pair(dev, a.device == "tnca" or a.compile, (a.fp8_kernel, dict(dot="exact", scale_div="ieee")),
                        meter=a.device == "tnca")
    res["device_seconds"] = round(time.time() - t0, 1)
    res["device"] = compare(dev_rows)
    # verdict: per step and per compared tensor, rel-L2 <= max(2 x the worst CPU floor, 1e-2) and finite
    ok, table = len(res["device"]) == len(calls), []
    for s, r in enumerate(res["device"]):
        if "vs_cpu_reference" not in r:
            ok = False
            continue
        for k, v in r["vs_cpu_reference"].items():
            fls = {f: res[f][s]["vs_cpu_reference"][k]["rel_l2"]
                   if s < len(res[f]) and "vs_cpu_reference" in res[f][s] else None
                   for f in ("floor_scale_recip", "floor_dot_fp32", "floor_cpu_inductor")}
            fl = max(x for x in fls.values() if x is not None)
            good = v["finite"] and v["dtype_equal"] and v["rel_l2"] <= max(2 * fl, 1e-2)
            ok &= good
            table.append(dict(step=s, tensor=k, rel_l2=v["rel_l2"], cosine=v["cosine"], floor=fl, floors=fls,
                              ok=good))
        tr = r.get("transfers_inside_step", {})
        # steps 0 and 1 also compile (prefill graph, decode graph); every later step must move nothing
        if s >= 2 and tr and (tr["host_read_bytes"] or tr["host_write_bytes"] or tr["staging_upload_bytes"]
                              or tr["staging_download_bytes"]):
            ok = False
        if s >= 2 and r.get("new_graphs"):
            ok = False
    res["summary"] = table
    res["pass"] = bool(ok)
    res["criterion"] = ("every step: outputs of both layers and every cache tensor finite, same dtype, rel-L2 vs the "
                        "CPU W8A8 whole-model run <= max(2 x CPU floor (scale=recip, dot=fp32, CPU Inductor), 1e-2); "
                        "zero host reads/writes and zero launch staging inside every step after the two compiling "
                        "calls; no new graph after decode step 1")
    write(a, res)
    print(json.dumps(dict(pass_=res["pass"], cpu_pair_reproduces_whole_model=res["cpu_pair_reproduces_whole_model"],
                          rows=[{k: r.get(k) for k in ("step", "seconds", "new_graphs", "transfers_inside_step",
                                                        "error")} for r in res["device"]]), indent=1, default=str))
    for t in table:
        print(json.dumps(t), flush=True)


# ---- the whole vision tower -----------------------------------------------------------------------------
def cmd_tower(a):
    from safetensors.torch import load_file
    spec = spec_of(a.model)
    dev = setup(a)
    if a.device == "tnca":
        import torch._dynamo as dynamo
        dynamo.config.capture_scalar_outputs = True
    rt = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
    model, _ = L.load_model(spec, "fp8")
    visual = L.vision_tower(model)
    grid = rt["image_grid_thw"]
    pre = L.vision_precompute(visual, grid)
    args = [rt["pixel_values"], grid, pre["interp_indices"], pre["interp_weights"], pre["position_ids"],
            pre["cu_seqlens"]]
    host = {1, 5}                     # grid_thw and cu_seqlens: integer host metadata, kept on the CPU (V4)
    res = dict(model=spec.name, device=a.device, knobs=knobs(), grid_thw=grid.tolist(),
               patches=int(rt["pixel_values"].shape[0]), depth=len(visual.blocks))
    t0 = time.time()
    with torch.no_grad():
        want = L.VisionStage(visual)(*args)
        f32 = copy.deepcopy(visual).float()
        want32 = L.VisionStage(f32)(*args)
    del f32
    res["cpu_seconds"] = round(time.time() - t0, 1)
    ref_keys = ["vision_embeds"] + [f"vision_deepstack_{i}" for i in range(len(want) - 1)]
    res["cpu_equals_reference_run"] = [bool(torch.equal(x, rt[k])) for x, k in zip(want, ref_keys)]
    res["floor_bf16_vs_fp32"] = [L.rel_l2(x, y) for x, y in zip(want, want32)]
    write(a, res)
    t0 = time.time()
    try:
        stage = L.VisionStage(copy.deepcopy(visual).to(dev))
        fn = torch.compile(stage, fullgraph=True) if a.device == "tnca" else stage
        with torch.no_grad():
            got = [t.cpu() for t in fn(*[t if i in host else t.to(dev) for i, t in enumerate(args)])]
        res["seconds"] = round(time.time() - t0, 1)
        res["outputs"] = [dict(name=k, shape=list(g.shape), vs_cpu=cmp(g, w), vs_fp32_rel_l2=L.rel_l2(g, w32),
                               floor=fl, finite=bool(torch.isfinite(g.float()).all()))
                          for k, g, w, w32, fl in zip(ref_keys, got, want, want32, res["floor_bf16_vs_fp32"])]
        res["pass"] = all(o["finite"] and o["vs_cpu"]["rel_l2"] <= max(2 * o["floor"], 1e-2) for o in res["outputs"])
        res["criterion"] = "finite and rel-L2 vs the CPU bf16 tower <= max(2 x (CPU bf16 vs CPU fp32), 1e-2)"
    except Exception as e:  # noqa: BLE001
        res.update(seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}"[:3000],
                   traceback=traceback.format_exc()[-6000:], **{"pass": False})
    write(a, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("knobs", "traceback")}, indent=1, default=str))


def knobs():
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("TNCA_")}


def write(a, res):
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("linears", "layer", "vision", "decode", "tower"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=("vl", "q35"), required=True)
        p.add_argument("--teacher-force", required=name != "decode", metavar="REF_DIR")
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
        elif name == "decode":
            p.add_argument("--index", type=int, required=True, help="first of the two consecutive layers")
            p.add_argument("--steps", type=int, default=4)
            p.add_argument("--prompt", default="Name three primary colors.")
        elif name in ("layer", "vision"):
            p.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    dict(linears=cmd_linears, layer=cmd_layer, vision=cmd_vision, decode=cmd_decode, tower=cmd_tower)[a.cmd](a)


if __name__ == "__main__":
    main()
