"""FLUX.2 [klein] 4B end to end through the standard torch path, stage by stage.

    run       prompt -> text encoder -> transformer (FP8 or BF16) -> VAE -> image, each stage on
              cpu (eager: the reference; or --cpu-compile inductor) or tnca (always
              torch.compile(fullgraph=True), one compiled module per stage: text-encoder forward,
              one denoising step, VAE decode); saves prompt embeddings, every step's latents and
              noise prediction, the final latent, the image, wall time / peak RSS (and tnca device
              peak) per stage.  --teacher-force REF feeds every stage the reference's inputs, so
              each stage is compared alone; --check-official asserts cpu stages bitwise equal to
              the official pipeline functions
    official  the unmodified diffusers Flux2KleinPipeline.__call__ on cpu, same settings, saved
              the same way; `compare` against a `run` directory proves the staged runner is the
              official computation
    compare   numeric comparison of two saved runs (e.g. tnca vs cpu, fp8 vs bf16)
    plan      no device: fullgraph capture of each stage (dynamo + the Inductor decomposition
              table, recorded, executed only on meta/cpu), op/dtype inventory, FLOPs, weight and
              activation bytes at given resolutions
    verify-weights          FP8 loader output against the official BF16 transformer, layer by layer
    layer-check             every FP8Linear alone on the target device, on its reference input, against
                            cpu eager and the exact float64 W8A8 result
    probe-official-loader   what diffusers' own from_single_file does with the FP8 file

Model code is the official public code: diffusers (Flux2Transformer2DModel, AutoencoderKLFlux2,
FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline's own helpers) and transformers (Qwen3ForCausalLM,
the tokenizer).  The FP8 checkpoint goes through tools/models/fp8_checkpoint.py, which is keyed on the
checkpoint's `_quantization_metadata` format only.  Each stage module is a thin wrapper that calls the
same functions __call__ calls; `run --check-official` asserts bitwise equality with them on cpu.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fp8_checkpoint as fc  # noqa: E402

MAIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
MAIN_REV = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
FP8_REPO = "black-forest-labs/FLUX.2-klein-4b-fp8"
FP8_REV = "5b4408e59397a4a37ccb46afe426d8ed86379441"
FP8_FILE = "flux-2-klein-4b-fp8.safetensors"
# Official example (model card of the main repo): 4 steps, guidance 1.0 (distilled), seed 0
DEFAULT_PROMPT = "A cat holding a sign that says hello world"
DEFAULT_STEPS = 4
DEFAULT_SEED = 0
TE_LAYERS = (9, 18, 27)          # Flux2KleinPipeline default text_encoder_out_layers
MAX_SEQ = 512                    # Flux2KleinPipeline default max_sequence_length


# ---- environment -----------------------------------------------------------------------------
def snapshot(repo, rev):
    """The pinned revision's directory in the HF cache (only the files this runner needs are downloaded)."""
    from huggingface_hub import constants
    path = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    if not path.is_dir():
        raise SystemExit(f"{repo}@{rev} is not in {constants.HF_HUB_CACHE}")
    return path


def versions():
    import diffusers
    import safetensors
    import transformers
    import accelerate
    import huggingface_hub
    return dict(python=platform.python_version(), torch=torch.__version__, diffusers=diffusers.__version__,
                transformers=transformers.__version__, accelerate=accelerate.__version__,
                safetensors=safetensors.__version__, huggingface_hub=huggingface_hub.__version__,
                numpy=np.__version__, machine=platform.machine(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
                threads=torch.get_num_threads())


def _status(field):
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    return 0


class Meter:
    """Wall time and peak RSS (VmHWM reset per stage through /proc/self/clear_refs) of each stage."""

    def __init__(self):
        self.stages = []

    @contextlib.contextmanager
    def __call__(self, name, **extra):
        gc.collect()
        try:
            open("/proc/self/clear_refs", "w").write("5")
            reset = True
        except OSError:
            reset = False
        rss0, t0 = _status("VmRSS"), time.perf_counter()
        row = dict(stage=name, **extra)
        try:
            yield row
        finally:
            row.update(seconds=round(time.perf_counter() - t0, 3), rss_before=rss0, rss_after=_status("VmRSS"),
                       peak_rss=_status("VmHWM"), peak_rss_is_stage_local=reset)
            self.stages.append(row)
            print(f"[stage] {name}: {row['seconds']:.2f}s peak_rss={row['peak_rss'] / 2**30:.2f}GiB", flush=True)


def rss_watchdog(limit_bytes):
    def watch():
        while True:
            rss = _status("VmRSS")
            if rss > limit_bytes:
                print(f"RSS {rss / 2**30:.1f} GiB over the {limit_bytes / 2**30:.1f} GiB limit; aborting", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=watch, daemon=True).start()


def require_available(min_bytes):
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
            if avail < min_bytes:
                raise SystemExit(f"MemAvailable {avail / 2**30:.1f} GiB < {min_bytes / 2**30:.0f} GiB; not starting")
            return avail


def digest(t):
    t = t.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(t):
    f = t.detach().float().cpu()
    return dict(shape=list(t.shape), dtype=str(t.dtype).replace("torch.", ""), sha256=digest(t),
                mean=f.mean().item(), std=f.std().item(), absmax=f.abs().max().item(),
                finite=bool(torch.isfinite(f).all()))


# ---- components (official loaders) -----------------------------------------------------------
def model_index(snap):
    return json.load(open(snap / "model_index.json"))


def component_class(snap, name):
    import importlib
    lib, cls = model_index(snap)[name]
    return getattr(importlib.import_module(lib), cls)


def load_tokenizer(snap):
    return component_class(snap, "tokenizer").from_pretrained(snap, subfolder="tokenizer")


def load_text_encoder(snap):
    return component_class(snap, "text_encoder").from_pretrained(snap, subfolder="text_encoder", dtype=torch.bfloat16).eval()


def load_transformer(snap, kind, *, fp8_path=None, mode="scaled_mm"):
    cls = component_class(snap, "transformer")
    if kind == "bf16":
        return cls.from_pretrained(snap, subfolder="transformer", torch_dtype=torch.bfloat16).eval(), {}
    model, report = fc.load_diffusers_single_file(cls, fp8_path, config=str(snap), subfolder="transformer",
                                                  torch_dtype=torch.bfloat16, mode=mode)
    return model, report


def load_vae(snap):
    return component_class(snap, "vae").from_pretrained(snap, subfolder="vae", torch_dtype=torch.bfloat16).eval()


def load_scheduler(snap):
    return component_class(snap, "scheduler").from_pretrained(snap, subfolder="scheduler")


def helper_pipeline(snap, scheduler, vae):
    """The official pipeline object, for its helpers only (prepare_latents, encode_prompt, ids, image processor)."""
    from diffusers import Flux2KleinPipeline
    return Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=None,
                              is_distilled=model_index(snap).get("is_distilled", False))


# ---- stage modules ---------------------------------------------------------------------------
def tokenize(tokenizer, prompt, max_sequence_length=MAX_SEQ):
    """Flux2KleinPipeline._get_qwen3_prompt_embeds, its host (tokenizer) half."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_sequence_length)
    return inputs["input_ids"], inputs["attention_mask"]


class TextEncodeStage(nn.Module):
    """Flux2KleinPipeline._get_qwen3_prompt_embeds, its device half: the text-encoder forward and the stack."""

    def __init__(self, text_encoder, layers=TE_LAYERS):
        super().__init__()
        self.text_encoder, self.layers = text_encoder, tuple(layers)

    def forward(self, input_ids, attention_mask):
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                   use_cache=False)
        out = torch.stack([output.hidden_states[k] for k in self.layers], dim=1)
        b, c, s, d = out.shape
        return out.permute(0, 2, 1, 3).reshape(b, s, c * d)


class DenoiseStage(nn.Module):
    """One iteration of Flux2KleinPipeline.__call__'s loop (no CFG: distilled) with
    FlowMatchEulerDiscreteScheduler.step written functionally: sample in fp32 + dt * model_output,
    cast to the model output dtype.  Returns (next latents, noise prediction)."""

    def __init__(self, transformer, compute_dtype=torch.bfloat16):
        super().__init__()
        self.transformer, self.compute_dtype = transformer, compute_dtype

    def forward(self, latents, t, dt, prompt_embeds, txt_ids, img_ids):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = self.transformer(hidden_states=latents.to(self.compute_dtype), timestep=timestep / 1000,
                                      guidance=None, encoder_hidden_states=prompt_embeds, txt_ids=txt_ids,
                                      img_ids=img_ids, return_dict=False)[0]
        noise_pred = noise_pred[:, : latents.size(1):]
        prev = (latents.to(torch.float32) + dt * noise_pred).to(noise_pred.dtype)
        return prev, noise_pred


class DecodeStage(nn.Module):
    """Flux2KleinPipeline.__call__ after the loop: unpack, BN de-normalisation, unpatchify, vae.decode."""

    def __init__(self, vae, latent_h, latent_w):
        super().__init__()
        self.vae, self.h, self.w = vae, latent_h, latent_w

    def forward(self, latents, latent_ids):
        from diffusers import Flux2KleinPipeline as P
        latents = P._unpack_latents_with_ids(latents, latent_ids, self.h, self.w)
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype)
        latents = P._unpatchify_latents(latents * std + mean)
        return self.vae.decode(latents, return_dict=False)[0]


def latent_hw(height, width, vae_scale_factor):
    lh = 2 * (int(height) // (vae_scale_factor * 2))
    lw = 2 * (int(width) // (vae_scale_factor * 2))
    return lh // 2, lw // 2


def timesteps_for(pipe, scheduler, steps, image_seq_len, device):
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    sigmas = np.linspace(1.0, 1 / steps, steps)
    if getattr(scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas, mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    return timesteps, mu


def declare_fp64_unsupported(devices):
    """Add device types to diffusers' own table of devices without float64 (mps, npu, neuron by default).
    Model code that asks `maybe_adjust_dtype_for_device(torch.float64, device)` then computes in float32 there.
    A device capability statement, generic for every diffusers model; not a model change."""
    if not devices:
        return None
    from diffusers.utils import torch_utils as tu
    table = tu._DTYPE_UNSUPPORTED_DEVICES
    table[torch.float64] = frozenset(table[torch.float64]) | frozenset(devices)
    return sorted(table[torch.float64])


# ---- device placement ---------------------------------------------------------------------------
_TNCA = False


def use_tnca():
    """Register the tnca device from THIS tree (ROOT/python etc.), before anything imports triton:
    the triton in torchenv discovers a tnca backend, and torch caches a failed triton probe."""
    global _TNCA
    if not _TNCA:
        paths = [ROOT / "python", ROOT / "python/_staged", ROOT / "compiler/python", ROOT / "tools/asm",
                 ROOT / "tools/performance_harness/systemc"]
        sys.path[:0] = [str(p) for p in paths if p.is_dir()]
        import tnca.torch_device as device
        if os.environ.get("TNCA_BACKEND", "").startswith("systemc-tlm"):
            import tlm_backend
            tlm_backend.register()
        device.register()
        _TNCA = True
    return torch.device("tnca")


def prepare_for_compile(stage):
    """Do once, outside the compiled region, what library code would otherwise do lazily on the first call.

    transformers (5.x) installs its output-capturing hooks (used for output_hidden_states) on the first
    forward, under a threading.Lock that dynamo cannot trace: a first call inside torch.compile(fullgraph=True)
    fails.  Installing them up front through transformers' own function is model-agnostic and needs no eager
    run on the device."""
    try:
        from transformers import PreTrainedModel
        from transformers.utils.output_capturing import maybe_install_capturing_hooks
    except ImportError:
        return 0
    n = 0
    for m in stage.modules():
        if isinstance(m, PreTrainedModel):
            maybe_install_capturing_hooks(m)
            n += 1
    return n


def place(stage, device, compile_mode):
    """Stage module on its device; tnca always torch.compile(fullgraph=True), cpu eager unless asked."""
    if device == "tnca" or compile_mode == "inductor":
        prepare_for_compile(stage)
    if device == "tnca":
        dev = use_tnca()
        stage = stage.to(dev)
        return torch.compile(stage, fullgraph=True), dev
    stage = stage.to("cpu")
    if compile_mode == "inductor":
        return torch.compile(stage, fullgraph=True), torch.device("cpu")
    return stage, torch.device("cpu")


def device_memory(dev):
    if dev.type == "tnca":
        return dict(device_peak=int(torch.tnca.max_memory_allocated()), device_live=int(torch.tnca.memory_allocated()))
    return {}


# ---- run --------------------------------------------------------------------------------------
def save_tensors(path, tensors):
    from safetensors.torch import save_file
    # own dense storage per tensor: safetensors refuses shared storages and non-contiguous views
    save_file({k: v.detach().cpu().clone(memory_format=torch.contiguous_format) for k, v in tensors.items()}, str(path))


def cmd_run(a):
    if "tnca" in (a.te_device, a.dit_device, a.vae_device):
        use_tnca()
    torch.set_num_threads(a.threads)
    if a.emulate_precision_casts:
        # Inductor option: round fused intermediates where eager would cast (e.g. to bf16)
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True
    fp64_table = declare_fp64_unsupported(a.fp64_unsupported_devices)
    rss_watchdog(int(a.rss_limit_gib * 2**30))
    avail = require_available(int(a.min_available_gib * 2**30))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main = snapshot(MAIN_REPO, MAIN_REV)
    fp8_path = snapshot(FP8_REPO, FP8_REV) / FP8_FILE
    meter = Meter()
    report = dict(schema="tnca-flux2-klein-run/1", mode="run", versions=versions(), mem_available_at_start=avail,
                  repos={MAIN_REPO: MAIN_REV, FP8_REPO: FP8_REV},
                  settings=dict(prompt=a.prompt, height=a.height, width=a.width, steps=a.steps, seed=a.seed,
                                guidance_scale=1.0, max_sequence_length=MAX_SEQ, te_layers=list(TE_LAYERS),
                                transformer=a.transformer, fp8_mode=a.fp8_mode, devices=dict(te=a.te_device,
                                dit=a.dit_device, vae=a.vae_device), cpu_compile=a.cpu_compile,
                                emulate_precision_casts=a.emulate_precision_casts,
                                fp64_unsupported_devices=fp64_table),
                  checks={})
    ref = None
    if a.teacher_force:
        # every stage takes the reference's inputs: per-stage error without drift accumulated upstream
        from safetensors.torch import load_file
        ref = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
        report["teacher_forced_from"] = str(a.teacher_force)
    t_all = time.perf_counter()
    scheduler = load_scheduler(main)
    with meter("load_vae"):
        vae = load_vae(main)
    pipe = helper_pipeline(main, scheduler, vae)
    tensors = {}

    # 1. text encoder
    with meter("load_text_encoder") as row:
        tokenizer = load_tokenizer(main)
        te = load_text_encoder(main)
        row["weight_bytes"] = weight_bytes(te)[0]          # tied embedding counted once
    input_ids, attention_mask = tokenize(tokenizer, a.prompt)
    tensors.update(input_ids=input_ids, attention_mask=attention_mask)
    stage, dev = place(TextEncodeStage(te), a.te_device, a.cpu_compile)
    with torch.no_grad(), meter("text_encode", device=str(dev)) as row:
        prompt_embeds = stage(input_ids.to(dev), attention_mask.to(dev)).cpu()
        row.update(device_memory(dev))
    if a.check_official and a.te_device == "cpu":
        with torch.no_grad(), meter("text_encode_official_check"):
            official = pipe._get_qwen3_prompt_embeds(te, tokenizer, a.prompt, dtype=prompt_embeds.dtype,
                                                     device=torch.device("cpu"), max_sequence_length=MAX_SEQ,
                                                     hidden_states_layers=TE_LAYERS)
        report["checks"]["prompt_embeds_bitwise_equal_official"] = bool(torch.equal(official, prompt_embeds))
    prompt_embeds, text_ids = pipe.encode_prompt(prompt=None, prompt_embeds=prompt_embeds, device=torch.device("cpu"),
                                                 num_images_per_prompt=1, max_sequence_length=MAX_SEQ)
    tensors.update(prompt_embeds=prompt_embeds, text_ids=text_ids)
    if ref is not None:
        prompt_embeds, text_ids = ref["prompt_embeds"], ref["text_ids"]
    del stage, te, tokenizer
    gc.collect()

    # 2. transformer
    with meter("load_transformer", kind=a.transformer) as row:
        transformer, load_report = load_transformer(main, a.transformer, fp8_path=fp8_path, mode=a.fp8_mode)
        row["weight_bytes"] = weight_bytes(transformer)[0]
        row["model_dtype_attr"] = str(transformer.dtype)
    if load_report:
        json.dump(load_report, open(out / "fp8_load_report.json", "w"), indent=1)
        report["fp8_load"] = {k: v for k, v in load_report.items() if k != "mapping"}
    generator = torch.Generator(device="cpu").manual_seed(a.seed)
    latents, latent_ids = pipe.prepare_latents(batch_size=1, num_latents_channels=transformer.config.in_channels // 4,
                                               height=a.height, width=a.width, dtype=prompt_embeds.dtype,
                                               device=torch.device("cpu"), generator=generator)
    timesteps, mu = timesteps_for(pipe, scheduler, a.steps, latents.shape[1], torch.device("cpu"))
    sigmas = scheduler.sigmas
    report["schedule"] = dict(mu=mu, timesteps=timesteps.tolist(), sigmas=sigmas.tolist())
    tensors.update(latents_init=latents, latent_ids=latent_ids)
    if ref is not None:
        report["checks"]["latents_init_equal_reference"] = bool(torch.equal(latents, ref["latents_init"]))
    stage, dev = place(DenoiseStage(transformer), a.dit_device, a.cpu_compile)
    on = lambda t: t.to(dev)  # noqa: E731
    pe_d, tid_d, lid_d = on(prompt_embeds), on(text_ids), on(latent_ids)
    step_equal = []
    for i, t in enumerate(timesteps):
        dt = (sigmas[i + 1] - sigmas[i]).to(torch.float32)
        if ref is not None:
            latents = ref["latents_init"] if i == 0 else ref[f"step{i - 1}_latents"]
        with torch.no_grad(), meter(f"denoise_step_{i}", device=str(dev)) as row:
            nxt, noise_pred = stage(on(latents), on(t), on(dt), pe_d, tid_d, lid_d)
            nxt, noise_pred = nxt.cpu(), noise_pred.cpu()
            row.update(device_memory(dev))
        if a.check_official:
            stepped = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            step_equal.append(bool(torch.equal(stepped, nxt)))
        latents = nxt
        tensors[f"step{i}_noise_pred"] = noise_pred
        tensors[f"step{i}_latents"] = latents
    if step_equal:
        report["checks"]["scheduler_step_bitwise_equal_official"] = step_equal
    tensors["final_latents_packed"] = latents
    del stage, transformer, pe_d
    gc.collect()

    # 3. VAE decode
    h, w = latent_hw(a.height, a.width, pipe.vae_scale_factor)
    if ref is not None:
        latents = ref["final_latents_packed"]
    stage, dev = place(DecodeStage(vae, h, w), a.vae_device, a.cpu_compile)
    with torch.no_grad(), meter("vae_decode", device=str(dev)) as row:
        image = stage(latents.to(dev), latent_ids.to(dev)).cpu()
        row.update(device_memory(dev))
    with torch.no_grad():
        from diffusers import Flux2KleinPipeline as P
        unpacked = P._unpack_latents_with_ids(latents, latent_ids, h, w)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(unpacked.dtype)
        tensors["final_latents_vae"] = P._unpatchify_latents(unpacked * std + mean)
    tensors["image_decoded"] = image
    image_pt = pipe.image_processor.postprocess(image, output_type="pt")
    tensors["image_pt"] = image_pt
    pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
    pil.save(out / "image.png")
    report["wall_seconds_total"] = round(time.perf_counter() - t_all, 3)
    report["stages"] = meter.stages
    report["tensors"] = {k: stats(v) for k, v in tensors.items()}
    save_tensors(out / "tensors.safetensors", tensors)
    report["image_png_sha256"] = hashlib.sha256((out / "image.png").read_bytes()).hexdigest()
    json.dump(report, open(out / "run.json", "w"), indent=1)
    print(json.dumps(dict(checks=report["checks"], total=report["wall_seconds_total"]), indent=1))


def cmd_official(a):
    """Unmodified Flux2KleinPipeline.__call__ on cpu with the same settings."""
    torch.set_num_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * 2**30))
    require_available(int(a.min_available_gib * 2**30))
    from diffusers import Flux2KleinPipeline
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main = snapshot(MAIN_REPO, MAIN_REV)
    t0 = time.perf_counter()
    transformer, _ = load_transformer(main, a.transformer, fp8_path=snapshot(FP8_REPO, FP8_REV) / FP8_FILE,
                                      mode=a.fp8_mode)
    pipe = Flux2KleinPipeline.from_pretrained(main, transformer=transformer, torch_dtype=torch.bfloat16)
    tensors = {}

    def capture(p, i, t, kw):
        tensors[f"step{i}_latents"] = kw["latents"].detach().clone()
        return {}
    t1 = time.perf_counter()
    image = pipe(prompt=a.prompt, height=a.height, width=a.width, guidance_scale=1.0, num_inference_steps=a.steps,
                 generator=torch.Generator(device="cpu").manual_seed(a.seed), output_type="pt",
                 callback_on_step_end=capture, callback_on_step_end_tensor_inputs=["latents"]).images
    tensors["image_pt"] = image
    report = dict(schema="tnca-flux2-klein-run/1", mode="official", versions=versions(),
                  settings=dict(prompt=a.prompt, height=a.height, width=a.width, steps=a.steps, seed=a.seed,
                                transformer=a.transformer, fp8_mode=a.fp8_mode),
                  pipeline_transformer_dtype=str(pipe.transformer.dtype),
                  load_seconds=round(t1 - t0, 3), call_seconds=round(time.perf_counter() - t1, 3),
                  tensors={k: stats(v) for k, v in tensors.items()})
    save_tensors(out / "tensors.safetensors", tensors)
    json.dump(report, open(out / "run.json", "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("load_seconds", "call_seconds", "pipeline_transformer_dtype")}))


def cmd_compare(a):
    from safetensors.torch import load_file
    A, B = load_file(str(Path(a.a) / "tensors.safetensors")), load_file(str(Path(a.b) / "tensors.safetensors"))
    rows = {}
    for k in sorted(set(A) & set(B)):
        x, y = A[k], B[k]
        if x.shape != y.shape:
            rows[k] = dict(shape_a=list(x.shape), shape_b=list(y.shape))
            continue
        if not torch.is_floating_point(x):
            rows[k] = dict(equal=bool(torch.equal(x, y)))
            continue
        xf, yf = x.double(), y.double()
        d = xf - yf
        rows[k] = dict(bitwise_equal=bool(torch.equal(x, y)), max_abs=d.abs().max().item(),
                       rel_l2=(d.norm() / yf.norm().clamp_min(1e-30)).item(),
                       cosine=torch.nn.functional.cosine_similarity(xf.flatten(), yf.flatten(), dim=0).item())
        if k == "image_pt":
            mse = (d ** 2).mean().item()
            rows[k]["psnr_db"] = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    result = dict(a=str(a.a), b=str(a.b), tensors=rows)
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)
    for k, r in rows.items():
        print(k, json.dumps(r))


def cmd_layer_check(a):
    """Every FP8Linear alone: the input it sees in a teacher-forced reference step (cpu eager), run on the
    target device (tnca: compiled fullgraph; cpu: eager or --cpu-compile inductor), against cpu eager and
    against the exact W8A8 result in float64.  A whole-step comparison cannot see a single faulty layer
    (evidence/r41/fp8/flux/compare/teacher_forced/sensitivity.json); this can."""
    import copy
    from safetensors.torch import load_file
    if a.device == "tnca":
        use_tnca()
    torch.set_num_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * 2**30))
    ref = load_file(str(Path(a.teacher_force) / "tensors.safetensors"))
    main = snapshot(MAIN_REPO, MAIN_REV)
    model, _ = load_transformer(main, "fp8", fp8_path=snapshot(FP8_REPO, FP8_REV) / FP8_FILE, mode="scaled_mm")
    scheduler = load_scheduler(main)
    pipe = helper_pipeline(main, scheduler, load_vae(main))
    timesteps, _ = timesteps_for(pipe, scheduler, DEFAULT_STEPS, ref["latents_init"].shape[1], torch.device("cpu"))
    i = a.step
    lat = ref["latents_init"] if i == 0 else ref[f"step{i - 1}_latents"]
    dt = (scheduler.sigmas[i + 1] - scheduler.sigmas[i]).float()
    captured = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: captured.__setitem__(n, (inp[0].detach().clone(), out.detach().clone())))
             for n, m in fc.fp8_modules(model).items()]
    with torch.no_grad():
        DenoiseStage(model)(lat, timesteps[i], dt, ref["prompt_embeds"], ref["text_ids"], ref["latent_ids"])
    for h in hooks:
        h.remove()
    dev = torch.device("tnca") if a.device == "tnca" else torch.device("cpu")
    compile_it = a.device == "tnca" or a.cpu_compile == "inductor"
    rows = {}
    for name, m in fc.fp8_modules(model).items():
        x, y_cpu = captured[name]
        with torch.no_grad():
            xq = fc.quantize_static(x, m.input_scale, m.weight.dtype).double() * m.input_scale.double()
            exact = xq.reshape(-1, m.in_features) @ (m.weight.double() * m.weight_scale.double()).t()
            exact = exact.reshape(y_cpu.shape)
            mod = copy.deepcopy(m).to(dev)
            fn = torch.compile(mod, fullgraph=True) if compile_it else mod
            y_dev = fn(x.to(dev)).cpu()

        def err(u, v):
            return ((u.double() - v.double()).norm() / v.double().norm()).item()
        rows[name] = dict(shape=[int(np.prod(x.shape[:-1])), m.in_features, m.out_features],
                          bitwise_equal_cpu=bool(torch.equal(y_dev, y_cpu)), rel_l2_vs_cpu=err(y_dev, y_cpu),
                          rel_l2_vs_exact=err(y_dev, exact), cpu_rel_l2_vs_exact=err(y_cpu, exact),
                          bf16_rounding_of_exact=err(exact.to(torch.bfloat16), exact))
        del mod, fn
    worst = max(rows.values(), key=lambda r: r["rel_l2_vs_exact"])
    result = dict(device=a.device, compiled=compile_it, step=i, layers=len(rows),
                  all_bitwise_equal_cpu=all(r["bitwise_equal_cpu"] for r in rows.values()),
                  max_rel_l2_vs_cpu=max(r["rel_l2_vs_cpu"] for r in rows.values()),
                  max_rel_l2_vs_exact=worst["rel_l2_vs_exact"],
                  max_cpu_rel_l2_vs_exact=max(r["cpu_rel_l2_vs_exact"] for r in rows.values()),
                  max_bf16_rounding_of_exact=max(r["bf16_rounding_of_exact"] for r in rows.values()),
                  criterion="rel_l2_vs_exact of every layer within ~2x bf16_rounding_of_exact", rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


def cmd_probe_official_loader(a):
    """What the official public code does with this checkpoint (evidence for the loader decision)."""
    import inspect
    import traceback
    import diffusers
    import transformers
    from safetensors import safe_open
    torch.set_num_threads(a.threads)
    main = snapshot(MAIN_REPO, MAIN_REV)
    fp8_path = snapshot(FP8_REPO, FP8_REV) / FP8_FILE
    cls = component_class(main, "transformer")
    result = dict(versions=versions(), checkpoint=str(fp8_path))
    # 1. from_single_file, as documented, on the FP8 file
    try:
        m = cls.from_single_file(str(fp8_path), config=str(main), subfolder="transformer", torch_dtype=torch.bfloat16)
        result["from_single_file"] = dict(outcome="loaded", to_q_dtype=str(m.transformer_blocks[0].attn.to_q.weight.dtype))
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        result["from_single_file"] = dict(outcome="raised", error=f"{type(e).__name__}: {e}",
                                          frames=[f"{Path(f.filename).name}:{f.lineno} {f.name}: {f.line}" for f in tb[-3:]])
    # 2. any reader of the format in the installed official libraries
    hits = {}
    for pkg in (diffusers, transformers):
        root = Path(pkg.__file__).parent
        hits[pkg.__name__] = sorted(str(f.relative_to(root)) for f in root.rglob("*.py")
                                    if "_quantization_metadata" in f.read_text(errors="ignore"))
    result["files_mentioning__quantization_metadata"] = hits
    from diffusers.quantizers.auto import AUTO_QUANTIZER_MAPPING
    result["diffusers_quantizers"] = sorted(AUTO_QUANTIZER_MAPPING)
    from diffusers.models import model_loading_utils
    src = inspect.getsource(model_loading_utils.load_model_dict_into_meta)
    result["load_model_dict_into_meta_keeps_float8_only_with_quantizer"] = (
        "hf_quantizer is not None and param.dtype == getattr(torch, \"float8_e4m3fn\"" in src)
    # 3. bypass the crash (drop the scale tensors) and see what from_single_file loads
    sd = {}
    with safe_open(str(fp8_path), "pt") as f:
        for k in f.keys():
            if k.rsplit(".", 1)[-1] not in fc.SCALE_KEYS:
                sd[k] = f.get_tensor(k)
        ws = f.get_tensor("double_blocks.0.img_attn.proj.weight_scale").item()
    m = cls.from_single_file(sd, config=str(main), subfolder="transformer", torch_dtype=torch.bfloat16)
    w = m.transformer_blocks[0].attn.to_out[0].weight
    bf16_file = main / "transformer" / "diffusion_pytorch_model.safetensors"
    with safe_open(str(bf16_file), "pt") as f:
        r = f.get_tensor("transformer_blocks.0.attn.to_out.0.weight")
    result["scales_dropped_from_single_file"] = dict(
        param="transformer_blocks.0.attn.to_out.0.weight", loaded_dtype=str(w.dtype),
        loaded_absmax=w.float().abs().max().item(), bf16_reference_absmax=r.float().abs().max().item(),
        checkpoint_weight_scale=ws, loaded_over_reference_rms=(w.float().norm() / r.float().norm()).item(),
        meaning="float8 codes cast to bf16 with no weight_scale applied: every quantized weight off by 1/weight_scale")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


def cmd_verify_weights(a):
    """Every FP8Linear's dequantized weight against the official BF16 transformer's parameter of the same name."""
    torch.set_num_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * 2**30))
    main = snapshot(MAIN_REPO, MAIN_REV)
    fp8, load_report = load_transformer(main, "fp8", fp8_path=snapshot(FP8_REPO, FP8_REV) / FP8_FILE)
    bf16, _ = load_transformer(main, "bf16")
    ref = bf16.state_dict()
    rows, worst = {}, None
    for name, m in fc.fp8_modules(fp8).items():
        w, r = m.dequantized_weight(torch.float32).double(), ref[f"{name}.weight"].double()
        d = w - r
        row = dict(rel_l2=(d.norm() / r.norm()).item(), cosine=torch.nn.functional.cosine_similarity(
            w.flatten(), r.flatten(), dim=0).item(), weight_scale=m.weight_scale.item(), input_scale=m.input_scale.item(),
            w_absmax=r.abs().max().item(), source=load_report["mapping"][f"{name}.weight"]["source"])
        rows[name] = row
        worst = row if worst is None or row["rel_l2"] > worst["rel_l2"] else worst
    others = {}
    for k, v in fp8.state_dict().items():
        if k.endswith(("weight_scale", "input_scale")) or v.dtype == torch.float8_e4m3fn:
            continue
        others[k] = bool(torch.equal(v, ref[k]))
    missing = sorted(set(ref) - set(fp8.state_dict()))
    result = dict(fp8_linears=len(rows), max_rel_l2=max(r["rel_l2"] for r in rows.values()),
                  min_cosine=min(r["cosine"] for r in rows.values()), worst=worst,
                  unquantized_tensors=len(others), unquantized_bitwise_equal=sum(others.values()),
                  unquantized_not_equal=[k for k, eq in others.items() if not eq],
                  bf16_keys_missing_in_fp8=missing, model_dtype_attr=str(fp8.dtype), layers=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=1))


# ---- plan: capture, ops, FLOPs, memory -----------------------------------------------------------------
class LiveBytes(torch.utils._python_dispatch.TorchDispatchMode):
    """Peak bytes of storages created inside the mode (eager activations; weights/inputs excluded)."""

    def __init__(self):
        super().__init__()
        self.live, self.current, self.peak = {}, 0, 0
        self.touched = set()        # storages read as inputs (to tell which weights the stage uses)

    def _release(self, key):
        entry = self.live.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] == 0:
            self.current -= entry[0]
            del self.live[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        import weakref
        from torch.utils._pytree import tree_leaves
        out = func(*args, **(kwargs or {}))
        inputs = {t.untyped_storage()._cdata for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)}
        self.touched |= inputs
        for t in tree_leaves(out):
            if not isinstance(t, torch.Tensor):
                continue
            key = t.untyped_storage()._cdata
            if key in self.live:
                self.live[key][1] += 1
            elif key in inputs:
                continue
            else:
                n = t.untyped_storage().nbytes()
                self.live[key] = [n, 1]
                self.current += n
                self.peak = max(self.peak, self.current)
            weakref.finalize(t, self._release, key)
        return out


def flop_mapping():
    """The cpu flash-attention op has no formula in torch's table; same count as the other SDPA kernels."""
    from torch.utils.flop_counter import sdpa_flop_count
    op = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
    if op is None:
        return {}

    def f(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):   # shapes, not tensors
        return sdpa_flop_count(query_shape, key_shape, value_shape)
    return {op: f}


def capture_graphs(stage, inputs):
    """torch.compile(fullgraph=True) with a recording backend: dynamo + AOT with Inductor's decompositions.

    The recorded graph is also executed (on the inputs' device: meta or cpu) under a FLOP counter and the
    live-bytes tracker, giving the compiled graph's FLOPs (after dead-code elimination), its unfused
    activation peak and the weights it actually reads."""
    from torch._dynamo.backends.common import aot_autograd
    from torch._inductor.decomposition import select_decomp_table
    from torch.utils.flop_counter import FlopCounterMode
    graphs, runs = [], []

    def fw(gm, example_inputs):
        graphs.append(gm)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

        def run(*args):
            fcm, live = FlopCounterMode(display=False, custom_mapping=flop_mapping()), LiveBytes()
            with fcm, live:
                out = gm.forward(*args)
            consumed = {tensor_key(a) for a, n in zip(args, placeholders) if isinstance(a, torch.Tensor) and n.users}
            runs.append(dict(flops=int(fcm.get_total_flops()), activation_peak=live.peak, consumed=consumed))
            return out
        return run
    torch._dynamo.reset()
    prepare_for_compile(stage)
    compiled = torch.compile(stage, fullgraph=True, backend=aot_autograd(fw_compiler=fw, decompositions=select_decomp_table()))
    with torch.no_grad():
        compiled(*inputs)
    return graphs, runs


def op_inventory(graphs):
    from torch.utils._pytree import tree_leaves
    ops = {}
    for gm in graphs:
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            name = str(n.target).replace("aten.", "")
            ins = tuple(sorted({str(x.meta["val"].dtype).replace("torch.", "") for x in n.all_input_nodes
                                if isinstance(x.meta.get("val"), torch.Tensor)}))
            outs = tuple(sorted({str(v.dtype).replace("torch.", "") for v in tree_leaves(n.meta.get("val"))
                                 if isinstance(v, torch.Tensor)}))
            key = f"{name} {','.join(ins) or '-'} -> {','.join(outs) or '-'}"
            ops[key] = ops.get(key, 0) + 1
    return dict(sorted(ops.items()))


def graph_dtypes(graphs):
    seen = set()
    for gm in graphs:
        for n in gm.graph.nodes:
            v = n.meta.get("val")
            if isinstance(v, torch.Tensor):
                seen.add(str(v.dtype).replace("torch.", ""))
    return sorted(seen)


def weight_bytes(module):
    seen, total, by_dtype = set(), 0, {}
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) in seen:            # tied parameters are one object; fused-QKV views are separate tensors
            continue
        seen.add(id(t))
        n = t.numel() * t.element_size()
        total += n
        d = str(t.dtype).replace("torch.", "")
        by_dtype[d] = by_dtype.get(d, 0) + n
    return total, by_dtype


def tensor_key(t):
    """One parameter view, even when many share a storage (transformers maps a shard as one storage)."""
    return (t.untyped_storage()._cdata, t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)


def consumed_bytes(module, consumed):
    used, seen = 0, set()
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen and tensor_key(t) in consumed:
            seen.add(id(t))
            used += t.numel() * t.element_size()
    return used


def touched_bytes(module, touched):
    used, seen = 0, set()
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen and t.untyped_storage()._cdata in touched:
            seen.add(id(t))
            used += t.numel() * t.element_size()
    return used


def measure(stage, inputs, label, *, eager_flops=True, graph=True):
    """Weights, inputs, eager activation peak (+ weights read), eager FLOPs, and the fullgraph capture."""
    from torch.utils.flop_counter import FlopCounterMode
    wbytes, wdtype = weight_bytes(stage)
    in_bytes = sum(t.numel() * t.element_size() for t in inputs if isinstance(t, torch.Tensor))
    row = dict(label=label, device=str(inputs[0].device), weight_bytes=wbytes, weight_bytes_by_dtype=wdtype,
               input_bytes=in_bytes)
    with torch.no_grad():
        if eager_flops:
            fcm = FlopCounterMode(display=False, custom_mapping=flop_mapping())
            with fcm:
                stage(*inputs)
            row.update(eager_flops=int(fcm.get_total_flops()),
                       eager_flops_by_op={str(k).replace("aten.", ""): int(v)
                                          for k, v in fcm.get_flop_counts().get("Global", {}).items()})
        live = LiveBytes()
        with live:
            out = stage(*inputs)
    row.update(output_bytes=sum(t.numel() * t.element_size() for t in (out if isinstance(out, tuple) else (out,))),
               eager_activation_peak_bytes=live.peak,
               eager_weight_bytes_read=touched_bytes(stage, live.touched))   # storage level: a shared shard storage counts whole
    if graph:
        try:
            graphs, runs = capture_graphs(stage, inputs)
            consumed = set().union(*(r["consumed"] for r in runs)) if runs else set()
            row.update(fullgraph="ok", graphs=len(graphs), graph_nodes=sum(len(g.graph.nodes) for g in graphs),
                       graph_flops=sum(r["flops"] for r in runs),
                       graph_activation_peak_bytes=max((r["activation_peak"] for r in runs), default=0),
                       graph_weight_bytes_read=consumed_bytes(stage, consumed),
                       ops=op_inventory(graphs), dtypes=graph_dtypes(graphs))
        except Exception as e:  # noqa: BLE001 - a graph break or capture failure is a finding
            row.update(fullgraph=f"FAILED: {type(e).__name__}: {str(e)[:2000]}")
    print(f"[plan] {label} ({row['device']}): weights {wbytes / 2**30:.3f} GiB, eager act {live.peak / 2**30:.3f} GiB, "
          f"graph act {row.get('graph_activation_peak_bytes', 0) / 2**30:.3f} GiB, eager {row.get('eager_flops', 0) / 1e12:.3f} T, "
          f"graph {row.get('graph_flops', 0) / 1e12:.3f} T, fullgraph {row.get('fullgraph', '-')[:60]}", flush=True)
    return row


def to_meta(module):
    module = module.to("meta")
    if hasattr(module, "tie_weights"):
        module.tie_weights()
    return module


def plan_inputs(pipe, scheduler, res, device):
    gen = torch.Generator(device="cpu").manual_seed(0)
    lat, lat_ids = pipe.prepare_latents(batch_size=1, num_latents_channels=128 // 4, height=res, width=res,
                                        dtype=torch.bfloat16, device=torch.device("cpu"), generator=gen)
    pe = torch.randn(1, MAX_SEQ, len(TE_LAYERS) * 2560, generator=gen).to(torch.bfloat16)
    _, txt_ids = pipe.encode_prompt(prompt=None, prompt_embeds=pe, device=torch.device("cpu"), num_images_per_prompt=1)
    timesteps, _ = timesteps_for(pipe, scheduler, DEFAULT_STEPS, lat.shape[1], torch.device("cpu"))
    dt = (scheduler.sigmas[1] - scheduler.sigmas[0]).float()
    dit = tuple(x.to(device) for x in (lat, timesteps[0], dt, pe, txt_ids, lat_ids))
    return dit, (lat.to(device), lat_ids.to(device)), int(lat.shape[1])


def cmd_plan(a):
    """Two passes.  meta: every stage x resolution x transformer, no data -- FLOPs, weights, fullgraph capture
    (SDPA decomposes to the math path on meta, so activations there include S x S scores: an upper bound).
    cpu (real tensors, --real-resolutions): fp8 transformer and VAE with the fused cpu SDPA kernel, and the
    text encoder (512 tokens, resolution independent)."""
    torch.set_num_threads(a.threads)
    rss_watchdog(int(a.rss_limit_gib * 2**30))
    main = snapshot(MAIN_REPO, MAIN_REV)
    fp8_path = snapshot(FP8_REPO, FP8_REV) / FP8_FILE
    scheduler = load_scheduler(main)
    pipe = helper_pipeline(main, scheduler, load_vae(main))
    report = dict(schema="tnca-flux2-klein-plan/2", versions=versions(), stages=[],
                  note="ops: the ATen graph after dynamo fullgraph capture and Inductor's decomposition table "
                       "(what Inductor lowers for a device). eager_*: op-by-op eager execution. graph_*: the "
                       "captured graph executed node by node (dead code removed, no fusion). Activation bytes: "
                       "storages created during the stage; weights and inputs excluded.")
    meta = torch.device("meta")
    # text encoder: 512 tokens whatever the resolution; real weights and the real prompt's tokens
    te = load_text_encoder(main)
    report["text_encoder_attn_implementation"] = getattr(te.config, "_attn_implementation", None)
    report["text_encoder_tied_embeddings"] = te.lm_head.weight is te.model.embed_tokens.weight
    ids, mask = tokenize(load_tokenizer(main), a.prompt)
    report["stages"].append(dict(stage="text_encode", resolution=None, **measure(TextEncodeStage(te), (ids, mask), "text_encode")))
    del te
    gc.collect()
    # meta pass
    dits = {kind: to_meta(load_transformer(main, kind, fp8_path=fp8_path)[0]) for kind in a.transformers}
    vae = to_meta(load_vae(main))
    for res in a.resolutions:
        dit_in, vae_in, tokens = plan_inputs(pipe, scheduler, res, meta)
        for kind, model in dits.items():
            report["stages"].append(dict(stage="denoise_step", resolution=res, transformer=kind, image_tokens=tokens,
                                         text_tokens=MAX_SEQ, **measure(DenoiseStage(model), dit_in, f"denoise_step[{kind}]@{res}")))
        h, w = latent_hw(res, res, pipe.vae_scale_factor)
        report["stages"].append(dict(stage="vae_decode", resolution=res,
                                     **measure(DecodeStage(vae, h, w), vae_in, f"vae_decode@{res}")))
    del dits, vae
    gc.collect()
    # cpu pass: fused SDPA, real tensors
    if a.real_resolutions:
        dit, _ = load_transformer(main, "fp8", fp8_path=fp8_path)
        vae = load_vae(main)
        for res in a.real_resolutions:
            dit_in, vae_in, tokens = plan_inputs(pipe, scheduler, res, torch.device("cpu"))
            report["stages"].append(dict(stage="denoise_step", resolution=res, transformer="fp8", image_tokens=tokens,
                                         text_tokens=MAX_SEQ, **measure(DenoiseStage(dit), dit_in, f"denoise_step[fp8]@{res}",
                                                                         eager_flops=False)))
            h, w = latent_hw(res, res, pipe.vae_scale_factor)
            report["stages"].append(dict(stage="vae_decode", resolution=res,
                                         **measure(DecodeStage(vae, h, w), vae_in, f"vae_decode@{res}", eager_flops=False)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--prompt", default=DEFAULT_PROMPT)
        p.add_argument("--height", type=int, default=256)
        p.add_argument("--width", type=int, default=256)
        p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--transformer", choices=("fp8", "bf16"), default="fp8")
        p.add_argument("--fp8-mode", choices=("scaled_mm", "emulate"), default="scaled_mm")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--rss-limit-gib", type=float, default=38.0)
        p.add_argument("--min-available-gib", type=float, default=40.0)
        p.add_argument("--out", required=True)

    p = sub.add_parser("run")
    common(p)
    for s in ("te", "dit", "vae"):
        p.add_argument(f"--{s}-device", choices=("cpu", "tnca"), default="cpu")
    p.add_argument("--cpu-compile", choices=("eager", "inductor"), default="eager")
    p.add_argument("--check-official", action="store_true", help="assert cpu stages bitwise equal to the official functions")
    p.add_argument("--emulate-precision-casts", action="store_true",
                   help="torch._inductor.config.emulate_precision_casts=True for compiled stages (cpu or tnca)")
    p.add_argument("--teacher-force", metavar="REF_DIR",
                   help="feed every stage the inputs saved in REF_DIR (a cpu reference run): per-stage comparison")
    p.add_argument("--fp64-unsupported-devices", nargs="*", default=[],
                   help="device types to add to diffusers' no-float64 table (e.g. tnca; cpu to measure the effect)")
    p = sub.add_parser("official")
    common(p)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    p = sub.add_parser("layer-check")
    p.add_argument("--teacher-force", required=True, metavar="REF_DIR")
    p.add_argument("--device", choices=("cpu", "tnca"), default="cpu")
    p.add_argument("--cpu-compile", choices=("eager", "inductor"), default="eager")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--rss-limit-gib", type=float, default=38.0)
    p = sub.add_parser("probe-official-loader")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("verify-weights")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--rss-limit-gib", type=float, default=38.0)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--resolutions", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--transformers", nargs="+", default=["fp8", "bf16"])
    p.add_argument("--real-resolutions", type=int, nargs="*", default=[256, 1024])
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--rss-limit-gib", type=float, default=38.0)
    a = ap.parse_args()
    dict(run=cmd_run, official=cmd_official, compare=cmd_compare, plan=cmd_plan,
         **{"verify-weights": cmd_verify_weights, "probe-official-loader": cmd_probe_official_loader,
            "layer-check": cmd_layer_check})[a.cmd](a)


if __name__ == "__main__":
    main()
