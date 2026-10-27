#!/usr/bin/env python3
"""Chat with the staged Qwen FP8 checkpoints through the standard Torch path.

One-shot:
  python tools/models/chat.py --model vl --prompt "Hello" --device tnca --out OUT
  python tools/models/chat.py --model q35 --sample --temperature 0.8 --top-p 0.95

Interactive:
  python tools/models/chat.py --model vl --device tnca --interactive

The official processor supplies the tokenizer and chat template. Model execution
uses transformers' modules and FP8Linear's official Hub route on TNCA. The CPU
W8A8 route is an independent teacher-forced reference, never a device fallback.
--layers is only a reduced-depth integration proof; full depth is the default.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import qwen_fp8_lib as L  # noqa: E402


CORE_PORT_BYTES_PER_SECOND = 2_400_000_000
CORE_PORT_SOURCE = "doc/r40_tasks_and_acceptance.md: 8 B/cycle/core, three cores at 100 MHz"


def spec_of(name):
    if name == "vl":
        from qwen3vl_run import SPEC
    elif name == "q35":
        from qwen35_run import SPEC
    else:
        raise ValueError(name)
    return SPEC


def validate_sampling(temperature, top_k, top_p):
    if not 0 < temperature:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top-k must be nonnegative")
    if not 0 < top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")


def select_token(logits, sample, temperature, top_k, top_p, generator):
    """Select on the host; no model operation is executed by this function."""
    validate_sampling(temperature, top_k, top_p)
    values = logits.detach().float().cpu().reshape(-1)
    greedy = int(values.argmax())
    if not sample:
        return greedy, greedy
    scaled = values / temperature
    if top_k:
        k = min(top_k, scaled.numel())
        threshold = scaled.topk(k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
    if top_p < 1:
        ordered, indices = scaled.sort(descending=True)
        probabilities = ordered.softmax(dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > top_p
        remove[0] = False
        ordered = ordered.masked_fill(remove, -torch.inf)
        scaled = torch.full_like(scaled, -torch.inf).scatter(0, indices, ordered)
    token = int(torch.multinomial(scaled.softmax(dim=-1), 1, generator=generator))
    return token, greedy


def compare_logits(reference, candidate):
    rows = []
    for index, (want, got) in enumerate(zip(reference, candidate)):
        x, y = want.double().flatten(), got.double().flatten()
        rows.append(dict(step=index,
                         rel_l2=float((y - x).norm() / x.norm().clamp_min(1e-300)),
                         reference_argmax=int(want.float().argmax()),
                         candidate_argmax=int(got.float().argmax()),
                         argmax_equal=bool(int(want.float().argmax()) == int(got.float().argmax()))))
    return rows


def generation_stats(rows):
    decode = [row for row in rows if row["kind"] == "decode"]
    seconds = sum(row["seconds"] for row in decode)
    return dict(generated_tokens=len(rows), decode_steps=len(decode),
                prefill_seconds=rows[0]["seconds"] if rows else 0.0,
                decode_seconds=seconds,
                mean_decode_seconds_per_token=seconds / len(decode) if decode else None,
                decode_tokens_per_second=len(decode) / seconds if decode and seconds > 0 else None,
                generation_seconds=sum(row["seconds"] for row in rows))


def make_inputs(spec, processor, prompt, image):
    content = ([{"type": "image", "image": image}] if image is not None else []) + [
        {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", **spec.chat_kwargs)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **spec.chat_kwargs)
    return dict(inputs), text


def transfer_meter(enabled):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext(None)
    from qwen_tnca_layers import TransferMeter
    return TransferMeter()


def run_generation(model, processor, spec, inputs, prompt_text, device, *, bucket, new_tokens,
                   sample=False, temperature=1.0, top_k=0, top_p=1.0, seed=0, forced_tokens=None):
    validate_sampling(temperature, top_k, top_p)
    if new_tokens < 1:
        raise ValueError("new-tokens must be positive")
    if forced_tokens is not None and not 0 < len(forced_tokens) <= new_tokens:
        raise ValueError("forced tokens must be nonempty and no longer than new-tokens")
    if device == "tnca":
        dev = L.use_tnca()
    else:
        dev = torch.device(device)
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    vision = None
    vision_seconds = 0.0
    if "pixel_values" in inputs:
        visual = L.vision_tower(model)
        pre = L.vision_precompute(visual, inputs["image_grid_thw"])
        stage, vdev = L.place(L.VisionStage(visual), device, "eager")
        args = [inputs["pixel_values"], inputs["image_grid_thw"], pre["interp_indices"],
                pre["interp_weights"], pre["position_ids"], pre["cu_seqlens"]]
        started = time.perf_counter()
        with torch.no_grad():
            vision = tuple(t.cpu() for t in stage(*[t.to(vdev) for t in args]))
        vision_seconds = time.perf_counter() - started
        del stage

    position_ids = L.prefill_position_ids(model, inputs)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    if prompt_tokens + new_tokens > bucket:
        raise ValueError(f"static bucket {bucket} is smaller than prompt {prompt_tokens} plus {new_tokens} tokens")
    prefill, pdev = L.place(L.PrefillStage(model, all_logits=False, hidden_states=False), device, "eager",
                            skip=L.vision_tower(model))
    decode, ddev = L.place(L.DecodeStage(model), device, "eager", skip=L.vision_tower(model))
    if ddev != pdev:
        raise ValueError("prefill and decode must use the same device")
    cache = L.make_cache(model, bucket, pdev)
    eos = L.generation_eos(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, logits_rows, tokens, candidate_tokens = [], [], [], []
    old_recompile = dynamo.config.error_on_recompile
    try:
        graph_start = counters["stats"]["unique_graphs"]
        started = time.perf_counter()
        with transfer_meter(device == "tnca") as meter, torch.no_grad():
            result = prefill(inputs["input_ids"].to(pdev), position_ids.to(pdev), cache,
                             None if vision is None else tuple(t.to(pdev) for t in vision))
            if meter is not None:
                meter.mem.synchronize()
        logits = result[0][0, -1].cpu()
        elapsed = time.perf_counter() - started
        token, greedy = select_token(logits, sample, temperature, top_k, top_p, generator)
        candidate_tokens.append(token)
        tokens.append(forced_tokens[0] if forced_tokens else token)
        logits_rows.append(logits)
        rows.append(dict(step=0, kind="prefill", input_tokens=prompt_tokens, seconds=elapsed,
                         new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                         token=tokens[-1], candidate_token=token, greedy_token=greedy,
                         transfers_inside_step=meter.summary() if meter is not None else None))
        position = position_ids[..., -1:]
        for index in range(1, new_tokens):
            if not forced_tokens and tokens[-1] in eos:
                break
            if forced_tokens and index < len(forced_tokens) and tokens[-1] in eos:
                pass
            graph_start = counters["stats"]["unique_graphs"]
            dynamo.config.error_on_recompile = index >= 2
            position = position + 1
            token_input = torch.tensor([[tokens[-1]]], dtype=torch.long)
            started = time.perf_counter()
            with transfer_meter(device == "tnca") as meter, torch.no_grad():
                logits = decode(token_input.to(ddev), position.to(ddev), cache)
                if meter is not None:
                    meter.mem.synchronize()
            logits = logits[0].cpu()
            elapsed = time.perf_counter() - started
            token, greedy = select_token(logits, sample, temperature, top_k, top_p, generator)
            candidate_tokens.append(token)
            forced = forced_tokens[index] if forced_tokens and index < len(forced_tokens) else None
            tokens.append(token if forced is None else forced)
            logits_rows.append(logits)
            rows.append(dict(step=index, kind="decode", input_tokens=1, seconds=elapsed,
                             new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                             token=tokens[-1], candidate_token=token, greedy_token=greedy,
                             transfers_inside_step=meter.summary() if meter is not None else None))
    finally:
        dynamo.config.error_on_recompile = old_recompile
    weight_bytes = L.module_bytes(model, lambda name: not name.startswith("model.visual."))
    report = dict(schema="tnca-qwen-chat-generation/1", status="pass", model=spec.name,
                  repo=spec.repo, revision=spec.revision, device=device,
                  decoder_layers=model.config.get_text_config().num_hidden_layers,
                  prompt_text=prompt_text, prompt_tokens=prompt_tokens, bucket=bucket,
                  requested_new_tokens=new_tokens, sampling=dict(sample=sample, temperature=temperature,
                  top_k=top_k, top_p=top_p, seed=seed), forced_tokens=forced_tokens,
                  generated_tokens=tokens, candidate_tokens=candidate_tokens,
                  generated_text=processor.batch_decode(torch.tensor(tokens)[None], skip_special_tokens=True)[0],
                  stopped_at_eos=bool(tokens and tokens[-1] in eos), eos_token_ids=sorted(eos),
                  vision_seconds=vision_seconds, steps=rows, statistics=generation_stats(rows),
                  logits=logits_rows, fp8_route_in_use=L.fp8_route_in_use(),
                  decode_weight_bytes=weight_bytes,
                  device_memory=L.device_memory(ddev), cache_bytes=L.cache_bytes(cache),
                  standard_torch_fullgraph=device == "tnca", cpu_numerical_fallback=False)
    report["theoretical_core_port_bound"] = dict(
        bytes_per_second=CORE_PORT_BYTES_PER_SECOND, source=CORE_PORT_SOURCE,
        decode_weight_bytes=weight_bytes,
        weight_stream_ceiling_tokens_per_second=CORE_PORT_BYTES_PER_SECOND / weight_bytes,
        interpretation="Theoretical interface ceiling only, not a measured bandwidth or performance prediction")
    return report


def public_report(report):
    return {k: v for k, v in report.items() if k != "logits"}


def save_report(report, out):
    if out is None:
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    if report.get("logits"):
        save_file({"step_%04d" % i: x.contiguous() for i, x in enumerate(report["logits"])},
                  str(out / "logits.safetensors"))
    (out / "report.json").write_text(json.dumps(public_report(report), indent=2) + "\n")


class ChatApplication:
    def __init__(self, args):
        self.args = args
        self.spec = spec_of(args.model)
        if args.device == "tnca":
            L.use_tnca(args.tnca_tree)
        L.set_threads(args.threads)
        self.model, self.load_report = L.load_model(self.spec, "fp8", layers=args.layers)
        L.prepare_for_compile(self.model)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(L.snapshot(self.spec))

    def generate(self, prompt, image=None):
        args = self.args
        inputs, text = make_inputs(self.spec, self.processor, prompt, image)
        reference = None
        forced = None
        if args.check_tokens:
            L.install_fp8_route("reference", dot="exact", scale_div="ieee")
            reference = run_generation(self.model, self.processor, self.spec, inputs, text, "cpu",
                                       bucket=args.bucket, new_tokens=args.check_tokens)
            forced = reference["generated_tokens"]
            gc.collect()
        L.install_fp8_route("reference" if args.device == "cpu" else "official", dot="exact", scale_div="ieee")
        result = run_generation(self.model, self.processor, self.spec, inputs, text, args.device,
                                bucket=args.bucket, new_tokens=args.new_tokens, sample=args.sample,
                                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                seed=args.seed, forced_tokens=forced)
        result["model_scope"] = "reduced-depth-proof" if args.layers else "full-checkpoint"
        result["reduced_depth_layers"] = args.layers
        result["load"] = self.load_report
        result["versions"] = L.versions()
        if reference is not None:
            comparison = compare_logits(reference["logits"], result["logits"][:args.check_tokens])
            passed = all(row["rel_l2"] <= args.max_rel_l2 for row in comparison)
            result["teacher_forced_cpu_w8a8"] = dict(
                reference_tokens=reference["generated_tokens"], checked_tokens=len(comparison),
                max_allowed_rel_l2=args.max_rel_l2, passed=passed, steps=comparison,
                interpretation="Reference tokens feed the device; copied tokens are not the correctness check")
            if not passed:
                result["status"] = "failed-teacher-forced-tolerance"
        return result

    def repl(self):
        print("Qwen TNCA chat. /help lists commands; /quit exits.")
        image = Path(self.args.image).expanduser() if self.args.image else None
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/help":
                print("/greedy; /sample TEMPERATURE [TOP_K TOP_P SEED]; /image PATH|none; /quit")
                continue
            if line == "/greedy":
                self.args.sample = False
                continue
            if line.startswith("/sample"):
                fields = line.split()
                self.args.sample = True
                if len(fields) > 1:
                    self.args.temperature = float(fields[1])
                if len(fields) > 2:
                    self.args.top_k = int(fields[2])
                if len(fields) > 3:
                    self.args.top_p = float(fields[3])
                if len(fields) > 4:
                    self.args.seed = int(fields[4])
                validate_sampling(self.args.temperature, self.args.top_k, self.args.top_p)
                continue
            if line.startswith("/image"):
                fields = line.split(maxsplit=1)
                image = None if len(fields) == 1 or fields[1] == "none" else Path(fields[1]).expanduser()
                continue
            from PIL import Image
            with (Image.open(image) if image is not None else _null_image()) as loaded:
                report = self.generate(line, loaded)
            print("assistant>", report["generated_text"], flush=True)
            stats = report["statistics"]
            print(json.dumps({k: stats[k] for k in ("decode_steps", "mean_decode_seconds_per_token",
                                                     "decode_tokens_per_second")}), flush=True)


class _null_image:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("vl", "q35"), required=True)
    ap.add_argument("--device", choices=("tnca", "cpu"), default="tnca")
    ap.add_argument("--prompt", default="Hello. Introduce yourself briefly.")
    ap.add_argument("--image", help="optional single image; multi-image/video dynamic shapes are not supported")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--bucket", type=int, default=256)
    ap.add_argument("--layers", type=int, help="reduced-depth proof only; not a full-model run")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-tokens", type=int, default=0,
                    help="teacher-force this many greedy CPU-W8A8 tokens before free generation")
    ap.add_argument("--max-rel-l2", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tnca-tree")
    ap.add_argument("--out", type=Path)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    validate_sampling(args.temperature, args.top_k, args.top_p)
    if args.check_tokens < 0 or args.check_tokens > args.new_tokens:
        raise ValueError("check-tokens must be in 0..new-tokens")
    if args.interactive and args.check_tokens:
        raise ValueError("interactive teacher forcing would move the model between CPU and TNCA; use one-shot mode")
    app = ChatApplication(args)
    if args.interactive:
        app.repl()
        return 0
    from PIL import Image
    image = Path(args.image).expanduser() if args.image else None
    try:
        with (Image.open(image) if image is not None else _null_image()) as loaded:
            report = app.generate(args.prompt, loaded)
        save_report(report, args.out)
        print(json.dumps(public_report(report), indent=2), flush=True)
        return 0 if report["status"] == "pass" else 2
    except BaseException:
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "failure.json").write_text(json.dumps(dict(schema="tnca-qwen-chat-generation/1",
                status="failed-preserved", model=app.spec.name, device=args.device,
                error=traceback.format_exc()), indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
