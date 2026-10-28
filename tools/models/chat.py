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


def make_inputs(spec, processor, messages):
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


def code_store_counters():
    """Modules the runtime has unloaded to make room in the device code store, and modules loaded now.
    A working set above the code store is served by least-recently-used reloads (a cost, not a fallback)."""
    try:
        from tnca.triton_backend import driver
    except ImportError:
        return None
    return dict(evicted_total=int(driver.evicted), loaded_modules=len(driver._loaded))


def _counter_delta(before, after):
    if before is None or after is None:
        return None
    return dict(evicted=after["evicted_total"] - before["evicted_total"], loaded_modules=after["loaded_modules"])


def image_rows(grid_thw):
    """Rows of pixel_values that belong to each image (one row per patch)."""
    return [int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()]


class Session:
    """One process-wide generation session on one device.

    The compiled prefill and decode stages and one static cache are created once and reused by every prompt, so
    a REPL conversation keeps its compiled graphs, its device allocations and, when the new conversation text
    extends what is already cached, the cached state itself. A prompt that does not extend the cache resets the
    cache in place (same tensors, zeroed). With prefill_chunk > 0, text is prefilled in fixed-size chunks with
    the remainder through the decode graph, so the number of compiled graphs does not depend on prompt length.
    Prompts that carry images are prefilled whole (their image tokens must stay in one call)."""

    def __init__(self, model, spec, device, bucket, prefill_chunk=0):
        self.model, self.spec, self.device, self.bucket, self.chunk = model, spec, device, bucket, prefill_chunk
        if prefill_chunk < 0:
            raise ValueError("prefill chunk must be nonnegative")
        self.dev = L.use_tnca() if device == "tnca" else torch.device(device)
        skip = L.vision_tower(model)
        self.prefill, pdev = L.place(L.PrefillStage(model, all_logits=False, hidden_states=False), device, "eager",
                                     skip=skip)
        self.decode, ddev = L.place(L.DecodeStage(model), device, "eager", skip=skip)
        if ddev != pdev:
            raise ValueError("prefill and decode must use the same device")
        self.cache = L.make_cache(model, bucket, pdev)
        self.vision = None
        self.cached = []          # token ids whose state the cache holds, in order
        self.last_position = None
        self.resets = 0
        self.prompts = 0

    def reset(self):
        self.cache.reset()
        self.cached, self.last_position = [], None
        self.resets += 1

    def vision_features(self, inputs):
        """The vision tower once per image (each call has exactly one attention segment), features concatenated
        per image the way get_image_features returns them."""
        if "pixel_values" not in inputs:
            return None, []
        visual = L.vision_tower(self.model)
        if self.vision is None:
            self.vision, _ = L.place(L.VisionStage(visual), self.device, "eager")
        grid = inputs["image_grid_thw"]
        rows, calls, start = image_rows(grid), [], 0
        per_image = []
        for index, count in enumerate(rows):
            g = grid[index:index + 1]
            pre = L.vision_precompute(visual, g)
            args = [inputs["pixel_values"][start:start + count], g, pre["interp_indices"], pre["interp_weights"],
                    pre["position_ids"], pre["cu_seqlens"]]
            start += count
            started = time.perf_counter()
            with torch.no_grad():
                out = tuple(t.cpu() for t in self.vision(*[t.to(self.dev) for t in args]))
            calls.append(dict(image=index, grid_thw=g.tolist()[0], seconds=time.perf_counter() - started))
            per_image.append(out)
        if start != inputs["pixel_values"].shape[0]:
            raise ValueError("pixel rows do not match image_grid_thw")
        return per_image, calls

    def _call_prefill(self, ids, positions, vision):
        result = self.prefill(ids.to(self.dev), positions.to(self.dev), self.cache,
                              None if vision is None else vision)
        return result[0][0, -1]

    def _call_decode(self, ids, positions):
        return self.decode(ids.to(self.dev), positions.to(self.dev), self.cache)[0]

    def prefill_prompt(self, inputs, per_image):
        """Bring the cache to the state after the whole prompt; return (last logits on host, call rows, reused)."""
        from torch._dynamo.utils import counters
        ids = inputs["input_ids"]
        positions = L.prefill_position_ids(self.model, inputs)
        full = ids[0].tolist()
        reused = len(self.cached)
        if per_image or full[:reused] != self.cached or reused >= len(full):
            if self.cached:
                self.reset()
            reused = 0
        if len(full) > self.bucket:
            raise ValueError(f"static bucket {self.bucket} is smaller than the {len(full)}-token conversation")
        vision = None
        if per_image:
            vision = (tuple(img[0] for img in per_image),
                      *[tuple(img[j] for img in per_image) for j in range(1, len(per_image[0]))])
            vision = _vision_to(vision, self.dev)
        rows, logits, at = [], None, reused
        segments = []
        if vision is not None or not self.chunk:
            segments.append(("prefill", at, len(full)))
        else:
            while len(full) - at >= self.chunk:
                segments.append(("prefill", at, at + self.chunk))
                at += self.chunk
            segments += [("token", i, i + 1) for i in range(at, len(full))]
        for kind, a, b in segments:
            graphs = counters["stats"]["unique_graphs"]
            started = time.perf_counter()
            with torch.no_grad():
                if kind == "prefill":
                    out = self._call_prefill(ids[:, a:b], positions[..., a:b], vision)
                else:
                    out = self._call_decode(ids[:, a:b], positions[..., a:b])
            rows.append(dict(kind=kind, first=a, tokens=b - a, seconds=time.perf_counter() - started,
                             new_graphs=counters["stats"]["unique_graphs"] - graphs))
            logits = out
        self.cached = full
        self.last_position = positions[..., -1:]
        return logits, rows, reused

    def step(self, token):
        position = self.last_position + 1
        with torch.no_grad():
            logits = self._call_decode(torch.tensor([[token]], dtype=torch.long), position)
        self.cached = self.cached + [token]
        self.last_position = position
        return logits


def _vision_to(vision, dev):
    return tuple(tuple(t.to(dev) for t in group) for group in vision)


def run_generation(session, processor, spec, inputs, prompt_text, *, new_tokens, per_image=None,
                   vision_calls=None, sample=False, temperature=1.0, top_k=0, top_p=1.0, seed=0,
                   forced_tokens=None):
    validate_sampling(temperature, top_k, top_p)
    if new_tokens < 1:
        raise ValueError("new-tokens must be positive")
    if forced_tokens is not None and not 0 < len(forced_tokens) <= new_tokens:
        raise ValueError("forced tokens must be nonempty and no longer than new-tokens")
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    model, device = session.model, session.device
    tnca = device == "tnca"
    prompt_tokens = int(inputs["input_ids"].shape[1])
    if prompt_tokens + new_tokens > session.bucket:
        raise ValueError(f"static bucket {session.bucket} is smaller than prompt {prompt_tokens} plus {new_tokens} tokens")
    eos = L.generation_eos(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, logits_rows, tokens, candidate_tokens = [], [], [], []
    old_recompile = dynamo.config.error_on_recompile
    session.prompts += 1
    graphs_at_start = counters["stats"]["unique_graphs"]
    try:
        graph_start = counters["stats"]["unique_graphs"]
        code = code_store_counters() if tnca else None
        started = time.perf_counter()
        with transfer_meter(tnca) as meter:
            logits, calls, reused = session.prefill_prompt(inputs, per_image)
            if meter is not None:
                meter.mem.synchronize()
            logits = logits.cpu()
        elapsed = time.perf_counter() - started
        token, greedy = select_token(logits, sample, temperature, top_k, top_p, generator)
        candidate_tokens.append(token)
        tokens.append(forced_tokens[0] if forced_tokens else token)
        logits_rows.append(logits)
        rows.append(dict(step=0, kind="prefill", input_tokens=prompt_tokens - reused, reused_cached_tokens=reused,
                         seconds=elapsed, calls=calls,
                         new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                         token=tokens[-1], candidate_token=token, greedy_token=greedy,
                         code_store=_counter_delta(code, code_store_counters() if tnca else None),
                         transfers_inside_step=meter.summary() if meter is not None else None))
        for index in range(1, new_tokens):
            if not forced_tokens and tokens[-1] in eos:
                break
            graph_start = counters["stats"]["unique_graphs"]
            # The first decode step may compile the decode graph; every later step must reuse it.
            dynamo.config.error_on_recompile = index >= 2
            code = code_store_counters() if tnca else None
            started = time.perf_counter()
            with transfer_meter(tnca) as meter:
                logits = session.step(tokens[-1])
                if meter is not None:
                    meter.mem.synchronize()
                logits = logits.cpu()
            elapsed = time.perf_counter() - started
            token, greedy = select_token(logits, sample, temperature, top_k, top_p, generator)
            candidate_tokens.append(token)
            forced = forced_tokens[index] if forced_tokens and index < len(forced_tokens) else None
            tokens.append(token if forced is None else forced)
            logits_rows.append(logits)
            rows.append(dict(step=index, kind="decode", input_tokens=1, seconds=elapsed,
                             new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                             token=tokens[-1], candidate_token=token, greedy_token=greedy,
                             code_store=_counter_delta(code, code_store_counters() if tnca else None),
                             transfers_inside_step=meter.summary() if meter is not None else None))
    finally:
        dynamo.config.error_on_recompile = old_recompile
    weight_bytes = L.module_bytes(model, lambda name: not name.startswith("model.visual."))
    report = dict(schema="tnca-qwen-chat-generation/2", status="pass", model=spec.name,
                  repo=spec.repo, revision=spec.revision, device=device,
                  decoder_layers=model.config.get_text_config().num_hidden_layers,
                  prompt_text=prompt_text, prompt_tokens=prompt_tokens, bucket=session.bucket,
                  prefill_chunk=session.chunk, requested_new_tokens=new_tokens,
                  sampling=dict(sample=sample, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed),
                  forced_tokens=forced_tokens, generated_tokens=tokens, candidate_tokens=candidate_tokens,
                  generated_text=processor.batch_decode(torch.tensor(tokens)[None], skip_special_tokens=True)[0],
                  stopped_at_eos=bool(tokens and tokens[-1] in eos), eos_token_ids=sorted(eos),
                  images=len(per_image or []), vision_calls=vision_calls or [],
                  vision_seconds=sum(c["seconds"] for c in vision_calls or []),
                  steps=rows, statistics=generation_stats(rows),
                  logits=logits_rows, fp8_route_in_use=L.fp8_route_in_use(),
                  decode_weight_bytes=weight_bytes,
                  session=dict(prompt_index=session.prompts, cache_resets=session.resets,
                               new_graphs_this_prompt=counters["stats"]["unique_graphs"] - graphs_at_start,
                               unique_graphs_total=counters["stats"]["unique_graphs"]),
                  device_memory=L.device_memory(session.dev), cache_bytes=L.cache_bytes(session.cache),
                  standard_torch_fullgraph=tnca, cpu_numerical_fallback=False)
    if tnca:
        report["device_target"] = device_target()
    report["theoretical_core_port_bound"] = dict(
        bytes_per_second=CORE_PORT_BYTES_PER_SECOND, source=CORE_PORT_SOURCE,
        decode_weight_bytes=weight_bytes,
        weight_stream_ceiling_tokens_per_second=CORE_PORT_BYTES_PER_SECOND / weight_bytes,
        interpretation="Theoretical interface ceiling only, not a measured bandwidth or performance prediction")
    return report


def device_target():
    """What the run compiled for and ran on (the operations the image offers are the compiler's business:
    the driver issues standard Torch operations and never selects kernels itself)."""
    keys = ("TNCA_BACKEND", "TNCA_PROFILE", "TNCA_BACKEND_ARGS", "TNCA_OPERATING_CLOCK_EVIDENCE")
    return {k: os.environ.get(k) for k in keys}


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
        self.session = None
        self.history = []
        self.transcript = None    # (rendered text, token ids) of the conversation so far, generated ids kept as is

    def device_session(self):
        """Created on first use, after any CPU reference has run (placing it moves the weights to the device)."""
        if self.session is None:
            L.install_fp8_route("reference" if self.args.device == "cpu" else "official", dot="exact",
                                scale_div="ieee")
            self.session = Session(self.model, self.spec, self.args.device, self.args.bucket,
                                   self.args.prefill_chunk)
        return self.session

    def reset_conversation(self):
        self.history = []
        self.transcript = None

    def continuation(self, messages, inputs, text):
        """Token ids for a text-only follow-up turn that keep the generated ids exactly as generated: the
        previous transcript's ids plus the tokenized new template text. Returns None when the new rendering does
        not extend the previous one (the session then resets and prefills the whole conversation)."""
        if self.transcript is None or "pixel_values" in inputs:
            return None
        old_text, old_ids = self.transcript
        if not text.startswith(old_text):
            return None
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        suffix = tokenizer(text[len(old_text):], add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([torch.tensor([old_ids], dtype=torch.long), suffix], dim=1)
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))

    def reference(self, inputs, text, per_image_cpu, new_tokens, forced=None):
        cpu = Session(self.model, self.spec, "cpu", self.args.bucket, 0)
        return run_generation(cpu, self.processor, self.spec, inputs, text, new_tokens=new_tokens,
                              per_image=per_image_cpu, forced_tokens=forced)

    def generate(self, prompt, images=(), forced=None, keep_history=False, check=True):
        args = self.args
        messages = (self.history if keep_history else []) + [user_message(prompt, images)]
        inputs, text = make_inputs(self.spec, self.processor, messages)
        if keep_history:
            inputs = self.continuation(messages, inputs, text) or inputs
        reference = scale_floor = None
        if check and args.check_tokens and forced is None:
            if self.session is not None:
                raise RuntimeError("the CPU reference must run before the model is placed on the device")
            L.install_fp8_route("reference", dot="exact", scale_div="ieee")
            cpu = Session(self.model, self.spec, "cpu", args.bucket, 0)
            per_image_cpu, _ = cpu.vision_features(inputs)
            reference = run_generation(cpu, self.processor, self.spec, inputs, text, new_tokens=args.check_tokens,
                                       per_image=per_image_cpu)
            forced = reference["generated_tokens"]
            L.install_fp8_route("reference", dot="exact", scale_div="recip")
            cpu = Session(self.model, self.spec, "cpu", args.bucket, 0)
            per_image_cpu, _ = cpu.vision_features(inputs)
            reciprocal = run_generation(cpu, self.processor, self.spec, inputs, text, new_tokens=args.check_tokens,
                                        per_image=per_image_cpu, forced_tokens=forced)
            scale_floor = compare_logits(reference["logits"], reciprocal["logits"])
            del cpu
            gc.collect()
        session = self.device_session()
        per_image, vision_calls = session.vision_features(inputs)
        result = run_generation(session, self.processor, self.spec, inputs, text, new_tokens=args.new_tokens,
                                per_image=per_image, vision_calls=vision_calls, sample=args.sample,
                                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                seed=args.seed, forced_tokens=forced)
        result["model_scope"] = "reduced-depth-proof" if args.layers else "full-checkpoint"
        result["reduced_depth_layers"] = args.layers
        result["load"] = self.load_report
        result["versions"] = L.versions()
        if reference is not None:
            comparison = compare_logits(reference["logits"], result["logits"][:args.check_tokens])
            # The gate is the fixed --max-rel-l2 limit, set before any device
            # output is seen (discipline 2.1.6). The CPU reciprocal-scale floor
            # is logged as a diagnostic only; it never raises the limit. A
            # floor-derived limit was introduced after TLM q35-v1 failed and
            # was withdrawn (evidence/r41/hbm_bw/AUDIT.md, R6).
            for row, floor in zip(comparison, scale_floor, strict=True):
                row["cpu_scale_rounding_floor"] = floor["rel_l2"]
                row["allowed_rel_l2"] = args.max_rel_l2
            passed = all(row["rel_l2"] <= args.max_rel_l2 for row in comparison)
            result["teacher_forced_cpu_w8a8"] = dict(
                reference_tokens=reference["generated_tokens"], checked_tokens=len(comparison),
                max_allowed_rel_l2=args.max_rel_l2, passed=passed, steps=comparison,
                interpretation="Reference tokens feed the device; the fixed max_allowed_rel_l2 is the gate; the CPU reciprocal-scale floor is a logged diagnostic and does not change the gate")
            if not passed:
                result["status"] = "failed-teacher-forced-tolerance"
        if keep_history:
            self.history = messages + [dict(role="assistant", content=[dict(type="text",
                                                                              text=result["generated_text"])])]
            generated = self.processor.batch_decode(torch.tensor(result["generated_tokens"])[None],
                                                    skip_special_tokens=False)[0]
            self.transcript = (text + generated, inputs["input_ids"][0].tolist() + result["generated_tokens"])
        return result

    def session_check(self, first, first_prompt, first_images, prompt):
        """Persistent-session check: in the same process, session and cache, run a different conversation and
        then the first prompt again with the first run's tokens forced. The repeat must reproduce the first run's
        tokens and every logit bit for bit (code-store reloads, allocator reuse and cache reset are exercised
        in between), and must compile nothing new."""
        other = self.generate(prompt, check=False)
        again = self.generate(first_prompt, first_images, forced=first["generated_tokens"])
        mismatched = [i for i, (x, y) in enumerate(zip(first["logits"], again["logits"], strict=True))
                      if not torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x,
                                         y.view(torch.int16) if y.dtype == torch.bfloat16 else y)]
        passed = not mismatched and again["candidate_tokens"] == first["candidate_tokens"]
        return dict(passed=passed, other_prompt=prompt, other=public_report(other), repeat=public_report(again),
                    bitwise_logit_mismatch_steps=mismatched,
                    repeat_new_graphs=again["session"]["new_graphs_this_prompt"],
                    code_store_evictions=sum((r.get("code_store") or {}).get("evicted", 0)
                                             for r in other["steps"] + again["steps"]),
                    interpretation="same-session repeat after an intervening conversation; bitwise equality of "
                                   "logits and candidate tokens is required")

    def repl(self, lines=None, transcript=None):
        """Interactive chat; with `lines` (script mode) the same commands come from a file, every turn's report
        is appended to `transcript`, and a failed turn stops the script."""
        print("Qwen TNCA chat. /help lists commands; /quit exits.")
        images = [Path(p).expanduser() for p in (self.args.image or [])]
        source = iter(lines) if lines is not None else None
        while True:
            try:
                line = (next(source) if source is not None else input("you> ")).strip()
                if source is not None:
                    print("you>", line, flush=True)
            except (EOFError, StopIteration):
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/help":
                print("/greedy; /sample TEMPERATURE [TOP_K TOP_P SEED]; /image PATH [PATH...]|none; /reset; /quit")
                continue
            if line == "/reset":
                self.reset_conversation()
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
                fields = line.split()[1:]
                images = [] if not fields or fields == ["none"] else [Path(f).expanduser() for f in fields]
                continue
            try:
                report = self.generate(line, open_images(images), keep_history=True)
            except ValueError as error:          # e.g. the conversation no longer fits the static bucket
                print("error>", error, "(use /reset)", flush=True)
                if transcript is not None:
                    transcript.append(dict(user=line, status="refused", error=str(error)))
                    return False
                continue
            images = []
            if transcript is not None:
                transcript.append(dict(user=line, assistant=report["generated_text"], report=public_report(report)))
            print("assistant>", report["generated_text"], flush=True)
            stats = report["statistics"]
            print(json.dumps({k: stats[k] for k in ("prefill_seconds", "decode_steps",
                                                     "mean_decode_seconds_per_token",
                                                     "decode_tokens_per_second")}
                             | {"reused_cached_tokens": report["steps"][0]["reused_cached_tokens"],
                                "new_graphs": report["session"]["new_graphs_this_prompt"]}), flush=True)
        return True


def user_message(prompt, images=()):
    return dict(role="user", content=[dict(type="image", image=image) for image in images]
                + [dict(type="text", text=prompt)])


def open_images(paths):
    from PIL import Image
    loaded = []
    for path in paths:
        with Image.open(path) as image:
            loaded.append(image.convert("RGB"))
    return loaded


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("vl", "q35"), required=True)
    ap.add_argument("--device", choices=("tnca", "cpu"), default="tnca")
    ap.add_argument("--prompt", default="Hello. Introduce yourself briefly.")
    ap.add_argument("--image", action="append",
                    help="an image for the prompt (repeatable); the vision tower runs once per image")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--script", type=Path,
                    help="run the interactive session from this file (one prompt or /command per line) and write "
                         "a transcript report to --out")
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
    ap.add_argument("--max-rel-l2", type=float, default=0.05,
                    help="fixed per-token relative-L2 limit; fixed before the run, never derived from its output")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="prefill text in chunks of this many tokens (remainder through the decode graph), so the "
                         "compiled graphs do not depend on prompt length; 0 prefills each prompt whole")
    ap.add_argument("--session-check", metavar="PROMPT",
                    help="after the one-shot prompt, run PROMPT and then the first prompt again in the same "
                         "session and require bitwise-identical logits and tokens")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tnca-tree")
    ap.add_argument("--out", type=Path)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    validate_sampling(args.temperature, args.top_k, args.top_p)
    if args.check_tokens < 0 or args.check_tokens > args.new_tokens:
        raise ValueError("check-tokens must be in 0..new-tokens")
    if args.max_rel_l2 < 0:
        raise ValueError("numerical allowances must be nonnegative")
    if (args.interactive or args.script) and args.check_tokens:
        raise ValueError("interactive teacher forcing would move the model between CPU and TNCA; use one-shot mode")
    app = ChatApplication(args)
    if args.interactive:
        app.repl()
        return 0
    if args.script:
        transcript = []
        try:
            completed = app.repl(args.script.read_text().splitlines(), transcript)
        finally:
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                session = app.session
                (args.out / "transcript.json").write_text(json.dumps(dict(
                    schema="tnca-qwen-chat-transcript/1", model=app.spec.name, device=args.device,
                    model_scope="reduced-depth-proof" if args.layers else "full-checkpoint",
                    script=str(args.script), turns=transcript, load=app.load_report, versions=L.versions(),
                    cache_resets=session.resets if session else None,
                    device_target=device_target() if args.device == "tnca" else None), indent=2) + "\n")
        return 0 if completed and all(t.get("report", {}).get("status") == "pass" for t in transcript) else 2
    images = open_images([Path(p).expanduser() for p in (args.image or [])])
    try:
        report = app.generate(args.prompt, images)
        if args.session_check:
            check = app.session_check(report, args.prompt, images, args.session_check)
            report["persistent_session_check"] = check
            if not check["passed"] and report["status"] == "pass":
                report["status"] = "failed-persistent-session-check"
        save_report(report, args.out)
        print(json.dumps(public_report(report), indent=2), flush=True)
        return 0 if report["status"] == "pass" else 2
    except BaseException:
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "failure.json").write_text(json.dumps(dict(schema="tnca-qwen-chat-generation/2",
                status="failed-preserved", model=app.spec.name, device=args.device,
                error=traceback.format_exc()), indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
