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
import time
_MODULE_STARTED = time.perf_counter()

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import threading
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import qwen_fp8_lib as L  # noqa: E402
from chat_timing import (compiler_delta, compiler_snapshot, finalize_user_visible_metrics,
                         user_visible_metrics)  # noqa: E402


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


def validate_device_sampling(device, sample):
    if device == 'tnca' and sample:
        raise ValueError('TNCA device-side sampling is not implemented; use greedy decoding')


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
    if not reference or len(reference) != len(candidate):
        raise ValueError('logit comparison requires matching nonempty step sets')
    rows = []
    for index, (want, got) in enumerate(zip(reference, candidate)):
        x, y = want.double().flatten(), got.double().flatten()
        if x.numel() == 0 or x.numel() != y.numel():
            raise ValueError(f'logit comparison step {index} has incompatible vocabulary sizes')
        rows.append(dict(step=index,
                         rel_l2=float((y - x).norm() / x.norm().clamp_min(1e-300)),
                         reference_argmax=int(want.float().argmax()),
                         candidate_argmax=int(got.float().argmax()),
                         argmax_equal=bool(int(want.float().argmax()) == int(got.float().argmax()))))
    return rows


def generation_stats(rows):
    decode = [row for row in rows if row["kind"] == "decode"]
    seconds = sum(row["seconds"] for row in decode)
    warm_prefill = (rows[0].get('prefill_seconds_in_warmup', 0.0) if rows else 0.0)
    return dict(generated_tokens=len(rows), decode_steps=len(decode),
                prefill_seconds=(rows[0]["seconds"] + warm_prefill) if rows else 0.0,
                prefill_seconds_already_in_warmup=warm_prefill,
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


def transfer_meter(enabled, *, deferred=False):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext(None)
    from qwen_tnca_layers import TransferMeter
    return TransferMeter(compact=True, deferred=deferred)


def run_with_code_capture(fn):
    """Capture both new and persistent-cache Inductor wrappers for fallback checks."""
    from unittest.mock import patch
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.graph import GraphLowering
    code = []
    def save_output_code(source):
        if source not in code:
            code.append(source)
    original_load = PyCodeCache.load_by_key_path
    def load_and_capture(cls, key, path, *args, **kwargs):
        module = original_load(key, path, *args, **kwargs)
        # Never treat an unreadable cached wrapper as proof of no host work.
        try:
            source = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"cannot inspect cached Inductor wrapper {path}") from error
        save_output_code(source)
        return module
    with patch.object(GraphLowering, "save_output_code", save_output_code), \
            patch.object(PyCodeCache, "load_by_key_path", classmethod(load_and_capture)):
        result = fn()
    return result, code


def capture_generation_role(role, fn, code_by_role):
    """Capture new wrappers once; reject later recompiles on stable roles.

    Decode and token selection have fixed shapes within a session. Their
    already-inspected wrappers need not be patched and copied on every
    token, but a new wrapper must never escape the no-fallback inspection.
    Prefill can legitimately change with prompt length, so it is always
    captured. An attempted steady-role recompile fails rather than running
    an uninspected graph.
    """
    if role in ('decode', 'selection') and code_by_role.get(role):
        config = torch._dynamo.config
        previous = config.error_on_recompile
        try:
            config.error_on_recompile = True
            return fn()
        finally:
            config.error_on_recompile = previous
    result, code = run_with_code_capture(fn)
    code_by_role.setdefault(role, []).extend(code)
    return result


def generation_fallbacks(session, code_by_role, inspect):
    """Reuse an inspection only while every captured wrapper is unchanged.

    New or replaced wrappers always force a complete inspection, including
    cross-wrapper host-metadata classification. Cached lists are copied so
    a caller cannot accidentally change the next prompt's verdict.
    """
    identity = tuple((role, tuple(sources)) for role, sources in code_by_role.items())
    cached = getattr(session, '_fallback_inspection', None)
    if cached is None or cached[0] != identity:
        found = {role: tuple(inspect(sources)) for role, sources in code_by_role.items()}
        session._fallback_inspection = (identity, found)
    else:
        found = cached[1]
    return {role: list(items) for role, items in found.items()}


def code_store_counters():
    """Process-wide successful code loads, evictions and current resident aliases."""
    try:
        from tnca.triton_backend import driver
    except ImportError:
        return None
    sessions = list(driver._Sessions.open.values())
    return dict(loaded_total=int(driver.loaded), evicted_total=int(driver.evicted),
                loaded_modules=len(driver._loaded),
                code_words_written=sum(getattr(getattr(s, 'dev', None), 'code_words_written', 0) for s in sessions),
                code_words_skipped=sum(getattr(getattr(s, 'dev', None), 'code_words_skipped', 0) for s in sessions))


def _counter_delta(before, after):
    if before is None or after is None:
        return None
    return dict(loads=after["loaded_total"] - before["loaded_total"],
                evicted=after["evicted_total"] - before["evicted_total"], loaded_modules=after["loaded_modules"],
                code_bytes_written=8 * (after.get('code_words_written', 0) - before.get('code_words_written', 0)),
                code_bytes_unchanged=8 * (after.get('code_words_skipped', 0) - before.get('code_words_skipped', 0)),
                scope='process-wide successful code loads and evictions')


def image_rows(grid_thw):
    """Rows of pixel_values that belong to each image (one row per patch)."""
    return [int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()]


class Session:
    """One process-wide generation session on one device.

    The compiled prefill and decode stages and one static cache are created once and reused by every prompt, so
    a REPL conversation keeps its compiled graphs, its device allocations and, when the new conversation text
    extends what is already cached, the cached state itself. A prompt that does not extend the cache resets the
    cache in place (same tensors, zeroed). With prefill_chunk > 0, text is prefilled in fixed-size chunks with
    the final short block through prefill, so no token in the prompt is silently treated as a decode step.
    Prompts that carry images are prefilled whole (their image tokens must stay in one call)."""

    def __init__(self, model, spec, device, bucket, prefill_chunk=0):
        self.model, self.spec, self.device, self.bucket, self.chunk = model, spec, device, bucket, prefill_chunk
        if prefill_chunk < 0:
            raise ValueError("prefill chunk must be nonnegative")
        self.dev = L.use_tnca() if device == "tnca" else torch.device(device)
        # A device session keeps the vision weights resident as well as the text weights.
        skip = None if device == "tnca" else L.vision_tower(model)
        prefill_stage = L.PrefillStage(model, all_logits=False, hidden_states=False)
        decode_stage = L.DecodeStage(model)
        self.prefill, pdev = L.place(prefill_stage, device, "eager", skip=skip)
        # Both stages wrap the identical model. Its parameters and buffers
        # were placed above; traversing the entire checkpoint again adds to
        # first-prompt latency without moving any additional tensor.
        self.decode, ddev = L.place(decode_stage, device, "eager", skip=skip,
                                    already_placed=decode_stage.model is prefill_stage.model)
        if ddev != pdev:
            raise ValueError("prefill and decode must use the same device")
        self.cache = L.make_cache(model, bucket, pdev)
        self.vision = None
        self.cached = []          # token ids whose state the cache holds, in order
        self._cache_tokens = 0    # host metadata; never read a device tensor
        self.last_position = None
        self._warm_prefill = None
        self._warm_prefill_mode = 'not-run'
        self._pending_reset = False
        self._cleanup_thread = None
        self._cleanup_error = None
        self.resets = 0
        self.prompts = 0
        self.greedy = (torch.compile(lambda x: x.float().argmax().reshape(1, 1),
                                     fullgraph=True, dynamic=False) if device == "tnca" else None)

    def reset(self):
        self.wait_for_cleanup()
        if not (self.device == 'tnca' and os.environ.get('TNCA_DEVICE_ZERO') == '1'
                and clear_static_attention_cache(self.cache)):
            self.cache.reset()
        self.cached, self.last_position = [], None
        self._cache_tokens = 0
        self._warm_prefill = None
        self._pending_reset = False
        self.resets += 1

    def wait_for_cleanup(self):
        """Join an earlier idle-time reset before any cache is used again."""
        worker = getattr(self, '_cleanup_thread', None)
        if worker is None or worker is threading.current_thread():
            return
        worker.join()
        self._cleanup_thread = None
        error, self._cleanup_error = getattr(self, '_cleanup_error', None), None
        if error is not None:
            raise RuntimeError('background cache cleanup failed') from error

    def schedule_idle_cleanup(self):
        """Reset discarded state while an interactive user considers a reply."""
        self.wait_for_cleanup()
        if not getattr(self, '_pending_reset', False):
            return False
        def cleanup():
            try:
                # Inference mode is thread-local; cached tensors may have
                # been created in that mode on the generation thread.
                with torch.inference_mode():
                    self.reset()
                    if self.device == 'tnca':
                        from tnca.torch_device import MEMORY
                        MEMORY.synchronize()
            except BaseException as error:
                self._cleanup_error = error
        self._cleanup_error = None
        worker = threading.Thread(target=cleanup, name='tnca-cache-cleanup')
        self._cleanup_thread = worker
        worker.start()
        return True

    def defer_reset(self):
        """Invalidate speculative state now; clean device storage on next use.

        A recurrent cache cannot be rewound after speculative post-EOS work.
        Its full reset may be expensive, but is unnecessary for returning
        this answer or when a one-shot process exits. A following prompt
        performs the same reset before it can consume any cache state.
        """
        self.wait_for_cleanup()
        self.cached, self.last_position = [], None
        self._cache_tokens = 0
        self._warm_prefill = None
        self._pending_reset = True

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
            with torch.no_grad(), transfer_meter(self.device == "tnca") as meter:
                if self.device == "tnca":
                    from tnca.inductor import host_fallbacks
                    result, code = run_with_code_capture(lambda: self.vision(*[t.to(self.dev) for t in args]))
                    self._vision_code = getattr(self, "_vision_code", []) + code
                    meter.mem.synchronize()
                    out = tuple(result)
                    fallback = host_fallbacks(self._vision_code) if self._vision_code else None
                    if any(t.device.type != "tnca" for t in out):
                        raise RuntimeError("vision returned a host activation")
                else:
                    out = tuple(self.vision(*[t.to(self.dev) for t in args]))
                    fallback = None
            calls.append(dict(image=index, grid_thw=g.tolist()[0], seconds=time.perf_counter() - started,
                              transfers=meter.summary() if meter is not None else None,
                              compiled_wrapper_fallbacks=fallback))
            per_image.append(out)
        if start != inputs["pixel_values"].shape[0]:
            raise ValueError("pixel rows do not match image_grid_thw")
        return per_image, calls

    def _call_prefill(self, ids, positions, vision):
        # Canonical strides: a slice of the whole prompt keeps the prompt length as the stride of its size-1 batch
        # dimension, and the compiled graph is guarded on input strides, so every new prompt length recompiled
        # (dynamo: "tensor 'input_ids' stride mismatch at index 0"; CPU graph-count check,
        # evidence/r41/chat/p3/graphcount.tgz).
        ids = ids.clone(memory_format=torch.contiguous_format)
        positions = positions.clone(memory_format=torch.contiguous_format)
        result = self.prefill(ids.to(self.dev), positions.to(self.dev), self.cache,
                              None if vision is None else vision)
        return result[0][0, -1]

    def _call_decode(self, ids, positions):
        # Greedy feedback is already a device tensor. Cloning its single
        # token launched a separate device copy on every decode step. A
        # one-element view has a canonical stride without copying bytes;
        # other input shapes retain the previous contiguous-copy behavior.
        if tuple(ids.shape) == (1, 1):
            ids = ids.as_strided((1, 1), (1, 1))
        else:
            ids = ids.clone(memory_format=torch.contiguous_format)
        if positions.device.type == 'tnca' and positions.is_contiguous():
            stride, strides = 1, []
            for size in reversed(positions.shape):
                strides.append(stride)
                stride *= size
            positions = positions.as_strided(positions.shape, tuple(reversed(strides)))
        else:
            positions = positions.clone(memory_format=torch.contiguous_format)
        return self.decode(ids.to(self.dev), positions.to(self.dev), self.cache)[0]

    def prefill_prompt(self, inputs, per_image):
        """Bring the cache to the state after the whole prompt; return (last logits on host, call rows, reused)."""
        self.wait_for_cleanup()
        if getattr(self, '_pending_reset', False):
            self.reset()
        from torch._dynamo.utils import counters
        ids = inputs["input_ids"]
        positions = L.prefill_position_ids(self.model, inputs)
        full = ids[0].tolist()
        reused = len(self.cached)
        if (per_image or full[:reused] != self.cached or reused >= len(full)
                or getattr(self, '_cache_tokens', 0) != reused):
            if self.cached or getattr(self, '_cache_tokens', 0):
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
            if at < len(full):
                segments.append(("prefill", at, len(full)))
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
        self._cache_tokens = len(full)
        self.last_position = positions[..., -1:]
        return logits, rows, reused

    def step(self, token):
        self.wait_for_cleanup()
        position = self.last_position + 1
        with torch.no_grad():
            logits = self._call_decode(torch.tensor([[token]], dtype=torch.long), position)
        self.cached = self.cached + [token]
        self._cache_tokens += 1
        self.last_position = position
        return logits

    def step_device(self, token, position_device=None, position_host=None):
        """Feed a device token without converting its value to a host scalar."""
        self.wait_for_cleanup()
        if self.last_position is None:
            raise RuntimeError('decode requires a populated prompt cache')
        if not torch.is_tensor(token) or token.device.type != self.dev.type:
            raise ValueError('decode feedback token must already be resident on the session device')
        if (position_device is None) != (position_host is None):
            raise ValueError('queued decode needs both host and device position metadata')
        if position_device is not None:
            if (not torch.is_tensor(position_device) or not torch.is_tensor(position_host)
                    or position_device.device.type != 'tnca' or position_host.device.type != 'cpu'
                    or position_device.shape != position_host.shape
                    or position_device.dtype != position_host.dtype
                    or position_host.shape != self.last_position.shape):
                raise ValueError('queued decode position metadata is incompatible with the cache')
        position = self.last_position + 1 if position_host is None else position_host
        with torch.no_grad():
            logits = self._call_decode(token, position if position_device is None else position_device)
        self.last_position = position
        self._cache_tokens += 1
        return logits

    def warmup_decode(self, inputs, per_image=None, capture=None):
        """Compile against a populated prompt; optionally retain its exact KV state."""
        self.wait_for_cleanup()
        if getattr(self, '_pending_reset', False):
            self.reset()
        if self.cached or self.last_position is not None:
            raise RuntimeError("decode warmup requires a fresh cache")
        capture = capture or (lambda role, fn: fn())
        retained = False
        # Reuse is the normal device path when the cache can be rewound
        # exactly; unsupported caches retain the existing reset fallback.
        reuse = (os.environ.get('TNCA_REUSE_WARM_PREFILL', '1') == '1'
                 and getattr(self, 'device', None) == 'tnca')
        self._warm_prefill_mode = 'fallback-reset' if reuse else 'disabled'
        try:
            prefill_started = time.perf_counter()
            prefill = capture("prefill", lambda: self.prefill_prompt(inputs, per_image))
            length, position = (self._cache_tokens, self.last_position) if reuse else (None, None)
            token = torch.zeros((1, 1), dtype=torch.long).to(self.dev)
            # Uploading the dummy token drains the prefill queue, so this
            # captures its full wall span instead of just enqueue time.
            prefill_seconds = time.perf_counter() - prefill_started
            hybrid = (snapshot_hybrid_prefill(self.cache, length)
                      if reuse and os.environ.get('TNCA_REUSE_HYBRID_PREFILL', '1') == '1' else None)
            result = capture("decode", lambda: self.step_device(token))
            if (reuse and position is not None and len(self.cached) == length
                    and self._cache_tokens == length + 1
                    and inputs['input_ids'].device.type == 'cpu'
                    and (restore_hybrid_prefill(hybrid) if hybrid is not None
                         else rewind_attention_cache(self.cache, length, dirty_length=length + 1))):
                # Prompt logits and cache state retain the same device bits;
                # do not recompute the whole prompt after the dummy decode.
                self.last_position, self._cache_tokens = position, length
                self._warm_prefill = (inputs, list(self.cached), per_image, prefill, prefill_seconds)
                self._warm_prefill_mode = 'hybrid-restored' if hybrid is not None else 'static-rewound'
                retained = True
            return result
        finally:
            if not retained:
                self.reset()

    def consume_warm_prefill(self, inputs, per_image):
        """Consume a retained warmup result only for the identical prompt."""
        saved, self._warm_prefill = self._warm_prefill, None
        if saved is None:
            return None
        original_inputs, tokens, images, prefill, seconds = saved
        if (original_inputs is inputs and images is per_image
                and tokens == inputs['input_ids'][0].tolist()
                and self.cached == tokens and self._cache_tokens == len(tokens)):
            return prefill, seconds
        return None


def _vision_to(vision, dev):
    return tuple(tuple(t.to(dev) for t in group) for group in vision)


def stage_decode_positions(last_position, count, device):
    """Prepare host and device position metadata once for queued decode."""
    if (not torch.is_tensor(last_position) or last_position.device.type != 'cpu'
            or last_position.dtype not in (torch.int32, torch.int64)
            or last_position.numel() == 0 or type(count) is not int or count < 2):
        raise ValueError('queued decode requires a CPU position and at least two tokens')
    shape = (count - 1,) + (1,) * last_position.ndim
    increments = torch.arange(1, count, dtype=last_position.dtype).reshape(shape)
    host = (last_position.unsqueeze(0) + increments).contiguous()
    return host, host.to(device)


def validate_queued_decode():
    """Reject controls that would silently turn the queued path synchronous."""
    if os.environ.get('TNCA_SUBMIT', 'batched') != 'batched':
        raise ValueError('queued decode requires TNCA_SUBMIT=batched')
    if os.environ.get('TNCA_TRANSFER_POLICY', 'resident') != 'resident':
        raise ValueError('queued decode requires resident device tensors')
    if os.environ.get('TNCA_LAUNCH') == 'skip':
        raise ValueError('queued decode cannot use compile-only skipped launches')
    if os.environ.get('TNCA_GUARD_CHECK') == '1' or int(os.environ.get('TNCA_GUARD_BYTES', '0')):
        raise ValueError('queued decode does not support host staging guards')


def clear_static_attention_cache(cache):
    """Batch complete standard KV clears before clearing length scalars.

    Transformers 5.20.2 resets each layer's keys, values, then its length.
    A host write to every scalar drains the device queue between each pair
    of cache clears. Reordering those independent writes keeps their exact
    final bits while allowing all device clears to queue together. Unknown,
    hybrid and future cache implementations retain their own reset method.
    """
    try:
        import transformers
        from transformers.cache_utils import StaticCache, StaticLayer
    except ImportError:
        return False
    if transformers.__version__ != '5.20.2' or type(cache) is not StaticCache:
        return False
    layers = getattr(cache, 'layers', None)
    if not layers:
        return False
    for layer in layers:
        if type(layer) is not StaticLayer or not getattr(layer, 'is_initialized', False):
            return False
        for tensor in (getattr(layer, 'keys', None), getattr(layer, 'values', None)):
            if not torch.is_tensor(tensor) or not tensor.is_contiguous() or tensor.storage_offset() != 0:
                return False
            nbytes = tensor.numel() * tensor.element_size()
            if (nbytes < 65536 or nbytes > (1 << 31) - (1 << 20) or nbytes % 8
                    or tensor.untyped_storage().nbytes() != nbytes):
                return False
        length = getattr(layer, 'cumulative_length', None)
        if (not torch.is_tensor(length) or length.dtype != torch.int64 or length.numel() != 1
                or not length.is_contiguous() or length.storage_offset() != 0
                or length.untyped_storage().nbytes() != length.element_size()):
            return False
    for layer in layers:
        layer.keys.zero_()
        layer.values.zero_()
    for layer in layers:
        layer.cumulative_length.zero_()
    return True


def _static_attention_lengths(layers, valid_length):
    if not layers or type(valid_length) is not int or valid_length < 0:
        return None
    try:
        from transformers.cache_utils import StaticLayer
    except ImportError:
        return None
    lengths = []
    for layer in layers:
        # Exact type rules out sliding windows and custom recurrent state.
        if type(layer) is not StaticLayer or not getattr(layer, 'is_initialized', False):
            return None
        length = getattr(layer, 'cumulative_length', None)
        keys, values = getattr(layer, 'keys', None), getattr(layer, 'values', None)
        if (not torch.is_tensor(length) or length.numel() != 1 or not length.is_contiguous()
                or length.dtype != torch.int64 or length.storage_offset() != 0
                or length.untyped_storage().nbytes() != length.element_size()
                or not torch.is_tensor(keys) or not torch.is_tensor(values)
                or keys.ndim != 4 or values.ndim != 4
                or keys.device != length.device or values.device != length.device
                or keys.shape[-2] != layer.max_cache_len or values.shape[-2] != layer.max_cache_len
                or valid_length > layer.max_cache_len):
            return None
        lengths.append(length)
    return lengths


def _write_static_lengths(lengths, valid_length):
    value = torch.tensor(valid_length, dtype=torch.int64)
    for length in lengths:
        length.copy_(value)


def _static_tail_views(layers, start, end):
    """Validate complete cache tensors before clearing speculative slots."""
    if end is None:
        return []
    if type(end) is not int or end < start:
        return None
    if end == start:
        return []
    views = []
    for layer in layers:
        if end > layer.max_cache_len:
            return None
        for tensor in (layer.keys, layer.values):
            item = tensor.element_size()
            if (not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or tensor.stride(-1) != 1 or tensor.shape[-1] * item % 8
                    or any(tensor.stride(axis) * item % 8 for axis in (0, 1, 2))
                    or tensor.untyped_storage().nbytes() != tensor.numel() * item):
                return None
            view = tensor[:, :, start:end, :]
            span = (1 + sum((n - 1) * stride for n, stride in zip(view.shape, view.stride()) if n)) * item
            if view.numel() * item > (1 << 31) - (1 << 20) or span > (1 << 31) - (1 << 20):
                return None
            views.append(view)
    return views


def _clear_static_tail(views):
    for view in views:
        if view.numel() == 0:
            continue
        if view.device.type == 'tnca':
            from tnca.torch_device.copy import zero_view_tensor
            zero_view_tensor(view)
        else:
            view.zero_()


def rewind_attention_cache(cache, valid_length, *, dirty_length=None):
    """Restore static KV and clear discarded slots without host tensor reads."""
    if dirty_length is None:
        return False  # Without the high-water mark, stale values cannot be cleared safely.
    layers = getattr(cache, 'layers', None)
    lengths = _static_attention_lengths(layers, valid_length)
    if lengths is None:
        return False
    tails = _static_tail_views(layers, valid_length, dirty_length)
    if tails is None:
        return False
    # Validate every layer before mutating any. Keep the marked static device
    # addresses; replacing the tensors would invalidate compiled graphs.
    _clear_static_tail(tails)
    _write_static_lengths(lengths, valid_length)
    return True


def _linear_prefill_signature(layer):
    """Only a fully initialized, stable standard linear cache is rewindable."""
    count = getattr(layer, 'number_of_states', None)
    if type(count) is not int or not 0 < count <= 16 or getattr(layer, 'record_past', None) is not False:
        return None
    names = ('is_conv_states_initialized', 'is_recurrent_states_initialized', 'has_previous_state',
             'conv_states', 'recurrent_states', 'conv_kernel_size')
    if any(not isinstance(getattr(layer, name, None), dict) for name in names):
        return None
    records = []
    for index in range(count):
        if (layer.is_conv_states_initialized.get(index) is not True
                or layer.is_recurrent_states_initialized.get(index) is not True
                or layer.has_previous_state.get(index) is not True):
            return None
        for name in ('conv_states', 'recurrent_states'):
            tensor = getattr(layer, name).get(index)
            if (not torch.is_tensor(tensor) or not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or tensor.numel() == 0
                    or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
                return None
            records.append((name, index, tensor, tuple(tensor.shape), tuple(tensor.stride()),
                            tensor.dtype, tensor.device, tensor.untyped_storage().data_ptr()))
    return (count, tuple(layer.conv_kernel_size.get(i) for i in range(count)), records)


def snapshot_hybrid_prefill(cache, valid_length):
    """Save standard hybrid recurrent state on-device before one dummy step.

    Unknown/sliding/uninitialized layers and snapshots over 64 MiB fall back
    to the original reset. No tensor contents are read by the host.
    """
    try:
        import transformers
        from transformers.cache_utils import LinearAttentionLayer, StaticCache, StaticLayer
    except ImportError:
        return None
    if transformers.__version__ != '5.20.2' or type(cache) is not StaticCache:
        return None
    layers = getattr(cache, 'layers', None) or []
    statics = [layer for layer in layers if type(layer) is StaticLayer]
    linear = [layer for layer in layers if type(layer) is LinearAttentionLayer]
    if not statics or not linear or len(statics) + len(linear) != len(layers):
        return None
    lengths = _static_attention_lengths(statics, valid_length)
    if lengths is None:
        return None
    signatures = [_linear_prefill_signature(layer) for layer in linear]
    if any(signature is None for signature in signatures):
        return None
    tensors = [record[2] for _, _, records in signatures for record in records]
    required = sum(t.numel() * t.element_size() for t in tensors)
    if (required > 64 * 1024 * 1024
            or len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors)
            or any(t.device != lengths[0].device for t in tensors)):
        return None
    if tensors[0].device.type == 'tnca':
        # Never exhaust the shared device heap for an optional speed path.
        # The allocator metadata is host-side; this does not read tensor data.
        from tnca.torch_device import LAUNCH_RESERVE, MEMORY
        session = MEMORY.session()
        with session._lock:
            free = session._heap.capacity - session._heap.used
        if free - required < max(LAUNCH_RESERVE, 16 * 1024 * 1024):
            return None
    from tnca.errors import BufferError as TncaBufferError
    copies = []
    try:
        for tensor in tensors:
            copies.append(tensor.clone(memory_format=torch.contiguous_format))
    except (TncaBufferError, torch.OutOfMemoryError, MemoryError):
        copies.clear()
        if tensors[0].device.type == 'tnca':
            MEMORY.synchronize()
        return None
    return dict(cache=cache, layers=tuple(layers), statics=statics, lengths=tuple(lengths),
                linear=linear, signatures=signatures, tensors=tensors, copies=copies, valid_length=valid_length)


def restore_hybrid_prefill(snapshot):
    """Restore the saved tensors only after every cache guard still matches."""
    if snapshot is None:
        return False
    current_layers = getattr(snapshot['cache'], 'layers', None) or []
    if (len(current_layers) != len(snapshot['layers']) or
            any(a is not b for a, b in zip(current_layers, snapshot['layers'], strict=True))):
        return False
    lengths = _static_attention_lengths(snapshot['statics'], snapshot['valid_length'])
    if lengths is None or any(a is not b for a, b in zip(lengths, snapshot['lengths'], strict=True)):
        return False
    tails = _static_tail_views(snapshot['statics'], snapshot['valid_length'], snapshot['valid_length'] + 1)
    if tails is None:
        return False
    for layer, before in zip(snapshot['linear'], snapshot['signatures'], strict=True):
        after = _linear_prefill_signature(layer)
        if after is None or before[:2] != after[:2] or len(before[2]) != len(after[2]):
            return False
        for old, new in zip(before[2], after[2], strict=True):
            if old[:2] != new[:2] or old[2] is not new[2] or old[3:] != new[3:]:
                return False
    for tensor, saved in zip(snapshot['tensors'], snapshot['copies'], strict=True):
        tensor.copy_(saved)
    _clear_static_tail(tails)
    _write_static_lengths(lengths, snapshot['valid_length'])
    return True


def run_device_generation(session, processor, spec, inputs, prompt_text, *, new_tokens, per_image,
                          vision_calls, sample, temperature, top_k, top_p, seed, forced_tokens,
                          retain_logits=True, logit_limit=None):
    """Run greedy feedback on the device; copy outputs only after the decode loop."""
    validate_device_sampling('tnca', sample)
    from torch._dynamo.utils import counters
    from tnca.inductor import host_fallbacks

    model = session.model
    eos = L.generation_eos(model)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    if prompt_tokens + new_tokens > session.bucket:
        raise ValueError(f"static bucket {session.bucket} is smaller than prompt {prompt_tokens} plus {new_tokens} tokens")
    placements = {"parameters": {}, "buffers": {}}
    misplaced = []
    for kind, entries in (("parameters", model.named_parameters()), ("buffers", model.named_buffers())):
        for name, value in entries:
            device = str(value.device)
            placements[kind][device] = placements[kind].get(device, 0) + 1
            if value.device.type != "tnca":
                misplaced.append(f"{kind}:{name}:{device}")
    if misplaced:
        raise RuntimeError("model tensors are not on TNCA: " + ", ".join(misplaced))

    # Teacher tokens are transferred before the measured loop, never read back in it.
    teacher = [torch.tensor([[t]], dtype=torch.long).to(session.dev) for t in (forced_tokens or [])]
    rows, device_logits, device_candidates, position_history = [], [], [], []
    queue_steps = os.environ.get('TNCA_QUEUE_STEPS') == '1' and new_tokens > 1
    if queue_steps:
        validate_queued_decode()
    position_batch, host_positions, queued_meters, queue_anchor = None, None, [], None
    code_by_role = getattr(session, "_generation_code", {})
    session._generation_code = code_by_role
    from tnca.code_residency import RepeatedCodeTrace
    if not hasattr(session, '_decode_code_trace'):
        session._decode_code_trace = RepeatedCodeTrace()
    def capture(role, fn):
        return capture_generation_role(role, fn, code_by_role)

    # Compilation can read its own constant-folding scalars. Do that once on
    # a dummy token after populating the prompt cache; record all transfers
    # separately and clear the cache before the actual prompt. Real decode
    # remains subject to the unchanged zero-read rule on every step.
    warmup = None
    if not code_by_role.get("decode") and session.prompts == 0:
        started = time.perf_counter()
        with transfer_meter(True) as warmup_meter:
            warmup_logits = session.warmup_decode(inputs, per_image, capture)
            warmup_meter.mem.synchronize()
        if warmup_logits.device.type != "tnca":
            raise RuntimeError("decode warmup returned a host tensor")
        warmup = dict(seconds=time.perf_counter() - started, transfers=warmup_meter.summary())
        warmup['prefill_retained'] = getattr(session, '_warm_prefill', None) is not None
        warmup['prefill_strategy'] = getattr(session, '_warm_prefill_mode', 'unavailable')

    old_recompile = torch._dynamo.config.error_on_recompile
    session.prompts += 1
    graphs_at_start = counters["stats"]["unique_graphs"]
    try:
        current = None
        for index in range(new_tokens):
            graph_start = counters["stats"]["unique_graphs"]
            code = code_store_counters()
            started = time.perf_counter()
            meter_context = (transfer_meter(True, deferred=True) if queue_steps and index
                             else transfer_meter(True))
            with meter_context as meter:
                if index == 0:
                    prior_prefill_code = len(code_by_role.get("prefill", []))
                    consume = getattr(session, 'consume_warm_prefill', None)
                    saved_prefill = consume(inputs, per_image) if consume is not None else None
                    if saved_prefill is None:
                        logits, calls, reused = capture("prefill", lambda: session.prefill_prompt(inputs, per_image))
                    else:
                        (logits, calls, reused), warm_prefill_seconds = saved_prefill
                    # A chunked prompt can compile the one-token decode graph before
                    # generation starts. Attribute that captured wrapper to decode too.
                    if any(c["kind"] == "token" and c["new_graphs"] for c in calls):
                        code_by_role.setdefault("decode", []).extend(
                            code_by_role["prefill"][prior_prefill_code:])
                else:
                    # A forced token is a test input; otherwise feedback remains on TNCA.
                    feed = teacher[index - 1] if index - 1 < len(teacher) else current
                    with session._decode_code_trace.step():
                        logits = capture("decode", lambda: session.step_device(
                            feed, position_batch[index - 1], host_positions[index - 1])
                            if queue_steps else session.step_device(feed))
                current = capture("selection", lambda: session.greedy(logits.reshape(-1)))
                if queue_steps and index == 0:
                    host_positions, position_batch = stage_decode_positions(session.last_position, new_tokens, session.dev)
                if not queue_steps or index == 0:
                    meter.mem.synchronize()
            if queue_steps and index == 0:
                queue_anchor = time.time()
            if logits.device.type != "tnca" or current.device.type != "tnca":
                raise RuntimeError("device model or token selection returned a host tensor")
            position_history.append(getattr(session, 'last_position', None))
            elapsed = time.perf_counter() - started
            if retain_logits and (logit_limit is None or len(device_logits) < logit_limit):
                device_logits.append(logits)
            device_candidates.append(current)
            row = dict(step=index, kind="prefill" if index == 0 else "decode",
                       input_tokens=prompt_tokens - reused if index == 0 else 1,
                       seconds=elapsed, new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                       code_store=_counter_delta(code, code_store_counters()), logits_download_bytes=0,
                       transfers_inside_step=None if queue_steps and index else meter.summary())
            if index == 0:
                row.update(calls=calls, reused_cached_tokens=reused)
                row['prefill_from_warmup'] = saved_prefill is not None
                if saved_prefill is not None:
                    row['prefill_seconds_in_warmup'] = warm_prefill_seconds
            rows.append(row)
            if queue_steps and index:
                queued_meters.append((row, meter))
        if queued_meters:
            # All input positions and token feedback are already resident.
            # One final synchronization replaces a host wait after each step.
            queued_meters[-1][1].mem.synchronize()
            previous = queue_anchor
            for row, meter in queued_meters:
                row['transfers_inside_step'] = meter.summary()
                complete = meter.log.completion_time
                if complete is None or complete < previous:
                    raise RuntimeError('queued decode lacks ordered completion timestamps')
                row['seconds'] = complete - previous
                previous = complete
    except BaseException:
        if queue_steps and position_batch is not None:
            try:
                meter.mem.synchronize()
            except Exception:
                pass
        raise
    finally:
        torch._dynamo.config.error_on_recompile = old_recompile

    # This is output retrieval after generation, reported separately and not hidden in the step counters.
    retrieval_start = time.perf_counter()
    with transfer_meter(True) as retrieval:
        candidates = []
        for candidate in device_candidates:
            value = int(candidate.cpu().item())
            candidates.append(value)
            # Candidate feedback has already completed on the device. Once
            # EOS is retrieved, later candidates cannot appear in the reply.
            if not forced_tokens and value in eos:
                break
        tokens = [forced_tokens[i] if forced_tokens and i < len(forced_tokens) else t
                  for i, t in enumerate(candidates)]
        # The device loop can execute beyond EOS. Those logits are discarded
        # from the public result, so do not transfer entire vocabulary vectors
        # for them. This is after generation and adds no steady-decode reads.
        logits_rows = [t.cpu() for t in device_logits[:len(tokens)]]
        retrieval.mem.synchronize()
    retrieval_seconds = time.perf_counter() - retrieval_start
    for i, row in enumerate(rows):
        row.update(token=(tokens[i] if i < len(tokens) else None),
                   candidate_token=(candidates[i] if i < len(candidates) else None),
                   greedy_token=(candidates[i] if i < len(candidates) else None))
    cache_rewind = 'not-needed'
    retained_prefix_tokens = 0
    cache_maintenance_start = time.perf_counter()
    if len(tokens) != new_tokens:
        # Rewind only caches whose full state is the standard static KV cache.
        # Hybrid/recurrent caches retain the original conservative reset.
        position = position_history[len(tokens) - 1] if tokens else None
        valid_length = len(session.cached) + max(0, len(tokens) - 1)
        expected_length = len(session.cached) + max(0, len(rows) - 1)
        if (position is not None and not forced_tokens and tokens and tokens[-1] in eos
                and getattr(session, '_cache_tokens', None) == expected_length
                and rewind_attention_cache(session.cache, valid_length, dirty_length=expected_length)):
            session.cached += tokens[:-1]
            session.last_position = position
            session._cache_tokens = valid_length
            cache_rewind = 'static-attention-prefix-retained'
            retained_prefix_tokens = len(session.cached)
        else:
            defer_reset = getattr(session, 'defer_reset', None)
            if defer_reset is None or os.environ.get('TNCA_DEFER_CACHE_RESET', '1') == '0':
                session.reset()
                cache_rewind = 'full-reset'
            else:
                defer_reset()
                cache_rewind = 'full-reset-deferred'
    else:
        session.cached += tokens[:-1]
    cache_maintenance_seconds = time.perf_counter() - cache_maintenance_start
    fallbacks = generation_fallbacks(session, code_by_role, host_fallbacks)
    covered = all(code_by_role.get(role) for role in ("prefill", "decode", "selection"))
    fallback = any(fallbacks.values()) if covered else None
    vision_unknown = any(c.get("compiled_wrapper_fallbacks") is None for c in (vision_calls or []))
    vision_fallback = any(c.get("compiled_wrapper_fallbacks") for c in (vision_calls or []))
    read_during_decode = any((row["transfers_inside_step"]["host_read_calls"] or
                              row["transfers_inside_step"]["staging_download_bytes"])
                             for row in rows if row["kind"] == "decode")
    if fallback or vision_fallback or read_during_decode:
        status = "failed-no-host-fallback"
    elif fallback is None or vision_unknown:
        status = "unverified-no-host-fallback"
    else:
        status = "pass"
    weight_bytes = L.module_bytes(model, lambda name: not name.startswith("model.visual."))
    stats = generation_stats(rows)
    stats.update(generated_tokens=len(tokens), executed_generation_steps=len(rows),
                 prefill_tokens_per_second=(rows[0]['input_tokens'] / stats['prefill_seconds']
                                            if stats['prefill_seconds'] > 0 else None),
                 returned_token_ids_per_generation_second=(len(tokens) / stats['generation_seconds']
                                                         if stats['generation_seconds'] > 0 else None),
                 decode_rate_scope='executed decode steps, including any work after EOS',
                 returned_rate_scope='returned token IDs including EOS; generation wall only, excludes output retrieval')
    if queue_steps:
        stats['decode_time_scope'] = 'successive device completion times after one queued submission span'
        stats['decode_submission'] = 'queued-resident-device-feedback'
    if rows[0].get('prefill_from_warmup'):
        stats['returned_rate_scope'] = ('generation loop after prefill retained from warmup; '
                                        'excludes warmup and output retrieval')
    clock_path = os.environ.get('TNCA_OPERATING_CLOCK_EVIDENCE')
    try:
        clock_hz = int(json.loads(Path(clock_path).read_text())['clock_hz']) if clock_path else None
    except (OSError, ValueError, KeyError, TypeError):
        clock_hz = None
    decode_rows = [row for row in rows if row['kind'] == 'decode']
    cycles = [row['transfers_inside_step']['device_cycles'] for row in decode_rows]
    if clock_hz and clock_hz > 0 and cycles and all(c is not None for c in cycles) and stats['decode_seconds'] > 0:
        stats['decode_device_cycles'] = sum(cycles)
        stats['decode_estimated_non_device_wall_fraction'] = 1 - sum(cycles) / clock_hz / stats['decode_seconds']
        stats['host_share_caveat'] = ('Wall residual after summed device cycles at the qualified clock; includes '
                                      'host work, transfers, waits and possible counter overlap, not host-only time')
    report = dict(schema="tnca-qwen-chat-generation/2", status=status, model=spec.name,
                  repo=spec.repo, revision=spec.revision, device="tnca",
                  decoder_layers=model.config.get_text_config().num_hidden_layers,
                  prompt_text=prompt_text, prompt_tokens=prompt_tokens, bucket=session.bucket,
                  prefill_chunk=session.chunk, requested_new_tokens=new_tokens,
                  sampling=dict(sample=sample, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed),
                  forced_tokens=forced_tokens, generated_tokens=tokens, candidate_tokens=candidates,
                  generated_text=processor.batch_decode(torch.tensor(tokens)[None], skip_special_tokens=True)[0],
                  stopped_at_eos=bool(tokens and tokens[-1] in eos), eos_token_ids=sorted(eos),
                  images=len(per_image or []), vision_calls=vision_calls or [],
                  vision_seconds=sum(c["seconds"] for c in vision_calls or []), steps=rows,
                  statistics=stats, logits=logits_rows, logits_retained=retain_logits,
                  logit_limit=logit_limit,
                  fp8_route_in_use=L.fp8_route_in_use(),
                  decode_weight_bytes=weight_bytes,
                  session=dict(prompt_index=session.prompts, cache_resets=session.resets,
                               cache_rewind=cache_rewind, retained_prefix_tokens=retained_prefix_tokens,
                               cache_maintenance_seconds=cache_maintenance_seconds,
                               new_graphs_this_prompt=counters["stats"]["unique_graphs"] - graphs_at_start,
                               unique_graphs_total=counters["stats"]["unique_graphs"]),
                  device_memory=L.device_memory(session.dev), cache_bytes=L.cache_bytes(session.cache),
                  standard_torch_fullgraph=True, cpu_numerical_fallback=fallback,
                  compiled_wrapper_fallbacks=fallbacks, compiled_wrapper_roles_covered=covered,
                  decode_warmup=warmup,
                  steady_decode_host_reads_detected=bool(read_during_decode),
                  model_tensor_placements=placements, output_retrieval=retrieval.summary(),
                  output_retrieval_seconds=retrieval_seconds,
                  executed_generation_steps=len(rows), returned_tokens=len(tokens),
                  eos_overrun_steps=len(rows) - len(tokens) if not forced_tokens else None,
                  eos_stop_observed=(len(rows) == len(tokens) if not forced_tokens and tokens and tokens[-1] in eos
                                     else None), device_target=device_target())
    report["theoretical_core_port_bound"] = dict(
        bytes_per_second=CORE_PORT_BYTES_PER_SECOND, source=CORE_PORT_SOURCE,
        decode_weight_bytes=weight_bytes,
        weight_stream_ceiling_tokens_per_second=CORE_PORT_BYTES_PER_SECOND / weight_bytes,
        interpretation="Theoretical interface ceiling only, not a measured bandwidth or performance prediction")
    return report


def run_generation(session, processor, spec, inputs, prompt_text, *, new_tokens, per_image=None,
                   vision_calls=None, sample=False, temperature=1.0, top_k=0, top_p=1.0, seed=0,
                   forced_tokens=None, retain_logits=True, logit_limit=None):
    validate_sampling(temperature, top_k, top_p)
    if new_tokens < 1:
        raise ValueError("new-tokens must be positive")
    if forced_tokens is not None and not 0 < len(forced_tokens) <= new_tokens:
        raise ValueError("forced tokens must be nonempty and no longer than new-tokens")
    if logit_limit is not None and (type(logit_limit) is not int or logit_limit < 0):
        raise ValueError('logit-limit must be a nonnegative integer or None')
    if session.device == "tnca":
        return run_device_generation(session, processor, spec, inputs, prompt_text, new_tokens=new_tokens,
                                     per_image=per_image, vision_calls=vision_calls, sample=sample,
                                     temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
                                     forced_tokens=forced_tokens, retain_logits=retain_logits,
                                     logit_limit=logit_limit)
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
        # The logits download for host-side token selection is the step's output; it is counted separately from
        # the transfers inside the model call, and its time is inside the step's wall time.
        logits_bytes = logits.numel() * logits.element_size() if tnca else 0
        logits = logits.cpu()
        elapsed = time.perf_counter() - started
        token, greedy = select_token(logits, sample, temperature, top_k, top_p, generator)
        candidate_tokens.append(token)
        tokens.append(forced_tokens[0] if forced_tokens else token)
        if retain_logits and (logit_limit is None or len(logits_rows) < logit_limit):
            logits_rows.append(logits)
        rows.append(dict(step=0, kind="prefill", input_tokens=prompt_tokens - reused, reused_cached_tokens=reused,
                         seconds=elapsed, calls=calls,
                         new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                         token=tokens[-1], candidate_token=token, greedy_token=greedy,
                         code_store=_counter_delta(code, code_store_counters() if tnca else None),
                         logits_download_bytes=logits_bytes,
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
            logits_bytes = logits.numel() * logits.element_size() if tnca else 0
            logits = logits.cpu()
            elapsed = time.perf_counter() - started
            token, greedy = select_token(logits, sample, temperature, top_k, top_p, generator)
            candidate_tokens.append(token)
            forced = forced_tokens[index] if forced_tokens and index < len(forced_tokens) else None
            tokens.append(token if forced is None else forced)
            if retain_logits and (logit_limit is None or len(logits_rows) < logit_limit):
                logits_rows.append(logits)
            rows.append(dict(step=index, kind="decode", input_tokens=1, seconds=elapsed,
                             new_graphs=counters["stats"]["unique_graphs"] - graph_start,
                             token=tokens[-1], candidate_token=token, greedy_token=greedy,
                             code_store=_counter_delta(code, code_store_counters() if tnca else None),
                             logits_download_bytes=logits_bytes,
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
                  logits=logits_rows, logits_retained=retain_logits, logit_limit=logit_limit,
                  fp8_route_in_use=L.fp8_route_in_use(),
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


def emit_user_report(report, prompt_started):
    """Return text first, then serialize the detailed diagnostic fields."""
    generated = report["generated_text"]
    timing = report["user_visible_timing"]
    sys.stdout.write('{\n  "generated_text": ' + json.dumps(generated))
    sys.stdout.flush()
    finalize_user_visible_metrics(timing, time.perf_counter() - prompt_started,
                                  "module entry through generated-text stdout flush; excludes interpreter startup")
    data = public_report(report)
    data.pop("generated_text")
    data.pop("user_visible_timing")
    if data:
        body = json.dumps(data, indent=2)[1:-1].strip('\n')
        sys.stdout.write(',\n' + body)
    sys.stdout.write(",\n  \"user_visible_timing\": " + json.dumps(timing, indent=2) + "\n}\n")
    sys.stdout.flush()


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
        validate_device_sampling(args.device, args.sample)
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

    def generate(self, prompt, images=(), forced=None, keep_history=False, check=True, prompt_started=None,
                 retain_logits=True):
        if prompt_started is None:
            prompt_started = time.perf_counter()
        from torch._dynamo.utils import compilation_time_metrics, counters
        compilation_before = compiler_snapshot(compilation_time_metrics, counters)
        reference_seconds = 0.0
        args = self.args
        messages = (self.history if keep_history else []) + [user_message(prompt, images)]
        inputs, text = make_inputs(self.spec, self.processor, messages)
        if keep_history:
            inputs = self.continuation(messages, inputs, text) or inputs
        reference = scale_floor = None
        if check and args.check_tokens and forced is None:
            reference_started = time.perf_counter()
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
                                        per_image=per_image_cpu, forced_tokens=forced,
                                        logit_limit=len(reference['logits']))
            scale_floor = compare_logits(reference["logits"], reciprocal["logits"])
            del cpu
            gc.collect()
            reference_seconds = time.perf_counter() - reference_started
        session = self.device_session()
        session.wait_for_cleanup()
        per_image, vision_calls = session.vision_features(inputs)
        # Teacher-forced diagnostics need only the reference's actual steps;
        # ordinary chat does not need to download any vocabulary vectors.
        required_logits = len(reference['logits']) if reference is not None else 0
        effective_retention = retain_logits or required_logits > 0
        logit_limit = None if retain_logits else required_logits
        result = run_generation(session, self.processor, self.spec, inputs, text, new_tokens=args.new_tokens,
                                per_image=per_image, vision_calls=vision_calls, sample=args.sample,
                                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                seed=args.seed, forced_tokens=forced,
                                retain_logits=effective_retention, logit_limit=logit_limit)
        result["model_scope"] = "reduced-depth-proof" if args.layers else "full-checkpoint"
        result["reduced_depth_layers"] = args.layers
        result["load"] = self.load_report
        result["versions"] = L.versions()
        if reference is not None:
            comparison = compare_logits(reference["logits"], result["logits"][:required_logits])
            # Default: the fixed --max-rel-l2 limit (the rule every run before 2026-10-28 is judged by; a
            # floor-derived limit used after TLM q35-v1 failed was withdrawn, evidence/r41/hbm_bw/AUDIT.md R6).
            # --cpu-floor-multiplier K > 0 selects the rule registered on 2026-10-28 before any run that uses it
            # (author decision, spec/r41/chat/s2-qwen-chat-v3.json "tolerance_rule"): per step
            # rel_l2 <= max(max_rel_l2, K x the CPU reference's own reciprocal-vs-IEEE scale-rounding floor).
            # --require-argmax adds that the device argmax equal the reference argmax at every step.
            for row, floor in zip(comparison, scale_floor, strict=True):
                row["cpu_scale_rounding_floor"] = floor["rel_l2"]
                row["allowed_rel_l2"] = max(args.max_rel_l2, args.cpu_floor_multiplier * floor["rel_l2"])
            passed = all(row["rel_l2"] <= row["allowed_rel_l2"] and (row["argmax_equal"] or not args.require_argmax)
                         for row in comparison)
            rule = ("fixed max_rel_l2" if args.cpu_floor_multiplier == 0 else
                    "max(max_rel_l2, cpu_floor_multiplier x CPU scale-rounding floor)") + \
                   (" and argmax equal" if args.require_argmax else "")
            result["teacher_forced_cpu_w8a8"] = dict(
                reference_tokens=reference["generated_tokens"], checked_tokens=len(comparison),
                max_allowed_rel_l2=args.max_rel_l2, cpu_floor_multiplier=args.cpu_floor_multiplier,
                require_argmax=args.require_argmax, rule=rule, passed=passed, steps=comparison,
                interpretation="Reference tokens feed the device; the gate is the rule named in 'rule', fixed before the run")
            if not passed:
                result["status"] = "failed-teacher-forced-tolerance"
        if keep_history:
            self.history = messages + [dict(role="assistant", content=[dict(type="text",
                                                                              text=result["generated_text"])])]
            generated = self.processor.batch_decode(torch.tensor(result["generated_tokens"])[None],
                                                    skip_special_tokens=False)[0]
            self.transcript = (text + generated, inputs["input_ids"][0].tolist() + result["generated_tokens"])
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        result["user_visible_timing"] = user_visible_metrics(
            result, time.perf_counter() - prompt_started,
            getattr(tokenizer, "all_special_ids", ()) or (), reference_seconds)
        result["user_visible_timing"]["compiler"] = compiler_delta(
            compilation_before, compilation_time_metrics, counters)
        return result

    def session_check(self, first, first_prompt, first_images, prompt):
        """Persistent-session check: in the same process, session and cache, run a different conversation and
        then the first prompt again with the first run's tokens forced. The repeat must reproduce the first run's
        tokens and every logit bit for bit (code-store reloads, allocator reuse and cache reset are exercised
        in between), and must compile nothing new."""
        # The intervening prompt exercises cache eviction, but its logits
        # are not compared. Keep the bitwise check on the first/repeated
        # prompt while avoiding an unnecessary vocabulary download here.
        other = self.generate(prompt, check=False, retain_logits=False)
        again = self.generate(first_prompt, first_images, forced=first["generated_tokens"])
        mismatched = [i for i, (x, y) in enumerate(zip(first["logits"], again["logits"], strict=True))
                      if not torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x,
                                         y.view(torch.int16) if y.dtype == torch.bfloat16 else y)]
        passed = (not mismatched and again["candidate_tokens"] == first["candidate_tokens"] and
                  all(r["status"] == "pass" for r in (first, other, again)))
        return dict(passed=passed, other_prompt=prompt, other=public_report(other), repeat=public_report(again),
                    bitwise_logit_mismatch_steps=mismatched,
                    repeat_new_graphs=again["session"]["new_graphs_this_prompt"],
                    code_store_evictions=sum((r.get("code_store") or {}).get("evicted", 0)
                                             for r in other["steps"] + again["steps"]),
                    interpretation="same-session repeat after an intervening conversation; bitwise equality of "
                                   "logits and candidate tokens is required")

    def repl(self, lines=None, transcript=None, on_turn=None):
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
                try:
                    validate_device_sampling(self.args.device, True)
                except ValueError as error:
                    print('error>', error, flush=True)
                    continue
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
            prompt_started = time.perf_counter()
            try:
                report = self.generate(line, open_images(images), keep_history=True, prompt_started=prompt_started,
                                       retain_logits=False)
            except ValueError as error:          # e.g. the conversation no longer fits the static bucket
                print("error>", error, "(use /reset)", flush=True)
                if transcript is not None:
                    transcript.append(dict(user=line, status="refused", error=str(error)))
                    if on_turn is not None:
                        on_turn(transcript[-1])
                    return False
                continue
            images = []
            if transcript is not None:
                transcript.append(dict(user=line, assistant=report["generated_text"], report=public_report(report)))
            if report["status"] != "pass":
                print("error> generation failed or lacks device evidence:", report["status"], flush=True)
                if on_turn is not None and transcript is not None:
                    on_turn(transcript[-1])
                return False
            print("assistant>", report["generated_text"], flush=True)
            finalize_user_visible_metrics(report["user_visible_timing"], time.perf_counter() - prompt_started,
                                          "interactive prompt through assistant stdout flush")
            if source is None and self.session is not None:
                self.session.schedule_idle_cleanup()
            stats = report["statistics"]
            visible = report["user_visible_timing"]
            print(json.dumps({k: stats[k] for k in ("prefill_seconds", "decode_steps",
                                                     "mean_decode_seconds_per_token",
                                                     "decode_tokens_per_second")}
                             | {"reused_cached_tokens": report["steps"][0]["reused_cached_tokens"],
                                "new_graphs": report["session"]["new_graphs_this_prompt"],
                                "prompt_to_output_seconds": visible["prompt_to_return_seconds"],
                                "returned_token_ids_per_second": visible["returned_token_ids_per_second"],
                                "non_special_token_ids_per_second": visible["non_special_token_ids_per_second"]}),
                  flush=True)
            if on_turn is not None and transcript is not None:
                on_turn(transcript[-1])
        if getattr(self, 'session', None) is not None:
            self.session.wait_for_cleanup()
        return True


def user_message(prompt, images=()):
    return dict(role="user", content=[dict(type="image", image=image) for image in images]
                + [dict(type="text", text=prompt)])


def open_images(paths):
    if not paths:
        return []
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
    ap.add_argument("--cpu-floor-multiplier", type=float, default=0.0,
                    help="0 (default): the fixed limit alone; K > 0: the 2026-10-28 registered rule, limit "
                         "max(max-rel-l2, K x the CPU reference's own scale-rounding floor)")
    ap.add_argument("--require-argmax", action="store_true",
                    help="also require the device argmax to equal the reference argmax at every checked step")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="prefill text in blocks of up to this many tokens; the last block may be shorter. "
                         "0 prefills each prompt whole")
    ap.add_argument("--session-check", metavar="PROMPT",
                    help="after the one-shot prompt, run PROMPT and then the first prompt again in the same "
                         "session and require bitwise-identical logits and tokens")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tnca-tree")
    ap.add_argument("--out", type=Path)
    return ap


def main(argv=None):
    prompt_started = _MODULE_STARTED if __name__ == '__main__' else time.perf_counter()
    args = parser().parse_args(argv)
    validate_sampling(args.temperature, args.top_k, args.top_p)
    if args.check_tokens < 0 or args.check_tokens > args.new_tokens:
        raise ValueError("check-tokens must be in 0..new-tokens")
    if args.max_rel_l2 < 0 or args.cpu_floor_multiplier < 0:
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
        # Ordinary one-shot chat returns text, not vocabulary vectors. Keep
        # logits only when an explicit comparison or artifact needs them.
        needs_logits = bool(args.out or args.session_check)
        report = app.generate(args.prompt, images, prompt_started=prompt_started,
                              retain_logits=needs_logits)
        if args.session_check:
            check = app.session_check(report, args.prompt, images, args.session_check)
            report["persistent_session_check"] = check
            if not check["passed"] and report["status"] == "pass":
                report["status"] = "failed-persistent-session-check"
        # Flush the generated answer before writing optional diagnostics;
        # saving vocabulary tensors must not delay the user-visible reply.
        emit_user_report(report, prompt_started)
        save_report(report, args.out)
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
